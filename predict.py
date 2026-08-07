import argparse
import gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from irg import save_raw
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader import create_dataloader
from main import get_training_mode, gpu_setting, prepare_data_dict
from models import build_unet
from trainer import CustomModel


def load_checkpoint(checkpoint_path, cfg) -> CustomModel:
    # 保存済みモデル全体を復元すると、推論には不要なoptimizerとslot変数も
    # メモリに載る。output.yamlからネットワークだけを構築し、重みだけを読む。
    input_shape = tuple(cfg.aug.crop_size_zyx) + (
        cfg.model.input_num_channel + cfg.model.num_channel,
    )
    model = build_unet(
        CustomModel,
        input_shape,
        cfg.model.num_channel,
        **cfg.model.unet,
        **cfg.model.renorm,
    )
    model.load_weights(checkpoint_path)
    return model


def enable_gpu_memory_growth():
    """Avoid reserving nearly all GPU memory before the first inference."""
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as error:
            # TensorFlowがすでにdeviceを初期化していた場合も推論は継続する。
            logging.warning(f"Could not enable memory growth for {device}: {error}")


def reverse_normalize_img(img, min_val, max_val):
    img = img * (max_val - min_val)
    img = img + min_val
    return img


def to_int16_img(img):
    return np.rint(img).astype(np.int16)


def concat_comparison_img(img_list, separator_width=4):
    separator_shape = list(img_list[0].shape)
    separator_shape[2] = separator_width
    separator = np.zeros(separator_shape, dtype=img_list[0].dtype)
    out = []
    for img in img_list:
        if len(out) > 0:
            out.append(separator)
        out.append(img)
    return np.concatenate(out, axis=2)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str, help="gpu num (default 0)")
    parser.add_argument(
        "--no-gpu-allow-growth",
        action="store_false",
        dest="gpu_allow_growth",
        default=True,
        help="GPUメモリの段階確保を無効化する（推論時は既定で有効）",
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="I2I-RFRのEuler更新回数。未指定なら学習時のoutput.yamlを使う",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="I2I-RFRのt下限。未指定なら学習時のoutput.yamlを使う",
    )
    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument(
        "--clip-output", action="store_true", help="推論出力を0-1にclipする"
    )
    clip_group.add_argument(
        "--no-clip-output", action="store_true", help="推論出力の0-1 clipを無効化する"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="推論データローダの並列数。OOM時は1を推奨",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=1,
        help="推論データローダのprefetch数。OOM時は1を推奨",
    )
    parser.add_argument(
        "--crop-size-zyx",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="推論時のcropサイズを上書きする。GPU OOM時は小さくする",
    )
    parser.add_argument(
        "--no-save-comparison",
        action="store_true",
        help="input/output/target結合画像を保存しない",
    )
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path

    # 実験時の設定ファイルを読み込む
    cfg_path = checkpoint_path.parents[1] / "output.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.batch_size = 1
    cfg.num_workers = args.num_workers
    cfg.prefetch_size = args.prefetch_size
    cfg.debug_dataloader = True
    if args.crop_size_zyx is not None:
        if any(size <= 0 for size in args.crop_size_zyx):
            parser.error("--crop-size-zyxには正の整数を指定してください")
        cfg.aug.crop_size_zyx = list(args.crop_size_zyx)
        downsample_factor = np.power(
            np.asarray(cfg.model.unet.pool_size_zyx, dtype=np.int64),
            int(cfg.model.unet.depth),
        )
        if np.any(np.asarray(args.crop_size_zyx) % downsample_factor):
            parser.error(
                "--crop-size-zyxはUNetのdownsample倍率 "
                f"{downsample_factor.tolist()} で割り切れる値にしてください"
            )
    if args.inference_steps is not None:
        cfg.i2i_rfr.inference_steps = args.inference_steps
    if args.t_min is not None:
        cfg.i2i_rfr.t_min = args.t_min
    if args.clip_output:
        cfg.i2i_rfr.clip_output = True
    if args.no_clip_output:
        cfg.i2i_rfr.clip_output = False

    # 保存場所を作成
    save_dir = checkpoint_path.parents[1] / "preds"
    save_dir.mkdir(exist_ok=True)

    # テスト時に使うGPUを設定
    gpu_setting(args.gpu, args.gpu_allow_growth)
    if args.gpu_allow_growth:
        enable_gpu_memory_growth()

    # データを準備
    val_dict = prepare_data_dict(
        cfg.source_data_dir, cfg.target_data_dir, training_mode=get_training_mode(cfg)
    )[1]
    test_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    # モデルを読み込む
    model = load_checkpoint(checkpoint_path, cfg)
    model.cfg = cfg
    logging.info(f"Loaded from: {checkpoint_path} (optimizer state skipped)")

    spacing_zyx = np.array(cfg.aug.affine.norm_spacing_zyx, np.float32)
    for data in tqdm(test_loader):
        pred = model.predict_step(data).numpy()
        source = data["imgs"].numpy()
        target = data.get("target_imgs")
        if target is not None:
            target = target.numpy()
        keys = [key.decode() for key in data["img_hdr_list"].numpy()]
        target_min_clip_vals = data["target_min_clip_vals"].numpy()
        target_max_clip_vals = data["target_max_clip_vals"].numpy()

        for idx, key in enumerate(keys):
            source_img = source[idx, :, :, :, 0]
            source_img = to_int16_img(source_img)
            save_raw(source_img, spacing_zyx, save_dir / f"{key}.input.hdr")

            pred_img = pred[idx, :, :, :, 0]
            pred_img = reverse_normalize_img(
                pred_img, target_min_clip_vals[idx], target_max_clip_vals[idx]
            )
            pred_img = to_int16_img(pred_img)
            save_raw(pred_img, spacing_zyx, save_dir / f"{key}.hdr")

            comparison_img_list = [source_img, pred_img]
            if target is not None:
                target_img = target[idx, :, :, :, 0]
                target_img = to_int16_img(target_img)
                save_raw(target_img, spacing_zyx, save_dir / f"{key}.target.hdr")
                comparison_img_list.append(target_img)

            if not args.no_save_comparison:
                comparison_img = concat_comparison_img(comparison_img_list)
                save_raw(
                    comparison_img, spacing_zyx, save_dir / f"{key}.comparison.hdr"
                )

        del pred, source, target
        gc.collect()
