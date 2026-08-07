import argparse
import gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw, save_raw
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from data.dataloader import create_dataloader
from data.utils import calculate_intensity
from main import get_training_mode, gpu_setting, prepare_data_dict
from models import build_unet
from sliding_window import resample_volume, sliding_window_inference
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


def _to_data_dir_list(data_dir):
    if isinstance(data_dir, (list, tuple, ListConfig)):
        return [Path(path) for path in data_dir]
    return [Path(data_dir)]


def _find_volume_hdr_paths(data_dir):
    hdr_paths = []
    for raw_path in sorted(data_dir.rglob("*.raw")):
        if raw_path.name.endswith(".mask.raw"):
            continue
        hdr_path = raw_path.with_suffix(".hdr")
        if hdr_path.exists():
            hdr_paths.append(hdr_path)
        else:
            logging.warning(f"hdrがないためスキップします: {raw_path}")
    return hdr_paths


def _intensity_range(img, cfg):
    if cfg.image.modality == "MR":
        return calculate_intensity(
            img.astype(np.float32),
            cfg.image.MR.min_percentile,
            cfg.image.MR.max_percentile,
        )
    window_level = float(cfg.image.CT.window_level)
    window_width = float(cfg.image.CT.window_width)
    return window_level - window_width / 2, window_level + window_width / 2


def predict_full_volumes(model, cfg, save_dir, overlap, seed):
    source_data_dirs = _to_data_dir_list(cfg.source_data_dir)
    window_size_zyx = tuple(int(size) for size in cfg.aug.crop_size_zyx)
    model_spacing_zyx = np.asarray(cfg.aug.affine.norm_spacing_zyx, np.float32)
    multiple_roots = len(source_data_dirs) > 1
    volume_count = 0

    for source_data_dir in source_data_dirs:
        if not source_data_dir.is_dir():
            raise FileNotFoundError(source_data_dir)
        hdr_paths = _find_volume_hdr_paths(source_data_dir)
        if not hdr_paths:
            raise FileNotFoundError(
                f"{source_data_dir} 以下に使用可能な.hdr/.raw画像が見つかりません"
            )

        for hdr_path in hdr_paths:
            size_zyx, _, source_spacing_zyx = read_hdr(hdr_path)
            source_spacing_zyx = np.asarray(source_spacing_zyx, np.float32)
            source_img = np.asarray(read_raw(hdr_path))
            if tuple(source_img.shape) != tuple(size_zyx):
                raise ValueError(
                    f"headerとrawのsizeが一致しません: {hdr_path}: "
                    f"{size_zyx} != {source_img.shape}"
                )

            min_clip_val, max_clip_val = _intensity_range(source_img, cfg)
            needs_rescale = not np.allclose(
                source_spacing_zyx, model_spacing_zyx, rtol=1e-4, atol=1e-5
            )
            if needs_rescale:
                model_size_zyx = (
                    np.rint(
                        (np.asarray(source_img.shape) - 1)
                        * source_spacing_zyx
                        / model_spacing_zyx
                    ).astype(np.int32)
                    + 1
                )
                model_size_zyx = np.maximum(model_size_zyx, 1)
                model_img = resample_volume(source_img, model_size_zyx)
            else:
                model_img = source_img

            padding_value = np.uint16(1 << int(cfg.bit_info.padding_bit))

            def predict_patch(image_patch, valid_patch, initial_noise):
                mask_patch = np.where(valid_patch, 0, padding_value).astype(np.uint16)
                data = {
                    "imgs": tf.convert_to_tensor(
                        image_patch[None, ..., None], tf.float32
                    ),
                    "msks": tf.convert_to_tensor(
                        mask_patch[None, ..., None], tf.uint16
                    ),
                    "min_clip_vals": tf.convert_to_tensor([min_clip_val], tf.float32),
                    "max_clip_vals": tf.convert_to_tensor([max_clip_val], tf.float32),
                }
                prediction = model.predict_step(
                    data,
                    initial_noise=tf.convert_to_tensor(initial_noise[None], tf.float32),
                )
                return prediction.numpy()[0]

            relative_path = hdr_path.relative_to(source_data_dir)
            description = str(relative_path.with_suffix(""))
            logging.info(
                f"Sliding-window inference: {hdr_path}, "
                f"size={model_img.shape}, window={window_size_zyx}, overlap={overlap}"
            )
            prediction = sliding_window_inference(
                model_img,
                window_size_zyx,
                overlap,
                predict_patch,
                num_output_channels=int(cfg.model.num_channel),
                seed=seed + volume_count,
                progress=lambda positions: tqdm(positions, desc=description),
            )[..., 0]
            prediction = reverse_normalize_img(prediction, min_clip_val, max_clip_val)
            prediction = to_int16_img(prediction)

            if needs_rescale:
                prediction = resample_volume(prediction, source_img.shape)
                prediction = to_int16_img(prediction)

            output_relative_path = relative_path
            if multiple_roots:
                output_relative_path = Path(source_data_dir.name) / relative_path
            output_path = (save_dir / output_relative_path).with_suffix(".hdr")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_raw(prediction, source_spacing_zyx, output_path)
            logging.info(f"Saved full-volume prediction: {output_path}")
            volume_count += 1

    return volume_count


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str, help="gpu num (default 0)")
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=None,
        help="推論対象のsourceフォルダ。未指定ならoutput.yamlの設定を使う",
    )
    parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=None,
        help="通常crop推論のpaired target。sliding-windowでは使用しない",
    )
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="入力画像全体をoverlap付きsliding-windowで推論する",
    )
    parser.add_argument(
        "--window-overlap",
        type=float,
        default=0.5,
        help="sliding-windowのoverlap率。0以上1未満（default: 0.5）",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="全volumeで共有するI2I-RFR初期ノイズのseed"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="出力先。未指定時はpredsまたはpreds_full",
    )
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
    training_mode = get_training_mode(cfg)
    if (
        not args.sliding_window
        and training_mode == "paired"
        and bool(args.source_data_dir) != bool(args.target_data_dir)
    ):
        parser.error(
            "pairedモードでデータフォルダを上書きする場合は、"
            "--source-data-dirと--target-data-dirを両方指定してください"
        )
    if training_mode == "self_supervised_deblur" and args.target_data_dir is not None:
        parser.error("self_supervised_deblurモードでは--target-data-dirは使用しません")
    if not 0 <= args.window_overlap < 1:
        parser.error("--window-overlapは0以上1未満にしてください")
    if args.source_data_dir is not None:
        if not args.source_data_dir.is_dir():
            parser.error(f"sourceフォルダが見つかりません: {args.source_data_dir}")
        cfg.source_data_dir = str(args.source_data_dir)
    if args.target_data_dir is not None:
        if not args.target_data_dir.is_dir():
            parser.error(f"targetフォルダが見つかりません: {args.target_data_dir}")
        cfg.target_data_dir = str(args.target_data_dir)
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
    default_save_name = "preds_full" if args.sliding_window else "preds"
    save_dir = args.output_dir or checkpoint_path.parents[1] / default_save_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # テスト時に使うGPUを設定
    gpu_setting(args.gpu, args.gpu_allow_growth)
    if args.gpu_allow_growth:
        enable_gpu_memory_growth()

    # モデルを読み込む
    model = load_checkpoint(checkpoint_path, cfg)
    model.cfg = cfg
    logging.info(f"Loaded from: {checkpoint_path} (optimizer state skipped)")

    if args.sliding_window:
        volume_count = predict_full_volumes(
            model, cfg, save_dir, args.window_overlap, args.seed
        )
        logging.info(f"Completed full-volume inference: {volume_count} volumes")
        raise SystemExit(0)

    # 従来のcrop推論用データを準備
    val_dict = prepare_data_dict(
        cfg.source_data_dir, cfg.target_data_dir, training_mode=training_mode
    )[1]
    test_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

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
