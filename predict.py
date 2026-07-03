import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from irg import save_raw
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader import create_dataloader
from main import gpu_setting, prepare_data_dict
from trainer import CustomModel

tf.config.run_functions_eagerly(True)


def load_checkpoint(checkpoint_path) -> tuple[CustomModel, int]:
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    step = model.optimizer.iterations.numpy()
    return model, step


def reverse_normalize_img(img, min_val, max_val):
    img = img * (max_val - min_val)
    img = img + min_val
    return img


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str, help="gpu num (default 0)")
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path

    # 実験時の設定ファイルを読み込む
    cfg_path = checkpoint_path.parents[1] / "output.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.batch_size = 1
    cfg.debug_dataloader = True

    # 保存場所を作成
    save_dir = checkpoint_path.parents[1] / "preds"
    save_dir.mkdir(exist_ok=True)

    # テスト時に使うGPUを設定
    gpu_setting(args.gpu, cfg.gpu_allow_growth)

    # データを準備
    val_dict = prepare_data_dict(cfg.source_data_dir, cfg.target_data_dir)[1]
    test_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    # モデルを読み込む
    model, step = load_checkpoint(checkpoint_path)
    model.cfg = cfg
    logging.info(f"Loaded from: {checkpoint_path} (step: {step})")

    spacing_zyx = np.array(cfg.aug.affine.norm_spacing_zyx, np.float32)
    for data in tqdm(test_loader):
        pred = model.predict_step(data).numpy()
        keys = [key.decode() for key in data["img_hdr_list"].numpy()]
        target_min_clip_vals = data["target_min_clip_vals"].numpy()
        target_max_clip_vals = data["target_max_clip_vals"].numpy()

        for idx, key in enumerate(keys):
            pred_img = pred[idx, :, :, :, 0]
            pred_img = reverse_normalize_img(
                pred_img, target_min_clip_vals[idx], target_max_clip_vals[idx]
            )
            pred_img = np.rint(pred_img).astype(np.int16)
            save_raw(pred_img, spacing_zyx, save_dir / f"{key}.hdr")
