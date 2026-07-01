import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
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

    for data in tqdm(test_loader):
        pred = model.predict_step(data).numpy()
        source = data["imgs"].numpy()
        target = data["target_imgs"].numpy()
        keys = [key.decode() for key in data["img_hdr_list"].numpy()]

        for idx, key in enumerate(keys):
            save_path = save_dir / f"{key}.npz"
            np.savez_compressed(
                save_path,
                pred=pred[idx],
                source=source[idx],
                target=target[idx],
                source_min_clip_val=data["min_clip_vals"][idx].numpy(),
                source_max_clip_val=data["max_clip_vals"][idx].numpy(),
                target_min_clip_val=data["target_min_clip_vals"][idx].numpy(),
                target_max_clip_val=data["target_max_clip_vals"][idx].numpy(),
            )
