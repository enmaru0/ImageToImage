import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import keras
import numpy as np
from absl import logging
from keras.api.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.api.optimizers import SGD, AdamW
from keras.api.optimizers.schedules import CosineDecay
from omegaconf import ListConfig, OmegaConf

from callbacks import ImageLogger, UnifiedTensorBoardLogger
from data.dataloader import create_dataloader
from models import build_unet
from trainer import CustomModel


def read_cfg_and_parse_arg():
    # コマンドライン引数と設定ファイルを読み込む関数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="設定を上書きするフォーマット (例: 'batch_size=12 aug.crop_size_zyx=[64,64,64]')",
    )
    args = parser.parse_args()
    cmd_overrides = args.overrides

    config_path = "conf/config.yaml"
    cfg = OmegaConf.load(config_path)

    # コマンドライン引数で設定を上書きする
    override_config = OmegaConf.from_dotlist(cmd_overrides)
    for key in override_config:
        if key not in cfg:
            raise KeyError(f"設定ファイルに存在しないキー: {key}")
    cfg = OmegaConf.merge(cfg, override_config)

    # ディレクトリを Path 型に変換
    cfg.exp_dir = Path(cfg.exp_dir)
    cfg.source_data_dir = Path(cfg.source_data_dir)
    cfg.target_data_dir = Path(cfg.target_data_dir)
    cfg.restore = Path(cfg.restore) if cfg.restore else None
    cfg.finetune = Path(cfg.finetune) if cfg.finetune else None

    if cfg.restore and cfg.finetune:
        raise ValueError("restoreとfinetuneの両方を指定することはできません")

    # リスケール済みのディレクトリを探す
    target_scale_zyx = np.array(cfg.aug.affine.norm_spacing_zyx)
    target_scale_zyx = target_scale_zyx.astype(np.float32)

    def _resolve_rescaled_dir(data_dir):
        rescaled_dir = data_dir.parent / (
            data_dir.name + "_" + "_".join(map(str, target_scale_zyx))
        )
        if rescaled_dir.exists():
            return rescaled_dir, True
        return data_dir, False

    cfg.source_data_dir, source_rescaled = _resolve_rescaled_dir(cfg.source_data_dir)
    cfg.target_data_dir, target_rescaled = _resolve_rescaled_dir(cfg.target_data_dir)
    if target_scale_zyx[0] > 2 and not (source_rescaled and target_rescaled):
        raise ValueError(
            "Thickスライスで学習する場合は./utils/rescale_dataset.pyで予めリスケールすることを推奨します"
        )

    # その他cfgのチェック
    assert cfg.image.modality in ["CT", "MR"], cfg.image.modality
    assert cfg.model.input_num_channel == 1, (
        "現在のDataLoaderは1チャンネル画像を想定しています"
    )
    assert cfg.model.num_channel == 1, (
        "現在のI2I-RFR実装は1チャンネル出力を想定しています"
    )
    return cfg


def gpu_setting(gpu_str: str, gpu_allow_growth: bool) -> None:
    if gpu_str == "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(os.getenv("SGE_GPU", 0))
    else:
        if isinstance(gpu_str, ListConfig):
            gpu_str = ",".join(map(str, gpu_str))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_str)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = str(gpu_allow_growth).lower()


def _iter_dataset_dirs(split_dir):
    dataset_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    if len(dataset_dirs) == 0:
        return [split_dir]
    return dataset_dirs


def prepare_data_dict(source_data_dir, target_data_dir):
    source_data_dir = Path(source_data_dir)
    target_data_dir = Path(target_data_dir)

    def _make_split_dict(split):
        source_split_dir = source_data_dir / split
        target_split_dir = target_data_dir / split
        if not source_split_dir.exists():
            raise FileNotFoundError(source_split_dir)
        if not target_split_dir.exists():
            raise FileNotFoundError(target_split_dir)

        split_dict = defaultdict(dict)
        dataset_dirs = _iter_dataset_dirs(source_split_dir)
        freq = 1.0 / len(dataset_dirs)
        for dataset_dir in dataset_dirs:
            data_name = (
                dataset_dir.name
                if dataset_dir != source_split_dir
                else source_split_dir.name
            )
            pair_list = []
            for source_raw_path in sorted(dataset_dir.glob("*.raw")):
                source_hdr_path = source_raw_path.with_suffix(".hdr")
                rel_hdr_path = source_hdr_path.relative_to(source_split_dir)
                target_hdr_path = target_split_dir / rel_hdr_path
                if not target_hdr_path.exists():
                    raise FileNotFoundError(
                        f"対応するtarget画像が見つかりません: {target_hdr_path}"
                    )
                pair_list.append((source_hdr_path, target_hdr_path))
            split_dict[data_name]["img_hdr_list"] = pair_list
            split_dict[data_name]["freq"] = freq
        return split_dict

    train_dict = _make_split_dict("train")
    val_dict = _make_split_dict("val")
    for value in val_dict.values():
        value["freq"] = -1
    return train_dict, val_dict


def select_optimizer(cfg):
    cfg_opt = cfg.optimizer[cfg.optimizer.name]
    # スケジューラーを設定
    warmup_steps = int(cfg_opt.warmup_ratio * cfg.num_train_steps)
    lr_schedule = CosineDecay(
        cfg_opt.warmup_lr,
        cfg.num_train_steps - warmup_steps,
        alpha=0.0,
        name="CosineDecay",
        warmup_target=cfg_opt.max_lr,
        warmup_steps=warmup_steps,
    )

    # オプティマイザを設定
    if cfg.optimizer.name == "sgd":
        optimizer = SGD(
            learning_rate=lr_schedule,
            momentum=cfg_opt.momentum,
            nesterov=cfg_opt.use_nesterov,
            weight_decay=cfg_opt.wd,
            clipvalue=cfg_opt.clip_value,  # 勾配クリッピング
        )
    elif cfg.optimizer.name == "adamw":
        optimizer = AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg_opt.wd,
            clipvalue=cfg_opt.clip_value,  # 勾配クリッピング
        )
    else:
        raise NotImplementedError(cfg.optimizer.name)
    return optimizer


if __name__ == "__main__":
    # ログのレベルを設定する：INFO以上を表示
    logging.set_verbosity(logging.INFO)

    cfg = read_cfg_and_parse_arg()

    # 実験フォルダを作成
    cfg.exp_dir.mkdir(exist_ok=True, parents=True)

    # すでにチェックポイントがあり、restoreが指定されていない場合は終了する
    _checkpoint_path = cfg.exp_dir / "checkpoints" / "model_latest.keras"
    if _checkpoint_path.exists() and not cfg.restore:
        logging.error(f"すでにチェックポイントが存在します。{_checkpoint_path}")
        logging.error("checkpointを削除するか、restoreを指定してください")
        exit(1)

    # 設定ファイルを保存
    OmegaConf.save(cfg, cfg.exp_dir / "output.yaml")

    gpu_setting(cfg.gpu, cfg.gpu_allow_growth)
    tensorboard_dir = cfg.exp_dir / "tensorboard_logs"

    # 学習データリストを準備
    """ 下記のような辞書を作成する
    {
       "DataSetA":
            {
                "img_hdr_list": [(source1.hdr, target1.hdr), ...]
                "freq": 0.8, # 80%の確率でDataSetAからサンプリング
            },
        "DataSetB": 
            {
                "img_hdr_list": [(source2.hdr, target2.hdr), ...]
                "freq": 0.2, # 20%の確率でDataSetAからサンプリング
            },
    }
    """
    train_dict, val_dict = prepare_data_dict(cfg.source_data_dir, cfg.target_data_dir)

    # トレーニングおよび検証用のDataLoaderを作成
    train_loader = create_dataloader(train_dict, is_training=True, cfg=cfg)
    val_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    # モデルを作成
    input_shape = tuple(cfg.aug.crop_size_zyx) + (
        cfg.model.input_num_channel + cfg.model.num_channel,
    )
    model: CustomModel = build_unet(
        CustomModel,
        input_shape,
        cfg.model.num_channel,
        **cfg.model.unet,
        **cfg.model.renorm,
    )
    model.cfg = cfg

    # オプティマイザを選択する
    optimizer = select_optimizer(cfg)

    # モデルをコンパイル
    # lossとmetricsはいろいろとカスタマイズしたい場所なので、
    # trainer.pyで手動設定する。
    model.compile(
        optimizer=optimizer,
        loss=None,
        metrics=None,
        weighted_metrics=None,
        jit_compile=True,  # 実行を JIT コンパイルで高速化
    )

    # TensorBoard コールバックを設定
    # trainとvalを同じログに記録するように変更している。
    # デフォルト仕様がいい場合はkeras.callback.TensorBoardに書き換える。
    profile_batch = (32, 64) if cfg.enable_profiling else 0  # プロファイリングする範囲
    tensorboard_callback = UnifiedTensorBoardLogger(
        log_dir=tensorboard_dir,
        step_per_epoch=cfg.eval_every,  # 1エポックあたりのステップ数
        write_images=True,  # 訓練中の画像をログに保存
        profile_batch=profile_batch,
        write_steps_per_second=True,
    )

    # 検証用データの1バッチ分をTensorBoardに記録するコールバック
    image_logger_callback = ImageLogger(
        val_data=next(iter(val_loader)), log_dir=tensorboard_dir, jit_compile=True
    )

    # ModelCheckpoint コールバックを設定
    best_checkpoint_callback = ModelCheckpoint(
        filepath=str(cfg.exp_dir / "checkpoints" / "model_best.keras"),
        save_best_only=True,  # 最良モデルのみを保存
        monitor="val_total_loss",  # CustomModelで初期化したMetricsの名前にval_をつけたもの
        mode="min",  # 指標を最小化するか最大にするか（min/max）
        save_weights_only=False,  # モデル全体（オプティマイザの状態を含む）を保存
    )
    latest_model_callback = ModelCheckpoint(
        filepath=str(cfg.exp_dir / "checkpoints" / "model_latest.keras"),
        save_best_only=False,
        save_weights_only=False,
    )

    if cfg.restore:
        # 学習途中のモデルを復元する場合
        assert cfg.restore.exists(), f"restore path not found: {cfg.restore}"
        model = keras.models.load_model(cfg.restore)
        model.cfg = cfg
        step = model.optimizer.iterations.numpy()
        logging.info(f"Restoring from {cfg.restore}. (step: {step})")
        initial_epoch = step // cfg.eval_every
    elif cfg.finetune:
        # 事前学習済みモデルをファインチューニングする場合
        assert cfg.finetune.exists(), f"finetune path not found: {cfg.finetune}"
        model.load_weights(cfg.finetune)
        logging.info(f"Finetuning from {cfg.finetune}")
        initial_epoch = 0
    else:
        initial_epoch = 0

    # 学習の実行
    model.fit(
        x=train_loader,
        validation_data=val_loader,
        epochs=math.ceil(cfg.num_train_steps / cfg.eval_every),
        steps_per_epoch=cfg.eval_every,
        callbacks=[
            tensorboard_callback,
            image_logger_callback,
            best_checkpoint_callback,
            latest_model_callback,
            TerminateOnNaN(),
        ],
        initial_epoch=initial_epoch,
    )
