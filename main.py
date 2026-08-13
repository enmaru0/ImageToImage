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


def _to_path_list(data_dir):
    if isinstance(data_dir, (list, tuple, ListConfig)):
        return [Path(path) for path in data_dir]
    return [Path(data_dir)]


def _to_config_path(path_list):
    if len(path_list) == 1:
        return str(path_list[0])
    return [str(path) for path in path_list]


def get_training_mode(cfg):
    """Return the data pairing mode, including compatibility with old configs."""
    return str(getattr(cfg, "training_mode", "paired"))


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
    training_mode = get_training_mode(cfg)
    if training_mode not in ["paired", "self_supervised_deblur"]:
        raise ValueError(
            "training_modeは'paired'または'self_supervised_deblur'を"
            f"指定してください: {training_mode}"
        )

    source_data_dirs = _to_path_list(cfg.source_data_dir)
    if training_mode == "paired":
        target_data_dirs = _to_path_list(cfg.target_data_dir)
        if len(source_data_dirs) != len(target_data_dirs):
            raise ValueError(
                "source_data_dirとtarget_data_dirは同じ数だけ指定してください"
            )
    else:
        # 単一画像集合をclean targetとして再利用する。source側にだけblurを加える。
        target_data_dirs = source_data_dirs
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

    source_rescaled_list = []
    target_rescaled_list = []
    source_rescaled_flags = []
    target_rescaled_flags = []
    for source_data_dir, target_data_dir in zip(source_data_dirs, target_data_dirs):
        source_rescaled, source_rescaled_flag = _resolve_rescaled_dir(source_data_dir)
        target_rescaled, target_rescaled_flag = _resolve_rescaled_dir(target_data_dir)
        source_rescaled_list.append(source_rescaled)
        target_rescaled_list.append(target_rescaled)
        source_rescaled_flags.append(source_rescaled_flag)
        target_rescaled_flags.append(target_rescaled_flag)

    cfg.source_data_dir = _to_config_path(source_rescaled_list)
    if training_mode == "paired":
        cfg.target_data_dir = _to_config_path(target_rescaled_list)
        rescaled_flags = source_rescaled_flags + target_rescaled_flags
    else:
        # targetはsourceと同じファイルなので、出力設定にも解決済みパスを記録する。
        cfg.target_data_dir = cfg.source_data_dir
        rescaled_flags = source_rescaled_flags
    if target_scale_zyx[0] > 5 and not all(rescaled_flags):
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
    if cfg.model.unet.downsample_type not in ["max_pool", "stride_conv"]:
        raise ValueError(
            "model.unet.downsample_typeはmax_poolまたはstride_convを指定してください"
        )
    if cfg.model.unet.upsample_type not in ["transpose_conv", "resize_conv"]:
        raise ValueError(
            "model.unet.upsample_typeはtranspose_convまたはresize_convを"
            "指定してください"
        )
    if training_mode == "self_supervised_deblur":
        degradation_type = str(
            getattr(cfg.self_supervised_deblur, "degradation_type", "gaussian")
        )
        allowed_degradations = ["gaussian", "cardiac_motion", "cardiac_motion_gaussian"]
        if degradation_type not in allowed_degradations:
            raise ValueError(
                "self_supervised_deblur.degradation_typeは"
                f"{allowed_degradations}から指定してください"
            )
        if degradation_type in ["gaussian", "cardiac_motion_gaussian"]:
            sigma_range = list(cfg.self_supervised_deblur.sigma_range)
            if (
                len(sigma_range) != 2
                or sigma_range[0] <= 0
                or sigma_range[0] >= sigma_range[1]
            ):
                raise ValueError(
                    "self_supervised_deblur.sigma_rangeは0より大きい"
                    "min < maxの[min, max]で指定してください"
                )
            if cfg.self_supervised_deblur.validation_sigma <= 0:
                raise ValueError(
                    "self_supervised_deblur.validation_sigmaは0より大きくしてください"
                )
        if degradation_type in ["cardiac_motion", "cardiac_motion_gaussian"]:
            motion_cfg = cfg.self_supervised_deblur.cardiac_motion
            if motion_cfg.num_phases < 3 or motion_cfg.num_phases % 2 == 0:
                raise ValueError("cardiac_motion.num_phasesは3以上の奇数にしてください")
            num_phases_range = getattr(motion_cfg, "num_phases_range", None)
            if num_phases_range is not None:
                if (
                    len(num_phases_range) != 2
                    or num_phases_range[0] < 3
                    or num_phases_range[0] > num_phases_range[1]
                    or any(value % 2 == 0 for value in num_phases_range)
                ):
                    raise ValueError(
                        "cardiac_motion.num_phases_rangeは3以上の奇数で"
                        "min <= maxとなる[min, max]にしてください"
                    )
            if (
                len(motion_cfg.max_translation_mm_yx) != 2
                or min(motion_cfg.max_translation_mm_yx) < 0
            ):
                raise ValueError(
                    "cardiac_motion.max_translation_mm_yxは非負の[Y, X]にしてください"
                )
            if motion_cfg.max_rotation_deg < 0:
                raise ValueError("cardiac_motion.max_rotation_degは非負にしてください")
            if not 0 <= motion_cfg.max_scale_delta < 1:
                raise ValueError(
                    "cardiac_motion.max_scale_deltaは0以上1未満にしてください"
                )
            if len(motion_cfg.roi_center_yx) != 2 or not all(
                0 <= value <= 1 for value in motion_cfg.roi_center_yx
            ):
                raise ValueError(
                    "cardiac_motion.roi_center_yxは0-1の[Y, X]にしてください"
                )
            if (
                len(motion_cfg.roi_sigma_ratio_yx) != 2
                or min(motion_cfg.roi_sigma_ratio_yx) <= 0
            ):
                raise ValueError(
                    "cardiac_motion.roi_sigma_ratio_yxは正の[Y, X]にしてください"
                )
            if len(motion_cfg.validation_translation_mm_yx) != 2:
                raise ValueError(
                    "cardiac_motion.validation_translation_mm_yxは[Y, X]にしてください"
                )
            if abs(motion_cfg.validation_scale_delta) >= 1:
                raise ValueError(
                    "cardiac_motion.validation_scale_deltaの絶対値は1未満にしてください"
                )
        slice_thickness_cfg = getattr(
            cfg.self_supervised_deblur, "slice_thickness", None
        )
        if slice_thickness_cfg is not None and slice_thickness_cfg.enabled:
            if slice_thickness_cfg.clean_thickness_mm <= 0:
                raise ValueError(
                    "slice_thickness.clean_thickness_mmは0より大きくしてください"
                )
            if (
                slice_thickness_cfg.degraded_thickness_mm
                <= slice_thickness_cfg.clean_thickness_mm
            ):
                raise ValueError(
                    "slice_thickness.degraded_thickness_mmは"
                    "clean_thickness_mmより大きくしてください"
                )
            if slice_thickness_cfg.gaussian_truncate <= 0:
                raise ValueError(
                    "slice_thickness.gaussian_truncateは0より大きくしてください"
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


def _has_raw_files(data_dir, recursive=False):
    if not data_dir.exists():
        return False
    raw_paths = data_dir.rglob("*.raw") if recursive else data_dir.glob("*.raw")
    return any(raw_paths)


def prepare_data_dict(source_data_dir, target_data_dir=None, training_mode="paired"):
    source_data_dirs = _to_path_list(source_data_dir)
    is_self_supervised = training_mode == "self_supervised_deblur"
    if training_mode == "paired":
        if target_data_dir is None:
            raise ValueError("pairedモードではtarget_data_dirが必要です")
        target_data_dirs = _to_path_list(target_data_dir)
        if len(source_data_dirs) != len(target_data_dirs):
            raise ValueError(
                "source_data_dirとtarget_data_dirは同じ数だけ指定してください"
            )
    elif training_mode == "self_supervised_deblur":
        # 各画像を(source, clean target)として自己ペアリングする。
        target_data_dirs = source_data_dirs
    else:
        raise ValueError(f"未対応のtraining_modeです: {training_mode}")

    def _make_split_dict(source_split_dir, target_split_dir, data_name):
        if not source_split_dir.exists():
            raise FileNotFoundError(source_split_dir)
        if not target_split_dir.exists():
            raise FileNotFoundError(target_split_dir)

        split_dict = defaultdict(dict)
        pair_list = []
        skip_count = 0
        if is_self_supervised:
            source_raw_paths = source_split_dir.rglob("*.raw")
        else:
            source_raw_paths = source_split_dir.glob("*.raw")
        for source_raw_path in sorted(source_raw_paths):
            source_hdr_path = source_raw_path.with_suffix(".hdr")
            if is_self_supervised:
                target_raw_path = source_raw_path
            else:
                target_raw_path = target_split_dir / source_raw_path.name
            target_hdr_path = target_raw_path.with_suffix(".hdr")
            if not source_hdr_path.exists():
                skip_count += 1
                logging.warning(
                    f"source hdrがないためスキップします: {source_hdr_path}"
                )
                continue
            if not target_raw_path.exists() or not target_hdr_path.exists():
                skip_count += 1
                logging.warning(
                    f"対応するtargetがないためスキップします: {source_raw_path.name}"
                )
                continue
            pair_list.append((source_hdr_path, target_hdr_path))
        if len(pair_list) == 0:
            if is_self_supervised:
                raise FileNotFoundError(
                    f"{source_split_dir} 以下に使用可能な画像が見つかりません。"
                    "同じ場所に同じbasenameの.rawと.hdrを置いてください"
                )
            raise FileNotFoundError(
                f"{source_split_dir} 直下に使用可能なペア画像が見つかりません"
            )
        if skip_count > 0:
            logging.warning(f"{source_split_dir}: {skip_count}件をスキップしました")

        split_dict[data_name]["img_hdr_list"] = pair_list
        split_dict[data_name]["freq"] = len(pair_list)
        return split_dict

    def _update_unique(dst_dict, src_dict):
        for data_name, value in src_dict.items():
            unique_name = data_name
            count = 1
            while unique_name in dst_dict:
                unique_name = f"{data_name}_{count}"
                count += 1
            dst_dict[unique_name] = value

    train_dict = defaultdict(dict)
    val_dict = defaultdict(dict)
    for source_data_dir, target_data_dir in zip(source_data_dirs, target_data_dirs):
        source_train_dir = source_data_dir / "train"
        target_train_dir = target_data_dir / "train"
        source_val_dir = source_data_dir / "val"
        target_val_dir = target_data_dir / "val"

        if source_train_dir.exists() or target_train_dir.exists():
            train_part = _make_split_dict(
                source_train_dir, target_train_dir, f"{source_data_dir.name}_train"
            )
            if source_val_dir.exists() or target_val_dir.exists():
                val_part = _make_split_dict(
                    source_val_dir, target_val_dir, f"{source_data_dir.name}_val"
                )
            else:
                logging.warning(
                    f"{source_data_dir} のvalフォルダが見つからないため、"
                    "trainデータをvalidationにも使用します"
                )
                val_part = _make_split_dict(
                    source_train_dir,
                    target_train_dir,
                    f"{source_data_dir.name}_train_as_val",
                )
        elif _has_raw_files(
            source_data_dir, recursive=is_self_supervised
        ) and _has_raw_files(target_data_dir, recursive=is_self_supervised):
            search_scope = "以下" if is_self_supervised else "直下"
            logging.warning(
                f"{source_data_dir} にtrain/valフォルダが見つからないため、"
                f"指定フォルダ{search_scope}の画像を"
                "train/validationの両方に使用します"
            )
            train_part = _make_split_dict(
                source_data_dir, target_data_dir, source_data_dir.name
            )
            val_part = _make_split_dict(
                source_data_dir, target_data_dir, f"{source_data_dir.name}_as_val"
            )
        else:
            if is_self_supervised:
                raise FileNotFoundError(
                    f"{source_data_dir} 以下、またはtrainフォルダ以下に"
                    ".rawファイルが見つかりません"
                )
            raise FileNotFoundError(
                f"{source_data_dir} と {target_data_dir} の直下、"
                "またはtrainフォルダ内に.rawファイルが見つかりません"
            )

        _update_unique(train_dict, train_part)
        _update_unique(val_dict, val_part)

    train_total = sum(value["freq"] for value in train_dict.values())
    for value in train_dict.values():
        value["freq"] = value["freq"] / train_total
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
    train_dict, val_dict = prepare_data_dict(
        cfg.source_data_dir, cfg.target_data_dir, training_mode=get_training_mode(cfg)
    )

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
