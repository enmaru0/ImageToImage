from pathlib import Path

import numpy as np
from irg import save_raw, save_re4

from data.dataloader import create_dataloader
from data.gpu_aug import normalize
from main import get_training_mode, prepare_data_dict, read_cfg_and_parse_arg
from trainer import CustomModel


def reverse_normalize_img(img, min_val, max_val) -> None:
    img *= max_val - min_val
    img += min_val


def restore_img(img, min_val, max_val):
    """Convert one normalized z-y-x-channel image back to an int16 volume."""
    img = np.array(img[:, :, :, 0], np.float32, copy=True)
    if (min_val is None) or (max_val is None):
        assert (min_val is None) and (max_val is None), (min_val, max_val)
    else:
        reverse_normalize_img(img, min_val, max_val)
    return img.astype(np.int16)


def save_imgs(key_batch, img_batch, save_root, spacing_zyx, min_val, max_val):
    assert img_batch.ndim == 5
    assert save_root.exists(), save_root
    if isinstance(min_val, (int, float)):
        min_val = [min_val] * len(img_batch)
        max_val = [max_val] * len(img_batch)
    for key, img, _min, _max in zip(key_batch, img_batch, min_val, max_val):
        print(f"saving img: {key}")
        img = restore_img(img, _min, _max)
        hdr_path = save_root / (key + ".hdr")
        save_raw(img, spacing_zyx, hdr_path)


def save_comparison_imgs(
    key_batch,
    source_batch,
    target_batch,
    save_root,
    spacing_zyx,
    source_min_vals,
    source_max_vals,
    target_min_vals,
    target_max_vals,
    separator_width=4,
):
    """Save source | separator | target volumes along the X axis."""
    assert source_batch.shape == target_batch.shape
    assert save_root.exists(), save_root
    values = zip(
        key_batch,
        source_batch,
        target_batch,
        source_min_vals,
        source_max_vals,
        target_min_vals,
        target_max_vals,
    )
    for key, source, target, source_min, source_max, target_min, target_max in values:
        print(f"saving comparison: {key}")
        source = restore_img(source, source_min, source_max)
        target = restore_img(target, target_min, target_max)
        separator_shape = list(source.shape)
        separator_shape[2] = separator_width
        separator = np.zeros(separator_shape, dtype=np.int16)
        comparison = np.concatenate([source, separator, target], axis=2)
        save_raw(comparison, spacing_zyx, save_root / (key + ".hdr"))


def save_msks(key_batch, msk_batch, save_root, spacing_zyx, bit_dict):
    assert msk_batch.ndim == 5
    assert save_root.exists()
    for key, msk in zip(key_batch, msk_batch):
        print(f"saving msk: {key}")
        msk = msk.astype(np.uint16)
        msk_hdr = save_root / (key + ".mask.hdr")
        if msk.shape[-1] == 1:
            # irg.save_re4は1channel + src_dst_bit_dict指定時に内部squeeze後も
            # 4Dとして扱うため、1channelは明示的に3Dへ変換する。
            save_re4(msk[..., 0], spacing_zyx, "mask", msk_hdr, bit_dict=bit_dict)
        else:
            src_dst_bit_dict = {i: i for i in range(msk.shape[-1])}
            save_re4(
                msk,
                spacing_zyx,
                "mask",
                msk_hdr,
                src_dst_bit_dict=src_dst_bit_dict,
                bit_dict=bit_dict,
            )


if __name__ == "__main__":
    cfg = read_cfg_and_parse_arg()
    cfg.debug_dataloader = True

    train_dict, val_dict = prepare_data_dict(
        cfg.source_data_dir, cfg.target_data_dir, training_mode=get_training_mode(cfg)
    )

    save_root = Path(cfg.exp_dir)
    save_root.mkdir(exist_ok=True, parents=True)
    spacing_zyx = cfg.aug.affine.norm_spacing_zyx
    for is_training in [True, False]:
        if is_training:
            img_hdr_dict = train_dict
            save_dir = save_root / "sample_train"
        else:
            img_hdr_dict = val_dict
            save_dir = save_root / "sample_val"
        save_dir.mkdir(exist_ok=True)
        loader = create_dataloader(
            img_hdr_dict,
            is_training=is_training,
            cfg=cfg,
            use_degradation_context=True,
        )
        for num, batch in enumerate(loader):
            if num == 2:
                # バッチサイズx2で停止
                break

            # GPUの処理を実行
            if is_training:
                batch["imgs"], batch["target_imgs"] = (
                    CustomModel.prepare_training_images(
                        batch["imgs"],
                        batch["target_imgs"],
                        CustomModel._get_img_msks(
                            batch["msks"], cfg.bit_info.padding_bit
                        ),
                        batch["min_clip_vals"],
                        batch["max_clip_vals"],
                        batch["target_min_clip_vals"],
                        batch["target_max_clip_vals"],
                        cfg,
                        heart_msks=CustomModel._get_heart_msks(
                            batch["msks"], cfg.bit_info.heart_bit
                        ),
                    )
                )
            else:
                batch["imgs"] = normalize(
                    batch["imgs"], batch["min_clip_vals"], batch["max_clip_vals"]
                )
                batch["imgs"] *= CustomModel._get_img_msks(
                    batch["msks"], cfg.bit_info.padding_bit
                )
                batch["imgs"] = CustomModel.apply_self_supervised_deblur(
                    batch["imgs"],
                    CustomModel._get_img_msks(batch["msks"], cfg.bit_info.padding_bit),
                    cfg,
                    is_training=False,
                    heart_msks=CustomModel._get_heart_msks(
                        batch["msks"], cfg.bit_info.heart_bit
                    ),
                )
                batch["target_imgs"] = CustomModel.normalize_target(
                    batch["target_imgs"],
                    CustomModel._get_img_msks(batch["msks"], cfg.bit_info.padding_bit),
                    batch["target_min_clip_vals"],
                    batch["target_max_clip_vals"],
                )
                batch["imgs"] = CustomModel.center_crop_to_model_size(
                    batch["imgs"], cfg
                )
                batch["target_imgs"] = CustomModel.center_crop_to_model_size(
                    batch["target_imgs"], cfg
                )
            batch["msks"] = CustomModel.center_crop_to_model_size(batch["msks"], cfg)
            # Numpyに変換
            batch = {k: v.numpy() for k, v in batch.items()}

            img_hdr_list = batch["img_hdr_list"]
            img_hdr_list = [i.decode() for i in img_hdr_list]

            source_save_dir = save_dir / "source"
            target_save_dir = save_dir / "target"
            comparison_save_dir = save_dir / "comparison"
            source_save_dir.mkdir(exist_ok=True)
            target_save_dir.mkdir(exist_ok=True)
            comparison_save_dir.mkdir(exist_ok=True)

            save_comparison_imgs(
                img_hdr_list,
                batch["imgs"],
                batch["target_imgs"],
                comparison_save_dir,
                spacing_zyx,
                batch["min_clip_vals"],
                batch["max_clip_vals"],
                batch["target_min_clip_vals"],
                batch["target_max_clip_vals"],
            )

            save_imgs(
                img_hdr_list,
                batch["imgs"],
                source_save_dir,
                spacing_zyx,
                batch["min_clip_vals"],
                batch["max_clip_vals"],
            )
            save_imgs(
                img_hdr_list,
                batch["target_imgs"],
                target_save_dir,
                spacing_zyx,
                batch["target_min_clip_vals"],
                batch["target_max_clip_vals"],
            )

            img_msks = ((batch["msks"] & (1 << cfg.bit_info.padding_bit)) == 0).astype(
                np.float32
            )
            heart_msks = ((batch["msks"] & (1 << cfg.bit_info.heart_bit)) > 0).astype(
                np.float32
            )
            msks = np.concatenate([img_msks, heart_msks], axis=-1)
            bit_dict = {0: "img_msks", 1: "heart_msks"}

            save_msks(img_hdr_list, msks, save_dir, spacing_zyx, bit_dict)
