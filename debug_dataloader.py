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


def save_imgs(key_batch, img_batch, save_root, spacing_zyx, min_val, max_val):
    assert img_batch.ndim == 5
    assert save_root.exists(), save_root
    if isinstance(min_val, (int, float)):
        min_val = [min_val] * len(img_batch)
        max_val = [max_val] * len(img_batch)
    for key, img, _min, _max in zip(key_batch, img_batch, min_val, max_val):
        print(f"saving img: {key}")
        img = img[:, :, :, 0]

        if (_min is None) or (_max is None):
            assert (_min is None) and (_max is None), (_min, max_val)
        else:
            reverse_normalize_img(img, _min, _max)

        hdr_path = save_root / (key + ".hdr")
        save_raw(img.astype(np.int16), spacing_zyx, hdr_path)


def save_msks(key_batch, msk_batch, save_root, spacing_zyx, bit_dict):
    assert msk_batch.ndim == 5
    assert save_root.exists()
    for key, msk in zip(key_batch, msk_batch):
        print(f"saving msk: {key}")
        msk = msk.astype(np.uint16)
        src_dst_bit_dict = {i: i for i in range(msk.shape[-1])}
        msk_hdr = save_root / (key + ".mask.hdr")
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
        loader = create_dataloader(img_hdr_dict, is_training=is_training, cfg=cfg)
        for num, batch in enumerate(loader):
            if num == 2:
                # バッチサイズx2で停止
                break

            # GPUの処理を実行
            if is_training:
                batch["imgs"] = CustomModel.gpu_aug(
                    batch["imgs"],
                    CustomModel._get_img_msks(batch["msks"], cfg.bit_info.padding_bit),
                    batch["min_clip_vals"],
                    batch["max_clip_vals"],
                    cfg,
                )
                batch["imgs"] = CustomModel.apply_self_supervised_deblur(
                    batch["imgs"],
                    CustomModel._get_img_msks(batch["msks"], cfg.bit_info.padding_bit),
                    cfg,
                    is_training=True,
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
                )
            batch["target_imgs"] = CustomModel.normalize_target(
                batch["target_imgs"],
                CustomModel._get_img_msks(batch["msks"], cfg.bit_info.padding_bit),
                batch["target_min_clip_vals"],
                batch["target_max_clip_vals"],
            )
            # Numpyに変換
            batch = {k: v.numpy() for k, v in batch.items()}

            img_hdr_list = batch["img_hdr_list"]
            img_hdr_list = [i.decode() for i in img_hdr_list]

            source_save_dir = save_dir / "source"
            target_save_dir = save_dir / "target"
            source_save_dir.mkdir(exist_ok=True)
            target_save_dir.mkdir(exist_ok=True)

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

            msks = ((batch["msks"] & (1 << cfg.bit_info.padding_bit)) == 0).astype(
                np.float32
            )
            bit_dict = {0: "img_msks"}

            save_msks(img_hdr_list, msks, save_dir, spacing_zyx, bit_dict)
