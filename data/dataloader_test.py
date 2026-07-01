from pathlib import Path

import commonlib
import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw, read_re4

from .dataloader_utils import load_intensity
from .utils import add_channel_dim, add_margin, extract_bb, get_pad_for_margin


def rescale_cpp_img_with_out_size(data_np, src_spacing, dst_spacing, out_size):
    rate_zyx = src_spacing / dst_spacing
    img_filter_z = commonlib.ImageFilter.ANTIALIASING(rate_zyx[0])
    img_filter_y = commonlib.ImageFilter.ANTIALIASING(rate_zyx[1])
    img_filter_x = commonlib.ImageFilter.ANTIALIASING(rate_zyx[2])

    rescaled_transform = commonlib.RescaleTransform3DWithFilter(
        img_filter_z,
        img_filter_y,
        img_filter_x,
        commonlib.ImageFilter.DEFAULT_OUT(np.int16),
        commonlib.ImageFilter.DEFAULT_IN,
        commonlib.ImageFilter.ZERO_OVER,
    )

    rescaled_transform.SetOrgImageSize(*data_np.shape[::-1])
    rescaled_transform.SetResultImageSize(*out_size[::-1])
    rescaled_transform.SetOrgImage(data_np)
    out = rescaled_transform.Transform()
    return out


def preprocess_image_np_test(img_hdr_list_with_data_name: list[bytes], cfg):
    img_hdr_path, dataname = img_hdr_list_with_data_name
    dataname = dataname.decode()
    img_hdr_path = Path(img_hdr_path.decode())

    img_size_zyx, _, spacing_zyx = read_hdr(img_hdr_path)

    target_spacing_zyx = cfg.aug.affine.norm_spacing_zyx

    """
    TODO このサンプルコードでは、２段階の臓器抽出（低解像度でおおよその臓器の位置を見つけ、高解像度で正確に抽出）する
    なので、target_spacing_zyx[0]が大きい場合（低解像度）は体表マスク、
    それ以外の場合（高解像度）は臓器マスクからクロップ範囲を決定する
    """
    if target_spacing_zyx[0] > 2:
        msk_hdr_path = img_hdr_path.with_suffix(".body.mask.hdr")
        margin = 0
    else:
        msk_hdr_path = img_hdr_path.with_suffix(".prostate.mask.hdr")
        margin = cfg.aug.margin

    msk = read_re4(msk_hdr_path, src_dst_bit_dict={0: 0}, type_flag="mask")
    msk = msk[:, :, :, 0].astype(np.bool_)
    crop_zyxzyx = extract_bb(msk, to_open=True)
    margin_mm_zyx, resize_zyx = get_pad_for_margin(
        crop_zyxzyx,
        dilation_mm_zyx=(margin,) * 3,
        size_zyx=msk.shape[:3],
        src_spacing_zyx=spacing_zyx,
        dst_spacing=target_spacing_zyx,
        num_pool=cfg.model.unet.depth,
        return_dst_size=True,
    )

    crop_zyxzyx = add_margin(
        crop_zyxzyx,
        msk.shape[:3],
        margin_mm_zyx,
        spacing_zyx,
        round_to_int=False,
        pad_remain=True,
    )
    crop_zyxzyx = crop_zyxzyx.astype(np.int32)

    img = read_raw(img_hdr_path, clip_zyxzyx=crop_zyxzyx)
    img = rescale_cpp_img_with_out_size(
        img, spacing_zyx, target_spacing_zyx, resize_zyx
    )

    if cfg.image.modality == "MR":
        intensity_path = img_hdr_path.with_suffix(
            f".intensity-{cfg.image.MR.min_percentile}-{cfg.image.MR.max_percentile}.txt"
        )  # save_intensityで作成
        min_val, max_val = load_intensity(intensity_path)
    else:
        window_level = float(cfg.image.CT.window_level)
        window_width = float(cfg.image.CT.window_width)
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2

    # チャンネルの次元を追加
    img = add_channel_dim(img)
    msks = np.zeros_like(img, np.uint16)

    return (
        img,
        msks,
        crop_zyxzyx,
        spacing_zyx,
        img_size_zyx,
        np.array(min_val, np.float32),
        np.array(max_val, np.float32),
        str(img_hdr_path.stem).encode(),
    )


def preprocess_image_test(img_hdr_path_with_data_name, cfg):
    def _preprocess_image_np(img_hdr_path_with_data_name):
        return preprocess_image_np_test(img_hdr_path_with_data_name, cfg)

    return tf.numpy_function(
        func=_preprocess_image_np,
        inp=[img_hdr_path_with_data_name],
        Tout=[
            tf.int16,  # img
            tf.uint16,  # msk
            tf.int32,  # crop_zyxzyx
            tf.float32,  # spacing_zyx
            tf.uint32,  # img_size_zyx
            tf.float32,  # min_val
            tf.float32,  # max_val
            tf.string,  # key
        ],
    )


def make_dict_test(
    img,
    msk,
    crop_zyxzyx,
    spacing_zyx,
    img_size_zyx,
    min_clip_val,
    max_clip_val,
    img_key,
):
    data = dict(
        img=tf.cast(img, tf.float32),
        msk=msk,
        crop_zyxzyx=crop_zyxzyx,
        spacing_zyx=spacing_zyx,
        img_size_zyx=img_size_zyx,
        min_clip_val=min_clip_val,
        max_clip_val=max_clip_val,
        img_key=img_key,
    )

    return data


def create_dataloader_test(img_hdr_dict: dict, cfg):
    img_hdr_path_list = []
    for value in img_hdr_dict.values():
        img_hdr_path_list += value["img_hdr_list"]

    dataset_list = []
    for data_name, value in img_hdr_dict.items():
        # データセットごとに処理を変えることを想定してデータセット名を付与する（処理はpreprocess_image_npで実装）
        img_hdr_list_with_data_name = [
            (str(path), data_name) for path in sorted(value["img_hdr_list"])
        ]
        _dataset = tf.data.Dataset.from_tensor_slices(img_hdr_list_with_data_name)

        dataset_list.append(_dataset)
        logging.info(f"Dataset {data_name} has {len(value['img_hdr_list'])} images.")

    dataset = tf.data.Dataset.sample_from_datasets(dataset_list)

    def _preprocess_image(img_hdr_path_with_data_name):
        return preprocess_image_test(img_hdr_path_with_data_name, cfg)

    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=cfg.num_workers
    )  # autotuneはなんか遅かった・・・

    # 他で使いやすいように辞書型で保持する
    dataset = dataset.map(make_dict_test)
    dataset = dataset.prefetch(buffer_size=cfg.prefetch_size)

    return dataset
