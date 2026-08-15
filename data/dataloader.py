import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw, read_re4
from tqdm import tqdm

from .dataloader_utils import (
    get_center,
    has_foreground_in_every_z_slice,
    load_intensity,
    load_organ_box,
    save_intensity,
    save_organ_box,
)
from .utils import (
    AffineTransform,
    add_channel_dim,
    calc_img_crop_region,
    check_mask_bit_number,
    crop_input,
    prepare_thin2thick,
    virtual_thick_generator,
)


def preprocess_image_np(
    img_hdr_list_with_data_name: list[bytes], is_training: bool, cfg
):
    img_hdr_path, target_hdr_path, _ = img_hdr_list_with_data_name
    img_hdr_path = Path(img_hdr_path.decode())
    target_hdr_path = Path(target_hdr_path.decode())

    crop_size_zyx = cfg.aug.crop_size_zyx
    # <image>.mask.hdrのheart_bitに心臓マスクが入っている。
    organ_hdr_path = img_hdr_path.with_suffix(".mask.hdr")
    organ_box_path = img_hdr_path.with_suffix(".heart.box.txt")
    body_box_path = img_hdr_path.with_suffix(".body.box.txt")  # save_organ_boxで作成

    img_size_zyx, img_dtype, spacing_zyx = read_hdr(img_hdr_path)
    target_size_zyx, target_dtype, target_spacing_zyx = read_hdr(target_hdr_path)
    if not np.array_equal(img_size_zyx, target_size_zyx):
        raise ValueError(
            f"source/target size mismatch: {img_hdr_path}, {target_hdr_path}"
        )
    if not np.allclose(spacing_zyx, target_spacing_zyx):
        raise ValueError(
            f"source/target spacing mismatch: {img_hdr_path}, {target_hdr_path}"
        )

    full_box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx), np.int32)
    body_box_zyxzyx = (
        load_organ_box(body_box_path) if body_box_path.exists() else full_box_zyxzyx
    )
    organ_box_zyxzyx = (
        load_organ_box(organ_box_path) if organ_box_path.exists() else body_box_zyxzyx
    )

    # アフィン変換のためのインスタンスを作成
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)

    foreground_crop_cfg = getattr(cfg.aug, "foreground_crop", None)
    enforce_foreground_crop = bool(
        is_training and foreground_crop_cfg is not None and foreground_crop_cfg.enabled
    )
    if enforce_foreground_crop and not organ_hdr_path.exists():
        raise FileNotFoundError(
            "各cropスライスの前景を保証するための心臓マスクがありません: "
            f"{organ_hdr_path}。aug.foreground_crop.enabled=falseで無効化できます。"
        )

    max_crop_attempts = (
        int(foreground_crop_cfg.max_attempts) if enforce_foreground_crop else 1
    )
    min_foreground_voxels = (
        int(foreground_crop_cfg.min_voxels_per_slice) if enforce_foreground_crop else 1
    )

    # crop中心とaugmentationを再抽選し、変換後の全Zスライスに心臓が残るものを採用する。
    # 変換前のboxだけで判定すると、X/Y軸回転後に端のスライスが背景だけになることがある。
    for crop_attempt in range(max_crop_attempts):
        # 心臓boxがない場合はorgan_box_zyxzyxがbody boxへフォールバックする。
        crop_center_zyx = get_center(
            img_size_zyx,
            spacing_zyx,
            is_training,
            body_box_zyxzyx,
            organ_box_zyxzyx,
            cfg.aug.random_crop_method,
            crop_size_zyx,
            cfg.aug.affine.norm_spacing_zyx,
            cfg.aug.margin,
            cfg.aug.crop_keep_ratio,
        )

        affine_matrix = affine_transform.get_affine(
            spacing_zyx, crop_center_zyx, is_training
        )
        img_region_zyxzyx, shift_start = calc_img_crop_region(
            crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
        )
        # 画像などは切り取って読み込むので、その分アフィン行列をシフトさせる。
        affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

        if not organ_hdr_path.exists():
            # 心臓マスクがないデータ。motion生成時はGaussian ROIへfallbackする。
            raw_msk = np.zeros(
                img_region_zyxzyx[3:6] - img_region_zyxzyx[:3], np.uint16
            )
        else:
            # 心臓bitを含むbit mask全体を保持し、GPU側でも利用する。
            raw_msk = read_re4(
                organ_hdr_path,
                clip_zyxzyx=img_region_zyxzyx,
                size_zyx=img_size_zyx,
                type_flag="mask",
            )

        assert cfg.bit_info.padding_bit < check_mask_bit_number(raw_msk)
        msk = affine_transform.apply(
            raw_msk, affine_matrix, order=0, cval=1 << cfg.bit_info.padding_bit
        )
        if not enforce_foreground_crop or has_foreground_in_every_z_slice(
            msk, int(cfg.bit_info.heart_bit), min_foreground_voxels
        ):
            break
    else:
        raise ValueError(
            f"{img_hdr_path}: {max_crop_attempts}回試行しても全Zスライスに"
            f"心臓マスクbit {cfg.bit_info.heart_bit}を"
            f"{min_foreground_voxels} voxel以上含むcropを作成できませんでした。"
            "crop_size_zyx[0]を小さくするか、回転・margin・organ_cropを弱めてください。"
        )

    # ここでは画像は読み込まずメモリマッピングをするだけ。アフィン変換で初めて画像を読む
    img = read_raw(
        img_hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=img_dtype,  # img_dtypeとsize_zyxを指定するとhdrの読み込みスキップできる
        size_zyx=img_size_zyx,
        use_memmap=True,
    )
    target_img = read_raw(
        target_hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=target_dtype,
        size_zyx=target_size_zyx,
        use_memmap=True,
    )

    # mskはcrop候補の判定時に同一affineで変換済み。
    # bitの取り出しやその他の操作はGPU上で行うのでこれ以上は操作しない。
    target_img = affine_transform.apply(target_img, affine_matrix, order=1)

    # thin->thick変換の準備
    # self-supervisedのsource劣化は、clean targetへの共通augmentation後に
    # trainer側で適用する。ここでsourceだけを変形しても後段で使われない。
    thick2thin_rate_zyx = cfg.aug.thick2thin_rate_zyx
    if str(getattr(cfg, "training_mode", "paired")) == "self_supervised_deblur":
        thick2thin_rate_zyx = [0.0, 0.0, 0.0]
    thin2thick_param = prepare_thin2thick(
        spacing_zyx,
        affine_transform.norm_spacing_zyx,
        crop_size_zyx,
        thick2thin_rate_zyx,
        is_training=is_training,
        spacing_max_val=2,
        thickness_range=[2, 6],
    )

    # imgをアフィン変換：thin->thick変換用にcrop_sizeを大きくしておく
    # なのでimgのアフィンは一番最後にやること
    crop_size_extra = crop_size_zyx.copy()
    crop_size_extra[thin2thick_param["axis"]] += thin2thick_param["extra_slice"]
    affine_transform.crop_size_zyx = crop_size_extra  # 上書きするので注意
    img = affine_transform.apply(img, affine_matrix, order=1)

    if thin2thick_param["apply_thin_thick"]:
        img = virtual_thick_generator(
            img, thin2thick_param["thickness"], order=1, axis=thin2thick_param["axis"]
        )
        img = crop_input(img, [0, 0, 0] + list(crop_size_zyx))

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
    if cfg.image.modality == "MR":
        target_intensity_path = target_hdr_path.with_suffix(
            f".intensity-{cfg.image.MR.min_percentile}-{cfg.image.MR.max_percentile}.txt"
        )
        target_min_val, target_max_val = load_intensity(target_intensity_path)
    else:
        target_min_val = min_val
        target_max_val = max_val

    # チャンネルの次元を追加
    img = add_channel_dim(img)
    target_img = add_channel_dim(target_img)
    msk = add_channel_dim(msk)
    return (
        img,
        target_img,
        msk,
        np.array(min_val, np.float32),
        np.array(max_val, np.float32),
        np.array(target_min_val, np.float32),
        np.array(target_max_val, np.float32),
        str(img_hdr_path.stem).encode(),
    )


def preprocess_image(img_hdr_path_with_data_name, is_training: bool, cfg):
    def _preprocess_image_np(img_hdr_path_with_data_name):
        return preprocess_image_np(img_hdr_path_with_data_name, is_training, cfg)

    (
        img,
        target_img,
        msk,
        min_clip_val,
        max_clip_val,
        target_min_clip_val,
        target_max_clip_val,
        img_hdr,
    ) = tf.numpy_function(
        func=_preprocess_image_np,
        inp=[img_hdr_path_with_data_name],
        Tout=[
            tf.int16,
            tf.int16,
            tf.uint16,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.string,
        ],
    )

    # tf.numpy_functionを使ったときはset_shapeでshapeを指定する
    img_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
    img.set_shape(img_shape)
    target_img.set_shape(img_shape)
    msk_shape = tuple(cfg.aug.crop_size_zyx) + (
        1,
    )  # trainer.pyでチャンネルの分割（one-hot化など）は行う
    msk.set_shape(msk_shape)
    min_clip_val.set_shape(())
    max_clip_val.set_shape(())
    target_min_clip_val.set_shape(())
    target_max_clip_val.set_shape(())

    return (
        img,
        target_img,
        msk,
        min_clip_val,
        max_clip_val,
        target_min_clip_val,
        target_max_clip_val,
        img_hdr,
    )


def make_batch_dict(
    imgs,
    target_imgs,
    msks,
    min_clip_vals,
    max_clip_vals,
    target_min_clip_vals,
    target_max_clip_vals,
    img_hdr_list,
    cfg,
):
    """
    一般的にはモデルを GPU や TPU などのアクセラレータ上で実行している場合でも、
    tf.data パイプラインは CPU 上で実行されています。
    https://www.tensorflow.org/guide/data_performance_analysis?hl=ja#3_cpu_%E4%BD%BF%E7%94%A8%E7%8E%87%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%EF%BC%9F
    """

    data = dict(
        imgs=tf.cast(imgs, tf.float32),
        target_imgs=tf.cast(target_imgs, tf.float32),
        msks=tf.cast(msks, tf.uint16),
        min_clip_vals=min_clip_vals,
        max_clip_vals=max_clip_vals,
        target_min_clip_vals=target_min_clip_vals,
        target_max_clip_vals=target_max_clip_vals,
    )
    if cfg.debug_dataloader:
        data["img_hdr_list"] = img_hdr_list

    return data


def create_dataloader(
    img_hdr_dict: dict,
    is_training: bool,
    cfg,
    batch_size: int | None = None,
    drop_remainder: bool = True,
):
    """
    複数のデータセットから異なる確率で読み込むデータローダーを作成する
    ミニバッチに必ず特定のデータセットが含まれるような実装にはしていないが、それほど問題にならないはず。
    img_hdr_dict:
    e.g.
    {
       "DataSetA":
            {
                "img_hdr_list": [path1.hdr, path2.hdr, ...]
                "freq": 0.8, # 80%の確率でDataSetAからサンプリング
            },
        "DataSetB":
            {
                "img_hdr_list": [path3.hdr, path4.hdr, ...]
                "freq": 0.2,
            },
    }
    """

    img_hdr_path_list = []
    for value in img_hdr_dict.values():
        img_hdr_path_list += value["img_hdr_list"]
    source_hdr_path_list = [pair[0] for pair in img_hdr_path_list]
    target_hdr_path_list = [pair[1] for pair in img_hdr_path_list]

    with multiprocessing.Pool(cfg.num_workers) as pool:

        def _run(func, desc):
            for _ in tqdm(
                pool.imap_unordered(func, source_hdr_path_list),
                total=len(source_hdr_path_list),
                desc=desc,
            ):
                pass

        # bit 6の心臓マスクからcrop中心決定用の矩形を計算しておく。
        if any(path.with_suffix(".mask.hdr").exists() for path in source_hdr_path_list):
            func = partial(
                save_organ_box,
                suffix="",
                src_bit=int(cfg.bit_info.heart_bit),
                box_suffix=".heart",
            )
            _run(func, "saving heart box")

        if any(
            path.with_suffix(".body.mask.hdr").exists() for path in source_hdr_path_list
        ):
            func = partial(save_organ_box, suffix=".body")
            _run(func, "saving body box")
        # MRデータはあらかじめ、min_intensityとmax_intensityを計算しておく
        if cfg.image.modality == "MR":
            func = partial(
                save_intensity,
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )
            _run(func, "saving source intensity")
            for _ in tqdm(
                pool.imap_unordered(func, target_hdr_path_list),
                total=len(target_hdr_path_list),
                desc="saving target intensity",
            ):
                pass

    dataset_list = []
    frequency_list = []
    for data_name, value in img_hdr_dict.items():
        # データセットごとに処理を変えることを想定してデータセット名を付与する（処理はpreprocess_image_npで実装）
        img_hdr_list_with_data_name = [
            (str(source_path), str(target_path), data_name)
            for source_path, target_path in sorted(value["img_hdr_list"])
        ]
        _dataset = tf.data.Dataset.from_tensor_slices(img_hdr_list_with_data_name)

        if is_training:
            # ここでrepeatしないと正しくサンプリングできない
            _dataset = _dataset.repeat()
        dataset_list.append(_dataset)
        frequency_list.append(value["freq"])
        logging.info(f"Dataset {data_name} has {len(value['img_hdr_list'])} images.")

    # データセットを結合する。学習時はここでサンプリングの重みを設定する。
    if is_training:
        dataset = tf.data.Dataset.sample_from_datasets(
            dataset_list, weights=frequency_list
        )
    else:
        dataset = tf.data.Dataset.sample_from_datasets(dataset_list)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(img_hdr_path_list))

    def _preprocess_image(img_hdr_path_with_data_name):
        return preprocess_image(img_hdr_path_with_data_name, is_training, cfg)

    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=cfg.num_workers
    )  # autotuneはなんか遅かった・・・

    # 学習・validationはjit用にdrop_remainder=True、画像ログは端数batchも許可する。
    batch_size = int(cfg.batch_size) if batch_size is None else int(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # 他で使いやすいように辞書型で保持する
    def _make_batch_dict(*args):
        return make_batch_dict(*args, cfg)

    dataset = dataset.map(_make_batch_dict)
    dataset = dataset.prefetch(buffer_size=cfg.prefetch_size)

    return dataset
