from pathlib import Path

import numpy as np
import tensorflow as tf


def make_difference_img(prediction, source):
    """Return signed prediction-source intensity without int16 wraparound."""
    difference = np.asarray(prediction, np.float32) - np.asarray(source, np.float32)
    int16_info = np.iinfo(np.int16)
    return np.clip(np.rint(difference), int16_info.min, int16_info.max).astype(np.int16)


def seed_save_dir(base_save_dir, seeds, seed):
    """Keep legacy paths for one seed and isolate outputs for seed comparisons."""
    base_save_dir = Path(base_save_dir)
    if len(seeds) == 1:
        return base_save_dir
    return base_save_dir / f"seed_{seed}"


def make_crop_initial_noise(data, num_output_channels, seed):
    """Create reproducible per-case I2I-RFR noise for crop inference."""
    image_shape = tf.shape(data["imgs"])
    sample_shape = tf.concat(
        [image_shape[1:-1], tf.constant([num_output_channels], tf.int32)], axis=0
    )
    case_hashes = tf.strings.to_hash_bucket_fast(
        data["img_hdr_list"], num_buckets=(1 << 31) - 1
    )
    base_seed = tf.cast(seed, tf.int32)

    def make_sample(case_hash):
        stateless_seed = tf.stack([base_seed, tf.cast(case_hash, tf.int32)])
        return tf.random.stateless_normal(
            sample_shape, seed=stateless_seed, dtype=tf.float32
        )

    return tf.map_fn(make_sample, case_hashes, fn_output_signature=tf.float32)
