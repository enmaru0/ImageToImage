from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.predict_utils import (
    make_crop_initial_noise,
    make_difference_img,
    seed_save_dir,
)


def test_difference_is_signed_and_clipped_without_int16_wraparound():
    prediction = np.array([120, -200, 32767], np.int16)
    source = np.array([20, 100, -32768], np.int16)

    difference = make_difference_img(prediction, source)

    np.testing.assert_array_equal(difference, [100, -300, 32767])
    assert difference.dtype == np.int16


def test_multiple_seeds_use_separate_output_directories():
    base = Path("preds")

    assert seed_save_dir(base, [0], 0) == base
    assert seed_save_dir(base, [0, 1], 0) == base / "seed_0"
    assert seed_save_dir(base, [0, 1], 1) == base / "seed_1"


def test_crop_noise_is_reproducible_per_case_and_changes_with_seed():
    data = {
        "imgs": tf.zeros((2, 3, 8, 8, 1), tf.float32),
        "img_hdr_list": tf.constant(["case_a", "case_b"]),
    }

    seed_zero_first = make_crop_initial_noise(data, 1, seed=0).numpy()
    seed_zero_second = make_crop_initial_noise(data, 1, seed=0).numpy()
    seed_one = make_crop_initial_noise(data, 1, seed=1).numpy()

    np.testing.assert_array_equal(seed_zero_first, seed_zero_second)
    assert not np.array_equal(seed_zero_first, seed_one)
    assert not np.array_equal(seed_zero_first[0], seed_zero_first[1])


if __name__ == "__main__":
    test_difference_is_signed_and_clipped_without_int16_wraparound()
    test_multiple_seeds_use_separate_output_directories()
    test_crop_noise_is_reproducible_per_case_and_changes_with_seed()
