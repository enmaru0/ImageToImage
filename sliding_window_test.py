import numpy as np

from sliding_window import (
    importance_map,
    resample_volume,
    sliding_window_inference,
    sliding_window_starts,
)


def test_resample_volume_uses_exact_requested_size():
    volume = np.arange(3 * 5 * 7, dtype=np.int16).reshape(3, 5, 7)
    output = resample_volume(volume, (4, 8, 10))
    assert output.shape == (4, 8, 10)
    assert output.dtype == np.float32


def test_starts_cover_axis_end_without_duplicates():
    assert sliding_window_starts(20, 8, 0.5) == [0, 4, 8, 12]
    assert sliding_window_starts(19, 8, 0.5) == [0, 4, 8, 11]
    assert sliding_window_starts(6, 8, 0.5) == [0]


def test_importance_map_is_positive_and_center_weighted():
    weight = importance_map((3, 8, 8))
    assert weight.shape == (3, 8, 8)
    assert np.min(weight) > 0
    assert weight[1, 3, 3] > weight[0, 0, 0]


def test_overlap_reconstruction_has_no_window_seams():
    volume = np.arange(7 * 13 * 15, dtype=np.float32).reshape(7, 13, 15)

    def double_patch(image_patch, valid_patch, initial_noise):
        del valid_patch, initial_noise
        return (image_patch * 2.0)[..., None]

    output = sliding_window_inference(
        volume, window_size_zyx=(4, 8, 8), overlap=0.5, predict_patch=double_patch
    )

    assert output.shape == volume.shape + (1,)
    np.testing.assert_allclose(output[..., 0], volume * 2.0, rtol=1e-6, atol=1e-4)


def test_volume_smaller_than_window_is_padded_and_cropped_back():
    volume = np.full((2, 5, 6), 3.0, np.float32)

    def identity_patch(image_patch, valid_patch, initial_noise):
        del valid_patch, initial_noise
        return image_patch[..., None]

    output = sliding_window_inference(
        volume, window_size_zyx=(4, 8, 8), overlap=0.25, predict_patch=identity_patch
    )

    assert output.shape == volume.shape + (1,)
    np.testing.assert_allclose(output[..., 0], volume, atol=1e-6)


def test_overlapping_windows_share_the_same_initial_noise():
    volume = np.zeros((4, 10, 10), np.float32)

    def return_noise(image_patch, valid_patch, initial_noise):
        del image_patch, valid_patch
        return initial_noise

    first = sliding_window_inference(
        volume,
        window_size_zyx=(4, 6, 6),
        overlap=0.5,
        predict_patch=return_noise,
        seed=123,
    )
    second = sliding_window_inference(
        volume,
        window_size_zyx=(4, 6, 6),
        overlap=0.5,
        predict_patch=return_noise,
        seed=123,
    )

    np.testing.assert_array_equal(first, second)
