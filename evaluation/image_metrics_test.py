import numpy as np
import tensorflow as tf

from .image_metrics import (
    masked_mae_with_scale,
    masked_psnr,
    masked_ssim_xy,
    masked_xy_edge_strength_ratio,
    masked_z_gradient_mae,
)


def _volume(values):
    return tf.constant(values, tf.float32)[None, :, :, :, None]


def test_identical_images_have_expected_reference_metrics():
    rng = np.random.default_rng(0)
    target = tf.constant(rng.random((2, 3, 16, 16, 1)), tf.float32)
    mask = tf.ones((2, 3, 16, 16, 1), tf.float32)

    ssim, ssim_valid = masked_ssim_xy(target, target, mask)
    psnr, psnr_valid = masked_psnr(target, target, mask)
    z_error, z_valid = masked_z_gradient_mae(target, target, mask)
    edge_ratio, edge_valid = masked_xy_edge_strength_ratio(target, target, mask)

    np.testing.assert_allclose(ssim.numpy(), 1.0, atol=1e-5)
    np.testing.assert_array_equal(ssim_valid.numpy(), [True, True])
    np.testing.assert_allclose(psnr.numpy(), 80.0, atol=1e-5)
    np.testing.assert_array_equal(psnr_valid.numpy(), [True, True])
    np.testing.assert_allclose(z_error.numpy(), 0.0, atol=1e-6)
    np.testing.assert_array_equal(z_valid.numpy(), [True, True])
    np.testing.assert_allclose(edge_ratio.numpy(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(edge_valid.numpy(), [True, True])


def test_metrics_use_only_masked_roi_and_convert_to_hu():
    target = tf.zeros((1, 2, 16, 16, 1), tf.float32)
    prediction = tf.tensor_scatter_nd_update(
        target, indices=[[0, 0, 8, 8, 0], [0, 1, 8, 8, 0]], updates=[0.25, 0.25]
    )
    heart_mask = tf.zeros_like(target)
    heart_mask = tf.tensor_scatter_nd_update(
        heart_mask, indices=[[0, 0, 8, 8, 0], [0, 1, 8, 8, 0]], updates=[1.0, 1.0]
    )

    mae_hu, valid = masked_mae_with_scale(
        target, prediction, heart_mask, intensity_range=[2048.0]
    )

    np.testing.assert_allclose(mae_hu.numpy(), 512.0)
    np.testing.assert_array_equal(valid.numpy(), [True])


def test_z_gradient_detects_slice_discontinuity():
    target = _volume(np.zeros((3, 4, 4), np.float32))
    prediction = _volume(
        np.stack(
            [
                np.zeros((4, 4), np.float32),
                np.ones((4, 4), np.float32),
                np.ones((4, 4), np.float32),
            ]
        )
    )
    mask = tf.ones_like(target)

    z_error, valid = masked_z_gradient_mae(target, prediction, mask)

    np.testing.assert_allclose(z_error.numpy(), 0.5)
    np.testing.assert_array_equal(valid.numpy(), [True])


def test_edge_ratio_reports_over_sharpening_direction():
    base = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (2, 3, 16, 1))
    target = tf.constant(base[..., None])
    prediction = target * 2.0
    mask = tf.ones_like(target)

    edge_ratio, valid = masked_xy_edge_strength_ratio(target, prediction, mask)

    np.testing.assert_allclose(edge_ratio.numpy(), 2.0, atol=1e-5)
    np.testing.assert_array_equal(valid.numpy(), [True, True])


def test_empty_mask_is_reported_invalid():
    target = tf.zeros((1, 2, 16, 16, 1), tf.float32)
    mask = tf.zeros_like(target)

    _, psnr_valid = masked_psnr(target, target, mask)
    _, ssim_valid = masked_ssim_xy(target, target, mask)
    _, z_valid = masked_z_gradient_mae(target, target, mask)
    _, edge_valid = masked_xy_edge_strength_ratio(target, target, mask)

    np.testing.assert_array_equal(psnr_valid.numpy(), [False])
    np.testing.assert_array_equal(ssim_valid.numpy(), [False])
    np.testing.assert_array_equal(z_valid.numpy(), [False])
    np.testing.assert_array_equal(edge_valid.numpy(), [False])


def test_metrics_are_xla_compilable():
    target = tf.reshape(tf.linspace(0.0, 1.0, 2 * 3 * 16 * 16), (2, 3, 16, 16, 1))
    prediction = target * 0.9
    mask = tf.ones_like(target)

    @tf.function(jit_compile=True)
    def compiled_metrics(target, prediction, mask):
        ssim, _ = masked_ssim_xy(target, prediction, mask)
        psnr, _ = masked_psnr(target, prediction, mask)
        z_error, _ = masked_z_gradient_mae(target, prediction, mask)
        edge_ratio, _ = masked_xy_edge_strength_ratio(target, prediction, mask)
        return ssim, psnr, z_error, edge_ratio

    values = compiled_metrics(target, prediction, mask)

    assert all(np.all(np.isfinite(value.numpy())) for value in values)


if __name__ == "__main__":
    test_identical_images_have_expected_reference_metrics()
    test_metrics_use_only_masked_roi_and_convert_to_hu()
    test_z_gradient_detects_slice_discontinuity()
    test_edge_ratio_reports_over_sharpening_direction()
    test_empty_mask_is_reported_invalid()
    test_metrics_are_xla_compilable()
