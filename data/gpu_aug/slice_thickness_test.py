import math

import numpy as np
import tensorflow as tf

from .slice_thickness import _additional_gaussian_sigma_vox, simulate_slice_thickness


def test_additional_sigma_accounts_for_existing_three_mm_profile():
    sigma_vox = _additional_gaussian_sigma_vox(3.0, 5.0, 3.0)
    expected = 4.0 / math.sqrt(8.0 * math.log(2.0)) / 3.0
    assert math.isclose(sigma_vox, expected)


def test_constant_signal_and_output_shape_are_preserved():
    imgs = tf.fill((2, 8, 6, 7, 1), 0.4)
    img_msks = tf.ones_like(imgs)
    output = simulate_slice_thickness(
        imgs,
        img_msks,
        spacing_mm_z=3.0,
        clean_thickness_mm=3.0,
        degraded_thickness_mm=5.0,
    )
    assert output.shape == imgs.shape
    np.testing.assert_allclose(output.numpy(), imgs.numpy(), atol=1e-6)


def test_z_high_frequency_is_reduced_without_xy_blur_and_is_xla_compatible():
    z_signal = tf.cast(tf.range(8) % 2, tf.float32)
    imgs = tf.broadcast_to(z_signal[None, :, None, None, None], (1, 8, 5, 6, 1))
    img_msks = tf.ones_like(imgs)

    apply_degradation = tf.function(
        lambda image, mask: simulate_slice_thickness(
            image,
            mask,
            spacing_mm_z=3.0,
            clean_thickness_mm=3.0,
            degraded_thickness_mm=5.0,
        ),
        jit_compile=True,
    )
    output = apply_degradation(imgs, img_msks)

    input_variation = tf.reduce_mean(tf.abs(imgs[:, 1:] - imgs[:, :-1]))
    output_variation = tf.reduce_mean(tf.abs(output[:, 1:] - output[:, :-1]))
    assert output.shape == imgs.shape
    assert float(output_variation) < float(input_variation)
    # Every XY position had the same Z signal, so Z-only processing must retain it.
    reference = tf.broadcast_to(output[:, :, :1, :1], tf.shape(output))
    np.testing.assert_allclose(reference.numpy(), output.numpy(), atol=1e-6)
