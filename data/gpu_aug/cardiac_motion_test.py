import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from trainer import CustomModel
from .cardiac_motion import (
    _localize_displacement,
    _phase_weight,
    _sample_num_phases,
    cardiac_motion_blur,
)


def _validation_kwargs(**overrides):
    kwargs = dict(
        spacing_mm_yx=(1.0, 1.0),
        num_phases=3,
        max_translation_mm_yx=(0.0, 0.0),
        max_rotation_deg=0.0,
        max_scale_delta=0.0,
        roi_center_yx=(0.5, 0.5),
        roi_sigma_ratio_yx=(0.3, 0.3),
        validation_translation_mm_yx=(0.0, 0.0),
        validation_rotation_deg=0.0,
        validation_scale_delta=0.0,
        is_training=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_identity_motion_keeps_image_unchanged():
    rng = np.random.default_rng(0)
    imgs = tf.constant(rng.random((2, 3, 12, 16, 1)), tf.float32)
    img_msks = tf.ones_like(imgs)

    output = cardiac_motion_blur(imgs, img_msks, **_validation_kwargs())

    np.testing.assert_allclose(output.numpy(), imgs.numpy(), atol=1e-6)


def test_mask_normalization_avoids_dark_padding_edge():
    img_msks = np.ones((1, 2, 16, 16, 1), np.float32)
    img_msks[:, :, :, :4] = 0.0
    imgs = tf.constant(img_msks)
    img_msks = tf.constant(img_msks)

    output = cardiac_motion_blur(
        imgs, img_msks, **_validation_kwargs(validation_translation_mm_yx=(0.0, 3.0))
    )

    valid_output = tf.boolean_mask(output, img_msks > 0).numpy()
    np.testing.assert_allclose(valid_output, 1.0, atol=1e-6)


def test_motion_is_consistent_between_identical_slices_and_xla_compatible():
    plane = tf.reshape(tf.linspace(0.0, 1.0, 16 * 16), (1, 1, 16, 16, 1))
    imgs = tf.repeat(plane, repeats=4, axis=1)
    img_msks = tf.ones_like(imgs)
    kwargs = _validation_kwargs(
        validation_translation_mm_yx=(2.0, -1.0),
        validation_rotation_deg=2.0,
        validation_scale_delta=0.03,
    )

    apply_motion = tf.function(
        lambda image, mask: cardiac_motion_blur(image, mask, **kwargs), jit_compile=True
    )
    output = apply_motion(imgs, img_msks)

    assert output.shape == imgs.shape
    for z_index in range(1, 4):
        np.testing.assert_allclose(
            output[:, 0].numpy(), output[:, z_index].numpy(), atol=1e-6
        )


def test_random_num_phases_uses_only_odd_values_and_validation_is_fixed():
    training_counts, max_loop_phases = _sample_num_phases(
        batch_size=128, num_phases=5, num_phases_range=(3, 7), is_training=True
    )
    assert max_loop_phases == 7
    assert set(training_counts.numpy()).issubset({3, 5, 7})

    validation_counts, max_loop_phases = _sample_num_phases(
        batch_size=8, num_phases=5, num_phases_range=(3, 7), is_training=False
    )
    assert max_loop_phases == 5
    np.testing.assert_array_equal(validation_counts.numpy(), np.full(8, 5))


def test_bimodal_phase_weight_emphasizes_separated_endpoint_phases():
    phase_position = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0])
    weights = _phase_weight(
        phase_position=phase_position,
        active=tf.ones(5, tf.bool),
        phase_weight_mode="bimodal",
        bimodal_peak_sigma=tf.fill((5,), 0.25),
        bimodal_balance=tf.fill((5,), 0.5),
        uniform_phase_weight_mix=0.05,
    ).numpy()

    assert weights[0] > weights[1] > weights[2]
    assert weights[4] > weights[3] > weights[2]
    np.testing.assert_allclose(weights[0], weights[4], atol=1e-6)


def test_z_phase_offset_creates_smooth_slice_dependent_motion():
    plane = tf.reshape(tf.linspace(0.0, 1.0, 20 * 20), (1, 1, 20, 20, 1))
    imgs = tf.repeat(plane, repeats=5, axis=1)
    img_msks = tf.ones_like(imgs)

    output = cardiac_motion_blur(
        imgs,
        img_msks,
        **_validation_kwargs(
            num_phases=5,
            validation_translation_mm_yx=(0.0, 4.0),
            validation_z_phase_offset=0.5,
        ),
    ).numpy()

    assert not np.allclose(output[:, 0], output[:, -1], atol=1e-5)
    adjacent_difference = np.mean(np.abs(output[:, 1:] - output[:, :-1]))
    endpoint_difference = np.mean(np.abs(output[:, -1] - output[:, 0]))
    assert 0 < adjacent_difference < endpoint_difference


def test_center_preserving_keeps_asymmetric_double_edge_centered():
    image = np.zeros((1, 1, 65, 65, 1), np.float32)
    image[0, 0, 32, 32, 0] = 1.0
    imgs = tf.constant(image)
    img_msks = tf.ones_like(imgs)
    kwargs = _validation_kwargs(
        num_phases=5,
        roi_sigma_ratio_yx=(1000.0, 1000.0),
        phase_weight_mode="bimodal",
        bimodal_peak_sigma_range=(0.25, 0.25),
        bimodal_balance_range=(0.8, 0.8),
        uniform_phase_weight_mix=0.05,
        validation_translation_mm_yx=(0.0, 8.0),
        validation_bimodal_peak_sigma=0.25,
        validation_bimodal_balance=0.8,
    )

    uncentered = cardiac_motion_blur(imgs, img_msks, center_preserving=False, **kwargs)
    centered = cardiac_motion_blur(imgs, img_msks, center_preserving=True, **kwargs)
    x_coords = tf.cast(tf.range(65), tf.float32)[None, :]
    uncentered_x = tf.reduce_sum(uncentered[0, 0, :, :, 0] * x_coords) / tf.reduce_sum(
        uncentered
    )
    centered_x = tf.reduce_sum(centered[0, 0, :, :, 0] * x_coords) / tf.reduce_sum(
        centered
    )

    np.testing.assert_allclose(centered_x.numpy(), 32.0, atol=0.05)
    assert abs(float(centered_x) - 32.0) < abs(float(uncentered_x) - 32.0)


def test_roi_attenuates_displacement_instead_of_blending_intensities():
    grid_y = tf.constant([[[0.0, 1.0, 2.0]]])
    grid_x = tf.constant([[[2.0, 3.0, 4.0]]])
    transformed_y = grid_y + 4.0
    transformed_x = grid_x - 2.0
    roi_weight = tf.constant([[[[0.0], [0.25], [1.0]]]])

    source_y, source_x = _localize_displacement(
        grid_y, grid_x, transformed_y, transformed_x, roi_weight
    )

    np.testing.assert_allclose(source_y.numpy(), [[[0.0, 2.0, 6.0]]])
    np.testing.assert_allclose(source_x.numpy(), [[[2.0, 2.5, 2.0]]])


def test_self_supervised_source_is_created_from_clean_target():
    class IdentitySignalAugModel(CustomModel):
        @staticmethod
        def gpu_aug(imgs, img_msks, min_clip_vals, max_clip_vals, cfg):
            del min_clip_vals, max_clip_vals, cfg
            return imgs * img_msks

    cfg = OmegaConf.create(
        {
            "training_mode": "self_supervised_deblur",
            "aug": {"affine": {"norm_spacing_zyx": [1.0, 1.0, 1.0]}},
            "self_supervised_deblur": {
                "degradation_type": "cardiac_motion",
                "cardiac_motion": _validation_kwargs(
                    is_training=True, validation_translation_mm_yx=(0.0, 0.0)
                ),
            },
        }
    )
    del cfg.self_supervised_deblur.cardiac_motion.spacing_mm_yx
    del cfg.self_supervised_deblur.cardiac_motion.is_training
    source_before_degradation = tf.zeros((1, 2, 8, 8, 1), tf.float32)
    clean_target = tf.fill((1, 2, 8, 8, 1), 0.4)
    img_msks = tf.ones_like(clean_target)
    clip_values = tf.constant([0.0])

    source, target = IdentitySignalAugModel.prepare_training_images(
        source_before_degradation,
        clean_target,
        img_msks,
        clip_values,
        clip_values,
        clip_values,
        clip_values,
        cfg,
    )

    np.testing.assert_allclose(target.numpy(), clean_target.numpy(), atol=1e-6)
    np.testing.assert_allclose(source.numpy(), clean_target.numpy(), atol=1e-6)
