import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from trainer import CustomModel
from .cardiac_motion import _sample_num_phases, cardiac_motion_blur


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
