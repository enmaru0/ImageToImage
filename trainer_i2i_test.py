from types import SimpleNamespace

from keras import Input, layers
from keras.api.optimizers import Adam
import numpy as np
from omegaconf import OmegaConf
import tensorflow as tf

from trainer import CustomModel


def _rfr_cfg(prediction_type="residual", gradient_time_reweight=False):
    return SimpleNamespace(
        model=SimpleNamespace(num_channel=1),
        i2i_rfr=SimpleNamespace(
            prediction_type=prediction_type,
            inference_steps=3,
            clip_output=True,
            p=1.0,
            t_min=0.01,
        ),
        loss=SimpleNamespace(
            rfr=SimpleNamespace(weight=1.0),
            gradient=SimpleNamespace(weight=0.01, time_reweight=gradient_time_reweight),
        ),
    )


def _self_supervised_cfg(identity_probability=0.0):
    return OmegaConf.create(
        {
            "training_mode": "self_supervised_deblur",
            "aug": {
                "random_normalize": {
                    "prob": 0.0,
                    "random_center_deviation": 0.0,
                    "random_range_deviation": 0.0,
                },
                "random_gamma_correction": {
                    "prob": 0.0,
                    "random_gamma_range": [1.0, 1.0],
                },
                "random_sharpness": {
                    "prob": 1.0,
                    "sigma": 3.0,
                    "alpha_range": [1.0, 1.5],
                },
                "random_gauss_filter": {"prob": 0.0, "sigma_range": [0.5, 1.0]},
                "random_gauss_noise": {"prob": 1.0, "stddev_range": [0.02, 0.03]},
            },
            "self_supervised_deblur": {"identity_probability": identity_probability},
        }
    )


def test_image_and_residual_prediction_spaces_round_trip_to_target():
    source = tf.fill((1, 2, 4, 4, 1), 0.25)
    target = tf.fill(source.shape, 0.8)

    residual_cfg = _rfr_cfg("residual")
    residual = CustomModel.make_rfr_target(source, target, residual_cfg)
    reconstructed = CustomModel.reconstruct_rfr_prediction(
        source, residual, residual_cfg
    )
    np.testing.assert_allclose(residual.numpy(), 0.55, atol=1e-6)
    np.testing.assert_allclose(reconstructed.numpy(), target.numpy(), atol=1e-6)

    image_cfg = _rfr_cfg("image")
    image_target = CustomModel.make_rfr_target(source, target, image_cfg)
    reconstructed = CustomModel.reconstruct_rfr_prediction(
        source, image_target, image_cfg
    )
    np.testing.assert_array_equal(image_target.numpy(), target.numpy())
    np.testing.assert_array_equal(reconstructed.numpy(), target.numpy())


def test_old_configs_keep_image_prediction_and_time_reweighted_gradient():
    old_cfg = SimpleNamespace(
        i2i_rfr=SimpleNamespace(),
        loss=SimpleNamespace(gradient=SimpleNamespace(weight=0.05)),
    )
    assert CustomModel.get_prediction_type(old_cfg) == "image"

    target = np.zeros((1, 1, 4, 4, 1), np.float32)
    target[:, :, :, 2:, :] = 1.0
    prediction = np.zeros_like(target)
    mask = tf.ones_like(target)
    model = CustomModel()
    model.cfg = old_cfg
    loss = model.compute_gradient_loss(
        target, prediction, mask, t=tf.constant([[[[[0.5]]]]])
    )
    unweighted = model.compute_gradient_loss(target, prediction, mask, t=None)
    np.testing.assert_allclose(loss.numpy(), 2.0 * unweighted.numpy())


def test_rfr_inference_adds_residual_state_to_source_only_in_residual_mode():
    class ConstantStateModel(CustomModel):
        def call(self, inputs, training=False):
            del training
            return tf.ones_like(inputs[0][..., :1]) * 0.2

    source = tf.fill((1, 2, 4, 4, 1), 0.3)
    mask = tf.ones_like(source)
    initial_noise = tf.zeros_like(source)

    residual_model = ConstantStateModel()
    residual_model.cfg = _rfr_cfg("residual")
    residual_prediction = residual_model.i2i_rfr_inference(
        source, mask, initial_noise=initial_noise
    )
    np.testing.assert_allclose(residual_prediction.numpy(), 0.5, atol=1e-6)

    image_model = ConstantStateModel()
    image_model.cfg = _rfr_cfg("image")
    image_prediction = image_model.i2i_rfr_inference(
        source, mask, initial_noise=initial_noise
    )
    np.testing.assert_allclose(image_prediction.numpy(), 0.2, atol=1e-6)


def test_gradient_loss_is_not_time_reweighted_in_new_configuration():
    target = np.zeros((1, 1, 8, 8, 1), np.float32)
    target[:, :, :, 4:, :] = 1.0
    prediction = np.zeros_like(target)
    prediction[:, :, :, 5:, :] = 1.0
    mask = tf.ones_like(target)

    model = CustomModel()
    model.cfg = _rfr_cfg(gradient_time_reweight=False)
    loss_at_quarter = model.compute_gradient_loss(
        target, prediction, mask, t=tf.constant([[[[[0.25]]]]])
    )
    loss_at_one = model.compute_gradient_loss(
        target, prediction, mask, t=tf.constant([[[[[1.0]]]]])
    )
    np.testing.assert_allclose(loss_at_quarter.numpy(), loss_at_one.numpy())

    model.cfg = _rfr_cfg(gradient_time_reweight=True)
    legacy_loss = model.compute_gradient_loss(
        target, prediction, mask, t=tf.constant([[[[[0.25]]]]])
    )
    np.testing.assert_allclose(legacy_loss.numpy(), 4.0 * loss_at_one.numpy())


def test_source_artifacts_do_not_modify_self_supervised_clean_target():
    class NoDegradationModel(CustomModel):
        @staticmethod
        def apply_self_supervised_deblur(
            imgs, img_msks, cfg, is_training, heart_msks=None
        ):
            del cfg, is_training, heart_msks
            return imgs * img_msks

    cfg = _self_supervised_cfg(identity_probability=0.0)
    clean = tf.reshape(tf.linspace(0.1, 0.9, 8 * 16 * 16), (1, 8, 16, 16, 1))
    mask = tf.ones_like(clean)
    clip_min = tf.constant([0.0])
    clip_max = tf.constant([1.0])

    source, target = NoDegradationModel.prepare_training_images(
        tf.zeros_like(clean), clean, mask, clip_min, clip_max, clip_min, clip_max, cfg
    )

    np.testing.assert_allclose(target.numpy(), clean.numpy(), atol=1e-6)
    assert not np.allclose(source.numpy(), target.numpy())


def test_identity_probability_can_skip_all_source_degradation():
    degraded = tf.zeros((4, 2, 4, 4, 1), tf.float32)
    clean = tf.ones_like(degraded)

    identity_cfg = _self_supervised_cfg(identity_probability=1.0)
    output = CustomModel.mix_identity_samples(
        degraded, clean, identity_cfg, is_training=True
    )
    np.testing.assert_array_equal(output.numpy(), clean.numpy())

    degraded_cfg = _self_supervised_cfg(identity_probability=0.0)
    output = CustomModel.mix_identity_samples(
        degraded, clean, degraded_cfg, is_training=True
    )
    np.testing.assert_array_equal(output.numpy(), degraded.numpy())

    validation_output = CustomModel.mix_identity_samples(
        degraded, clean, identity_cfg, is_training=False
    )
    np.testing.assert_array_equal(validation_output.numpy(), degraded.numpy())

    quarter_cfg = _self_supervised_cfg(identity_probability=0.25)
    large_degraded = tf.zeros((256, 2, 4, 4, 1), tf.float32)
    large_clean = tf.ones_like(large_degraded)
    mixed = CustomModel.mix_identity_samples(
        large_degraded, large_clean, quarter_cfg, is_training=True
    ).numpy()
    per_sample = mixed.reshape(256, -1)
    assert np.all((per_sample == 0.0) | (per_sample == 1.0))
    assert np.all(np.ptp(per_sample, axis=1) == 0.0)
    identity_fraction = float(np.mean(per_sample[:, 0]))
    assert 0.15 < identity_fraction < 0.35


def test_residual_train_step_is_xla_compatible():
    class NoAugModel(CustomModel):
        @classmethod
        def prepare_training_images(
            cls,
            imgs,
            target_imgs,
            img_msks,
            min_clip_vals,
            max_clip_vals,
            target_min_clip_vals,
            target_max_clip_vals,
            cfg,
            heart_msks=None,
        ):
            del (
                cls,
                min_clip_vals,
                max_clip_vals,
                target_min_clip_vals,
                target_max_clip_vals,
                cfg,
                heart_msks,
            )
            return imgs * img_msks, target_imgs * img_msks

    image_input = Input((2, 4, 4, 2), name="image")
    mask_input = Input((2, 4, 4, 1), name="mask")
    prediction = layers.Conv3D(1, 1)(image_input)
    prediction = layers.Add()([prediction, layers.Rescaling(0.0)(mask_input)])
    model = NoAugModel(inputs=[image_input, mask_input], outputs=prediction)
    model.cfg = SimpleNamespace(
        bit_info=SimpleNamespace(heart_bit=6, padding_bit=15),
        model=SimpleNamespace(num_channel=1),
        i2i_rfr=SimpleNamespace(prediction_type="residual", p=1.0, t_min=0.01),
        loss=SimpleNamespace(
            rfr=SimpleNamespace(weight=1.0),
            gradient=SimpleNamespace(weight=0.01, time_reweight=False),
        ),
    )
    model.compile(optimizer=Adam(1e-3))
    data = {
        "imgs": tf.fill((1, 2, 4, 4, 1), 0.3),
        "target_imgs": tf.fill((1, 2, 4, 4, 1), 0.5),
        "msks": tf.zeros((1, 2, 4, 4, 1), tf.uint16),
        "min_clip_vals": tf.constant([0.0]),
        "max_clip_vals": tf.constant([1.0]),
        "target_min_clip_vals": tf.constant([0.0]),
        "target_max_clip_vals": tf.constant([1.0]),
    }

    train_step = tf.function(model.train_step, jit_compile=True)
    results = train_step(data)

    assert int(model.optimizer.iterations.numpy()) == 1
    assert all(np.isfinite(float(value.numpy())) for value in results.values())


if __name__ == "__main__":
    test_image_and_residual_prediction_spaces_round_trip_to_target()
    test_old_configs_keep_image_prediction_and_time_reweighted_gradient()
    test_rfr_inference_adds_residual_state_to_source_only_in_residual_mode()
    test_gradient_loss_is_not_time_reweighted_in_new_configuration()
    test_source_artifacts_do_not_modify_self_supervised_clean_target()
    test_identity_probability_can_skip_all_source_degradation()
    test_residual_train_step_is_xla_compatible()
