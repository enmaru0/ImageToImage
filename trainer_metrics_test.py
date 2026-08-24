from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from trainer import CustomModel


def _metric_model():
    model = CustomModel()
    model.cfg = SimpleNamespace(
        evaluation_metrics=SimpleNamespace(
            validation_seed=7,
            ssim_filter_size=11,
            ssim_filter_sigma=1.5,
            edge_epsilon=1e-6,
        ),
        model=SimpleNamespace(num_channel=1),
    )
    _ = model.metrics_dict
    return model


def test_validation_metrics_are_logged_and_train_results_exclude_them():
    model = _metric_model()
    target = tf.reshape(tf.linspace(0.0, 1.0, 2 * 3 * 16 * 16), (2, 3, 16, 16, 1))
    prediction = target * 0.9
    image_mask = tf.ones_like(target)
    heart_mask = tf.concat(
        [tf.ones((2, 3, 16, 8, 1), tf.float32), tf.zeros((2, 3, 16, 8, 1), tf.float32)],
        axis=3,
    )

    @tf.function(jit_compile=True)
    def update_metrics():
        model._update_evaluation_metrics(
            target,
            prediction,
            image_mask,
            heart_mask,
            intensity_range=tf.constant([2048.0, 2048.0]),
        )
        return model._get_metrics_result(include_evaluation=True)

    results = update_metrics()

    assert set(CustomModel.EVALUATION_METRIC_NAMES) <= set(results)
    assert not set(CustomModel.EVALUATION_METRIC_NAMES) & set(
        model._get_metrics_result(include_evaluation=False)
    )
    assert all(
        np.isfinite(float(results[name].numpy()))
        for name in CustomModel.EVALUATION_METRIC_NAMES
    )
    np.testing.assert_allclose(
        results["xy_edge_strength_ratio"].numpy(), 0.9, atol=1e-5
    )


def test_validation_noise_is_reproducible():
    model = _metric_model()
    images = tf.zeros((2, 3, 16, 16, 1), tf.float32)

    first = model._make_validation_initial_noise(images)
    second = model._make_validation_initial_noise(images)

    np.testing.assert_array_equal(first.numpy(), second.numpy())


if __name__ == "__main__":
    test_validation_metrics_are_logged_and_train_results_exclude_them()
    test_validation_noise_is_reproducible()
