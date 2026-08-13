import numpy as np
import tensorflow as tf

from .image import masked_xy_gradient_loss


def test_gradient_loss_is_zero_for_identical_images():
    image = tf.reshape(tf.linspace(0.0, 1.0, 2 * 4 * 5), (1, 2, 4, 5, 1))
    mask = tf.ones_like(image)

    loss = masked_xy_gradient_loss(image, image, mask)

    np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-7)


def test_gradient_loss_detects_double_edge_without_penalizing_constant_offset():
    target = np.zeros((1, 1, 8, 8, 1), np.float32)
    target[:, :, :, 4:, :] = 1.0
    double_edge = np.zeros_like(target)
    double_edge[:, :, :, 3:5, :] = 0.5
    double_edge[:, :, :, 5:, :] = 1.0
    constant_offset = np.clip(target + 0.2, 0.0, 1.0)
    mask = tf.ones_like(target)

    double_edge_loss = masked_xy_gradient_loss(target, double_edge, mask)
    offset_loss = masked_xy_gradient_loss(target, constant_offset, mask)

    assert float(double_edge_loss) > float(offset_loss)
