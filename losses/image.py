import tensorflow as tf
from keras.src import ops


def masked_xy_gradient_loss(target_imgs, preds, img_msks, time_weight=None):
    """L1 distance between target/prediction gradients in the in-plane axes."""

    def axis_loss(axis):
        if axis == 2:
            target_gradient = target_imgs[:, :, 1:, :, :] - target_imgs[:, :, :-1, :, :]
            pred_gradient = preds[:, :, 1:, :, :] - preds[:, :, :-1, :, :]
            pair_mask = img_msks[:, :, 1:, :, :] * img_msks[:, :, :-1, :, :]
        else:
            target_gradient = target_imgs[:, :, :, 1:, :] - target_imgs[:, :, :, :-1, :]
            pred_gradient = preds[:, :, :, 1:, :] - preds[:, :, :, :-1, :]
            pair_mask = img_msks[:, :, :, 1:, :] * img_msks[:, :, :, :-1, :]

        error = ops.abs(target_gradient - pred_gradient)
        if time_weight is not None:
            error = error / tf.maximum(time_weight, tf.cast(1e-6, error.dtype))
        num_channels = tf.cast(tf.shape(target_imgs)[-1], error.dtype)
        denominator = tf.maximum(ops.sum(pair_mask) * num_channels, 1.0)
        return ops.sum(error * pair_mask) / denominator

    return 0.5 * (axis_loss(2) + axis_loss(3))
