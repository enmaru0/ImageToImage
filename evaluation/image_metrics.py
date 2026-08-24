import tensorflow as tf


def _per_sample_denominator(mask, num_channels):
    mask = tf.cast(mask, tf.float32)
    denominator = tf.reduce_sum(mask, axis=(1, 2, 3, 4))
    denominator *= tf.cast(num_channels, denominator.dtype)
    valid = denominator > 0
    return tf.maximum(denominator, 1.0), valid


def masked_psnr(target, prediction, mask, max_val=1.0):
    """Return per-volume PSNR and whether each volume contains valid voxels."""
    error = tf.square(tf.cast(target, tf.float32) - tf.cast(prediction, tf.float32))
    mask = tf.cast(mask, error.dtype)
    denominator, valid = _per_sample_denominator(mask, tf.shape(error)[-1])
    mse = tf.reduce_sum(error * mask, axis=(1, 2, 3, 4)) / denominator
    max_val = tf.cast(max_val, mse.dtype)
    psnr = (
        10.0
        * tf.math.log(tf.square(max_val) / tf.maximum(mse, tf.cast(1e-8, mse.dtype)))
        / tf.math.log(tf.cast(10.0, mse.dtype))
    )
    return psnr, valid


def masked_mae_with_scale(target, prediction, mask, intensity_range):
    """Return per-volume MAE converted from normalized to physical intensity."""
    error = tf.abs(tf.cast(target, tf.float32) - tf.cast(prediction, tf.float32))
    mask = tf.cast(mask, error.dtype)
    denominator, valid = _per_sample_denominator(mask, tf.shape(error)[-1])
    normalized_mae = tf.reduce_sum(error * mask, axis=(1, 2, 3, 4)) / denominator
    intensity_range = tf.reshape(tf.cast(intensity_range, error.dtype), (-1,))
    return normalized_mae * intensity_range, valid


def _gaussian_kernel_2d(filter_size, filter_sigma, num_channels, dtype):
    radius = (filter_size - 1) / 2.0
    coordinates = tf.cast(tf.range(filter_size), dtype) - tf.cast(radius, dtype)
    gaussian = tf.exp(-0.5 * tf.square(coordinates / tf.cast(filter_sigma, dtype)))
    gaussian /= tf.reduce_sum(gaussian)
    kernel = gaussian[:, None] * gaussian[None, :]
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, tf.stack([1, 1, num_channels, 1]))


def _ssim_index_map(target_2d, prediction_2d, max_val, filter_size, filter_sigma):
    """Standard Gaussian-window SSIM map, retaining channels and spatial axes."""
    target_2d = tf.cast(target_2d, tf.float32)
    prediction_2d = tf.cast(prediction_2d, tf.float32)
    max_val = tf.cast(max_val, target_2d.dtype)
    target_2d = tf.clip_by_value(target_2d, 0.0, max_val)
    prediction_2d = tf.clip_by_value(prediction_2d, 0.0, max_val)

    kernel = _gaussian_kernel_2d(
        filter_size, filter_sigma, tf.shape(target_2d)[-1], target_2d.dtype
    )

    def weighted_mean(image):
        return tf.nn.depthwise_conv2d(
            image, kernel, strides=[1, 1, 1, 1], padding="VALID"
        )

    target_mean = weighted_mean(target_2d)
    prediction_mean = weighted_mean(prediction_2d)
    target_variance = weighted_mean(tf.square(target_2d)) - tf.square(target_mean)
    prediction_variance = weighted_mean(tf.square(prediction_2d)) - tf.square(
        prediction_mean
    )
    covariance = weighted_mean(target_2d * prediction_2d) - (
        target_mean * prediction_mean
    )

    c1 = tf.square(tf.cast(0.01, target_2d.dtype) * max_val)
    c2 = tf.square(tf.cast(0.03, target_2d.dtype) * max_val)
    luminance = (2.0 * target_mean * prediction_mean + c1) / (
        tf.square(target_mean) + tf.square(prediction_mean) + c1
    )
    contrast_structure = (2.0 * covariance + c2) / (
        target_variance + prediction_variance + c2
    )
    return luminance * contrast_structure


def masked_ssim_xy(
    target, prediction, mask, max_val=1.0, filter_size=11, filter_sigma=1.5
):
    """Return spacing-safe axial 2D SSIM averaged per 3D volume.

    The mask is averaged over the same Gaussian-window support and then used to
    weight the local SSIM map. This avoids background-dominated scores without
    introducing an artificial zero-valued boundary around the heart ROI.
    """
    target = tf.cast(target, tf.float32)
    prediction = tf.cast(prediction, tf.float32)
    mask = tf.cast(mask, tf.float32)
    batch_size = tf.shape(target)[0]
    height = tf.shape(target)[2]
    width = tf.shape(target)[3]
    num_channels = tf.shape(target)[4]

    target_2d = tf.reshape(target, (-1, height, width, num_channels))
    prediction_2d = tf.reshape(prediction, (-1, height, width, num_channels))
    mask_2d = tf.reshape(mask, (-1, height, width, 1))
    ssim_map = _ssim_index_map(
        target_2d,
        prediction_2d,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
    )
    ssim_map = tf.reduce_mean(ssim_map, axis=-1, keepdims=True)
    mask_kernel = _gaussian_kernel_2d(
        filter_size, filter_sigma, tf.constant(1, tf.int32), mask_2d.dtype
    )
    local_mask = tf.nn.depthwise_conv2d(
        mask_2d, mask_kernel, strides=[1, 1, 1, 1], padding="VALID"
    )

    weighted_ssim = tf.reshape(ssim_map * local_mask, (batch_size, -1))
    local_mask = tf.reshape(local_mask, (batch_size, -1))
    denominator = tf.reduce_sum(local_mask, axis=1)
    valid = denominator > 0
    score = tf.reduce_sum(weighted_ssim, axis=1) / tf.maximum(denominator, 1.0)
    return score, valid


def masked_z_gradient_mae(target, prediction, mask):
    """Return per-volume error of adjacent-slice differences inside the ROI."""
    target = tf.cast(target, tf.float32)
    prediction = tf.cast(prediction, tf.float32)
    mask = tf.cast(mask, tf.float32)
    target_gradient = target[:, 1:] - target[:, :-1]
    prediction_gradient = prediction[:, 1:] - prediction[:, :-1]
    pair_mask = mask[:, 1:] * mask[:, :-1]
    denominator, valid = _per_sample_denominator(
        pair_mask, tf.shape(target_gradient)[-1]
    )
    error = tf.abs(target_gradient - prediction_gradient)
    value = tf.reduce_sum(error * pair_mask, axis=(1, 2, 3, 4)) / denominator
    return value, valid


def masked_xy_edge_strength_ratio(target, prediction, mask, epsilon=1e-6):
    """Return prediction/target in-plane absolute-gradient strength per volume."""
    target = tf.cast(target, tf.float32)
    prediction = tf.cast(prediction, tf.float32)
    mask = tf.cast(mask, tf.float32)

    target_dy = target[:, :, 1:] - target[:, :, :-1]
    prediction_dy = prediction[:, :, 1:] - prediction[:, :, :-1]
    mask_dy = mask[:, :, 1:] * mask[:, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    prediction_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    reduce_axes = (1, 2, 3, 4)
    target_strength = tf.reduce_sum(tf.abs(target_dy) * mask_dy, axis=reduce_axes)
    target_strength += tf.reduce_sum(tf.abs(target_dx) * mask_dx, axis=reduce_axes)
    prediction_strength = tf.reduce_sum(
        tf.abs(prediction_dy) * mask_dy, axis=reduce_axes
    )
    prediction_strength += tf.reduce_sum(
        tf.abs(prediction_dx) * mask_dx, axis=reduce_axes
    )

    epsilon = tf.cast(epsilon, target_strength.dtype)
    valid = target_strength > epsilon
    ratio = prediction_strength / tf.maximum(target_strength, epsilon)
    return ratio, valid
