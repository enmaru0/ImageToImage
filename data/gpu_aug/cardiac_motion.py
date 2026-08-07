import math

import tensorflow as tf


def _gather_pixels(images, y_indices, x_indices):
    """Gather pixels from a channels-last image batch using flattened indices."""
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    channels = tf.shape(images)[3]

    batch_offsets = tf.reshape(tf.range(batch_size) * height * width, (-1, 1, 1))
    flat_indices = batch_offsets + y_indices * width + x_indices
    flat_images = tf.reshape(images, (-1, channels))
    return tf.gather(flat_images, flat_indices)


def bilinear_sample_2d(images, source_y, source_x):
    """Sample 2D images at floating-point source coordinates with zero fill."""
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    dtype = images.dtype

    max_y = tf.cast(height - 1, dtype)
    max_x = tf.cast(width - 1, dtype)
    valid = (
        (source_y >= 0) & (source_y <= max_y) & (source_x >= 0) & (source_x <= max_x)
    )

    y0_float = tf.floor(source_y)
    x0_float = tf.floor(source_x)
    y1_float = y0_float + 1.0
    x1_float = x0_float + 1.0

    y0 = tf.cast(tf.clip_by_value(y0_float, 0.0, max_y), tf.int32)
    x0 = tf.cast(tf.clip_by_value(x0_float, 0.0, max_x), tf.int32)
    y1 = tf.cast(tf.clip_by_value(y1_float, 0.0, max_y), tf.int32)
    x1 = tf.cast(tf.clip_by_value(x1_float, 0.0, max_x), tf.int32)

    top_left = _gather_pixels(images, y0, x0)
    top_right = _gather_pixels(images, y0, x1)
    bottom_left = _gather_pixels(images, y1, x0)
    bottom_right = _gather_pixels(images, y1, x1)

    wy = source_y - y0_float
    wx = source_x - x0_float
    top = top_left * (1.0 - wx[..., None]) + top_right * wx[..., None]
    bottom = bottom_left * (1.0 - wx[..., None]) + bottom_right * wx[..., None]
    sampled = top * (1.0 - wy[..., None]) + bottom * wy[..., None]
    return sampled * tf.cast(valid[..., None], dtype)


def _sample_endpoint_parameters(
    batch_size,
    dtype,
    spacing_mm_yx,
    max_translation_mm_yx,
    max_rotation_deg,
    max_scale_delta,
    is_training,
    validation_translation_mm_yx,
    validation_rotation_deg,
    validation_scale_delta,
):
    spacing = tf.cast(tf.convert_to_tensor(tuple(spacing_mm_yx)), dtype)
    if is_training:
        max_translation = tf.cast(
            tf.convert_to_tensor(tuple(max_translation_mm_yx)), dtype
        )
        translation_mm = tf.random.uniform(
            (batch_size, 2), -max_translation, max_translation, dtype=dtype
        )
        rotation_deg = tf.random.uniform(
            (batch_size,), -max_rotation_deg, max_rotation_deg, dtype=dtype
        )
        scale_delta = tf.random.uniform(
            (batch_size,), -max_scale_delta, max_scale_delta, dtype=dtype
        )
    else:
        translation_mm = tf.broadcast_to(
            tf.cast(tf.convert_to_tensor(tuple(validation_translation_mm_yx)), dtype),
            (batch_size, 2),
        )
        rotation_deg = tf.fill((batch_size,), tf.cast(validation_rotation_deg, dtype))
        scale_delta = tf.fill((batch_size,), tf.cast(validation_scale_delta, dtype))

    translation_px = translation_mm / spacing[None]
    rotation_rad = rotation_deg * tf.cast(math.pi / 180.0, dtype)
    return translation_px, rotation_rad, scale_delta


def _sample_num_phases(batch_size, num_phases, num_phases_range, is_training):
    """Sample an odd phase count per volume while keeping an XLA-static max loop."""
    if not is_training or num_phases_range is None:
        return tf.fill((batch_size,), tf.cast(num_phases, tf.int32)), int(num_phases)

    min_phases, max_phases = (int(value) for value in num_phases_range)
    num_choices = (max_phases - min_phases) // 2 + 1
    choice = tf.random.uniform(
        (batch_size,), minval=0, maxval=num_choices, dtype=tf.int32
    )
    return min_phases + choice * 2, max_phases


def cardiac_motion_blur(
    imgs,
    img_msks,
    spacing_mm_yx,
    num_phases=5,
    num_phases_range=None,
    max_translation_mm_yx=(3.0, 3.0),
    max_rotation_deg=3.0,
    max_scale_delta=0.04,
    roi_center_yx=(0.5, 0.5),
    roi_sigma_ratio_yx=(0.25, 0.25),
    validation_translation_mm_yx=(2.0, -2.0),
    validation_rotation_deg=2.0,
    validation_scale_delta=0.025,
    is_training=True,
):
    """Approximate cardiac CT motion by averaging smooth in-plane heart motion.

    The same transform is applied to every Z slice, preventing independently
    sampled slice jitter. A Gaussian ROI blends the motion into the stationary
    surroundings. Warped values are normalized by a warped validity mask so
    padding does not create a dark edge that the model could learn to overshoot.
    """
    imgs = tf.convert_to_tensor(imgs)
    img_msks = tf.cast(img_msks, imgs.dtype)
    dtype = imgs.dtype
    batch_size = tf.shape(imgs)[0]
    depth = tf.shape(imgs)[1]
    height = tf.shape(imgs)[2]
    width = tf.shape(imgs)[3]

    # Treat Z as channels so one in-plane transform is shared by all slices.
    imgs_2d = tf.transpose(tf.squeeze(imgs, axis=-1), (0, 2, 3, 1))
    msks_2d = tf.transpose(tf.squeeze(img_msks, axis=-1), (0, 2, 3, 1))
    image_and_mask = tf.concat([imgs_2d * msks_2d, msks_2d], axis=-1)

    y = tf.cast(tf.range(height), dtype)
    x = tf.cast(tf.range(width), dtype)
    grid_y, grid_x = tf.meshgrid(y, x, indexing="ij")
    grid_y = grid_y[None]
    grid_x = grid_x[None]

    roi_center = tf.cast(tf.convert_to_tensor(tuple(roi_center_yx)), dtype)
    center_y = roi_center[0] * tf.cast(height - 1, dtype)
    center_x = roi_center[1] * tf.cast(width - 1, dtype)
    roi_sigma = tf.cast(tf.convert_to_tensor(tuple(roi_sigma_ratio_yx)), dtype)
    sigma_y = tf.maximum(roi_sigma[0] * tf.cast(height, dtype), 1.0)
    sigma_x = tf.maximum(roi_sigma[1] * tf.cast(width, dtype), 1.0)
    roi_weight = tf.exp(
        -0.5
        * (
            tf.square((grid_y - center_y) / sigma_y)
            + tf.square((grid_x - center_x) / sigma_x)
        )
    )[..., None]

    translation_px, rotation_rad, scale_delta = _sample_endpoint_parameters(
        batch_size,
        dtype,
        spacing_mm_yx,
        max_translation_mm_yx,
        max_rotation_deg,
        max_scale_delta,
        is_training,
        validation_translation_mm_yx,
        validation_rotation_deg,
        validation_scale_delta,
    )
    translation_y = translation_px[:, 0, None, None]
    translation_x = translation_px[:, 1, None, None]
    rotation_rad = rotation_rad[:, None, None]
    scale_delta = scale_delta[:, None, None]

    accumulated = tf.zeros_like(imgs_2d)
    phase_counts, max_loop_phases = _sample_num_phases(
        batch_size, num_phases, num_phases_range, is_training
    )
    phase_denominator = tf.cast(tf.maximum(phase_counts - 1, 1), dtype)
    for phase_index in range(max_loop_phases):
        active = phase_index < phase_counts
        bounded_index = tf.minimum(phase_index, phase_counts - 1)
        phase_position = -1.0 + 2.0 * tf.cast(bounded_index, dtype) / phase_denominator
        phase_position = phase_position[:, None, None]
        angle = phase_position * rotation_rad
        scale = 1.0 + phase_position * scale_delta
        cos_angle = tf.cos(angle) / scale
        sin_angle = tf.sin(angle) / scale

        shifted_x = grid_x - center_x - phase_position * translation_x
        shifted_y = grid_y - center_y - phase_position * translation_y
        source_x = cos_angle * shifted_x + sin_angle * shifted_y + center_x
        source_y = -sin_angle * shifted_x + cos_angle * shifted_y + center_y

        sampled = bilinear_sample_2d(image_and_mask, source_y, source_x)
        warped_img = sampled[..., :depth]
        warped_msk = sampled[..., depth:]
        warped_img = warped_img / tf.maximum(warped_msk, tf.cast(1e-6, dtype))
        warped_img = tf.where(warped_msk > 1e-6, warped_img, imgs_2d)
        phase_img = roi_weight * warped_img + (1.0 - roi_weight) * imgs_2d
        accumulated += phase_img * tf.cast(active[:, None, None, None], dtype)

    blurred_2d = accumulated / tf.cast(phase_counts[:, None, None, None], dtype)
    blurred = tf.transpose(blurred_2d, (0, 3, 1, 2))[..., None]
    blurred = tf.reshape(blurred, (batch_size, depth, height, width, 1))
    return tf.clip_by_value(blurred, 0.0, 1.0) * img_msks
