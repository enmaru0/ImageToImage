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


def _sample_scalar_range(batch_size, value_range, validation_value, is_training, dtype):
    """Sample one scalar per volume, or use a reproducible validation value."""
    if not is_training:
        return tf.fill((batch_size,), tf.cast(validation_value, dtype))

    min_value, max_value = (float(value) for value in value_range)
    return tf.random.uniform(
        (batch_size,), minval=min_value, maxval=max_value, dtype=dtype
    )


def _phase_weight(
    phase_position,
    active,
    phase_weight_mode,
    bimodal_peak_sigma,
    bimodal_balance,
    uniform_phase_weight_mix,
):
    """Return per-volume exposure weights for one simulated cardiac phase."""
    dtype = phase_position.dtype
    if phase_weight_mode == "uniform":
        weight = tf.ones_like(phase_position)
    elif phase_weight_mode == "bimodal":
        sigma = tf.maximum(bimodal_peak_sigma, tf.cast(1e-3, dtype))
        negative_peak = tf.exp(-0.5 * tf.square((phase_position + 1.0) / sigma))
        positive_peak = tf.exp(-0.5 * tf.square((phase_position - 1.0) / sigma))
        bimodal = (
            bimodal_balance * negative_peak + (1.0 - bimodal_balance) * positive_peak
        )
        uniform_mix = tf.cast(uniform_phase_weight_mix, dtype)
        weight = (1.0 - uniform_mix) * bimodal + uniform_mix
    else:
        raise ValueError(f"Unsupported phase_weight_mode: {phase_weight_mode}")

    return weight * tf.cast(active, dtype)


def _localize_displacement(grid_y, grid_x, transformed_y, transformed_x, roi_weight):
    """Attenuate the displacement field, not the warped image intensity."""
    roi_weight = tf.squeeze(roi_weight, axis=-1)
    source_y = grid_y + roi_weight * (transformed_y - grid_y)
    source_x = grid_x + roi_weight * (transformed_x - grid_x)
    return source_y, source_x


def _soften_motion_mask(motion_msks, softening_px):
    """Create a soft in-plane motion ROI without mixing adjacent CT slices."""
    motion_msks = tf.cast(motion_msks > 0, tf.float32)
    if softening_px <= 0:
        return motion_msks

    kernel_size = 2 * int(softening_px) + 1
    # Repeated box filtering approximates a smooth bell-shaped transition and is
    # inexpensive/XLA-compatible. The Z kernel remains one because slice spacing
    # is substantially coarser than in-plane spacing.
    for _ in range(2):
        motion_msks = tf.nn.avg_pool3d(
            motion_msks,
            ksize=(1, 1, kernel_size, kernel_size, 1),
            strides=(1, 1, 1, 1, 1),
            padding="SAME",
        )
    return tf.clip_by_value(motion_msks, 0.0, 1.0)


def _motion_source_coordinates(
    phase_position,
    z_position,
    z_phase_offset,
    temporal_asymmetry,
    rotation_rad,
    scale_delta,
    translation_y,
    translation_x,
    grid_y,
    grid_x,
    center_y,
    center_x,
    roi_weight,
):
    """Create a smooth slice-dependent inverse displacement field for one phase."""
    phase_position_z = tf.clip_by_value(
        phase_position[:, None] + z_position * z_phase_offset[:, None], -1.0, 1.0
    )
    # Keep both endpoints fixed while bending the trajectory between them.
    motion_position = phase_position_z + temporal_asymmetry[:, None] * (
        tf.square(phase_position_z) - 1.0
    )
    motion_position = motion_position[:, :, None, None]

    angle = motion_position * rotation_rad[:, :, None, None]
    scale = 1.0 + motion_position * scale_delta[:, :, None, None]
    cos_angle = tf.cos(angle) / scale
    sin_angle = tf.sin(angle) / scale

    shifted_x = grid_x[None] - center_x - motion_position * translation_x[:, None]
    shifted_y = grid_y[None] - center_y - motion_position * translation_y[:, None]
    transformed_x = cos_angle * shifted_x + sin_angle * shifted_y + center_x
    transformed_y = -sin_angle * shifted_x + cos_angle * shifted_y + center_y
    return _localize_displacement(
        grid_y[None], grid_x[None], transformed_y, transformed_x, roi_weight
    )


def cardiac_motion_blur(
    imgs,
    img_msks,
    spacing_mm_yx,
    motion_msks=None,
    num_phases=5,
    num_phases_range=None,
    max_translation_mm_yx=(3.0, 3.0),
    max_rotation_deg=3.0,
    max_scale_delta=0.04,
    roi_center_yx=(0.5, 0.5),
    roi_sigma_ratio_yx=(0.25, 0.25),
    phase_weight_mode="uniform",
    bimodal_peak_sigma_range=(0.2, 0.4),
    bimodal_balance_range=(0.35, 0.65),
    uniform_phase_weight_mix=0.1,
    max_temporal_asymmetry=0.0,
    max_z_phase_offset=0.0,
    center_preserving=False,
    heart_mask_softening_px=6,
    validation_translation_mm_yx=(2.0, -2.0),
    validation_rotation_deg=2.0,
    validation_scale_delta=0.025,
    validation_bimodal_peak_sigma=0.3,
    validation_bimodal_balance=0.5,
    validation_temporal_asymmetry=0.0,
    validation_z_phase_offset=0.0,
    is_training=True,
):
    """Approximate cardiac CT motion by averaging smooth in-plane heart motion.

    Phase weights can emphasize two separated cardiac states to create double
    contours. When center_preserving is enabled, the exposure-weighted mean
    displacement is removed at every voxel, making the clean image the unique
    center of the synthetic motion. Temporal asymmetry produces a non-uniform
    trajectory, while a smooth Z phase offset approximates the phase drift of a
    helical non-gated acquisition. A Gaussian ROI attenuates the displacement
    field so each phase contains one continuously deformed image rather than an
    intensity blend of original and moved contours. Warped values are normalized
    by a warped validity mask so padding does not create a dark edge that the
    model could learn to overshoot.
    """
    imgs = tf.convert_to_tensor(imgs)
    img_msks = tf.cast(img_msks, imgs.dtype)
    dtype = imgs.dtype
    batch_size = tf.shape(imgs)[0]
    depth = tf.shape(imgs)[1]
    height = tf.shape(imgs)[2]
    width = tf.shape(imgs)[3]

    image_and_mask = tf.concat([imgs * img_msks, img_msks], axis=-1)
    image_and_mask = tf.reshape(image_and_mask, (-1, height, width, 2))

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
    roi_weight = tf.broadcast_to(
        roi_weight[:, None], (batch_size, depth, height, width, 1)
    )
    if motion_msks is not None:
        motion_msks = tf.cast(motion_msks, dtype) * img_msks
        soft_motion_msks = tf.cast(
            _soften_motion_mask(motion_msks, heart_mask_softening_px), dtype
        )
        has_heart_mask = tf.reduce_any(
            motion_msks > 0, axis=(1, 2, 3, 4), keepdims=True
        )
        # Keep compatibility with data without a heart mask by using the
        # configured Gaussian ROI only for those samples.
        roi_weight = tf.where(has_heart_mask, soft_motion_msks, roi_weight)

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
    rotation_rad = rotation_rad[:, None]
    scale_delta = scale_delta[:, None]

    bimodal_peak_sigma = _sample_scalar_range(
        batch_size,
        bimodal_peak_sigma_range,
        validation_bimodal_peak_sigma,
        is_training,
        dtype,
    )
    bimodal_balance = _sample_scalar_range(
        batch_size,
        bimodal_balance_range,
        validation_bimodal_balance,
        is_training,
        dtype,
    )
    if is_training:
        temporal_asymmetry = tf.random.uniform(
            (batch_size,), -max_temporal_asymmetry, max_temporal_asymmetry, dtype=dtype
        )
        z_phase_offset = tf.random.uniform(
            (batch_size,), -max_z_phase_offset, max_z_phase_offset, dtype=dtype
        )
    else:
        temporal_asymmetry = tf.fill(
            (batch_size,), tf.cast(validation_temporal_asymmetry, dtype)
        )
        z_phase_offset = tf.fill(
            (batch_size,), tf.cast(validation_z_phase_offset, dtype)
        )

    z_position = tf.linspace(tf.cast(-1.0, dtype), tf.cast(1.0, dtype), depth)
    z_position = z_position[None]

    phase_counts, max_loop_phases = _sample_num_phases(
        batch_size, num_phases, num_phases_range, is_training
    )
    phase_denominator = tf.cast(tf.maximum(phase_counts - 1, 1), dtype)

    def get_phase_state(phase_index):
        active = phase_index < phase_counts
        bounded_index = tf.minimum(phase_index, phase_counts - 1)
        phase_position = -1.0 + 2.0 * tf.cast(bounded_index, dtype) / phase_denominator
        phase_weights = _phase_weight(
            phase_position,
            active,
            phase_weight_mode,
            bimodal_peak_sigma,
            bimodal_balance,
            uniform_phase_weight_mix,
        )
        source_y, source_x = _motion_source_coordinates(
            phase_position,
            z_position,
            z_phase_offset,
            temporal_asymmetry,
            rotation_rad,
            scale_delta,
            translation_y,
            translation_x,
            grid_y,
            grid_x,
            center_y,
            center_x,
            roi_weight,
        )
        return phase_weights, source_y, source_x

    # Remove the weighted mean displacement field, not merely the mean affine
    # parameter. This also centers rotations, scaling and Z-dependent motion.
    if center_preserving:
        mean_displacement_y = tf.zeros((batch_size, depth, height, width), dtype=dtype)
        mean_displacement_x = tf.zeros((batch_size, depth, height, width), dtype=dtype)
        center_weight = tf.zeros((batch_size, 1, 1, 1), dtype=dtype)
        for phase_index in range(max_loop_phases):
            phase_weights, source_y, source_x = get_phase_state(phase_index)
            phase_weights = phase_weights[:, None, None, None]
            mean_displacement_y += (source_y - grid_y[None]) * phase_weights
            mean_displacement_x += (source_x - grid_x[None]) * phase_weights
            center_weight += phase_weights
        center_weight = tf.maximum(center_weight, tf.cast(1e-6, dtype))
        mean_displacement_y /= center_weight
        mean_displacement_x /= center_weight
    else:
        mean_displacement_y = tf.cast(0.0, dtype)
        mean_displacement_x = tf.cast(0.0, dtype)

    accumulated = tf.zeros_like(imgs)
    accumulated_weight = tf.zeros((batch_size, 1, 1, 1, 1), dtype=dtype)
    for phase_index in range(max_loop_phases):
        phase_weights, source_y, source_x = get_phase_state(phase_index)
        source_y -= mean_displacement_y
        source_x -= mean_displacement_x

        source_y = tf.reshape(source_y, (-1, height, width))
        source_x = tf.reshape(source_x, (-1, height, width))
        sampled = bilinear_sample_2d(image_and_mask, source_y, source_x)
        sampled = tf.reshape(sampled, (batch_size, depth, height, width, 2))
        warped_img = sampled[..., :1]
        warped_msk = sampled[..., 1:]
        warped_img = warped_img / tf.maximum(warped_msk, tf.cast(1e-6, dtype))
        warped_img = tf.where(warped_msk > 1e-6, warped_img, imgs)
        phase_weights = phase_weights[:, None, None, None, None]
        accumulated += warped_img * phase_weights
        accumulated_weight += phase_weights

    blurred = accumulated / tf.maximum(accumulated_weight, tf.cast(1e-6, dtype))
    return tf.clip_by_value(blurred, 0.0, 1.0) * img_msks
