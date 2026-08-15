import math

import tensorflow as tf


def _additional_gaussian_sigma_vox(
    clean_thickness_mm,
    degraded_thickness_mm,
    spacing_mm_z,
    profile_model="gaussian_fwhm",
):
    """Return additional Z Gaussian sigma for the selected thickness model."""
    additional_width_mm = math.sqrt(
        max(degraded_thickness_mm**2 - clean_thickness_mm**2, 0.0)
    )
    if profile_model == "gaussian_fwhm":
        # Treat nominal slice thickness as the FWHM of a Gaussian SSP.
        sigma_mm = additional_width_mm / math.sqrt(8.0 * math.log(2.0))
    elif profile_model == "box_variance":
        # Treat nominal thickness as a boxcar aperture and match its variance.
        # A width-T box has variance T^2 / 12.
        sigma_mm = additional_width_mm / math.sqrt(12.0)
    else:
        raise ValueError(
            f"profile_model must be 'gaussian_fwhm' or 'box_variance': {profile_model}"
        )
    return sigma_mm / spacing_mm_z


def _linear_sample_z(volume, coordinates):
    """Linearly sample a B-Z-Y-X-C tensor at Z coordinates in voxel units."""
    depth = tf.shape(volume)[1]
    dtype = volume.dtype
    coordinates = tf.cast(coordinates, dtype)
    coordinates = tf.clip_by_value(
        coordinates, tf.cast(0.0, dtype), tf.cast(depth - 1, dtype)
    )
    lower_float = tf.floor(coordinates)
    upper_float = tf.minimum(lower_float + 1.0, tf.cast(depth - 1, dtype))
    lower = tf.cast(lower_float, tf.int32)
    upper = tf.cast(upper_float, tf.int32)
    weight = (coordinates - lower_float)[None, :, None, None, None]
    lower_values = tf.gather(volume, lower, axis=1)
    upper_values = tf.gather(volume, upper, axis=1)
    return lower_values * (1.0 - weight) + upper_values * weight


def _gaussian_blur_z(imgs, img_msks, sigma_vox, truncate=3.0):
    """Apply mask-normalized Gaussian blur only along the Z axis."""
    if sigma_vox <= 0:
        return imgs
    radius = max(int(math.ceil(truncate * sigma_vox)), 1)
    coordinates = tf.range(-radius, radius + 1, dtype=tf.float32)
    kernel = tf.exp(-tf.square(coordinates) / (2.0 * sigma_vox**2))
    kernel /= tf.reduce_sum(kernel)
    kernel = tf.cast(tf.reshape(kernel, (-1, 1, 1, 1, 1)), imgs.dtype)
    paddings = [[0, 0], [radius, radius], [0, 0], [0, 0], [0, 0]]
    weighted_imgs = tf.pad(imgs * img_msks, paddings)
    padded_msks = tf.pad(img_msks, paddings)
    strides = [1, 1, 1, 1, 1]
    blurred = tf.nn.conv3d(weighted_imgs, kernel, strides, padding="VALID")
    blurred_msk = tf.nn.conv3d(padded_msks, kernel, strides, padding="VALID")
    return blurred / tf.maximum(blurred_msk, tf.cast(1e-6, imgs.dtype))


def simulate_slice_thickness(
    imgs,
    img_msks,
    spacing_mm_z,
    clean_thickness_mm=3.0,
    degraded_thickness_mm=5.0,
    profile_model="gaussian_fwhm",
    gaussian_truncate=3.0,
    enabled=True,
):
    """Simulate thicker slices and interpolate back to the original Z grid.

    ``profile_model='gaussian_fwhm'`` treats nominal thickness as the FWHM of a
    Gaussian SSP. ``profile_model='box_variance'`` treats it as the width of a
    boxcar slice aperture and selects a Gaussian with the same added variance.
    The blurred volume is sampled on an exact ``degraded_thickness_mm`` grid,
    then linearly sampled back onto the original ``spacing_mm_z`` grid without
    changing output shape.
    """
    del enabled
    imgs = tf.convert_to_tensor(imgs)
    img_msks = tf.cast(img_msks, imgs.dtype)
    input_depth = imgs.shape[1]
    if input_depth is None:
        raise ValueError("simulate_slice_thickness requires a static Z size")
    if input_depth <= 1:
        return imgs * img_msks

    spacing_mm_z = float(spacing_mm_z)
    clean_thickness_mm = float(clean_thickness_mm)
    degraded_thickness_mm = float(degraded_thickness_mm)
    sigma_vox = _additional_gaussian_sigma_vox(
        clean_thickness_mm,
        degraded_thickness_mm,
        spacing_mm_z,
        profile_model=str(profile_model),
    )
    blurred = _gaussian_blur_z(
        imgs, img_msks, sigma_vox, truncate=float(gaussian_truncate)
    )

    extent_mm = (input_depth - 1) * spacing_mm_z
    degraded_depth = max(int(round(extent_mm / degraded_thickness_mm)) + 1, 2)
    degraded_span_mm = (degraded_depth - 1) * degraded_thickness_mm
    degraded_start_mm = (extent_mm - degraded_span_mm) / 2.0
    degraded_positions_mm = degraded_start_mm + tf.range(
        degraded_depth, dtype=tf.float32
    ) * tf.cast(degraded_thickness_mm, tf.float32)
    degraded = _linear_sample_z(blurred, degraded_positions_mm / spacing_mm_z)

    original_positions_mm = tf.range(input_depth, dtype=tf.float32) * tf.cast(
        spacing_mm_z, tf.float32
    )
    degraded_coordinates = (
        original_positions_mm - degraded_start_mm
    ) / degraded_thickness_mm
    restored = _linear_sample_z(degraded, degraded_coordinates)
    return restored * img_msks
