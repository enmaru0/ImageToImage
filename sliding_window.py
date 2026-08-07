from itertools import product

import numpy as np
from scipy import ndimage


def resample_volume(volume, output_size_zyx):
    """Linearly resample a 3D volume to an exact output size."""
    volume = np.asarray(volume)
    output_size_zyx = np.asarray(output_size_zyx, np.int32)
    if np.array_equal(volume.shape, output_size_zyx):
        return volume.copy()
    zoom_factors = output_size_zyx / np.asarray(volume.shape, np.float64)
    working = volume.astype(np.float32, copy=False)

    # Suppress aliasing before reducing resolution. Upsampling axes use sigma=0.
    antialias_sigma = np.maximum((1.0 / zoom_factors - 1.0) * 0.5, 0.0)
    if np.any(antialias_sigma > 0):
        working = ndimage.gaussian_filter(
            working, sigma=antialias_sigma, mode="nearest"
        )
    output = ndimage.zoom(
        working,
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=False,
        grid_mode=False,
    )
    if tuple(output.shape) != tuple(output_size_zyx):
        raise RuntimeError(
            f"resampling size mismatch: {output.shape} != {tuple(output_size_zyx)}"
        )
    return output


def sliding_window_starts(image_size, window_size, overlap):
    """Return start indices that cover an axis including its final voxel."""
    if image_size <= window_size:
        return [0]
    stride = max(int(round(window_size * (1.0 - overlap))), 1)
    starts = list(range(0, image_size - window_size + 1, stride))
    final_start = image_size - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def importance_map(window_size_zyx):
    """Create a positive center-weighted map for seamless overlap blending."""
    axes = []
    for size in window_size_zyx:
        if size == 1:
            axes.append(np.ones(1, np.float32))
            continue
        center = (size - 1) / 2.0
        sigma = max(size * 0.25, 1.0)
        coordinate = (np.arange(size, dtype=np.float32) - center) / sigma
        axes.append(np.exp(-0.5 * coordinate**2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight /= np.max(weight)
    return np.maximum(weight, 1e-3).astype(np.float32)


def _pad_to_window(volume, window_size_zyx):
    pad_width = []
    for image_size, window_size in zip(volume.shape, window_size_zyx):
        required = max(window_size - image_size, 0)
        before = required // 2
        pad_width.append((before, required - before))
    padded = np.pad(volume, pad_width, mode="edge")
    valid = np.pad(
        np.ones(volume.shape, np.bool_),
        pad_width,
        mode="constant",
        constant_values=False,
    )
    return padded, valid, tuple(pad_width)


def sliding_window_inference(
    volume,
    window_size_zyx,
    overlap,
    predict_patch,
    num_output_channels=1,
    seed=0,
    progress=None,
):
    """Run overlap-weighted sliding-window inference on a 3D volume.

    ``predict_patch`` receives ``(image_patch, valid_mask, initial_noise)``.
    A single noise volume is shared across windows, so overlapping voxels use
    the same I2I-RFR initial state.
    """
    volume = np.asarray(volume)
    window_size_zyx = tuple(int(size) for size in window_size_zyx)
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D: {volume.shape}")
    if len(window_size_zyx) != 3 or min(window_size_zyx) <= 0:
        raise ValueError(
            f"window_size_zyx must contain three positive values: {window_size_zyx}"
        )
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1: {overlap}")

    padded, valid, pad_width = _pad_to_window(volume, window_size_zyx)
    starts_per_axis = [
        sliding_window_starts(image_size, window_size, overlap)
        for image_size, window_size in zip(padded.shape, window_size_zyx)
    ]
    window_positions = list(product(*starts_per_axis))
    iterator = progress(window_positions) if progress is not None else window_positions

    accumulated = np.zeros(padded.shape + (num_output_channels,), np.float32)
    accumulated_weight = np.zeros(padded.shape, np.float32)
    weight = importance_map(window_size_zyx)
    rng = np.random.default_rng(seed)
    shared_noise = rng.standard_normal(
        padded.shape + (num_output_channels,), dtype=np.float32
    ).astype(np.float16)

    for start_zyx in iterator:
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(start_zyx, window_size_zyx)
        )
        image_patch = padded[slices]
        valid_patch = valid[slices]
        noise_patch = shared_noise[slices].astype(np.float32)
        prediction = np.asarray(
            predict_patch(image_patch, valid_patch, noise_patch), np.float32
        )
        expected_shape = window_size_zyx + (num_output_channels,)
        if prediction.shape != expected_shape:
            raise ValueError(
                f"predict_patch returned {prediction.shape}; expected {expected_shape}"
            )
        accumulated[slices] += prediction * weight[..., None]
        accumulated_weight[slices] += weight

    output = accumulated / np.maximum(accumulated_weight[..., None], 1e-8)
    crop_slices = tuple(
        slice(before, padded_size - after if after > 0 else None)
        for (before, after), padded_size in zip(pad_width, padded.shape)
    )
    return output[crop_slices]
