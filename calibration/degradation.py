"""Dependency-light measurements used by the degradation calibration CLI.

The real and simulated volumes do not need to be paired.  Each volume is reduced
to intensity, edge, inter-slice, and frequency-domain descriptors inside a heart
ROI.  Distribution distances and a held-out linear domain classifier then
measure how easily the two collections can be separated.
"""

from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = (
    "intensity_mean",
    "intensity_std",
    "intensity_p05",
    "intensity_p50",
    "intensity_p95",
    "xy_gradient_mean",
    "xy_gradient_p90",
    "xy_gradient_p99",
    "z_gradient_mean",
    "z_gradient_p90",
    "z_gradient_p99",
    "xy_laplacian_abs_mean",
    "gradient_y_x_log_ratio",
    "z_xy_gradient_ratio",
    "fft_mid_power_ratio",
    "fft_high_power_ratio",
    "fft_spectral_centroid",
    "edge_autocorrelation_peak",
    "edge_autocorrelation_lag_px",
)


def _as_volume(array, name):
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [Z,Y,X] or [Z,Y,X,1]: {array.shape}")
    return array


def dilate_mask_xy(mask, margin_px):
    """Dilate a 3D mask only in-plane without mixing adjacent slices."""
    mask = _as_volume(mask, "mask").astype(bool, copy=False)
    margin_px = int(margin_px)
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    if margin_px == 0:
        return mask.copy()

    dilated = mask.copy()
    for _ in range(margin_px):
        previous = dilated
        expanded = previous.copy()
        expanded[:, 1:] |= previous[:, :-1]
        expanded[:, :-1] |= previous[:, 1:]
        expanded[:, :, 1:] |= previous[:, :, :-1]
        expanded[:, :, :-1] |= previous[:, :, 1:]
        dilated = expanded
    return dilated


def _quantiles(values, probabilities):
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return [math.nan] * len(probabilities)
    return np.quantile(values, probabilities).tolist()


def _masked_difference(volume, roi, axis):
    values = np.abs(np.diff(volume, axis=axis))
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    valid = roi[tuple(left)] & roi[tuple(right)]
    return values[valid]


def _fft_features(volume, roi):
    mid_ratios = []
    high_ratios = []
    centroids = []
    for z_index in range(volume.shape[0]):
        slice_mask = roi[z_index]
        coordinates = np.argwhere(slice_mask)
        if len(coordinates) < 32:
            continue
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
        if y1 - y0 < 8 or x1 - x0 < 8:
            continue

        patch = np.asarray(volume[z_index, y0:y1, x0:x1], np.float64)
        patch_mask = slice_mask[y0:y1, x0:x1]
        patch = np.where(patch_mask, patch, 0.0)
        patch -= np.sum(patch) / max(np.count_nonzero(patch_mask), 1)
        patch *= patch_mask
        patch *= np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))

        power = np.abs(np.fft.rfft2(patch)) ** 2
        fy = np.fft.fftfreq(patch.shape[0])[:, None]
        fx = np.fft.rfftfreq(patch.shape[1])[None, :]
        radius = np.sqrt(fy**2 + fx**2)
        usable = radius > 0
        total = float(np.sum(power[usable]))
        if total <= 1e-12:
            continue
        mid_ratios.append(
            float(np.sum(power[(radius >= 0.10) & (radius < 0.25)])) / total
        )
        high_ratios.append(float(np.sum(power[radius >= 0.25])) / total)
        centroids.append(float(np.sum(power[usable] * radius[usable])) / total)

    if not mid_ratios:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(mid_ratios)),
        float(np.mean(high_ratios)),
        float(np.mean(centroids)),
    )


def _edge_autocorrelation_features(volume, roi, max_lag_px=12):
    gx = np.zeros_like(volume, dtype=np.float64)
    gy = np.zeros_like(volume, dtype=np.float64)
    gx[:, :, :-1] = np.diff(volume, axis=2)
    gy[:, :-1, :] = np.diff(volume, axis=1)
    edge = np.sqrt(gx**2 + gy**2)
    if np.any(roi):
        edge = edge - float(np.mean(edge[roi]))
    edge = np.where(roi, edge, 0.0)

    correlations = []
    maximum_lag = min(int(max_lag_px), volume.shape[1] - 1, volume.shape[2] - 1)
    for lag in range(2, maximum_lag + 1):
        axis_correlations = []
        for axis in (1, 2):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis] = slice(None, -lag)
            right[axis] = slice(lag, None)
            valid = roi[tuple(left)] & roi[tuple(right)]
            a = edge[tuple(left)][valid]
            b = edge[tuple(right)][valid]
            if len(a) < 32:
                continue
            denominator = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
            if denominator > 1e-12:
                axis_correlations.append(float(np.sum(a * b)) / denominator)
        if axis_correlations:
            correlations.append((lag, float(np.mean(axis_correlations))))
    if not correlations:
        return math.nan, math.nan
    lag, peak = max(correlations, key=lambda item: item[1])
    return peak, float(lag)


def extract_volume_features(
    volume, heart_mask, valid_mask=None, roi_margin_px=8, max_autocorrelation_lag_px=12
):
    """Extract one unpaired-domain feature vector from a normalized CT crop."""
    volume = _as_volume(volume, "volume").astype(np.float64, copy=False)
    heart_mask = _as_volume(heart_mask, "heart_mask").astype(bool, copy=False)
    if valid_mask is None:
        valid_mask = np.ones_like(heart_mask)
    else:
        valid_mask = _as_volume(valid_mask, "valid_mask").astype(bool, copy=False)
    if volume.shape != heart_mask.shape or volume.shape != valid_mask.shape:
        raise ValueError("volume, heart_mask, and valid_mask must have the same shape")
    if not np.any(heart_mask & valid_mask):
        raise ValueError("heart_mask has no valid voxels")

    roi = dilate_mask_xy(heart_mask, roi_margin_px) & valid_mask
    intensity = volume[roi]
    intensity_q = _quantiles(intensity, (0.05, 0.50, 0.95))

    gradient_x = _masked_difference(volume, roi, axis=2)
    gradient_y = _masked_difference(volume, roi, axis=1)
    gradient_z = _masked_difference(volume, roi, axis=0)
    gradient_xy = np.concatenate([gradient_x, gradient_y])
    xy_q = _quantiles(gradient_xy, (0.90, 0.99))
    z_q = _quantiles(gradient_z, (0.90, 0.99))

    center = volume[:, 1:-1, 1:-1]
    laplacian = (
        4.0 * center
        - volume[:, :-2, 1:-1]
        - volume[:, 2:, 1:-1]
        - volume[:, 1:-1, :-2]
        - volume[:, 1:-1, 2:]
    )
    laplacian_mask = (
        roi[:, 1:-1, 1:-1]
        & roi[:, :-2, 1:-1]
        & roi[:, 2:, 1:-1]
        & roi[:, 1:-1, :-2]
        & roi[:, 1:-1, 2:]
    )
    laplacian_values = np.abs(laplacian[laplacian_mask])

    mean_x = float(np.mean(gradient_x)) if gradient_x.size else math.nan
    mean_y = float(np.mean(gradient_y)) if gradient_y.size else math.nan
    mean_xy = float(np.mean(gradient_xy)) if gradient_xy.size else math.nan
    mean_z = float(np.mean(gradient_z)) if gradient_z.size else math.nan
    epsilon = 1e-8
    direction_ratio = math.log((mean_y + epsilon) / (mean_x + epsilon))
    z_xy_ratio = (mean_z + epsilon) / (mean_xy + epsilon)
    fft_mid, fft_high, fft_centroid = _fft_features(volume, roi)
    autocorrelation_peak, autocorrelation_lag = _edge_autocorrelation_features(
        volume, roi, max_lag_px=max_autocorrelation_lag_px
    )

    feature_values = (
        float(np.mean(intensity)),
        float(np.std(intensity)),
        *intensity_q,
        mean_xy,
        *xy_q,
        mean_z,
        *z_q,
        float(np.mean(laplacian_values)) if laplacian_values.size else math.nan,
        direction_ratio,
        z_xy_ratio,
        fft_mid,
        fft_high,
        fft_centroid,
        autocorrelation_peak,
        autocorrelation_lag,
    )
    return dict(zip(FEATURE_NAMES, feature_values))


def _quantile_wasserstein(a, b, num_quantiles=101):
    probabilities = np.linspace(0.0, 1.0, num_quantiles)
    return float(
        np.mean(np.abs(np.quantile(a, probabilities) - np.quantile(b, probabilities)))
    )


def compare_feature_distributions(simulated, real, feature_names=FEATURE_NAMES):
    """Summarize standardized distances between two unpaired feature matrices."""
    simulated = np.asarray(simulated, np.float64)
    real = np.asarray(real, np.float64)
    if simulated.ndim != 2 or real.ndim != 2:
        raise ValueError("simulated and real must be two-dimensional")
    if simulated.shape[1] != len(feature_names) or real.shape[1] != len(feature_names):
        raise ValueError("feature matrix width does not match feature_names")

    rows = []
    for index, name in enumerate(feature_names):
        sim_values = simulated[:, index]
        real_values = real[:, index]
        sim_values = sim_values[np.isfinite(sim_values)]
        real_values = real_values[np.isfinite(real_values)]
        if not len(sim_values) or not len(real_values):
            rows.append(
                {
                    "feature": name,
                    "simulated_mean": math.nan,
                    "simulated_std": math.nan,
                    "real_mean": math.nan,
                    "real_std": math.nan,
                    "standardized_mean_difference": math.nan,
                    "normalized_wasserstein": math.nan,
                }
            )
            continue
        simulated_mean = float(np.mean(sim_values))
        real_mean = float(np.mean(real_values))
        simulated_std = float(np.std(sim_values))
        real_std = float(np.std(real_values))
        pooled_scale = math.sqrt((simulated_std**2 + real_std**2) / 2.0)
        pooled_scale = max(pooled_scale, 1e-8)
        rows.append(
            {
                "feature": name,
                "simulated_mean": simulated_mean,
                "simulated_std": simulated_std,
                "real_mean": real_mean,
                "real_std": real_std,
                "standardized_mean_difference": (simulated_mean - real_mean)
                / pooled_scale,
                "normalized_wasserstein": _quantile_wasserstein(sim_values, real_values)
                / pooled_scale,
            }
        )
    return rows


def _fit_logistic_regression(features, labels, l2=0.1, max_iterations=50):
    features = np.asarray(features, np.float64)
    labels = np.asarray(labels, np.float64)
    design = np.concatenate([np.ones((len(features), 1)), features], axis=1)
    weights = np.zeros(design.shape[1], np.float64)
    class_counts = np.bincount(labels.astype(np.int64), minlength=2)
    sample_weights = np.where(
        labels > 0.5, 0.5 / max(class_counts[1], 1), 0.5 / max(class_counts[0], 1)
    )
    regularization = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    regularization[0, 0] = 0.0

    for _ in range(int(max_iterations)):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weights * (probabilities - labels))
        gradient += regularization @ weights
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + regularization
        hessian += np.eye(hessian.shape[0]) * 1e-8
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= update
        if float(np.linalg.norm(update)) < 1e-7:
            break
    return weights


def _auc(labels, scores):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return math.nan
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def _group_folds(groups, num_folds, seed):
    groups = np.asarray(groups, object)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    assignment = {group: index % num_folds for index, group in enumerate(unique_groups)}
    return np.asarray([assignment[group] for group in groups], np.int32)


def cross_validated_domain_auc(
    simulated,
    real,
    simulated_groups=None,
    real_groups=None,
    num_folds=5,
    seed=0,
    l2=0.1,
):
    """Return group-aware held-out AUC for simulated-vs-real classification."""
    simulated = np.asarray(simulated, np.float64)
    real = np.asarray(real, np.float64)
    if simulated.ndim != 2 or real.ndim != 2 or simulated.shape[1] != real.shape[1]:
        raise ValueError("simulated and real must have compatible [N,F] shapes")
    if simulated_groups is None:
        simulated_groups = [f"sim-{index}" for index in range(len(simulated))]
    if real_groups is None:
        real_groups = [f"real-{index}" for index in range(len(real))]
    if len(simulated_groups) != len(simulated) or len(real_groups) != len(real):
        raise ValueError("group list length must match its feature matrix")

    available_folds = min(
        int(num_folds), len(np.unique(simulated_groups)), len(np.unique(real_groups))
    )
    if available_folds < 2:
        return {
            "auc": math.nan,
            "raw_auc": math.nan,
            "balanced_accuracy": math.nan,
            "num_folds": 0,
            "num_features": 0,
        }

    features = np.concatenate([simulated, real], axis=0)
    labels = np.concatenate([np.zeros(len(simulated)), np.ones(len(real))])
    sim_folds = _group_folds(simulated_groups, available_folds, seed)
    real_folds = _group_folds(real_groups, available_folds, seed + 1)
    folds = np.concatenate([sim_folds, real_folds])
    held_out_scores = np.full(len(features), np.nan, np.float64)
    used_feature_counts = []

    for fold in range(available_folds):
        train = folds != fold
        test = ~train
        train_features = features[train]
        train_mean = np.nanmean(train_features, axis=0)
        train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0)
        train_std = np.nanstd(train_features, axis=0)
        usable = np.isfinite(train_std) & (train_std > 1e-8)
        if not np.any(usable):
            held_out_scores[test] = 0.5
            used_feature_counts.append(0)
            continue
        train_normalized = (train_features[:, usable] - train_mean[usable]) / train_std[
            usable
        ]
        test_normalized = (features[test][:, usable] - train_mean[usable]) / train_std[
            usable
        ]
        train_normalized = np.nan_to_num(
            train_normalized, nan=0.0, posinf=5.0, neginf=-5.0
        )
        test_normalized = np.nan_to_num(
            test_normalized, nan=0.0, posinf=5.0, neginf=-5.0
        )
        train_normalized = np.clip(train_normalized, -8.0, 8.0)
        test_normalized = np.clip(test_normalized, -8.0, 8.0)
        weights = _fit_logistic_regression(train_normalized, labels[train], l2=l2)
        test_design = np.concatenate(
            [np.ones((len(test_normalized), 1)), test_normalized], axis=1
        )
        logits = np.clip(test_design @ weights, -30.0, 30.0)
        held_out_scores[test] = 1.0 / (1.0 + np.exp(-logits))
        used_feature_counts.append(int(np.count_nonzero(usable)))

    raw_auc = _auc(labels, held_out_scores)
    auc = max(raw_auc, 1.0 - raw_auc) if np.isfinite(raw_auc) else math.nan
    predictions = held_out_scores >= 0.5
    true_positive_rate = float(np.mean(predictions[labels == 1]))
    true_negative_rate = float(np.mean(~predictions[labels == 0]))
    return {
        "auc": auc,
        "raw_auc": raw_auc,
        "balanced_accuracy": (true_positive_rate + true_negative_rate) / 2.0,
        "num_folds": available_folds,
        "num_features": int(round(float(np.mean(used_feature_counts)))),
    }


def feature_dicts_to_matrix(records, feature_names=FEATURE_NAMES):
    return np.asarray(
        [[record[name] for name in feature_names] for record in records], np.float64
    )


def aggregate_calibration_score(summary_rows, domain_auc, distance_clip=10.0):
    """Combine interpretable feature distances and domain separability; lower is better."""
    smd = np.asarray(
        [abs(row["standardized_mean_difference"]) for row in summary_rows], np.float64
    )
    wasserstein = np.asarray(
        [row["normalized_wasserstein"] for row in summary_rows], np.float64
    )
    distance_clip = float(distance_clip)
    if distance_clip <= 0:
        raise ValueError("distance_clip must be positive")
    # Near-zero within-domain variance can otherwise make one descriptor dominate
    # the entire search by many orders of magnitude. Raw values remain available
    # in feature_summary.csv; only the aggregate ranking is winsorized.
    mean_abs_smd = float(np.nanmean(np.clip(smd, 0.0, distance_clip)))
    mean_wasserstein = float(np.nanmean(np.clip(wasserstein, 0.0, distance_clip)))
    auc_penalty = 0.0
    if np.isfinite(domain_auc):
        auc_penalty = 2.0 * abs(float(domain_auc) - 0.5)
    return {
        "mean_abs_standardized_mean_difference": mean_abs_smd,
        "mean_normalized_wasserstein": mean_wasserstein,
        "domain_auc_penalty": auc_penalty,
        "score": mean_abs_smd + mean_wasserstein + auc_penalty,
    }
