import numpy as np

from .degradation import (
    FEATURE_NAMES,
    aggregate_calibration_score,
    compare_feature_distributions,
    cross_validated_domain_auc,
    extract_volume_features,
    feature_dicts_to_matrix,
)


def _phantom(shape=(4, 40, 40)):
    z, y, x = np.indices(shape)
    mask = ((y - 20) ** 2 + (x - 20) ** 2) < 12**2
    image = np.where(mask, 0.75, 0.2).astype(np.float32)
    image += z.astype(np.float32) * 0.01
    return image[..., None], mask[..., None]


def test_extract_volume_features_returns_finite_expected_vector():
    image, mask = _phantom()
    features = extract_volume_features(image, mask, roi_margin_px=2)

    assert tuple(features) == FEATURE_NAMES
    assert np.all(np.isfinite(list(features.values())))
    assert features["intensity_p05"] < features["intensity_p95"]
    assert features["xy_gradient_mean"] > 0


def test_identical_feature_distributions_have_zero_distance_and_score():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(12, len(FEATURE_NAMES)))
    summary = compare_feature_distributions(matrix, matrix.copy())
    score = aggregate_calibration_score(summary, domain_auc=0.5)

    assert np.isclose(score["mean_abs_standardized_mean_difference"], 0.0)
    assert np.isclose(score["mean_normalized_wasserstein"], 0.0)
    assert np.isclose(score["score"], 0.0)


def test_cross_validated_domain_auc_detects_separated_domains():
    rng = np.random.default_rng(11)
    simulated = rng.normal(loc=-1.5, scale=0.3, size=(20, len(FEATURE_NAMES)))
    real = rng.normal(loc=1.5, scale=0.3, size=(20, len(FEATURE_NAMES)))
    result = cross_validated_domain_auc(simulated, real, num_folds=5, seed=9)

    assert result["auc"] > 0.95
    assert result["balanced_accuracy"] > 0.9
    assert result["num_folds"] == 5


def test_feature_dicts_to_matrix_follows_declared_column_order():
    records = [{name: float(index) for index, name in enumerate(FEATURE_NAMES)}]
    matrix = feature_dicts_to_matrix(records)

    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.array_equal(matrix[0], np.arange(len(FEATURE_NAMES)))


if __name__ == "__main__":
    test_extract_volume_features_returns_finite_expected_vector()
    test_identical_feature_distributions_have_zero_distance_and_score()
    test_cross_validated_domain_auc_detects_separated_domains()
    test_feature_dicts_to_matrix_follows_declared_column_order()
