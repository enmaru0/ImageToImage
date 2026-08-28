import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from calibrate_degradation import (
    _best_override_config,
    sample_trial_config,
    save_montages,
    simulate_cases,
)


def _small_config():
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.training_mode = "self_supervised_deblur"
    cfg.aug.crop_size_zyx = [8, 32, 32]
    cfg.self_supervised_deblur.degradation_type = "cardiac_motion"
    cfg.self_supervised_deblur.identity_probability = 0.0
    cfg.self_supervised_deblur.slice_thickness.enabled = False
    cfg.self_supervised_deblur.cardiac_motion.max_translation_mm_yx = [2.0, 2.0]
    cfg.self_supervised_deblur.cardiac_motion.max_rotation_deg = 0.0
    cfg.self_supervised_deblur.cardiac_motion.max_scale_delta = 0.0
    cfg.degradation_calibration.batch_size = 2
    cfg.degradation_calibration.simulations_per_clean = 1
    cfg.degradation_calibration.montages_per_trial = 1
    cfg.degradation_calibration.apply_identity_mixture = False
    return cfg


def _cases():
    cases = []
    z, y, x = np.indices((8, 32, 32))
    heart = ((y - 16) ** 2 + (x - 16) ** 2 < 8**2)[..., None]
    for index in range(2):
        image = np.full((8, 32, 32, 1), 0.2, np.float32)
        image[heart] = 0.7 + index * 0.05
        cases.append(
            {
                "case_id": f"case{index}",
                "image": image,
                "heart_mask": heart.astype(np.float32),
                "valid_mask": np.ones_like(image),
            }
        )
    return cases


def test_trial_sampling_is_reproducible_and_valid():
    cfg = _small_config()
    first = sample_trial_config(cfg, trial_index=3, seed=7)
    second = sample_trial_config(cfg, trial_index=3, seed=7)

    assert OmegaConf.to_container(first) == OmegaConf.to_container(second)
    assert first.self_supervised_deblur.cardiac_motion.max_scale_delta < 1
    assert first.self_supervised_deblur.cardiac_motion.bimodal_peak_sigma_range[0] > 0
    assert first.self_supervised_deblur.identity_probability == 0.0


def test_simulation_and_montage_smoke():
    cfg = _small_config()
    clean_cases = _cases()
    simulated = simulate_cases(clean_cases, cfg, seed=5)
    repeated = simulate_cases(clean_cases, cfg, seed=5)

    assert len(simulated) == len(clean_cases)
    assert simulated[0]["image"].shape == clean_cases[0]["image"].shape
    assert np.max(np.abs(simulated[0]["image"] - clean_cases[0]["image"])) > 0
    assert np.array_equal(simulated[0]["image"], repeated[0]["image"])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        save_montages(simulated, clean_cases, output_dir, cfg)
        montage_paths = list(output_dir.glob("*.png"))
        assert len(montage_paths) == 1
        montage = tf.io.decode_png(tf.io.read_file(str(montage_paths[0])))
        assert tuple(montage.shape) == (32, 5 * 32 + 4 * 4, 3)
        assert "real non-gated-clean difference" in (
            output_dir / "README.txt"
        ).read_text(encoding="utf-8")


def test_best_override_contains_only_training_degradation_section():
    override = OmegaConf.to_container(_best_override_config(_small_config()))
    assert tuple(override) == ("self_supervised_deblur",)
    assert override["self_supervised_deblur"]["degradation_type"] == "cardiac_motion"


if __name__ == "__main__":
    test_trial_sampling_is_reproducible_and_valid()
    test_simulation_and_montage_smoke()
    test_best_override_contains_only_training_degradation_section()
