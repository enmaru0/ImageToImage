"""End-to-end CLI smoke test for the generic med3d-dl container."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from irg import save_raw, save_re4

import calibrate_degradation


def _write_case(data_dir, case_index, real_domain):
    shape = (8, 32, 32)
    spacing = (3.0, 0.5, 0.5)
    z, y, x = np.indices(shape)
    heart = ((y - 16) ** 2 + (x - 16) ** 2) < (7 + case_index) ** 2
    image = np.full(shape, -300, np.int16)
    image[heart] = 150 + 20 * case_index
    if real_domain:
        image = np.round(
            0.5 * np.roll(image, -2, axis=2) + 0.5 * np.roll(image, 2, axis=2)
        ).astype(np.int16)
    domain_name = "nongated" if real_domain else "gated"
    image_path = data_dir / f"patient{case_index}_{domain_name}.hdr"
    save_raw(image, spacing, image_path)

    packed_mask = heart.astype(np.uint16) * (1 << 6)
    save_re4(
        packed_mask,
        spacing,
        "mask",
        image_path.with_suffix(".mask.hdr"),
        bit_dict={6: "heart"},
    )


def test_calibration_cli_creates_complete_report():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        clean_dir = root / "clean"
        real_dir = root / "real"
        output_dir = root / "output"
        clean_dir.mkdir()
        real_dir.mkdir()
        for case_index in range(2):
            _write_case(clean_dir, case_index, real_domain=False)
            _write_case(real_dir, case_index, real_domain=True)

        previous_argv = sys.argv
        sys.argv = [
            "calibrate_degradation.py",
            "--clean-data-dir",
            str(clean_dir),
            "--real-data-dir",
            str(real_dir),
            "--output-dir",
            str(output_dir),
            "--overrides",
            "self_supervised_deblur.degradation_type=cardiac_motion",
            "self_supervised_deblur.slice_thickness.enabled=false",
            "degradation_calibration.max_cases_per_domain=2",
            "degradation_calibration.batch_size=2",
            "degradation_calibration.search.num_trials=2",
            "degradation_calibration.montages_per_trial=1",
            "degradation_calibration.apply_identity_mixture=false",
            "degradation_calibration.pairing.enabled=true",
            "aug.crop_size_zyx=[8,32,32]",
            "num_workers=1",
        ]
        try:
            calibrate_degradation.main()
        finally:
            sys.argv = previous_argv

        assert (output_dir / "trials.csv").exists()
        assert (output_dir / "matched_patients.csv").exists()
        assert (output_dir / "best_config.yaml").exists()
        assert (output_dir / "report.json").exists()
        assert (output_dir / "trial_000" / "feature_summary.csv").exists()
        assert (output_dir / "trial_000" / "paired_feature_summary.csv").exists()
        assert (
            output_dir / "trial_000" / "paired_patient_feature_details.csv"
        ).exists()
        assert len(list((output_dir / "trial_001" / "montages").glob("*.png"))) == 1
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        assert report["pairing_enabled"] is True
        assert report["num_matched_patient_ids"] == 2
        assert report["paired_normalized_mae"] is not None


if __name__ == "__main__":
    test_calibration_cli_creates_complete_report()
