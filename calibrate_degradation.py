"""Calibrate the synthetic degradation distribution against real non-gated CT.

This command intentionally treats the two domains as unpaired.  It creates
heart-centred crops with the existing DataLoader, synthesizes degradations from
clean crops, measures per-volume descriptors, and searches simulator ranges that
minimize distribution distances and held-out domain-classifier AUC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from omegaconf import OmegaConf

from calibration.degradation import (
    FEATURE_NAMES,
    aggregate_calibration_score,
    compare_paired_feature_groups,
    compare_feature_distributions,
    cross_validated_domain_auc,
    extract_volume_features,
    feature_dicts_to_matrix,
)
from data.gpu_aug import normalize
from trainer import CustomModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="実non-gated CTへ合成degradationの分布を校正します"
    )
    parser.add_argument("--config", default="conf/config.yaml")
    parser.add_argument("--clean-data-dir", default=None)
    parser.add_argument("--real-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help=(
            "設定上書き。例: self_supervised_deblur.degradation_type="
            "cardiac_motion_gaussian degradation_calibration.search.num_trials=20"
        ),
    )
    return parser.parse_args()


def load_config(args):
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    calibration_cfg = cfg.degradation_calibration
    if args.clean_data_dir is not None:
        calibration_cfg.clean_data_dir = args.clean_data_dir
    if args.real_data_dir is not None:
        calibration_cfg.real_data_dir = args.real_data_dir
    if args.output_dir is not None:
        calibration_cfg.output_dir = args.output_dir
    cfg.training_mode = "self_supervised_deblur"
    cfg.debug_dataloader = True

    if not calibration_cfg.clean_data_dir:
        raise ValueError(
            "--clean-data-dirまたはdegradation_calibration.clean_data_dirが必要です"
        )
    if not calibration_cfg.real_data_dir:
        raise ValueError(
            "--real-data-dirまたはdegradation_calibration.real_data_dirが必要です"
        )
    for name in ("clean_data_dir", "real_data_dir"):
        path = Path(getattr(calibration_cfg, name))
        if not path.is_dir():
            raise FileNotFoundError(f"{name}が見つかりません: {path}")
    if int(calibration_cfg.batch_size) < 1:
        raise ValueError("degradation_calibration.batch_sizeは1以上にしてください")
    if int(calibration_cfg.simulations_per_clean) < 1:
        raise ValueError("simulations_per_cleanは1以上にしてください")
    if int(calibration_cfg.search.num_trials) < 1:
        raise ValueError("search.num_trialsは1以上にしてください")
    if int(calibration_cfg.classifier_folds) < 2:
        raise ValueError("classifier_foldsは2以上にしてください")
    if float(calibration_cfg.score_distance_clip) <= 0:
        raise ValueError("score_distance_clipは0より大きくしてください")
    pairing_cfg = calibration_cfg.pairing
    if not isinstance(pairing_cfg.enabled, bool):
        raise ValueError("degradation_calibration.pairing.enabledはboolにしてください")
    if not str(pairing_cfg.delimiter):
        raise ValueError("degradation_calibration.pairing.delimiterは空にできません")
    if float(pairing_cfg.score_weight) < 0:
        raise ValueError(
            "degradation_calibration.pairing.score_weightは0以上にしてください"
        )
    return cfg


def _resolve_heart_bit(configured_bit, default_bit, padding_bit, name):
    bit = int(default_bit if configured_bit is None else configured_bit)
    if not 0 <= bit < int(padding_bit):
        raise ValueError(f"{name}は0以上padding bit未満にしてください: {bit}")
    return bit


def _all_image_pairs(data_dir):
    # Import lazily so feature/simulator tests do not require the site's optional
    # compiled IRG reader until actual HDR/RAW files are requested.
    from main import prepare_unpaired_data_dict

    data_dict = prepare_unpaired_data_dict(data_dir, require_heart_mask=True)
    pairs = []
    for value in data_dict.values():
        pairs.extend(value["img_hdr_list"])
    return sorted(pairs)


def _make_calibration_data_dict(data_dir, pairs):
    return {Path(data_dir).name: {"img_hdr_list": sorted(pairs), "freq": -1}}


def _patient_id_from_stem(stem, delimiter="_"):
    patient_id = str(stem).split(str(delimiter), maxsplit=1)[0]
    if not patient_id:
        raise ValueError(f"患者IDを抽出できないファイル名です: {stem}")
    return patient_id


def _selected_unpaired_dict(data_dir, max_cases, seed):
    pairs = _all_image_pairs(data_dir)
    if max_cases > 0 and len(pairs) > max_cases:
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(
            rng.choice(len(pairs), size=max_cases, replace=False)
        )
        pairs = [pairs[index] for index in selected_indices]
    return _make_calibration_data_dict(data_dir, pairs)


def prepare_calibration_data_dicts(cfg, max_cases, seed):
    """Select independent cases or patient-matched gated/non-gated collections."""
    calibration_cfg = cfg.degradation_calibration
    if not bool(calibration_cfg.pairing.enabled):
        return (
            _selected_unpaired_dict(calibration_cfg.clean_data_dir, max_cases, seed),
            _selected_unpaired_dict(calibration_cfg.real_data_dir, max_cases, seed + 1),
            [],
        )

    delimiter = str(calibration_cfg.pairing.delimiter)
    clean_by_patient = defaultdict(list)
    real_by_patient = defaultdict(list)
    for pair in _all_image_pairs(calibration_cfg.clean_data_dir):
        clean_by_patient[_patient_id_from_stem(pair[0].stem, delimiter)].append(pair)
    for pair in _all_image_pairs(calibration_cfg.real_data_dir):
        real_by_patient[_patient_id_from_stem(pair[0].stem, delimiter)].append(pair)

    common_patients = sorted(set(clean_by_patient) & set(real_by_patient))
    if max_cases > 0 and len(common_patients) > max_cases:
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(
            rng.choice(len(common_patients), size=max_cases, replace=False)
        )
        common_patients = [common_patients[index] for index in selected_indices]
    if len(common_patients) < 2:
        raise ValueError(
            "患者対応校正には、ファイル名を'"
            f"{delimiter}'区切りした先頭が一致する患者が2名以上必要です。"
            f" clean IDs={len(clean_by_patient)}, real IDs={len(real_by_patient)}, "
            f"matched={len(common_patients)}"
        )

    clean_pairs = [
        pair for patient in common_patients for pair in clean_by_patient[patient]
    ]
    real_pairs = [
        pair for patient in common_patients for pair in real_by_patient[patient]
    ]
    clean_only = sorted(set(clean_by_patient) - set(real_by_patient))
    real_only = sorted(set(real_by_patient) - set(clean_by_patient))
    if clean_only:
        logging.warning(f"対応するreal画像がないclean患者を除外: {clean_only}")
    if real_only:
        logging.warning(f"対応するclean画像がないreal患者を除外: {real_only}")
    logging.info(
        f"患者ID照合: matched={len(common_patients)}, "
        f"clean images={len(clean_pairs)}, real images={len(real_pairs)}"
    )
    return (
        _make_calibration_data_dict(calibration_cfg.clean_data_dir, clean_pairs),
        _make_calibration_data_dict(calibration_cfg.real_data_dir, real_pairs),
        common_patients,
    )


def make_matching_manifest(clean_data_dict, real_data_dict, delimiter="_"):
    """Describe the exact file collections assigned to each matched patient."""
    grouped = {"clean": defaultdict(list), "real": defaultdict(list)}
    for domain, data_dict in (("clean", clean_data_dict), ("real", real_data_dict)):
        for value in data_dict.values():
            for source_path, _ in value["img_hdr_list"]:
                patient_id = _patient_id_from_stem(source_path.stem, delimiter)
                grouped[domain][patient_id].append(str(source_path))
    rows = []
    for patient_id in sorted(set(grouped["clean"]) & set(grouped["real"])):
        clean_files = sorted(grouped["clean"][patient_id])
        real_files = sorted(grouped["real"][patient_id])
        rows.append(
            {
                "patient_id": patient_id,
                "num_clean_files": len(clean_files),
                "num_real_files": len(real_files),
                "clean_files": "|".join(clean_files),
                "real_files": "|".join(real_files),
            }
        )
    return rows


def load_cases(
    data_dir, cfg, heart_bit, max_cases, seed, data_dict=None, group_by_patient=False
):
    """Load deterministic heart-centred crops through the production DataLoader."""
    from data.dataloader import create_dataloader

    loader_cfg = OmegaConf.merge(
        cfg,
        {
            "bit_info": {"heart_bit": int(heart_bit)},
            "batch_size": int(cfg.degradation_calibration.batch_size),
            "debug_dataloader": True,
        },
    )
    if data_dict is None:
        data_dict = _selected_unpaired_dict(data_dir, max_cases, seed)
    loader = create_dataloader(
        data_dict,
        is_training=False,
        cfg=loader_cfg,
        batch_size=int(cfg.degradation_calibration.batch_size),
        drop_remainder=False,
    )

    cases = []
    for batch in loader:
        valid_masks = CustomModel._get_img_msks(
            batch["msks"], loader_cfg.bit_info.padding_bit
        )
        heart_masks = CustomModel._get_heart_msks(batch["msks"], heart_bit)
        images = normalize(
            batch["imgs"], batch["min_clip_vals"], batch["max_clip_vals"]
        )
        images = images * valid_masks
        names = [name.decode() for name in batch["img_hdr_list"].numpy()]
        for name, image, heart_mask, valid_mask in zip(
            names, images.numpy(), heart_masks.numpy(), valid_masks.numpy()
        ):
            if not np.any(heart_mask * valid_mask):
                logging.warning(f"心臓マスクが空のためスキップします: {name}")
                continue
            case_id = f"{len(cases):04d}_{name}"
            patient_id = _patient_id_from_stem(
                name, cfg.degradation_calibration.pairing.delimiter
            )
            cases.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "group_id": patient_id if group_by_patient else case_id,
                    "image": image.astype(np.float32, copy=False),
                    "heart_mask": heart_mask.astype(np.float32, copy=False),
                    "valid_mask": valid_mask.astype(np.float32, copy=False),
                }
            )
    if len(cases) < 2:
        raise ValueError(f"校正には心臓マスク付き症例が2件以上必要です: {data_dir}")
    return cases


def _range(search_cfg, name):
    values = getattr(search_cfg, name, None)
    if values is None or len(values) != 2:
        raise ValueError(
            f"degradation_calibration.search.{name}は[min,max]にしてください"
        )
    low, high = (float(value) for value in values)
    if low > high:
        raise ValueError(f"search.{name}はmin <= maxにしてください")
    return low, high


def _sample_uniform(rng, search_cfg, name):
    low, high = _range(search_cfg, name)
    return float(rng.uniform(low, high))


def sample_trial_config(base_cfg, trial_index, seed):
    """Create trial 0 from the current config and randomized later trials."""
    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    if trial_index == 0:
        return trial_cfg

    rng = np.random.default_rng(int(seed) + int(trial_index) * 1009)
    search_cfg = trial_cfg.degradation_calibration.search
    motion_cfg = trial_cfg.self_supervised_deblur.cardiac_motion
    motion_cfg.max_translation_mm_yx = [
        _sample_uniform(rng, search_cfg, "translation_mm_y_range"),
        _sample_uniform(rng, search_cfg, "translation_mm_x_range"),
    ]
    motion_cfg.max_rotation_deg = _sample_uniform(rng, search_cfg, "rotation_deg_range")
    motion_cfg.max_scale_delta = _sample_uniform(rng, search_cfg, "scale_delta_range")
    motion_cfg.max_temporal_asymmetry = _sample_uniform(
        rng, search_cfg, "temporal_asymmetry_range"
    )
    motion_cfg.max_z_phase_offset = _sample_uniform(
        rng, search_cfg, "z_phase_offset_range"
    )
    softening_low, softening_high = _range(search_cfg, "heart_mask_softening_px_range")
    motion_cfg.heart_mask_softening_px = int(
        rng.integers(int(math.ceil(softening_low)), int(math.floor(softening_high)) + 1)
    )
    motion_cfg.uniform_phase_weight_mix = _sample_uniform(
        rng, search_cfg, "uniform_phase_weight_mix_range"
    )
    peak_center = _sample_uniform(rng, search_cfg, "bimodal_peak_center_range")
    peak_half_width = _sample_uniform(rng, search_cfg, "bimodal_peak_half_width_range")
    motion_cfg.bimodal_peak_sigma_range = [
        max(peak_center - peak_half_width, 0.01),
        peak_center + peak_half_width,
    ]
    balance_half_width = _sample_uniform(
        rng, search_cfg, "bimodal_balance_half_width_range"
    )
    motion_cfg.bimodal_balance_range = [
        max(0.5 - balance_half_width, 0.0),
        min(0.5 + balance_half_width, 1.0),
    ]

    sigma_min = _sample_uniform(rng, search_cfg, "gaussian_sigma_min_range")
    sigma_max = _sample_uniform(rng, search_cfg, "gaussian_sigma_max_range")
    sigma_min, sigma_max = sorted((sigma_min, sigma_max))
    if sigma_max - sigma_min < 1e-3:
        sigma_max = sigma_min + 1e-3
    trial_cfg.self_supervised_deblur.sigma_range = [sigma_min, sigma_max]
    if bool(trial_cfg.degradation_calibration.apply_identity_mixture):
        trial_cfg.self_supervised_deblur.identity_probability = _sample_uniform(
            rng, search_cfg, "identity_probability_range"
        )
    return trial_cfg


def _trial_parameters(cfg):
    motion = cfg.self_supervised_deblur.cardiac_motion
    return {
        "degradation_type": str(cfg.self_supervised_deblur.degradation_type),
        "identity_probability": float(cfg.self_supervised_deblur.identity_probability),
        "sigma_min": float(cfg.self_supervised_deblur.sigma_range[0]),
        "sigma_max": float(cfg.self_supervised_deblur.sigma_range[1]),
        "translation_y_mm": float(motion.max_translation_mm_yx[0]),
        "translation_x_mm": float(motion.max_translation_mm_yx[1]),
        "rotation_deg": float(motion.max_rotation_deg),
        "scale_delta": float(motion.max_scale_delta),
        "temporal_asymmetry": float(motion.max_temporal_asymmetry),
        "z_phase_offset": float(motion.max_z_phase_offset),
        "heart_mask_softening_px": int(motion.heart_mask_softening_px),
        "uniform_phase_weight_mix": float(motion.uniform_phase_weight_mix),
        "bimodal_peak_sigma_min": float(motion.bimodal_peak_sigma_range[0]),
        "bimodal_peak_sigma_max": float(motion.bimodal_peak_sigma_range[1]),
        "bimodal_balance_min": float(motion.bimodal_balance_range[0]),
        "bimodal_balance_max": float(motion.bimodal_balance_range[1]),
    }


def _uniform(rng, values):
    low, high = (float(value) for value in values)
    return float(rng.uniform(low, high))


def _sample_fixed_case_config(cfg, rng):
    """Convert training ranges to one deterministic validation-style sample."""
    case_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    degradation_cfg = case_cfg.self_supervised_deblur
    motion = degradation_cfg.cardiac_motion
    motion.validation_translation_mm_yx = [
        float(
            rng.uniform(
                -float(motion.max_translation_mm_yx[0]),
                float(motion.max_translation_mm_yx[0]),
            )
        ),
        float(
            rng.uniform(
                -float(motion.max_translation_mm_yx[1]),
                float(motion.max_translation_mm_yx[1]),
            )
        ),
    ]
    motion.validation_rotation_deg = float(
        rng.uniform(-float(motion.max_rotation_deg), float(motion.max_rotation_deg))
    )
    motion.validation_scale_delta = float(
        rng.uniform(-float(motion.max_scale_delta), float(motion.max_scale_delta))
    )
    motion.validation_bimodal_peak_sigma = _uniform(
        rng, motion.bimodal_peak_sigma_range
    )
    motion.validation_bimodal_balance = _uniform(rng, motion.bimodal_balance_range)
    motion.validation_temporal_asymmetry = float(
        rng.uniform(
            -float(motion.max_temporal_asymmetry), float(motion.max_temporal_asymmetry)
        )
    )
    motion.validation_z_phase_offset = float(
        rng.uniform(-float(motion.max_z_phase_offset), float(motion.max_z_phase_offset))
    )
    if motion.num_phases_range is not None:
        minimum, maximum = (int(value) for value in motion.num_phases_range)
        choices = np.arange(minimum, maximum + 1, 2)
        motion.num_phases = int(rng.choice(choices))
    motion.num_phases_range = None
    degradation_cfg.validation_sigma = _uniform(rng, degradation_cfg.sigma_range)
    return case_cfg


def _apply_seeded_source_artifacts(image, valid_mask, cfg, rng):
    """Reproduce optional source-only artifacts without stateful TensorFlow RNG."""
    from data.gpu_aug.filter import gaussian_filter

    artifact_cfg = cfg.aug
    random_value = float(rng.random())
    sharpness_probability = float(artifact_cfg.random_sharpness.prob)
    gaussian_probability = float(artifact_cfg.random_gauss_filter.prob)
    tensor = tf.convert_to_tensor(image[None], tf.float32)
    if random_value < sharpness_probability:
        blurred = gaussian_filter(
            tensor, sigma=float(artifact_cfg.random_sharpness.sigma)
        )
        alpha = _uniform(rng, artifact_cfg.random_sharpness.alpha_range)
        tensor = tensor + alpha * (tensor - blurred)
    elif random_value < sharpness_probability + gaussian_probability:
        sigma = _uniform(rng, artifact_cfg.random_gauss_filter.sigma_range)
        tensor = gaussian_filter(tensor, sigma=sigma)
    result = tensor.numpy()[0]

    if float(rng.random()) < float(artifact_cfg.random_gauss_noise.prob):
        stddev = _uniform(rng, artifact_cfg.random_gauss_noise.stddev_range)
        result += rng.normal(0.0, stddev, size=result.shape).astype(np.float32)
    return np.clip(result, 0.0, 1.0) * valid_mask


def simulate_cases(clean_cases, cfg, seed):
    calibration_cfg = cfg.degradation_calibration
    repetitions = int(calibration_cfg.simulations_per_clean)
    rng = np.random.default_rng(int(seed))
    simulated_cases = []

    for repetition in range(repetitions):
        for case in clean_cases:
            case_cfg = _sample_fixed_case_config(cfg, rng)
            image = tf.convert_to_tensor(case["image"][None], tf.float32)
            valid_mask = tf.convert_to_tensor(case["valid_mask"][None], tf.float32)
            heart_mask = tf.convert_to_tensor(case["heart_mask"][None], tf.float32)
            degraded = CustomModel.apply_self_supervised_deblur(
                image, valid_mask, case_cfg, is_training=False, heart_msks=heart_mask
            ).numpy()[0]
            degraded = _apply_seeded_source_artifacts(
                degraded, case["valid_mask"], case_cfg, rng
            )
            if bool(calibration_cfg.apply_identity_mixture) and float(
                rng.random()
            ) < float(case_cfg.self_supervised_deblur.identity_probability):
                degraded = case["image"].copy()
            simulated_cases.append(
                {
                    "case_id": f"{case['case_id']}_sim{repetition}",
                    "patient_id": case.get("patient_id", case["case_id"]),
                    "group_id": case.get("group_id", case["case_id"]),
                    "image": degraded,
                    "clean_image": case["image"],
                    "heart_mask": case["heart_mask"],
                    "valid_mask": case["valid_mask"],
                }
            )
    return simulated_cases


def measure_cases(cases, domain, cfg, trial_index=None):
    calibration_cfg = cfg.degradation_calibration
    records = []
    for case in cases:
        features = extract_volume_features(
            case["image"],
            case["heart_mask"],
            valid_mask=case["valid_mask"],
            roi_margin_px=int(calibration_cfg.roi_margin_px),
            max_autocorrelation_lag_px=int(calibration_cfg.max_autocorrelation_lag_px),
        )
        record = {
            "domain": domain,
            "trial": "" if trial_index is None else int(trial_index),
            "case_id": case["case_id"],
            "patient_id": case.get("patient_id", case["case_id"]),
            "group_id": case.get("group_id", case["case_id"]),
            **features,
        }
        records.append(record)
    return records


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _central_heart_slice(case):
    mask = np.asarray(case["heart_mask"])[..., 0] > 0
    return int(np.argmax(np.count_nonzero(mask, axis=(1, 2))))


def _gray_panel(image, z_index):
    gray = np.clip(np.asarray(image)[z_index, ..., 0], 0.0, 1.0)
    gray = np.round(gray * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _difference_panel(reference, comparison, reference_z, scale, comparison_z=None):
    if comparison_z is None:
        comparison_z = reference_z
    difference = (
        np.asarray(comparison)[comparison_z, ..., 0]
        - np.asarray(reference)[reference_z, ..., 0]
    ) * float(scale)
    positive = np.clip(difference, 0.0, 1.0)
    negative = np.clip(-difference, 0.0, 1.0)
    neutral = 1.0 - np.clip(np.abs(difference), 0.0, 1.0)
    rgb = np.stack(
        [positive + 0.5 * neutral, neutral * 0.5, negative + 0.5 * neutral], axis=-1
    )
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_montages(simulated_cases, real_cases, output_dir, cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(
        int(cfg.degradation_calibration.montages_per_trial), len(simulated_cases)
    )
    separator = np.zeros((int(cfg.aug.crop_size_zyx[1]), 4, 3), dtype=np.uint8)
    pair_by_patient = bool(cfg.degradation_calibration.pairing.enabled)
    real_by_patient = defaultdict(list)
    for real_case in real_cases:
        real_by_patient[real_case.get("patient_id")].append(real_case)
    patient_use_count = defaultdict(int)
    for index in range(count):
        simulated = simulated_cases[index]
        if pair_by_patient:
            patient_id = simulated["patient_id"]
            candidates = real_by_patient[patient_id]
            if not candidates:
                raise ValueError(f"montage用の対応real患者がありません: {patient_id}")
            real = candidates[patient_use_count[patient_id] % len(candidates)]
            patient_use_count[patient_id] += 1
        else:
            real = real_cases[index % len(real_cases)]
        clean_z = _central_heart_slice(simulated)
        real_z = _central_heart_slice(real)
        panels = [
            _gray_panel(simulated["clean_image"], clean_z),
            _gray_panel(simulated["image"], clean_z),
            _gray_panel(real["image"], real_z),
            _difference_panel(
                simulated["clean_image"],
                simulated["image"],
                clean_z,
                cfg.degradation_calibration.difference_display_scale,
            ),
            _difference_panel(
                simulated["clean_image"],
                real["image"],
                clean_z,
                cfg.degradation_calibration.difference_display_scale,
                comparison_z=real_z,
            ),
        ]
        montage_parts = []
        for panel in panels:
            if montage_parts:
                montage_parts.append(separator)
            montage_parts.append(panel)
        montage = np.concatenate(montage_parts, axis=1)
        filename = _safe_filename(simulated["case_id"]) + ".png"
        encoded = tf.io.encode_png(tf.convert_to_tensor(montage))
        tf.io.write_file(str(output_dir / filename), encoded)
    real_description = "patient-matched" if pair_by_patient else "unpaired"
    (output_dir / "README.txt").write_text(
        f"Left to right: clean | simulated degradation | {real_description} "
        "real non-gated | "
        "simulated-clean difference | real non-gated-clean difference "
        "(differences: red=positive, blue=negative).\n",
        encoding="utf-8",
    )


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _best_override_config(cfg):
    motion = cfg.self_supervised_deblur.cardiac_motion
    return OmegaConf.create(
        {
            "self_supervised_deblur": {
                "identity_probability": float(
                    cfg.self_supervised_deblur.identity_probability
                ),
                "degradation_type": str(cfg.self_supervised_deblur.degradation_type),
                "sigma_range": [
                    float(value) for value in cfg.self_supervised_deblur.sigma_range
                ],
                "cardiac_motion": {
                    "max_translation_mm_yx": [
                        float(value) for value in motion.max_translation_mm_yx
                    ],
                    "max_rotation_deg": float(motion.max_rotation_deg),
                    "max_scale_delta": float(motion.max_scale_delta),
                    "bimodal_peak_sigma_range": [
                        float(value) for value in motion.bimodal_peak_sigma_range
                    ],
                    "bimodal_balance_range": [
                        float(value) for value in motion.bimodal_balance_range
                    ],
                    "uniform_phase_weight_mix": float(motion.uniform_phase_weight_mix),
                    "max_temporal_asymmetry": float(motion.max_temporal_asymmetry),
                    "max_z_phase_offset": float(motion.max_z_phase_offset),
                    "heart_mask_softening_px": int(motion.heart_mask_softening_px),
                },
            }
        }
    )


def main():
    from main import gpu_setting

    logging.set_verbosity(logging.INFO)
    args = parse_args()
    cfg = load_config(args)
    calibration_cfg = cfg.degradation_calibration
    output_dir = Path(calibration_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "calibration_config.yaml")
    gpu_setting(str(cfg.gpu), bool(cfg.gpu_allow_growth))

    clean_heart_bit = _resolve_heart_bit(
        calibration_cfg.clean_heart_bit,
        cfg.bit_info.heart_bit,
        cfg.bit_info.padding_bit,
        "clean_heart_bit",
    )
    real_heart_bit = _resolve_heart_bit(
        calibration_cfg.real_heart_bit,
        cfg.bit_info.heart_bit,
        cfg.bit_info.padding_bit,
        "real_heart_bit",
    )
    max_cases = int(calibration_cfg.max_cases_per_domain)
    seed = int(calibration_cfg.seed)
    clean_data_dict, real_data_dict, matched_patient_ids = (
        prepare_calibration_data_dicts(cfg, max_cases, seed)
    )
    pairing_enabled = bool(calibration_cfg.pairing.enabled)
    if pairing_enabled:
        write_csv(
            output_dir / "matched_patients.csv",
            make_matching_manifest(
                clean_data_dict,
                real_data_dict,
                delimiter=calibration_cfg.pairing.delimiter,
            ),
        )
    logging.info("cleanデータを読み込みます")
    clean_cases = load_cases(
        calibration_cfg.clean_data_dir,
        cfg,
        clean_heart_bit,
        max_cases,
        seed,
        data_dict=clean_data_dict,
        group_by_patient=pairing_enabled,
    )
    logging.info("実non-gatedデータを読み込みます")
    real_cases = load_cases(
        calibration_cfg.real_data_dir,
        cfg,
        real_heart_bit,
        max_cases,
        seed + 1,
        data_dict=real_data_dict,
        group_by_patient=pairing_enabled,
    )
    logging.info(f"clean={len(clean_cases)} cases, real={len(real_cases)} cases")

    real_records = measure_cases(real_cases, "real", cfg)
    real_matrix = feature_dicts_to_matrix(real_records)
    real_groups = [record["group_id"] for record in real_records]
    write_csv(
        output_dir / "real_case_features.csv",
        real_records,
        ["domain", "trial", "case_id", "patient_id", "group_id", *FEATURE_NAMES],
    )

    trial_rows = []
    trial_configs = []
    num_trials = int(calibration_cfg.search.num_trials)
    for trial_index in range(num_trials):
        trial_cfg = sample_trial_config(cfg, trial_index, seed)
        trial_configs.append(trial_cfg)
        logging.info(f"trial {trial_index + 1}/{num_trials}: simulation")
        # Use one base seed so complete calibration runs are reproducible.
        simulated_cases = simulate_cases(clean_cases, trial_cfg, seed)
        simulated_records = measure_cases(
            simulated_cases, "simulated", trial_cfg, trial_index=trial_index
        )
        simulated_matrix = feature_dicts_to_matrix(simulated_records)
        simulated_groups = [record["group_id"] for record in simulated_records]
        summary_rows = compare_feature_distributions(simulated_matrix, real_matrix)
        classifier = cross_validated_domain_auc(
            simulated_matrix,
            real_matrix,
            simulated_groups=simulated_groups,
            real_groups=real_groups,
            num_folds=int(calibration_cfg.classifier_folds),
            seed=seed,
            l2=float(calibration_cfg.classifier_l2),
        )
        paired_summary_rows = []
        paired_detail_rows = []
        paired_metrics = {
            "num_paired_patients": 0,
            "paired_normalized_mae": math.nan,
            "paired_mean_feature_correlation": math.nan,
        }
        if pairing_enabled:
            paired_summary_rows, paired_detail_rows, paired_metrics = (
                compare_paired_feature_groups(
                    simulated_matrix,
                    real_matrix,
                    simulated_groups,
                    real_groups,
                    distance_clip=float(calibration_cfg.score_distance_clip),
                )
            )
        score_parts = aggregate_calibration_score(
            summary_rows,
            classifier["auc"],
            distance_clip=float(calibration_cfg.score_distance_clip),
            paired_distance=paired_metrics["paired_normalized_mae"],
            paired_distance_weight=(
                float(calibration_cfg.pairing.score_weight) if pairing_enabled else 0.0
            ),
        )
        trial_dir = output_dir / f"trial_{trial_index:03d}"
        trial_dir.mkdir(exist_ok=True)
        write_csv(
            trial_dir / "simulated_case_features.csv",
            simulated_records,
            ["domain", "trial", "case_id", "patient_id", "group_id", *FEATURE_NAMES],
        )
        write_csv(trial_dir / "feature_summary.csv", summary_rows)
        if pairing_enabled:
            write_csv(trial_dir / "paired_feature_summary.csv", paired_summary_rows)
            write_csv(
                trial_dir / "paired_patient_feature_details.csv", paired_detail_rows
            )
        save_montages(simulated_cases, real_cases, trial_dir / "montages", trial_cfg)

        trial_row = {
            "trial": trial_index,
            **_trial_parameters(trial_cfg),
            **classifier,
            **paired_metrics,
            **score_parts,
        }
        trial_rows.append(trial_row)
        trial_message = (
            f"trial {trial_index}: score={score_parts['score']:.4f}, "
            f"AUC={classifier['auc']:.4f}, "
            f"|SMD|={score_parts['mean_abs_standardized_mean_difference']:.4f}"
        )
        if pairing_enabled:
            trial_message += f", paired={paired_metrics['paired_normalized_mae']:.4f}"
        logging.info(trial_message)

    trial_rows.sort(key=lambda row: row["score"])
    write_csv(output_dir / "trials.csv", trial_rows)
    best_trial_index = int(trial_rows[0]["trial"])
    best_cfg = trial_configs[best_trial_index]
    OmegaConf.save(_best_override_config(best_cfg), output_dir / "best_config.yaml")
    report = {
        "best_trial": best_trial_index,
        "best_score": trial_rows[0]["score"],
        "feature_domain_auc": trial_rows[0]["auc"],
        "mean_abs_standardized_mean_difference": trial_rows[0][
            "mean_abs_standardized_mean_difference"
        ],
        "mean_normalized_wasserstein": trial_rows[0]["mean_normalized_wasserstein"],
        "num_clean_cases": len(clean_cases),
        "num_real_cases": len(real_cases),
        "pairing_enabled": pairing_enabled,
        "num_matched_patient_ids": len(matched_patient_ids),
        "paired_normalized_mae": trial_rows[0]["paired_normalized_mae"],
        "paired_mean_feature_correlation": trial_rows[0][
            "paired_mean_feature_correlation"
        ],
        "simulations_per_clean": int(calibration_cfg.simulations_per_clean),
        "notes": [
            "Lower score is better.",
            "Feature-domain AUC approaches 0.5 when the measured domains overlap.",
            "A low domain score alone does not prove anatomically correct motion.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info(
        f"完了: best trial={best_trial_index}, "
        f"score={trial_rows[0]['score']:.4f}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
