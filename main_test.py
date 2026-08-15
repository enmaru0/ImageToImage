from pathlib import Path

import pytest
from omegaconf import OmegaConf

from main import get_test_heart_bit, prepare_unpaired_data_dict


def _touch_volume(root: Path, relative_stem: str):
    raw_path = root / f"{relative_stem}.raw"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch()
    raw_path.with_suffix(".hdr").touch()
    return raw_path.with_suffix(".hdr")


def _touch_heart_mask(image_hdr: Path):
    mask_hdr = image_hdr.with_suffix(".mask.hdr")
    mask_hdr.touch()
    mask_hdr.with_suffix(".raw").touch()
    return mask_hdr


def test_prepare_unpaired_data_dict_finds_nested_images_and_ignores_masks(tmp_path):
    first_hdr = _touch_volume(tmp_path, "patient_a/image")
    second_hdr = _touch_volume(tmp_path, "patient_b/image")
    _touch_volume(tmp_path, "patient_a/image.mask")

    data_dict = prepare_unpaired_data_dict(tmp_path)

    pairs = next(iter(data_dict.values()))["img_hdr_list"]
    assert pairs == [(first_hdr, first_hdr), (second_hdr, second_hdr)]


def test_prepare_unpaired_data_dict_requires_hdr(tmp_path):
    (tmp_path / "image.raw").touch()

    with pytest.raises(FileNotFoundError):
        prepare_unpaired_data_dict(tmp_path)


def test_prepare_unpaired_data_dict_requires_heart_mask_when_enabled(tmp_path):
    image_hdr = _touch_volume(tmp_path, "patient/image")

    with pytest.raises(
        FileNotFoundError, match=str(image_hdr.with_suffix(".mask.hdr"))
    ):
        prepare_unpaired_data_dict(tmp_path, require_heart_mask=True)


def test_prepare_unpaired_data_dict_accepts_heart_mask_when_required(tmp_path):
    image_hdr = _touch_volume(tmp_path, "patient/image")
    _touch_heart_mask(image_hdr)

    data_dict = prepare_unpaired_data_dict(tmp_path, require_heart_mask=True)

    pairs = next(iter(data_dict.values()))["img_hdr_list"]
    assert pairs == [(image_hdr, image_hdr)]


def test_test_heart_bit_falls_back_to_training_bit():
    cfg = OmegaConf.create(
        {
            "bit_info": {"heart_bit": 6, "padding_bit": 15},
            "test_image_log": {"heart_bit": None},
        }
    )

    assert get_test_heart_bit(cfg) == 6


def test_test_heart_bit_can_differ_from_training_bit():
    cfg = OmegaConf.create(
        {
            "bit_info": {"heart_bit": 6, "padding_bit": 15},
            "test_image_log": {"heart_bit": 3},
        }
    )

    assert get_test_heart_bit(cfg) == 3


@pytest.mark.parametrize("heart_bit", [-1, 15])
def test_test_heart_bit_rejects_invalid_value(heart_bit):
    cfg = OmegaConf.create(
        {
            "bit_info": {"heart_bit": 6, "padding_bit": 15},
            "test_image_log": {"heart_bit": heart_bit},
        }
    )

    with pytest.raises(ValueError):
        get_test_heart_bit(cfg)
