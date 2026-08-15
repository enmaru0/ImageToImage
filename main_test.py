from pathlib import Path

import pytest

from main import prepare_unpaired_data_dict


def _touch_volume(root: Path, relative_stem: str):
    raw_path = root / f"{relative_stem}.raw"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.touch()
    raw_path.with_suffix(".hdr").touch()
    return raw_path.with_suffix(".hdr")


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
