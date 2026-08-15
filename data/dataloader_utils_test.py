import numpy as np
import pytest

from .dataloader_utils import get_center, has_foreground_in_every_z_slice


def test_has_foreground_in_every_z_slice_accepts_packed_mask():
    msk = np.zeros((3, 4, 5), np.uint16)
    msk[0, 0, 0] = 1 << 6
    msk[1, 1, 1] = (1 << 6) | (1 << 15)
    msk[2, 2, 2] = 1 << 6

    assert has_foreground_in_every_z_slice(msk, foreground_bit=6)


def test_has_foreground_in_every_z_slice_rejects_one_empty_slice():
    msk = np.zeros((3, 4, 5, 1), np.uint16)
    msk[0, :2, :2, 0] = 1 << 6
    msk[2, :2, :2, 0] = 1 << 6

    assert not has_foreground_in_every_z_slice(msk, foreground_bit=6)


def test_has_foreground_in_every_z_slice_respects_minimum_voxels():
    msk = np.full((2, 2, 2), 1 << 6, np.uint16)
    msk[1] = 0
    msk[1, 0, 0] = 1 << 6

    assert has_foreground_in_every_z_slice(msk, 6, min_voxels_per_slice=1)
    assert not has_foreground_in_every_z_slice(msk, 6, min_voxels_per_slice=2)


@pytest.mark.parametrize("min_voxels", [0, -1])
def test_has_foreground_in_every_z_slice_validates_minimum(min_voxels):
    with pytest.raises(ValueError):
        has_foreground_in_every_z_slice(
            np.zeros((1, 1, 1), np.uint16),
            foreground_bit=6,
            min_voxels_per_slice=min_voxels,
        )


def test_non_training_crop_uses_organ_box_center():
    organ_box = np.array([10, 20, 30, 20, 40, 50])
    center = get_center(
        img_size_zyx=np.array([40, 100, 100]),
        img_spacing_zyx=np.ones(3),
        is_training=False,
        body_box_zyxzyx=np.array([0, 0, 0, 40, 100, 100]),
        organ_box_zyxzyx=organ_box,
        random_crop_method={"image": 1.0},
        crop_size_zyx=np.array([8, 16, 16]),
        target_spacing_zyx=np.ones(3),
        margin=0,
        crop_keep_ratio=0.3,
    )

    np.testing.assert_allclose(center, [15, 30, 40])
