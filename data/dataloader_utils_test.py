import numpy as np
import pytest

from .dataloader_utils import has_foreground_in_every_z_slice


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
