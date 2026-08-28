"""Minimal HDR/RAW reader used only by the generic-container CLI smoke test.

The production project supplies its own ``irg`` package.  This module is found
only when the test explicitly prepends ``calibration/test_support`` to
``PYTHONPATH``.
"""

from pathlib import Path

import numpy as np


def _paths(hdr_path, data_path, suffix):
    hdr_path = Path(hdr_path)
    if data_path is None:
        data_path = hdr_path.with_suffix(suffix)
    return hdr_path, Path(data_path)


def _write_header(hdr_path, array, spacing_zyx, flag=""):
    dtype = np.dtype(array.dtype)
    size_xyz = [array.shape[2], array.shape[1], array.shape[0]]
    spacing_xyz = [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]]
    fields = [*size_xyz, dtype.itemsize, *spacing_xyz]
    text = " ".join(str(value) for value in fields)
    if flag:
        text += " " + flag
    hdr_path.write_text(text + "\n", encoding="utf-8")


def read_hdr(hdr_path, return_flag=False, return_bit_dict=False):
    lines = Path(hdr_path).read_text(encoding="utf-8").splitlines()
    fields = lines[0].split()
    size_zyx = np.asarray([fields[2], fields[1], fields[0]], np.uint32)
    flag = fields[7] if len(fields) > 7 else ""
    if flag in ("mask", "label"):
        dtype = np.dtype(f"u{fields[3]}")
    else:
        dtype = np.dtype(f"i{fields[3]}")
    spacing_zyx = np.asarray([fields[6], fields[5], fields[4]], np.float32)
    bit_dict = {}
    result = [size_zyx, dtype, spacing_zyx]
    if return_flag:
        result.append(flag)
    if return_bit_dict:
        result.append(bit_dict)
    return tuple(result)


def save_raw(input_array, spacing_zyx, hdr_path=None, raw_path=None, save_type=None):
    del save_type
    hdr_path, raw_path = _paths(hdr_path, raw_path, ".raw")
    _write_header(hdr_path, input_array, spacing_zyx)
    np.asarray(input_array).tofile(raw_path)


def save_re4(
    input_array,
    spacing_zyx,
    type_flag,
    hdr_path=None,
    re4_path=None,
    bit_dict=None,
    src_dst_bit_dict=None,
):
    del bit_dict, src_dst_bit_dict
    hdr_path, re4_path = _paths(hdr_path, re4_path, ".re4")
    _write_header(hdr_path, input_array, spacing_zyx, flag=type_flag)
    np.asarray(input_array).tofile(re4_path)


def _crop(array, clip_zyxzyx):
    if clip_zyxzyx is None:
        return array
    z0, y0, x0, z1, y1, x1 = (int(value) for value in clip_zyxzyx)
    return array[z0:z1, y0:y1, x0:x1]


def read_raw(
    hdr_path=None,
    raw_path=None,
    return_spacing_zyx=False,
    clip_zyxzyx=None,
    size_zyx=None,
    img_dtype=None,
    use_memmap=True,
):
    del use_memmap
    hdr_path, raw_path = _paths(hdr_path, raw_path, ".raw")
    if size_zyx is None or img_dtype is None or return_spacing_zyx:
        size_zyx, img_dtype, spacing_zyx = read_hdr(hdr_path)
    array = np.fromfile(raw_path, img_dtype).reshape(tuple(size_zyx))
    array = _crop(array, clip_zyxzyx)
    if return_spacing_zyx:
        return array, spacing_zyx
    return array


def read_re4(
    hdr_path=None,
    re4_path=None,
    return_spacing_zyx=False,
    src_dst_bit_dict=None,
    clip_zyxzyx=None,
    size_zyx=None,
    type_flag=None,
    **kwargs,
):
    del kwargs, type_flag
    hdr_path, re4_path = _paths(hdr_path, re4_path, ".re4")
    if size_zyx is None or return_spacing_zyx:
        size_zyx, dtype, spacing_zyx = read_hdr(hdr_path)
    else:
        dtype = read_hdr(hdr_path)[1]
    array = np.fromfile(re4_path, dtype).reshape(tuple(size_zyx))
    array = _crop(array, clip_zyxzyx)
    if src_dst_bit_dict is not None:
        channels = max(src_dst_bit_dict.values()) + 1
        output = np.zeros((*array.shape, channels), np.uint32)
        for source_bit, destination_channel in src_dst_bit_dict.items():
            output[..., destination_channel] = (array >> source_bit) & 1
        array = output
    if return_spacing_zyx:
        return array, spacing_zyx
    return array
