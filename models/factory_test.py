from models.factory import canonicalize_model_name, get_downsample_factor_zyx


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def test_model_name_aliases():
    assert canonicalize_model_name("unet") == "unet"
    assert canonicalize_model_name("Pix2Pix") == "pix2pix_generator"
    assert canonicalize_model_name("pix2pix_generator") == "pix2pix_generator"


def test_pix2pix_downsample_factor_uses_its_own_depth_and_strides():
    cfg = AttrDict(
        name="pix2pix_generator",
        pix2pix_generator=AttrDict(strides_zyx=[1, 2, 2], depth=4),
    )

    assert get_downsample_factor_zyx(cfg).tolist() == [1, 16, 16]
