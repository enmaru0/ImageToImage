import numpy as np

from .pix2pix_generator import build_pix2pix_generator
from .unet import build_unet


_MODEL_NAME_ALIASES = {
    "unet": "unet",
    "pix2pix": "pix2pix_generator",
    "pix2pixgenerator": "pix2pix_generator",
}


def canonicalize_model_name(name):
    key = str(name).lower().replace("_", "").replace("-", "")
    if key not in _MODEL_NAME_ALIASES:
        choices = sorted(set(_MODEL_NAME_ALIASES.values()))
        raise ValueError(f"Unsupported model.name: {name}. Choose from {choices}")
    return _MODEL_NAME_ALIASES[key]


def build_model(CustomModel, input_shape, num_channel, model_cfg):
    model_name = canonicalize_model_name(model_cfg.name)
    if model_name == "unet":
        return build_unet(
            CustomModel, input_shape, num_channel, **model_cfg.unet, **model_cfg.renorm
        )
    return build_pix2pix_generator(
        CustomModel,
        input_shape,
        num_channel,
        **model_cfg.pix2pix_generator,
        **model_cfg.renorm,
    )


def get_downsample_factor_zyx(model_cfg):
    model_name = canonicalize_model_name(model_cfg.name)
    if model_name == "unet":
        strides = np.asarray(model_cfg.unet.pool_size_zyx, dtype=np.int64)
        depth = int(model_cfg.unet.depth)
    else:
        strides = np.asarray(model_cfg.pix2pix_generator.strides_zyx, dtype=np.int64)
        depth = int(model_cfg.pix2pix_generator.depth)
    return np.power(strides, depth)
