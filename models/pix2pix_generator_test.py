import numpy as np
import tensorflow as tf
from keras import Model

from .pix2pix_generator import build_pix2pix_generator
from .unet import build_unet


RENORM = {
    "r_max": 1.0,
    "d_max": 0.0,
    "warmup_steps": 5,
    "change_d_steps": 10,
    "change_r_steps": 10,
}


def test_pix2pix_generator_preserves_shape_and_masks_padding():
    model = build_pix2pix_generator(
        Model,
        input_shape=(4, 64, 64, 2),
        num_channel=1,
        start_ch=8,
        depth=3,
        max_ch=32,
        dropout_depth=1,
        **RENORM,
    )
    image = tf.random.normal((1, 4, 64, 64, 2))
    mask = np.ones((1, 4, 64, 64, 1), np.float32)
    mask[:, :, :, 48:] = 0.0
    output = model([image, tf.constant(mask)], training=False)

    assert output.shape == (1, 4, 64, 64, 1)
    np.testing.assert_allclose(output.numpy()[:, :, :, 48:], 0.0, atol=1e-7)


def test_default_pix2pix_generator_is_smaller_than_default_unet():
    pix2pix = build_pix2pix_generator(
        Model,
        input_shape=(8, 192, 192, 2),
        num_channel=1,
        start_ch=16,
        depth=4,
        max_ch=128,
        dropout_depth=2,
        **RENORM,
    )
    unet = build_unet(
        Model,
        input_shape=(8, 192, 192, 2),
        num_channel=1,
        start_ch=32,
        depth=4,
        num_encode_blocks=2,
        num_decode_blocks=1,
        conv_kernel_size_zyx=(1, 3, 3),
        z_conv_kernel_size_zyx=(3, 3, 3),
        z_conv_interval=3,
        pool_size_zyx=(1, 2, 2),
        up_kernel_size_zyx=(1, 4, 4),
        up_strides_zyx=(1, 2, 2),
        **RENORM,
    )

    assert pix2pix.count_params() < unet.count_params() / 4
