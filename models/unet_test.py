import numpy as np
import tensorflow as tf
from keras import Model

from .unet import build_unet


RENORM = {
    "r_max": 1.0,
    "d_max": 0.0,
    "warmup_steps": 5,
    "change_d_steps": 10,
    "change_r_steps": 10,
}


def _build_factorized_unet():
    return build_unet(
        Model,
        input_shape=(8, 32, 32, 2),
        num_channel=1,
        start_ch=4,
        depth=3,
        num_encode_blocks=2,
        num_decode_blocks=1,
        conv_kernel_size_zyx=(1, 3, 3),
        z_conv_kernel_size_zyx=(3, 3, 3),
        z_conv_interval=3,
        factorized_z_conv=True,
        factorized_residual=True,
        pool_size_zyx=(1, 2, 2),
        up_kernel_size_zyx=(1, 4, 4),
        up_strides_zyx=(1, 2, 2),
        z_downsample_stages=(1, 2),
        z_down_kernel_size_zyx=(4, 1, 1),
        z_down_strides_zyx=(2, 1, 1),
        z_upsample_type="transpose_conv",
        z_up_kernel_size_zyx=(4, 1, 1),
        **RENORM,
    )


def test_factorized_unet_preserves_shape_and_runs_forward():
    model = _build_factorized_unet()
    image = tf.random.normal((1, 8, 32, 32, 2))
    mask = tf.ones((1, 8, 32, 32, 1), tf.float32)

    @tf.function(jit_compile=True)
    def predict(image, mask):
        return model([image, mask], training=False)

    output = predict(image, mask)

    assert output.shape == (1, 8, 32, 32, 1)
    assert np.all(np.isfinite(output.numpy()))


def test_factorized_unet_has_requested_xy_z_residual_and_sampling_layers():
    model = _build_factorized_unet()

    assert model.get_layer("enc0_conv0_xy").kernel_size == (1, 3, 3)
    assert model.get_layer("enc0_conv0_z").kernel_size == (3, 1, 1)
    assert model.get_layer("enc0_conv0_residual_projection") is not None
    assert model.get_layer("enc0_conv0_residual_add") is not None

    assert model.get_layer("enc1_z_downsample_strideconv").strides == (2, 1, 1)
    assert model.get_layer("enc2_z_downsample_strideconv").strides == (2, 1, 1)
    assert model.get_layer("dec1_z_up_transposeconv").strides == (2, 1, 1)
    assert model.get_layer("dec2_z_up_transposeconv").strides == (2, 1, 1)

    layer_names = {layer.name for layer in model.layers}
    assert "enc0_z_downsample_strideconv" not in layer_names
    assert "dec0_z_up_transposeconv" not in layer_names


def test_disabled_options_keep_legacy_unet_layer_structure():
    model = build_unet(
        Model,
        input_shape=(4, 16, 16, 2),
        num_channel=1,
        start_ch=4,
        depth=2,
        num_encode_blocks=1,
        num_decode_blocks=1,
        conv_kernel_size_zyx=(1, 3, 3),
        z_conv_kernel_size_zyx=(3, 3, 3),
        z_conv_interval=1,
        factorized_z_conv=False,
        factorized_residual=False,
        pool_size_zyx=(1, 2, 2),
        up_kernel_size_zyx=(1, 4, 4),
        up_strides_zyx=(1, 2, 2),
        z_downsample_stages=(),
        **RENORM,
    )

    layer_names = {layer.name for layer in model.layers}
    assert "enc0_conv0" in layer_names
    assert "enc0_conv0_xy" not in layer_names
    assert not any("z_downsample" in name for name in layer_names)
    assert not any("z_up" in name for name in layer_names)


if __name__ == "__main__":
    test_factorized_unet_preserves_shape_and_runs_forward()
    test_factorized_unet_has_requested_xy_z_residual_and_sampling_layers()
    test_disabled_options_keep_legacy_unet_layer_structure()
