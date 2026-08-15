from keras import Input, Model, initializers, layers
from keras.src import ops

from .layers import BatchRenormalization


def _to_tuple3(value):
    return tuple(int(v) for v in value)


def _validate_renorm_args(renorm):
    allowed_keys = {
        "r_max",
        "d_max",
        "warmup_steps",
        "change_d_steps",
        "change_r_steps",
        "momentum",
        "epsilon",
        "center",
        "scale",
        "dtype",
        "trainable",
    }
    unexpected_args = sorted(set(renorm) - allowed_keys)
    if unexpected_args:
        raise TypeError(f"Unrecognized Pix2Pix Generator arguments: {unexpected_args}")


def _renormalize(x, mask, renorm, name):
    mask = ops.squeeze(mask, axis=-1)
    return BatchRenormalization(**renorm, synchronized=False, name=name)(x, mask=mask)


def build_pix2pix_generator(
    CustomModel: Model,
    input_shape: tuple[int, int, int, int],
    num_channel: int,
    start_ch: int = 16,
    depth: int = 4,
    max_ch: int = 128,
    down_kernel_size_zyx: tuple[int, int, int] = (1, 4, 4),
    strides_zyx: tuple[int, int, int] = (1, 2, 2),
    up_kernel_size_zyx: tuple[int, int, int] = (1, 4, 4),
    dropout_depth: int = 2,
    dropout_rate: float = 0.3,
    leaky_relu_alpha: float = 0.2,
    **renorm: dict,
) -> Model:
    """Build a compact anisotropic 3D Pix2Pix-style U-Net generator.

    The original Pix2Pix block pattern is retained (strided convolution,
    LeakyReLU encoder, transposed-convolution decoder, skip connections), while
    Z is not downsampled for thick-slice cardiac CT. The output stays linear
    because I2I-RFR predicts x0 rather than a directly normalized tanh image.
    """
    _validate_renorm_args(renorm)
    if depth < 2:
        raise ValueError("Pix2Pix Generator depth must be at least 2")
    if start_ch <= 0 or max_ch < start_ch:
        raise ValueError("start_ch must be positive and max_ch >= start_ch")
    if not 0 <= dropout_depth < depth:
        raise ValueError("dropout_depth must satisfy 0 <= dropout_depth < depth")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")

    down_kernel = _to_tuple3(down_kernel_size_zyx)
    strides = _to_tuple3(strides_zyx)
    up_kernel = _to_tuple3(up_kernel_size_zyx)

    def kernel_initializer():
        # Use a fresh initializer per layer so kernels do not receive identical
        # values when Keras reuses an unseeded initializer instance.
        return initializers.RandomNormal(mean=0.0, stddev=0.02)

    input_img = Input(shape=input_shape, name="image")
    input_img_msk = Input(shape=input_shape[:3] + (1,), name="mask")
    x = input_img
    mask = input_img_msk
    skips = []
    skip_masks = []

    # Encoder: one stride-2 convolution per resolution, as in Pix2Pix.
    for block_index in range(depth):
        filters = min(start_ch << block_index, max_ch)
        mask = layers.MaxPooling3D(
            pool_size=strides,
            strides=strides,
            padding="same",
            name=f"pix2pix_enc{block_index}_mask_down",
        )(mask)
        x = layers.Conv3D(
            filters=filters,
            kernel_size=down_kernel,
            strides=strides,
            padding="same",
            use_bias=block_index == 0,
            kernel_initializer=kernel_initializer(),
            name=f"pix2pix_enc{block_index}_strideconv",
        )(x)
        if block_index > 0:
            x = _renormalize(x, mask, renorm, f"pix2pix_enc{block_index}_bn")
        x = layers.LeakyReLU(
            negative_slope=leaky_relu_alpha, name=f"pix2pix_enc{block_index}_lrelu"
        )(x)
        x = x * mask
        if block_index < depth - 1:
            skips.append(x)
            skip_masks.append(mask)

    # Decoder: learned upsampling followed by the corresponding encoder skip.
    for decoder_index, block_index in enumerate(reversed(range(depth - 1))):
        filters = min(start_ch << block_index, max_ch)
        output_mask = skip_masks.pop()
        x = layers.Conv3DTranspose(
            filters=filters,
            kernel_size=up_kernel,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_initializer(),
            name=f"pix2pix_dec{block_index}_transposeconv",
        )(x)
        x = _renormalize(x, output_mask, renorm, f"pix2pix_dec{block_index}_bn")
        if decoder_index < dropout_depth and dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"pix2pix_dec{block_index}_dropout")(
                x
            )
        x = layers.Activation("relu", name=f"pix2pix_dec{block_index}_relu")(x)
        x = x * output_mask
        x = layers.Concatenate(name=f"pix2pix_dec{block_index}_skip")([x, skips.pop()])

    outputs = layers.Conv3DTranspose(
        filters=num_channel,
        kernel_size=up_kernel,
        strides=strides,
        padding="same",
        use_bias=True,
        kernel_initializer=kernel_initializer(),
        name="pix2pix_output_transposeconv",
    )(x)
    outputs = outputs * input_img_msk
    return CustomModel(inputs=[input_img, input_img_msk], outputs=outputs)
