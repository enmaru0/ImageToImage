from keras import Input, Model, layers
from keras.src import ops

from .layers import BatchRenormalization


_DOWNSAMPLE_TYPE_ALIASES = {
    "maxpool": "max_pool",
    "maxpooling": "max_pool",
    "pooling": "max_pool",
    "strideconv": "stride_conv",
    "stridedconv": "stride_conv",
}
_UPSAMPLE_TYPE_ALIASES = {
    "transposeconv": "transpose_conv",
    "transposedconv": "transpose_conv",
    "transposeupconv": "transpose_conv",
    "resizeconv": "resize_conv",
    "resizeupconv": "resize_conv",
}


def _canonicalize_type(value, aliases, option_name):
    key = str(value).lower().replace("_", "").replace("-", "")
    if key not in aliases:
        choices = sorted(set(aliases.values()))
        raise ValueError(f"Unsupported {option_name}: {value}. Choose from {choices}")
    return aliases[key]


def _to_tuple3(value):
    return tuple(int(v) for v in value)


def _select_conv_kernel(
    block_index: int,
    conv_kernel_size_zyx: tuple[int, int, int],
    z_conv_kernel_size_zyx: tuple[int, int, int] | None,
    z_conv_interval: int,
):
    if z_conv_kernel_size_zyx is None or z_conv_interval <= 0:
        return conv_kernel_size_zyx
    if block_index % z_conv_interval == 0:
        return z_conv_kernel_size_zyx
    return conv_kernel_size_zyx


def conv_block(
    img,
    img_msk,
    filters: int,
    name_conv: str,
    renorm: dict,
    conv_kernel_size_zyx: tuple[int, int, int],
):
    x = layers.Conv3D(
        filters=filters,
        kernel_size=_to_tuple3(conv_kernel_size_zyx),
        padding="same",
        use_bias=False,
        name=name_conv,
        kernel_initializer="he_uniform",
    )(img)
    squeeze_img_msk = ops.squeeze(img_msk)
    x = BatchRenormalization(
        **renorm, synchronized=False, name=name_conv.replace("conv", "bn")
    )(x, mask=squeeze_img_msk)
    x = x * img_msk
    x = layers.Activation("relu")(x)
    return x


def encode_downsample(
    img,
    img_msk,
    name: str,
    downsample_type: str,
    pool_size_zyx: tuple[int, int, int],
    down_kernel_size_zyx: tuple[int, int, int],
):
    pool_size_zyx = _to_tuple3(pool_size_zyx)
    msk_downsampled = layers.MaxPooling3D(
        pool_size=pool_size_zyx,
        strides=pool_size_zyx,
        padding="same",
        name=f"{name}_mask_pool",
    )(img_msk)

    if downsample_type == "max_pool":
        img_downsampled = layers.MaxPooling3D(
            pool_size=pool_size_zyx,
            strides=pool_size_zyx,
            padding="same",
            name=f"{name}_pool",
        )(img)
    elif downsample_type == "stride_conv":
        # チャンネル数を維持し、純粋にdownsampling演算だけを比較可能にする。
        img_downsampled = layers.Conv3D(
            filters=int(img.shape[-1]),
            kernel_size=_to_tuple3(down_kernel_size_zyx),
            strides=pool_size_zyx,
            padding="same",
            use_bias=True,
            kernel_initializer="he_uniform",
            name=f"{name}_strideconv",
        )(img)
        img_downsampled = img_downsampled * msk_downsampled
    else:
        raise ValueError(f"Unsupported downsample_type: {downsample_type}")

    return img_downsampled, msk_downsampled


def decode_up(
    img,
    filters: int,
    name: str,
    upsample_type: str,
    up_kernel_size_zyx: tuple[int, int, int],
    up_strides_zyx: tuple[int, int, int],
    resize_conv_kernel_size_zyx: tuple[int, int, int],
):
    if upsample_type == "transpose_conv":
        return layers.Conv3DTranspose(
            filters=filters,
            kernel_size=_to_tuple3(up_kernel_size_zyx),
            strides=_to_tuple3(up_strides_zyx),
            padding="same",
            name=name,
            kernel_initializer="he_uniform",
            use_bias=True,
        )(img)
    if upsample_type == "resize_conv":
        resize_name = name.replace("transposeconv", "resize")
        conv_name = name.replace("transposeconv", "resizeconv")
        x = layers.UpSampling3D(size=_to_tuple3(up_strides_zyx), name=resize_name)(img)
        return layers.Conv3D(
            filters=filters,
            kernel_size=_to_tuple3(resize_conv_kernel_size_zyx),
            padding="same",
            name=conv_name,
            kernel_initializer="he_uniform",
            use_bias=True,
        )(x)
    raise ValueError(f"Unsupported upsample_type: {upsample_type}")


def decoder_last(
    img,
    img_msk,
    skip_tensor,
    filters: int,
    num_channel: int,
    num_decode_blocks: int,
    renorm: dict,
    conv_kernel_size_zyx: tuple[int, int, int],
    z_conv_kernel_size_zyx: tuple[int, int, int] | None,
    z_conv_interval: int,
    block_index: int,
    upsample_type: str,
    up_kernel_size_zyx: tuple[int, int, int],
    up_strides_zyx: tuple[int, int, int],
    resize_conv_kernel_size_zyx: tuple[int, int, int],
):
    x = decode_up(
        img,
        filters,
        "dec0_transposeconv0",
        upsample_type,
        up_kernel_size_zyx,
        up_strides_zyx,
        resize_conv_kernel_size_zyx,
    )
    x = layers.Concatenate()([x, skip_tensor])

    for e in range(num_decode_blocks - 1):
        kernel_size = _select_conv_kernel(
            block_index + e,
            conv_kernel_size_zyx,
            z_conv_kernel_size_zyx,
            z_conv_interval,
        )
        x = conv_block(x, img_msk, filters, f"dec0_conv{e}", renorm, kernel_size)

    x = layers.Conv3D(
        filters=num_channel,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name=f"dec0_conv{num_decode_blocks - 1}",
        kernel_initializer="he_uniform",
    )(x)
    return x


def build_unet(
    CustomModel: Model,
    input_shape: tuple[int, int, int, int],  # z,y,x,c
    num_channel: int,
    start_ch: int,
    depth: int,
    num_encode_blocks: int,
    num_decode_blocks: int,
    conv_kernel_size_zyx: tuple[int, int, int] = (3, 3, 3),
    z_conv_kernel_size_zyx: tuple[int, int, int] | None = None,
    z_conv_interval: int = 0,
    downsample_type: str = "max_pool",
    conv_type: str | None = None,
    pool_size_zyx: tuple[int, int, int] = (2, 2, 2),
    down_kernel_size_zyx: tuple[int, int, int] = (3, 3, 3),
    upsample_type: str = "transpose_conv",
    up_type: str | None = None,
    up_kernel_size_zyx: tuple[int, int, int] = (4, 4, 4),
    up_strides_zyx: tuple[int, int, int] = (2, 2, 2),
    resize_conv_kernel_size_zyx: tuple[int, int, int] = (3, 3, 3),
    **renorm: dict,
) -> Model:
    if conv_type is not None:
        downsample_type = conv_type
    if up_type is not None:
        upsample_type = up_type
    downsample_type = _canonicalize_type(
        downsample_type, _DOWNSAMPLE_TYPE_ALIASES, "downsample_type/conv_type"
    )
    upsample_type = _canonicalize_type(
        upsample_type, _UPSAMPLE_TYPE_ALIASES, "upsample_type/up_type"
    )

    allowed_renorm_keys = {
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
    unexpected_args = sorted(set(renorm) - allowed_renorm_keys)
    if unexpected_args:
        raise TypeError(f"Unrecognized U-Net arguments: {unexpected_args}")

    input_img = Input(shape=input_shape, name="image")
    input_img_msk = Input(shape=input_shape[:3] + (1,), name="mask")

    ## Encoder
    skips = []
    skips_msk = [input_img_msk]
    x = input_img
    img_msk = input_img_msk
    conv_block_index = 0
    for d in range(depth):
        for e in range(num_encode_blocks):
            kernel_size = _select_conv_kernel(
                conv_block_index,
                conv_kernel_size_zyx,
                z_conv_kernel_size_zyx,
                z_conv_interval,
            )
            x = conv_block(
                x, img_msk, start_ch << d, f"enc{d}_conv{e}", renorm, kernel_size
            )
            conv_block_index += 1
        x_pooled, msk_pooled = encode_downsample(
            x,
            img_msk,
            f"enc{d}_downsample",
            downsample_type,
            pool_size_zyx,
            down_kernel_size_zyx,
        )
        skips.append(x)
        skips_msk.append(msk_pooled)
        x, img_msk = x_pooled, msk_pooled

    # Bottleneck
    for b in range(num_encode_blocks):
        kernel_size = _select_conv_kernel(
            conv_block_index,
            conv_kernel_size_zyx,
            z_conv_kernel_size_zyx,
            z_conv_interval,
        )
        x = conv_block(
            x, img_msk, start_ch << depth, f"bottom_conv{b}", renorm, kernel_size
        )
        conv_block_index += 1

    skips_msk = skips_msk[:-1]  # delete the last one
    # Decoder
    for d in reversed(range(1, depth)):
        _skip_x = skips.pop()
        _skip_msk = skips_msk.pop()

        x = decode_up(
            x,
            start_ch << d,
            f"dec{d}_transposeconv0",
            upsample_type,
            up_kernel_size_zyx,
            up_strides_zyx,
            resize_conv_kernel_size_zyx,
        )
        x = layers.Concatenate()([x, _skip_x])
        for e in range(num_decode_blocks):
            kernel_size = _select_conv_kernel(
                conv_block_index,
                conv_kernel_size_zyx,
                z_conv_kernel_size_zyx,
                z_conv_interval,
            )
            x = conv_block(
                x, _skip_msk, start_ch << d, f"dec{d}_conv{e}", renorm, kernel_size
            )
            conv_block_index += 1

    # last decode
    _skip_x = skips.pop()
    _skip_msk = skips_msk.pop()
    outputs = decoder_last(
        x,
        _skip_msk,
        _skip_x,
        start_ch,
        num_channel,
        num_decode_blocks,
        renorm,
        conv_kernel_size_zyx,
        z_conv_kernel_size_zyx,
        z_conv_interval,
        conv_block_index,
        upsample_type,
        up_kernel_size_zyx,
        up_strides_zyx,
        resize_conv_kernel_size_zyx,
    )

    return CustomModel(inputs=[input_img, input_img_msk], outputs=outputs)
