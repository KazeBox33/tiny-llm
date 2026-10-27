import mlx.core as mx
from typing import Any

from extensions import tiny_llm_ext


def dequantize_weights(
    weight: mx.array, # (1, 3, 320)
    scales: mx.array, # (1,3,320*8//128)
    biases: mx.array, # (1,3,320*8//128)
    group_size: int, # 128
    bits: int,
) -> mx.array:
    """恢复选中的 W4/G128 权重行，保留输入的所有前导维度。"""
    if bits != 4 or group_size != 128:
        raise ValueError("dequantize_weights requires 4-bit weights and group_size=128")
    if weight.ndim < 1 or weight.dtype != mx.uint32:
        raise ValueError("packed weights must have a last dimension and dtype uint32")
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise ValueError("scales and biases must use bfloat16")
    width = weight.shape[-1] * 8
    group_shape = (*weight.shape[:-1], width // group_size)
    if width % group_size or scales.shape != group_shape or biases.shape != group_shape:
        raise ValueError("scale/bias shapes must match the packed rows and quantization groups")

    # 一个 uint32 从低位到高位依次保存八个四位编码。
    shifts = mx.arange(8, dtype=mx.uint32) * 4
    codes = (weight[..., None] >> shifts) & 0xF # (1,3,320,8)
    codes = codes.reshape(*group_shape, group_size).astype(mx.float32)
    # 在每组内广播缩放和偏移，先用 FP32 恢复，最后统一舍入为 BF16。
    values = (
        codes * scales[..., None].astype(mx.float32)
        + biases[..., None].astype(mx.float32)
    )
    return values.reshape(*weight.shape[:-1], width).astype(mx.bfloat16)


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array, #八个权重公用一个int32
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size  # 这个项目里 128个权重共用一组 scale/bias
        self.bits = bits  # 表示每个量化值用多少bit
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(  # 从已经加载好的量化层面提取出量化权重信息
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *batch_dims, hidden_dim = a.shape
    a = a.reshape(-1, hidden_dim)
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)
    a = mx.contiguous(a)
    b = mx.contiguous(b)
    return tiny_llm_ext.quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
    ).reshape(*batch_dims, -1) #reshape 为-1表示自己推断


def quantized_linear(  # 4bit  bf16
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    output = quantized_matmul(
        scales=w.scales,
        biases=w.biases,
        group_size=w.group_size,
        bits=w.bits,
        a=x,
        b=w.weight,
        transpose_b=True,
    )

    if bias is not None:
        output = output + bias

    return output
