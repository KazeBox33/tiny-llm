import mlx.core as mx
from typing import Any


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
        scales: mx.array,
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
    pass


def quantized_linear(
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
