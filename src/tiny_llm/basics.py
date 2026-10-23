import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    output=x@w.T
    if bias is not None:
        output =output+bias
    return output


def silu(x: mx.array) -> mx.array:
    # 负数分支直接计算小概率，避免低精度下用两个接近 1 的数相减。
    z = mx.exp(-mx.abs(x))
    sigmoid = mx.where(x < 0, z / (1 + z), 1 / (1 + z))
    return x * sigmoid
