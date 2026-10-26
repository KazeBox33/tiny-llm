import mlx.core as mx
import math
from functools import partial


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


@partial(mx.compile, shapeless=True)
def silu(x: mx.array) -> mx.array:
    # 与参考模型使用同一原语，保留当前设备在模型精度下的舍入顺序。
    return x * mx.sigmoid(x)
