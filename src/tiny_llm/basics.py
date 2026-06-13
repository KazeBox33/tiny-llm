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


def silu(x: mx.array) -> mx.array:  # sigmoid性质 sigmoid(-a) = 1 - sigmoid(a)  这里需要实现更加稳定的版本
    y=1/(1+mx.exp(-mx.abs(x)))
    sigmoid=mx.where(x>=0,y,1-y)
    return x*sigmoid
