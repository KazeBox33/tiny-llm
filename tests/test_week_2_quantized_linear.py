"""Week 2 Day 2 Task 2：原生量化算子的输入约定及压缩线性层接线。"""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from extensions import tiny_llm_ext
from tiny_llm.embedding import QuantizedEmbedding
from tiny_llm.quantize import QuantizedWeights, quantized_linear


@pytest.fixture
def linear_fixture():
    # 两个量化组、五个输出通道，覆盖分组索引和非整块输出。
    dense = mx.sin(mx.arange(5 * 256, dtype=mx.float32)).reshape(5, 256)
    packed, scales, biases = mx.quantize(
        dense.astype(mx.bfloat16), group_size=128, bits=4
    )
    weights = QuantizedWeights(scales, biases, 128, 4, packed)
    activation = mx.cos(mx.arange(256, dtype=mx.float32)).reshape(1, 256)
    return activation.astype(mx.bfloat16), weights


def reference_linear(activation, weights):
    """小尺寸测试用 NumPy 解包并以 FP32 点积，独立于两种 GPU 实现。"""
    packed = np.array(weights.weight)
    rows, packed_width = packed.shape
    shifts = np.arange(8, dtype=np.uint32) * 4
    codes = ((packed[..., None] >> shifts) & 15).reshape(rows, -1, 128)
    restored = (
        codes.astype(np.float32) * np.array(weights.scales.astype(mx.float32))[..., None]
        + np.array(weights.biases.astype(mx.float32))[..., None]
    ).reshape(rows, packed_width * 8)
    output = np.array(activation.astype(mx.float32)) @ restored.T
    return mx.array(output, dtype=activation.dtype)


def test_native_rejects_mixed_metadata_dtype_before_evaluation(linear_fixture):
    activation, weights = linear_fixture
    # 只创建计算图，不执行非法 GPU 输入；错误应在原生入口处报告。
    with pytest.raises(RuntimeError, match="same dtype"):
        tiny_llm_ext.quantized_matmul(
            weights.scales.astype(mx.float32), weights.biases, 128, 4,
            activation, weights.weight, True,
        )


@pytest.mark.parametrize(
    "argument,change,message",
    [
        ("a", lambda x: x[0], "rank-2"),
        ("scales", lambda x: x.reshape(-1), "rank-2"),
        ("a", lambda x: x.astype(mx.float32), "float16 or bfloat16"),
        ("b", lambda x: x.astype(mx.int32), "uint32"),
        ("a", lambda x: x[:, :128], "packed b must have shape"),
        ("a", lambda x: x[:, :255], "divisible by group_size"),
        ("scales", lambda x: x[:, :1], "scales and biases must have shape"),
        ("biases", lambda x: x[:4], "scales and biases must have shape"),
        ("biases", lambda x: x.astype(mx.float32), "same dtype"),
    ],
    ids=[
        "activation-rank", "metadata-rank", "activation-dtype", "packed-dtype",
        "packed-width", "partial-group", "group-count", "metadata-rows", "bias-dtype",
    ],
)
def test_native_rejects_invalid_input_before_evaluation(
    linear_fixture, argument, change, message
):
    activation, weights = linear_fixture
    arguments = dict(
        scales=weights.scales, biases=weights.biases, group_size=128, bits=4,
        a=activation, b=weights.weight, transpose_b=True,
    )
    arguments[argument] = change(arguments[argument])
    with pytest.raises(RuntimeError, match=message):
        tiny_llm_ext.quantized_matmul(**arguments)


@pytest.mark.parametrize("shape", [(1, 256), (1, 3, 256), (2, 3, 256)])
def test_quantized_linear_preserves_leading_dimensions_and_bias(linear_fixture, shape):
    _, weights = linear_fixture
    activation = mx.cos(mx.arange(256, dtype=mx.float32)).astype(mx.bfloat16)
    activation = mx.broadcast_to(activation, shape)
    bias = mx.arange(5, dtype=mx.bfloat16)
    expected = reference_linear(activation, weights) + bias
    actual = quantized_linear(activation, weights, bias)
    mx.eval(actual, expected)
    assert actual.shape == (*shape[:-1], 5)
    assert actual.dtype == mx.bfloat16
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


def test_tied_vocabulary_projection_keeps_weights_packed(linear_fixture):
    activation, weights = linear_fixture
    embedding = QuantizedEmbedding(5, 256, weights)
    expected = reference_linear(activation, weights)
    with (
        patch("tiny_llm.embedding.dequantize_weights", side_effect=AssertionError("keep W4")),
        patch.object(mx, "dequantize", side_effect=AssertionError("keep W4")),
        patch.object(tiny_llm_ext, "quantized_matmul", wraps=tiny_llm_ext.quantized_matmul) as native,
    ):
        actual = embedding.as_linear(activation)
        mx.eval(actual, expected)
    native.assert_called_once()
    packed = native.call_args.args[5]
    assert packed.shape == (5, 32) and packed.dtype == mx.uint32
    assert mx.array_equal(packed, weights.weight).item()
    assert actual.shape == (1, 5)
    assert actual.dtype == mx.bfloat16
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()


@pytest.mark.parametrize("metadata", ["scales", "biases"])
def test_gpu_rejects_strided_metadata_and_wrapper_makes_it_contiguous(linear_fixture, metadata):
    activation, weights = linear_fixture
    original = getattr(weights, metadata)
    # 保留原值和形状，但令相邻量化组在物理存储中隔一个元素。
    strided = mx.stack([original, original], axis=-1).reshape(5, 4)[:, ::2]
    mx.eval(strided)
    arguments = dict(
        scales=weights.scales, biases=weights.biases, group_size=128, bits=4,
        a=activation, b=weights.weight, transpose_b=True, stream=mx.gpu,
    )
    arguments[metadata] = strided
    with pytest.raises(RuntimeError, match="scales and biases must be contiguous"):
        mx.eval(tiny_llm_ext.quantized_matmul(**arguments))

    expected = quantized_linear(activation, weights)
    view_weights = QuantizedWeights(
        arguments["scales"], arguments["biases"], 128, 4, weights.weight
    )
    actual = quantized_linear(activation, view_weights)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_native_preserves_float16_support(linear_fixture):
    activation, weights = linear_fixture
    scales, biases = weights.scales.astype(mx.float16), weights.biases.astype(mx.float16)
    activation = activation.astype(mx.float16)
    actual = tiny_llm_ext.quantized_matmul(
        scales, biases, 128, 4, activation, weights.weight, True, stream=mx.gpu
    )
    expected = reference_linear(
        activation, QuantizedWeights(scales, biases, 128, 4, weights.weight)
    )
    mx.eval(actual, expected)
    assert actual.dtype == mx.float16
    assert mx.allclose(actual, expected, rtol=2e-2, atol=2e-2).item()
