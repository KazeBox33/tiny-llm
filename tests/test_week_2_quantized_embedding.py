"""Week 2 Day 2 Task 1：四位解包、分组恢复及只解量化选中行。"""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm.embedding import QuantizedEmbedding
from tiny_llm.quantize import QuantizedWeights, dequantize_weights


@pytest.fixture
def packed_fixture():
    # 手工构造编码和带正负符号的两组缩放值，不依赖量化算法生成期望值。
    codes = (np.arange(256)[None, :] + np.arange(6)[:, None]) % 16
    scales = np.array([
        [0.125, -0.25], [-0.5, 0.25], [1, -1],
        [-0.125, 0.5], [0.25, -0.125], [-0.25, 1],
    ], dtype=np.float32)
    biases = np.array([
        [0.5, 1], [-1, 2], [0, 4],
        [0.25, -2], [1, 0.5], [-0.5, 0],
    ], dtype=np.float32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = np.bitwise_or.reduce(
        codes.astype(np.uint32).reshape(6, 32, 8) << shifts, axis=-1
    )
    expected = (
        codes.reshape(6, 2, 128) * scales[..., None] + biases[..., None]
    ).reshape(6, 256).astype(np.float32)
    weights = QuantizedWeights(
        mx.array(scales, dtype=mx.bfloat16), mx.array(biases, dtype=mx.bfloat16),
        128, 4, mx.array(packed, dtype=mx.uint32),
    )
    return weights, expected


def test_manual_unpack_preserves_code_order_groups_and_signed_scales(packed_fixture):
    weights, expected = packed_fixture
    with patch.object(mx, "dequantize", side_effect=AssertionError("use explicit unpacking")):
        result = dequantize_weights(
            weights.weight, weights.scales, weights.biases, 128, 4
        )
        mx.eval(result)
    assert result.shape == (6, 256)
    assert result.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(result.astype(mx.float32)), expected)


@pytest.mark.parametrize("seed", [0, 2, 7])
def test_manual_dequantization_matches_mlx_rounding(seed):
    mx.random.seed(seed)
    original = mx.random.normal((7, 256)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(original, group_size=128, bits=4)
    expected = mx.dequantize(packed, scales, biases, group_size=128, bits=4)
    with patch.object(mx, "dequantize", side_effect=AssertionError("use explicit unpacking")):
        result = dequantize_weights(packed, scales, biases, 128, 4)
        mx.eval(result, expected)
    np.testing.assert_array_equal(
        np.array(result.astype(mx.float32)), np.array(expected.astype(mx.float32))
    )


@pytest.mark.parametrize(
    "ids,dtype",
    [([[5, 1, 5]], mx.int32), ([[2]], mx.uint32), ([[5, 0], [2, 5]], mx.int32)],
    ids=["prefill-with-repeat", "decode-uint32", "batch"],
)
def test_embedding_only_dequantizes_selected_rows(packed_fixture, ids, dtype):
    weights, dense = packed_fixture
    embedding = QuantizedEmbedding(6, 256, weights)
    tokens = mx.array(ids, dtype=dtype)
    with (
        patch("tiny_llm.embedding.dequantize_weights", wraps=dequantize_weights) as restore,
        patch.object(mx, "dequantize", side_effect=AssertionError("use explicit unpacking")),
    ):
        result = embedding(tokens)
        mx.eval(result)
    restore.assert_called_once()
    packed, scales, biases, group, bits = restore.call_args.args
    assert packed.shape == (*tokens.shape, 32)
    assert scales.shape == biases.shape == (*tokens.shape, 2)
    assert (group, bits) == (128, 4)
    assert result.shape == (*tokens.shape, 256)
    assert result.dtype == mx.bfloat16
    assert weights.weight.shape == (6, 32) and weights.weight.dtype == mx.uint32
    np.testing.assert_array_equal(np.array(result.astype(mx.float32)), dense[np.array(ids)])
