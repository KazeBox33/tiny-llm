"""First cooperative W4 matvec: two output columns per SIMD group."""

import mlx.core as mx
import numpy as np
import pytest

from extensions import tiny_llm_ext
from tiny_llm.quantize import quantized_matmul_vanilla


def packed_case(rows, width, outputs, dtype):
    codes = (np.arange(outputs * width).reshape(outputs, width) * 7 + 3) % 16
    groups = width // 128
    indices = np.arange(outputs * groups).reshape(outputs, groups)
    scales = np.where(indices % 2, -0.125, 0.25).astype(np.float32)
    biases = ((indices % 5 - 2) * 0.5).astype(np.float32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    packed = np.bitwise_or.reduce(
        codes.astype(np.uint32).reshape(outputs, width // 8, 8) << shifts,
        axis=-1,
    )
    activation = mx.sin(mx.arange(rows * width, dtype=mx.float32)).reshape(rows, width)
    activation = activation.astype(dtype)
    restored = (
        codes.reshape(outputs, groups, 128) * scales[..., None] + biases[..., None]
    ).reshape(outputs, width).astype(np.float32)
    expected = np.einsum(
        "mn,kn->mk", np.array(activation.astype(mx.float32)), restored,
        dtype=np.float32, optimize=False,
    )
    assert np.isfinite(expected).all()
    arguments = dict(
        scales=mx.array(scales, dtype=dtype), biases=mx.array(biases, dtype=dtype),
        group_size=128, bits=4, a=activation,
        b=mx.array(packed, dtype=mx.uint32), transpose_b=True,
    )
    return arguments, mx.array(expected, dtype=dtype)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize(
    "rows,width,outputs",
    [(1, 128, 1), (1, 256, 3), (2, 384, 5), (8, 1024, 9), (9, 256, 5), (128, 256, 5)],
    ids=["idle-lanes", "odd-output", "packed-tail", "matvec-limit", "matrix-cutoff", "prefill"],
)
def test_gpu_simd_and_vanilla_match_numpy(rows, width, outputs, dtype):
    arguments, expected = packed_case(rows, width, outputs, dtype)
    with mx.stream(mx.gpu):
        actual = tiny_llm_ext.quantized_matmul(**arguments)
        control = tiny_llm_ext.quantized_matmul(**arguments, use_simd=False)
        mx.eval(actual, control, expected)
    for result in (actual, control):
        assert result.shape == (rows, outputs)
        assert result.dtype == dtype
        assert mx.all(mx.isfinite(result)).item()
        np.testing.assert_allclose(
            np.array(result.astype(mx.float32)), np.array(expected.astype(mx.float32)),
            rtol=2e-2, atol=2e-2,
        )


def test_python_vanilla_preserves_leading_dimensions():
    arguments, expected = packed_case(6, 256, 5, mx.bfloat16)
    arguments["a"] = arguments["a"].reshape(2, 3, 256)
    with mx.stream(mx.gpu):
        actual = quantized_matmul_vanilla(**arguments)
        mx.eval(actual)
    assert actual.shape == (2, 3, 5)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.reshape(2, 3, 5).astype(mx.float32)),
        rtol=2e-2, atol=2e-2,
    )


def test_cpu_path_is_preserved():
    arguments, expected = packed_case(1, 256, 5, mx.bfloat16)
    actual = tiny_llm_ext.quantized_matmul(**arguments, stream=mx.cpu)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32)),
        rtol=2e-2, atol=2e-2,
    )
