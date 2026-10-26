"""使用已缓存的 4B 权重，复查曾经暴露激活函数舍入差异的输入。"""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

from tiny_llm.qwen3_week2 import Qwen3ModelWeek2

from .utils import qwen3_4b_model_exists


@pytest.fixture(scope="module")
def models_4b():
    if not qwen3_4b_model_exists():
        pytest.skip("Qwen3-4B-4bit model not found locally")
    reference, tokenizer = load("Qwen/Qwen3-4B-MLX-4bit")
    return Qwen3ModelWeek2(reference), reference, tokenizer


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_qwen3_4b_prefill_matches_reference(models_4b, seed):
    model, reference, tokenizer = models_4b
    mx.random.seed(seed)
    tokens = mx.random.randint(0, tokenizer.vocab_size, (1, 10))
    cache = model.create_kv_cache()
    actual = model(tokens, 0, cache)
    expected = reference(tokens)

    # 沿用课程整段预填充测试的归一化方式及容差。
    actual = actual - mx.logsumexp(actual, keepdims=True)
    expected = expected - mx.logsumexp(expected, keepdims=True)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.1,
        atol=1.0,
    )
    assert all(entry.offset == 10 for entry in cache)
