"""用两层微型 Qwen3 验证缓存、位置编码和模型输出的完整连接。"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.qwen3 import Model, ModelArgs

from tiny_llm.qwen3_week2 import Qwen3ModelWeek2


@pytest.fixture
def models():
    mx.random.seed(7)
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=32,
        max_position_embeddings=64,
        rope_theta=1000000,
        tie_word_embeddings=True,
    )
    reference = Model(args)
    reference.set_dtype(mx.bfloat16)
    nn.quantize(reference, group_size=128, bits=4)
    mx.eval(reference.parameters())
    return Qwen3ModelWeek2(reference), reference


def test_cache_factory_isolates_layers_and_requests(models):
    model, _ = models
    first = model.create_kv_cache()
    second = model.create_kv_cache()
    assert len(first) == len(second) == 2
    assert len({id(cache) for cache in first + second}) == 4

    key = mx.zeros((1, 2, 3, 32), dtype=mx.bfloat16)
    first[0].update_and_fetch(key, key)
    assert [cache.offset for cache in first] == [3, 0]
    assert [cache.offset for cache in second] == [0, 0]


@pytest.mark.parametrize("chunk_lengths", [(6,), (1, 1, 1, 1, 1, 1), (3, 1, 2)])
def test_cached_model_matches_full_sequence(models, chunk_lengths):
    model, reference = models
    tokens = mx.array([[1, 4, 2, 5, 3, 6]], dtype=mx.int32)
    expected = reference(tokens)
    cache = model.create_kv_cache()
    outputs = []
    offset = 0
    for length in chunk_lengths:
        outputs.append(model(tokens[:, offset : offset + length], offset, cache))
        offset += length
        assert [entry.offset for entry in cache] == [offset, offset]

    actual = mx.concatenate(outputs, axis=1)
    assert actual.shape == expected.shape == (1, 6, 32)
    assert actual.dtype == mx.bfloat16
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        rtol=0.05,
        atol=0.05,
    )


def test_wrong_offset_is_rejected_before_any_layer_changes(models):
    model, _ = models
    cache = model.create_kv_cache()
    key = mx.zeros((1, 2, 1, 32), dtype=mx.bfloat16)
    cache[1].update_and_fetch(key, key)
    with pytest.raises(ValueError, match="layer 1 cache offset"):
        model(mx.array([[1]], dtype=mx.int32), 0, cache)
    assert cache[0].key_values is None
    assert [entry.offset for entry in cache] == [0, 1]


def test_incomplete_cache_list_is_rejected(models):
    model, _ = models
    with pytest.raises(ValueError, match="expected 2 per-layer caches"):
        model(mx.array([[1]], dtype=mx.int32), 0, model.create_kv_cache()[:1])
