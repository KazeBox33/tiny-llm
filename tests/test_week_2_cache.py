"""第二周第一天：用小张量验证键值缓存，不加载模型权重。"""

import mlx.core as mx

from tiny_llm.kv_cache import TinyKvFullCache


def test_kv_cache_first_write():
    cache = TinyKvFullCache()
    assert cache.key_values is None
    assert cache.offset == 0

    key = mx.zeros((1, 2, 6, 4), dtype=mx.bfloat16)
    value = mx.ones((1, 2, 6, 4), dtype=mx.bfloat16)

    k, v, offset, mask = cache.update_and_fetch(key, value)

    assert k.shape == v.shape == (1, 2, 6, 4)
    assert k.dtype == v.dtype == mx.bfloat16
    assert offset == cache.offset == 6
    assert mask is None
    assert mx.array_equal(k, key).item()
    assert mx.array_equal(v, value).item()

    cached_key, cached_value = cache.key_values
    assert mx.array_equal(cached_key, key).item()
    assert mx.array_equal(cached_value, value).item()


def test_kv_cache_append_preserves_history():
    cache = TinyKvFullCache()
    key = mx.zeros((1, 2, 6, 4), dtype=mx.bfloat16)
    value = mx.ones((1, 2, 6, 4), dtype=mx.bfloat16)
    cache.update_and_fetch(key, value)

    for new_length in (1, 2):
        previous_key, previous_value = cache.key_values
        previous_length = cache.offset
        new_key = mx.full((1, 2, new_length, 4), 2, dtype=mx.bfloat16)
        new_value = mx.full((1, 2, new_length, 4), 3, dtype=mx.bfloat16)

        k, v, offset, mask = cache.update_and_fetch(new_key, new_value)

        assert k.shape == v.shape == (1, 2, previous_length + new_length, 4)
        assert k.dtype == v.dtype == mx.bfloat16
        assert offset == cache.offset == previous_length + new_length
        assert mask is None
        assert mx.array_equal(k[:, :, :previous_length, :], previous_key).item()
        assert mx.array_equal(v[:, :, :previous_length, :], previous_value).item()
        assert mx.array_equal(k[:, :, previous_length:, :], new_key).item()
        assert mx.array_equal(v[:, :, previous_length:, :], new_value).item()

        cached_key, cached_value = cache.key_values
        assert mx.array_equal(cached_key, k).item()
        assert mx.array_equal(cached_value, v).item()
