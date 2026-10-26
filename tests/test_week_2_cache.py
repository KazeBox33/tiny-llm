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
