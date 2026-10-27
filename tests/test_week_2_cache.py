"""第二周第一天：用小张量验证键值缓存，不加载模型权重。"""

import mlx.core as mx
import pytest

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


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["f32", "bf16"])
def test_capacity_cache_exposes_only_the_logical_prefix(device, dtype):
    with mx.stream(device):
        cache = TinyKvFullCache(capacity=8)
        all_keys = mx.arange(48, dtype=mx.float32).reshape(1, 2, 6, 4).astype(dtype)
        all_values = all_keys + 100
        for start, end in ((0, 3), (3, 4), (4, 6)):
            key, value, offset, mask = cache.update_and_fetch(
                all_keys[:, :, start:end, :],
                all_values[:, :, start:end, :],
                mask="causal",
            )

            assert key.shape == value.shape == (1, 2, end, 4)
            assert key.dtype == value.dtype == dtype
            assert offset == cache.offset == end
            assert mask == "causal"
            assert mx.array_equal(key, all_keys[:, :, :end, :]).item()
            assert mx.array_equal(value, all_values[:, :, :end, :]).item()
            stored_key, stored_value = cache.key_values
            assert stored_key.shape == stored_value.shape == (1, 2, 8, 4)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("initial_length", [0, 2], ids=["empty", "populated"])
def test_capacity_overflow_leaves_the_cache_usable(device, initial_length):
    with mx.stream(device):
        cache = TinyKvFullCache(capacity=3)
        key = mx.arange(6, dtype=mx.float32).reshape(1, 1, 3, 2)
        value = key + 100
        if initial_length:
            cache.update_and_fetch(
                key[:, :, :initial_length, :], value[:, :, :initial_length, :]
            )
            mx.eval(*cache.key_values)
        previous_storage = cache.key_values
        previous_counts = _movement_counts(cache)

        too_many = mx.zeros((1, 1, 4 - initial_length, 2))
        with pytest.raises(ValueError, match="capacity 3 exceeded"):
            cache.update_and_fetch(too_many, too_many)

        assert cache.offset == initial_length
        assert cache.key_values is previous_storage
        assert _movement_counts(cache) == previous_counts
        actual_key, actual_value, offset, _ = cache.update_and_fetch(
            key[:, :, initial_length:, :], value[:, :, initial_length:, :]
        )
        assert offset == 3
        assert mx.array_equal(actual_key, key).item()
        assert mx.array_equal(actual_value, value).item()


@pytest.mark.parametrize("capacity", [-1, True, 1.5])
def test_capacity_rejects_invalid_bounds(capacity):
    with pytest.raises(ValueError, match="non-negative integer"):
        TinyKvFullCache(capacity=capacity)


def test_zero_capacity_accepts_only_empty_updates():
    cache = TinyKvFullCache(capacity=0)
    empty = mx.zeros((1, 2, 0, 4), dtype=mx.bfloat16)
    key, value, offset, _ = cache.update_and_fetch(empty, empty)
    assert key.shape == value.shape == (1, 2, 0, 4)
    assert offset == cache.offset == 0
    with pytest.raises(ValueError, match="capacity 0 exceeded"):
        one_token = mx.ones((1, 2, 1, 4), dtype=mx.bfloat16)
        cache.update_and_fetch(one_token, one_token)
    assert cache.offset == 0


def _movement_counts(cache):
    return (
        cache.logical_copy_bytes,
        cache.physical_growth_copy_bytes,
        cache.slice_write_bytes,
        cache.growth_copy_bytes,
    )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize(
    "dtype,bytes_per_element", [(mx.float32, 4), (mx.bfloat16, 2)], ids=["f32", "bf16"]
)
@pytest.mark.parametrize("capacity", [None, 8], ids=["concat", "bounded"])
def test_cache_movement_counters_follow_the_update_path(
    device, dtype, bytes_per_element, capacity
):
    with mx.stream(device):
        cache = TinyKvFullCache(capacity=capacity)
        assert _movement_counts(cache) == (0, 0, 0, 0)
        token_bytes = 2 * 1 * 2 * 4 * bytes_per_element
        for length, total_length, copied_tokens in ((3, 3, 0), (1, 4, 3), (2, 6, 7)):
            key = mx.zeros((1, 2, length, 4), dtype=dtype)
            value = mx.ones((1, 2, length, 4), dtype=dtype)
            cache.update_and_fetch(key, value)
            if capacity is None:
                copied_bytes = copied_tokens * token_bytes
                assert _movement_counts(cache) == (
                    copied_bytes, copied_bytes, 0, copied_bytes
                )
            else:
                assert _movement_counts(cache) == (0, 0, total_length * token_bytes, 0)
        mx.eval(*cache.key_values)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("capacity", [None, 8], ids=["concat", "bounded"])
def test_rewind_overwrites_the_discarded_suffix(device, capacity):
    with mx.stream(device):
        cache = TinyKvFullCache(capacity=capacity)
        key = mx.arange(56).reshape(1, 2, 7, 4).astype(mx.bfloat16)
        value = key + 128
        cache.update_and_fetch(key[:, :, :6], value[:, :, :6])
        cache.update_and_fetch(key[:, :, 6:], value[:, :, 6:])
        mx.eval(*cache.key_values)
        previous_storage = cache.key_values
        previous_counts = _movement_counts(cache)

        cache.rewind(2)
        assert cache.offset == 5
        retained_key, retained_value = cache._logical_key_values()
        assert retained_key.shape == retained_value.shape == (1, 2, 5, 4)
        assert mx.array_equal(retained_key, key[:, :, :5]).item()
        assert mx.array_equal(retained_value, value[:, :, :5]).item()
        assert _movement_counts(cache) == previous_counts
        if capacity is not None:
            assert cache.key_values is previous_storage

        new_key = mx.full((1, 2, 1, 4), 200, dtype=mx.bfloat16)
        new_value = mx.full((1, 2, 1, 4), 250, dtype=mx.bfloat16)
        actual_key, actual_value, offset, _ = cache.update_and_fetch(new_key, new_value)
        assert offset == 6
        assert actual_key.shape == actual_value.shape == (1, 2, 6, 4)
        assert mx.array_equal(
            actual_key, mx.concat([key[:, :, :5], new_key], axis=2)
        ).item()
        assert mx.array_equal(
            actual_value, mx.concat([value[:, :, :5], new_value], axis=2)
        ).item()


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("capacity", [None, 8], ids=["concat", "bounded"])
@pytest.mark.parametrize("operation", ["reset", "rewind_all"])
def test_reset_and_full_rewind_keep_lifetime_counters(device, capacity, operation):
    with mx.stream(device):
        cache = TinyKvFullCache(capacity=capacity)
        key = mx.arange(12, dtype=mx.float32).reshape(1, 1, 6, 2)
        cache.update_and_fetch(key[:, :, :5], key[:, :, :5])
        cache.update_and_fetch(key[:, :, 5:], key[:, :, 5:])
        mx.eval(*cache.key_values)
        previous_storage = cache.key_values
        previous_counts = _movement_counts(cache)

        if operation == "reset":
            cache.reset()
        else:
            cache.rewind(6)
        assert cache.offset == 0
        assert _movement_counts(cache) == previous_counts
        if capacity is None:
            assert cache.key_values is None
        else:
            assert cache.key_values is previous_storage
            assert cache.key_values[0].shape == (1, 1, 8, 2)
        cache.rewind(0)

        replacement_key = mx.full((1, 1, 1, 2), 42, dtype=mx.float32)
        replacement_value = -replacement_key
        actual_key, actual_value, offset, _ = cache.update_and_fetch(
            replacement_key, replacement_value
        )
        assert offset == 1
        assert actual_key.shape == actual_value.shape == (1, 1, 1, 2)
        assert mx.array_equal(actual_key, replacement_key).item()
        assert mx.array_equal(actual_value, replacement_value).item()
        if capacity is None:
            assert _movement_counts(cache) == previous_counts
        else:
            old_copy, old_growth, old_writes, old_legacy = previous_counts
            assert _movement_counts(cache) == (
                old_copy, old_growth, old_writes + 16, old_legacy
            )


@pytest.mark.parametrize("capacity", [None, 8], ids=["concat", "bounded"])
@pytest.mark.parametrize("length", [-1, 4, True, 1.5])
def test_invalid_rewind_does_not_change_the_cache(capacity, length):
    cache = TinyKvFullCache(capacity=capacity)
    key = mx.ones((1, 1, 3, 2))
    cache.update_and_fetch(key[:, :, :2], key[:, :, :2])
    cache.update_and_fetch(key[:, :, 2:], key[:, :, 2:])
    mx.eval(*cache.key_values)
    previous_storage = cache.key_values
    previous_counts = _movement_counts(cache)

    with pytest.raises(ValueError, match="rewind length"):
        cache.rewind(length)
    assert cache.offset == 3
    assert cache.key_values is previous_storage
    assert _movement_counts(cache) == previous_counts
    actual_key, actual_value = cache._logical_key_values()
    assert mx.array_equal(actual_key, key).item()
    assert mx.array_equal(actual_value, key).item()
