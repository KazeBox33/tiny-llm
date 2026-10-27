from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        # 累计记录代码路径中的数据移动量，不代表硬件实测流量。
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[mx.array, mx.array]:
        """只返回有效历史，不把尚未使用的容量交给注意力。"""
        key, value = self.key_values
        return key[:, :, : self.offset, :], value[:, :, : self.offset, :]

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        assert key.shape==value.shape
        B,H,S,D=key.shape

        if self.uses_capacity:
            end = self.offset + S
            if end > self.capacity:
                raise ValueError(
                    f"KV cache capacity {self.capacity} exceeded by append ending at {end}"
                )

            if self.key_values is None:
                assert self.offset == 0
                self.key_values = (
                    mx.zeros((B, H, self.capacity, D), dtype=key.dtype),
                    mx.zeros((B, H, self.capacity, D), dtype=value.dtype),
                )
            else:
                stored_key, stored_value = self.key_values
                assert stored_key.shape == stored_value.shape == (B, H, self.capacity, D)
                assert stored_key.dtype == key.dtype
                assert stored_value.dtype == value.dtype

            if S:
                stored_key, stored_value = self.key_values
                start = mx.array([self.offset], dtype=mx.int32)
                self.key_values = (
                    mx.slice_update(stored_key, key, start_indices=start, axes=(2,)),
                    mx.slice_update(stored_value, value, start_indices=start, axes=(2,)),
                )
                self.slice_write_bytes += key.nbytes + value.nbytes
            self.offset = end
            key, value = self._logical_key_values()
            return key, value, self.offset, mask

        if self.key_values is None:
            assert self.offset==0
            self.key_values=(key,value)
            self.offset=S
            return key,value,self.offset,mask

        prev_key,prev_value=self.key_values
        assert prev_key.shape==(B,H,self.offset,D)
        assert prev_value.shape==(B,H,self.offset,D)

        copied_bytes = prev_key.nbytes + prev_value.nbytes
        key=mx.concat([prev_key,key],axis=2)
        value=mx.concat([prev_value,value],axis=2)
        self.key_values=(key,value)
        self.offset+=S
        self.logical_copy_bytes += copied_bytes
        self.physical_growth_copy_bytes += copied_bytes
        self.growth_copy_bytes += copied_bytes
        return key,value,self.offset,mask

    def reset(self):
        """清空有效历史；容量模式保留存储，累计计数保持不变。"""
        self.offset = 0
        if not self.uses_capacity:
            self.key_values = None

    def rewind(self, n: int):
        """撤回末尾 n 个词元，让下一次追加从新的有效长度开始。"""
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= self.offset:
            raise ValueError("rewind length must be between zero and the cache length")
        if n == 0:
            return
        self.offset -= n
        if not self.uses_capacity:
            if self.offset == 0:
                self.key_values = None
            else:
                self.key_values = self._logical_key_values()
