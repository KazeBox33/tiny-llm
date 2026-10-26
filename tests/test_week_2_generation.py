"""验证提示词只输入一次、逐词元解码和请求结束时的文本输出。"""

import mlx.core as mx
import pytest

from tiny_llm.generate import simple_generate_with_kv_cache
from tiny_llm.kv_cache import TinyKvFullCache


class BufferedTokenizer:
    eos_token_id = 0

    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.detokenizer_reads = 0

    def encode(self, prompt, add_special_tokens=True):
        assert add_special_tokens is False
        return self.prompt_tokens

    @property
    def detokenizer(self):
        self.detokenizer_reads += 1
        return self

    def reset(self):
        self.last_segment = ""
        self.pending = ""

    def add_token(self, token):
        self.last_segment = self.pending
        self.pending = {2: "A", 3: "B"}[token]

    def finalize(self):
        self.last_segment = self.pending
        self.pending = ""


class ScriptedCachedModel:
    def __init__(self, next_tokens):
        self.next_tokens = iter(next_tokens)
        self.calls = []
        self.cache_creations = 0

    def create_kv_cache(self):
        self.cache_creations += 1
        self.cache = [TinyKvFullCache(), TinyKvFullCache()]
        return self.cache

    def __call__(self, tokens, offset, cache):
        assert cache is self.cache
        assert [entry.offset for entry in cache] == [offset, offset]
        self.calls.append((tokens.tolist(), offset))
        key = mx.zeros((1, 1, tokens.shape[1], 2), dtype=mx.bfloat16)
        for entry in cache:
            entry.update_and_fetch(key, key)
        logits = mx.full((1, tokens.shape[1], 8), -1000.0)
        logits[:, -1, next(self.next_tokens)] = 0
        return logits


def test_generation_prefills_once_and_decodes_single_tokens(capsys):
    model = ScriptedCachedModel([2, 3, 0])
    tokenizer = BufferedTokenizer([1, 2, 3, 4, 5, 6])
    assert simple_generate_with_kv_cache(model, tokenizer, "prompt") is None
    assert model.calls == [([list(range(1, 7))], 0), ([[2]], 6), ([[3]], 7)]
    assert capsys.readouterr().out == "AB"
    assert model.cache_creations == tokenizer.detokenizer_reads == 1
    assert [entry.offset for entry in model.cache] == [8, 8]


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_cached_generation_stops_at_output_limit(capsys, limit):
    model = ScriptedCachedModel([2] * (limit + 1))
    tokenizer = BufferedTokenizer([5, 0])
    simple_generate_with_kv_cache(model, tokenizer, "prompt", max_tokens=limit)
    assert capsys.readouterr().out == "A" * limit
    assert len(model.calls) == limit
    assert model.cache_creations == (1 if limit else 0)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_output_limit_is_rejected_before_cache_creation(limit):
    model = ScriptedCachedModel([2])
    with pytest.raises(ValueError, match="non-negative integer"):
        simple_generate_with_kv_cache(
            model, BufferedTokenizer([5]), "prompt", max_tokens=limit
        )
    assert model.cache_creations == 0


def test_empty_prompt_is_rejected_before_cache_creation():
    model = ScriptedCachedModel([2])
    with pytest.raises(ValueError, match="at least one token"):
        simple_generate_with_kv_cache(model, BufferedTokenizer([]), "prompt")
    assert model.cache_creations == 0


def test_generation_creates_fresh_caches_for_each_request(capsys):
    model = ScriptedCachedModel([2, 0, 3, 0])
    tokenizer = BufferedTokenizer([5, 6])
    simple_generate_with_kv_cache(model, tokenizer, "first")
    simple_generate_with_kv_cache(model, tokenizer, "second")
    assert model.calls == [([[5, 6]], 0), ([[2]], 2), ([[5, 6]], 0), ([[3]], 2)]
    assert model.cache_creations == tokenizer.detokenizer_reads == 2
    assert capsys.readouterr().out == "AB"
