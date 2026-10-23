"""第一周推理边界的回归测试，不依赖下载的模型权重。"""

import mlx.core as mx
import numpy as np
import pytest

from tiny_llm import make_sampler, scaled_dot_product_attention_grouped, silu
from tiny_llm import simple_generate


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["f16", "bf16"])
def test_silu_preserves_small_negative_values(device, dtype):
    values = np.array([-9.0, -7.0, -5.0, 0.0, 5.0, 9.0, 1000.0])
    with mx.stream(device):
        output = silu(mx.array(values).astype(dtype))
        actual = np.array(output.astype(mx.float32))
    expected = values / (1 + np.exp(-values))
    assert output.dtype == dtype
    assert np.all(actual[:3] < 0)
    np.testing.assert_allclose(actual, expected, rtol=0.03, atol=1e-6)


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu], ids=["cpu", "gpu"])
def test_causal_attention_rejects_queries_longer_than_history(device):
    with mx.stream(device):
        query = mx.ones((1, 2, 3, 4))
        key = mx.ones((1, 1, 2, 4))
        with pytest.raises(ValueError, match="S >= L"):
            scaled_dot_product_attention_grouped(query, key, key, mask="causal")


class BufferedDetokenizer:
    def reset(self):
        self.last_segment = ""
        self.pending = ""

    def add_token(self, token):
        self.last_segment = self.pending
        self.pending = {2: "A", 3: "B"}[token]

    def finalize(self):
        self.last_segment = self.pending
        self.pending = ""


class Tokenizer:
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
        return BufferedDetokenizer()


class ScriptedModel:
    def __init__(self, next_tokens):
        self.next_tokens = iter(next_tokens)
        self.inputs = []

    def __call__(self, tokens):
        self.inputs.append(tokens.tolist())
        logits = np.full((1, tokens.shape[1], 8), -1000.0)
        logits[:, -1, next(self.next_tokens)] = 0
        return mx.array(logits)


def test_generation_keeps_prompt_eos_and_flushes_one_detokenizer(capsys):
    tokenizer = Tokenizer([5, 0])
    model = ScriptedModel([2, 3, 0])
    assert simple_generate(model, tokenizer, "prompt", sampler=None) is None
    assert capsys.readouterr().out == "AB"
    assert tokenizer.detokenizer_reads == 1
    assert model.inputs == [[[5, 0]], [[5, 0, 2]], [[5, 0, 2, 3]]]


@pytest.mark.parametrize("limit", [0, 1, 3, 256])
def test_generation_stops_at_requested_limit(capsys, limit):
    tokenizer = Tokenizer([5])
    model = ScriptedModel([2] * (limit + 1))
    simple_generate(model, tokenizer, "prompt", sampler=None, max_tokens=limit)
    assert capsys.readouterr().out == "A" * limit
    assert len(model.inputs) == limit


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_generation_rejects_invalid_limits_before_model_execution(limit):
    model = ScriptedModel([2])
    with pytest.raises(ValueError, match="non-negative integer"):
        simple_generate(model, Tokenizer([5]), "prompt", None, max_tokens=limit)
    assert not model.inputs


def test_generation_rejects_empty_prompt_before_model_execution():
    model = ScriptedModel([2])
    with pytest.raises(ValueError, match="at least one token"):
        simple_generate(model, Tokenizer([]), "prompt", None)
    assert not model.inputs


@pytest.mark.parametrize(
    "probabilities,top_p,top_k,expected",
    [
        ([0.5, 0.3, 0.15, 0.05], 0.6, None, [0.625, 0.375, 0, 0]),
        ([0.1, 0.2, 0.3, 0.4], 1.0, 4, [0.1, 0.2, 0.3, 0.4]),
        ([0.55, 0.25, 0.15, 0.05], 0.7, 3, [0.6875, 0.3125, 0, 0]),
    ],
)
def test_sampling_keeps_threshold_crossing_token_without_mutating_input(
    probabilities, top_p, top_k, expected
):
    logprobs = mx.log(mx.array([probabilities]))
    original = np.array(logprobs)
    sampler = make_sampler(1.0, top_p, top_k)
    mx.random.seed(2026)
    draws = [int(sampler(logprobs).item()) for _ in range(384)]
    frequencies = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_array_equal(frequencies[np.array(expected) == 0], 0)
    np.testing.assert_allclose(frequencies, expected, rtol=0, atol=0.08)
    np.testing.assert_array_equal(np.array(logprobs), original)
