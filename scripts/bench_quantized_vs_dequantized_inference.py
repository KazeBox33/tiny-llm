import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import mlx.core as mx
from mlx_lm import load

from tiny_llm.attention import scaled_dot_product_attention_grouped
from tiny_llm.basics import linear, silu
from tiny_llm.embedding import Embedding
from tiny_llm.kv_cache import TinyKvFullCache
from tiny_llm.layer_norm import RMSNorm
from tiny_llm.models import dispatch_model, shortcut_name_to_full_name
from tiny_llm.positional_encoding import RoPE
from tiny_llm.quantize import dequantize_linear


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


class DequantizedQwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int,
        theta: int,
        rms_norm_eps: float,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = mx.rsqrt(head_dim)

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.q_norm = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(head_dim, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvFullCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, slice(offset, offset + L))
        k = self.rope(k, slice(offset, offset + L))

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        k, v, _, mask = cache.update_and_fetch(k, v, mask_length=L, mask=mask)

        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return linear(out, self.wo)


class DequantizedQwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        return linear(gate * up, self.w_down)


class DequantizedQwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int,
        theta: int,
    ):
        self.input_layernorm = RMSNorm(
            hidden_size,
            w_input_layernorm,
            eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            eps=rms_norm_eps,
        )
        self.self_attn = DequantizedQwen3MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = DequantizedQwen3MLP(
            dim=hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvFullCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask=mask)
        h = x + r

        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class DequantizedQwen3ModelWeek2:
    """Week2 model with KV cache, but ordinary dequantized linear weights."""

    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size

        self.embedding = Embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )

        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            self.layers_inner.append(
                DequantizedQwen3TransformerBlock(
                    num_attention_heads=mlx_model.args.num_attention_heads,
                    num_kv_heads=mlx_model.args.num_key_value_heads,
                    hidden_size=mlx_model.args.hidden_size,
                    head_dim=mlx_model.args.head_dim,
                    intermediate_size=mlx_model.args.intermediate_size,
                    rms_norm_eps=mlx_model.args.rms_norm_eps,
                    wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                    wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                    wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                    wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                    q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                    k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                    w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                    w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                    w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                    w_input_layernorm=mlx_model.model.layers[
                        i
                    ].input_layernorm.weight,
                    w_post_attention_layernorm=mlx_model.model.layers[
                        i
                    ].post_attention_layernorm.weight,
                    max_seq_len=mlx_model.args.max_position_embeddings,
                    theta=mlx_model.args.rope_theta,
                )
            )

        self.norm = RMSNorm(
            self.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )

        if mlx_model.args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvFullCache],
    ) -> mx.array:
        h = self.embedding(inputs)
        mask = "causal" if inputs.shape[-1] > 1 else None

        for layer, layer_cache in zip(self.layers_inner, cache):
            h = layer(h, offset, layer_cache, mask=mask)

        h = self.norm(h)

        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)

        return self.embedding.as_linear(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full inference with dequantized weights vs packed 4-bit quantized weights."
    )
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--num-seqs", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def run_command(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": run_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "machine": platform.machine(),
    }


def random_token_id(rng: Random, low: int, high: int, eos_token_id: int) -> int:
    token = rng.randint(low, high)
    if token != eos_token_id:
        return token
    if token == low:
        return low + 1
    return token - 1


def build_requests(
    *,
    rng: Random,
    num_seqs: int,
    vocab_size: int,
    eos_token_id: int,
    input_len: int,
    output_len: int,
) -> list[BenchRequest]:
    token_low = 256 if vocab_size > 512 else 0
    token_high = vocab_size - 1
    return [
        BenchRequest(
            prompt_token_ids=[
                random_token_id(rng, token_low, token_high, eos_token_id)
                for _ in range(input_len)
            ],
            max_new_tokens=output_len,
        )
        for _ in range(num_seqs)
    ]


def sample_next(model: Any, y: mx.array, offset: int, kv_cache: list) -> mx.array:
    logits = model(y[None, :], offset, kv_cache)[:, -1, :]
    return mx.argmax(logits, axis=-1)


def run_one_request(model: Any, request: BenchRequest) -> tuple[int, float, float]:
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    offset = 0

    t0 = perf_counter()
    token = sample_next(model, context, offset, kv_cache)
    mx.eval(token)
    prefill_time = perf_counter() - t0
    offset += context.size

    generated_tokens = 1
    decode_time = 0.0

    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        token = sample_next(model, token, offset, kv_cache)
        mx.eval(token)
        decode_time += perf_counter() - t1
        offset += 1
        generated_tokens += 1

    return generated_tokens, prefill_time, decode_time


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def benchmark_model(
    *,
    name: str,
    model: Any,
    requests: list[BenchRequest],
    warmup: int,
) -> dict[str, float | int | str]:
    for i in range(warmup):
        run_one_request(model, requests[i % len(requests)])

    total_prompt_tokens = sum(len(request.prompt_token_ids) for request in requests)
    total_generated_tokens = 0
    total_decode_tokens = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0

    t0 = perf_counter()
    for request in requests:
        generated_tokens, prefill_time, decode_time = run_one_request(model, request)
        total_generated_tokens += generated_tokens
        total_decode_tokens += max(0, generated_tokens - 1)
        total_prefill_time += prefill_time
        total_decode_time += decode_time
    total_time = perf_counter() - t0

    total_model_tokens = total_prompt_tokens + total_generated_tokens
    return {
        "name": name,
        "requests": len(requests),
        "prompt_tokens": total_prompt_tokens,
        "generated_tokens": total_generated_tokens,
        "time_s": total_time,
        "output_tok_s": safe_div(total_generated_tokens, total_time),
        "total_tok_s": safe_div(total_model_tokens, total_time),
        "prefill_tok_s": safe_div(total_prompt_tokens, total_prefill_time),
        "decode_tok_s": safe_div(total_decode_tokens, total_decode_time),
    }


def main() -> None:
    args = parse_args()
    model_name = shortcut_name_to_full_name(args.model)
    mlx_model, tokenizer = load(model_name)

    with mx.stream(mx.gpu):
        dequantized_model = DequantizedQwen3ModelWeek2(mlx_model)
        quantized_model = dispatch_model(model_name, mlx_model, week=2)

        results = {
            "benchmark": "full_inference_dequantized_vs_quantized",
            "model": model_name,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "warmup": args.warmup,
            "seed": args.seed,
            "system": system_info(),
            "results": [],
        }

        for num_seqs in args.num_seqs:
            rng = Random(args.seed)
            eos_token_id = tokenizer.eos_token_id or 0
            requests = build_requests(
                rng=rng,
                num_seqs=num_seqs,
                vocab_size=mlx_model.args.vocab_size,
                eos_token_id=eos_token_id,
                input_len=args.input_len,
                output_len=args.output_len,
            )
            dequantized = benchmark_model(
                name="dequantized_week2_linear",
                model=dequantized_model,
                requests=requests,
                warmup=args.warmup,
            )
            quantized = benchmark_model(
                name="packed_int4_quantized_linear",
                model=quantized_model,
                requests=requests,
                warmup=args.warmup,
            )
            speedup = quantized["output_tok_s"] / dequantized["output_tok_s"]

            row = {
                "num_seqs": num_seqs,
                "dequantized": dequantized,
                "quantized": quantized,
                "quantized_vs_dequantized_output_speedup": speedup,
            }
            results["results"].append(row)

            print(
                f"num_seqs={num_seqs:<3d} "
                f"dequantized={dequantized['output_tok_s']:.2f} tok/s "
                f"quantized={quantized['output_tok_s']:.2f} tok/s "
                f"speedup={speedup:.2f}x "
                f"decode_speedup={quantized['decode_tok_s'] / dequantized['decode_tok_s']:.2f}x"
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
