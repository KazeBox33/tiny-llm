# Quantized Linear Benchmark - Qwen3 0.6B

Date: 2026-06-20

## Goal

Measure the performance impact of the Week 2 quantized inference path:

- Baseline: dequantize the 4-bit weight once, then run ordinary `linear(x, w)`.
- Quantized path: keep packed 4-bit weights and run `quantized_linear(x, w)`, which dispatches the custom C++ / Metal `quantized_matmul` primitive.

This benchmark isolates the linear layer path. It does not include tokenizer overhead.

## Environment

Raw system metadata is recorded in:

```text
benchmarks/quantized_linear_qwen3_0_6b_metal.json
benchmarks/quantized_vs_dequantized_inference_qwen3_0_6b.json
```

Observed environment:

```text
Platform: macOS-26.5.1-arm64-arm-64bit
CPU: Apple M5
Device: MLX GPU / Metal
Model: Qwen/Qwen3-0.6B-MLX-4bit
```

## Reproduction

Build extensions:

```bash
DEBUG=0 pdm run build-ext
```

Run isolated linear benchmark:

```bash
pdm run python scripts/bench_quantized_linear.py \
  --model qwen3-0.6b \
  --layers q_proj up_proj down_proj lm_head \
  --rows 1 16 128 \
  --warmup 10 \
  --iters 50 \
  --seed 0 \
  --output-json benchmarks/quantized_linear_qwen3_0_6b_metal.json
```

Run end-to-end Week 2 throughput benchmark:

```bash
pdm bench --solution tiny_llm --loader week2 --model qwen3-0.6b \
  --num-seqs 4 \
  --min-input-len 64 --max-input-len 64 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 \
  --seed 0
```

Run full inference quantized-vs-dequantized comparison:

```bash
pdm run python scripts/bench_quantized_vs_dequantized_inference.py \
  --model qwen3-0.6b \
  --num-seqs 1 4 8 \
  --input-len 64 \
  --output-len 32 \
  --warmup 1 \
  --seed 0 \
  --output-json benchmarks/quantized_vs_dequantized_inference_qwen3_0_6b.json
```

This comparison keeps the Week 2 KV-cache inference path fixed. The only intended difference is:

- Dequantized path: fully dequantized weights + ordinary `linear`.
- Quantized path: packed int4 weights + custom `quantized_linear`.

Reference implementation comparison:

```bash
DEBUG=0 pdm run build-ext-ref

pdm bench --solution tiny_llm_ref --loader week2 --model qwen3-0.6b \
  --num-seqs 4 \
  --min-input-len 64 --max-input-len 64 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 \
  --seed 0
```

Additional pressure runs varied `num_seqs` while keeping input/output lengths fixed:

```bash
for solution in tiny_llm tiny_llm_ref; do
  for n in 1 8; do
    pdm bench --solution "$solution" --loader week2 --model qwen3-0.6b \
      --num-seqs "$n" \
      --min-input-len 64 --max-input-len 64 \
      --min-output-len 32 --max-output-len 32 \
      --warmup 1 \
      --seed 0
  done
done
```

## Isolated Linear Results

Median latency is reported in milliseconds. Speedup is:

```text
baseline_median_ms / quantized_median_ms
```

So `> 1.0x` means the custom quantized path is faster.

| Layer | Source | Rows M | Baseline median ms | Quantized median ms | Speedup | Max abs diff |
|---|---|---:|---:|---:|---:|---:|
| q_proj | layers.0.self_attn.q_proj | 1 | 0.227 | 0.219 | 1.04x | 0.0156 |
| q_proj | layers.0.self_attn.q_proj | 16 | 0.281 | 0.285 | 0.99x | 0.0312 |
| q_proj | layers.0.self_attn.q_proj | 128 | 0.345 | 0.796 | 0.43x | 0.0625 |
| up_proj | layers.0.mlp.up_proj | 1 | 0.246 | 0.226 | 1.09x | 0.0156 |
| up_proj | layers.0.mlp.up_proj | 16 | 0.312 | 0.309 | 1.01x | 0.0156 |
| up_proj | layers.0.mlp.up_proj | 128 | 0.425 | 1.127 | 0.38x | 0.0156 |
| down_proj | layers.0.mlp.down_proj | 1 | 0.244 | 0.271 | 0.90x | 0.0156 |
| down_proj | layers.0.mlp.down_proj | 16 | 0.361 | 0.341 | 1.06x | 0.0312 |
| down_proj | layers.0.mlp.down_proj | 128 | 0.470 | 1.406 | 0.33x | 0.0312 |
| lm_head | embed_tokens(tied_lm_head) | 1 | 3.768 | 1.171 | 3.22x | 0.0156 |
| lm_head | embed_tokens(tied_lm_head) | 16 | 7.385 | 7.457 | 0.99x | 0.0312 |
| lm_head | embed_tokens(tied_lm_head) | 128 | 13.607 | 60.783 | 0.22x | 0.0312 |

## End-to-End Week 2 Results

Configuration:

```text
num_seqs=4
input_len=64
output_len=32
warmup=1
seed=0
flash_attention=False
```

| Implementation | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s | Raw output |
|---|---:|---:|---:|---:|---|
| tiny_llm | 47.09 | 141.26 | 491.71 | 56.45 | `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b.txt` |
| tiny_llm_ref | 49.24 | 147.73 | 488.10 | 59.80 | `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b.txt` |

The current implementation reaches:

```text
47.09 / 49.24 = 95.6% of reference output throughput
56.45 / 59.80 = 94.4% of reference decode throughput
```

## End-to-End Pressure Matrix

All runs used:

```text
input_len=64
output_len=32
warmup=1
seed=0
flash_attention=False
```

| Implementation | num_seqs | Prompt tokens | Generated tokens | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s | Raw output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny_llm | 1 | 64 | 32 | 49.25 | 147.76 | 470.13 | 60.40 | `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b_numseq1.txt` |
| tiny_llm_ref | 1 | 64 | 32 | 46.85 | 140.56 | 486.59 | 56.26 | `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b_numseq1.txt` |
| tiny_llm | 4 | 256 | 128 | 47.09 | 141.26 | 491.71 | 56.45 | `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b.txt` |
| tiny_llm_ref | 4 | 256 | 128 | 49.24 | 147.73 | 488.10 | 59.80 | `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b.txt` |
| tiny_llm | 8 | 512 | 256 | 47.77 | 143.32 | 481.90 | 57.76 | `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b_numseq8.txt` |
| tiny_llm_ref | 8 | 512 | 256 | 47.61 | 142.84 | 493.38 | 57.19 | `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b_numseq8.txt` |

Pressure takeaway:

- Across `num_seqs=1/4/8`, the current implementation remains in the same throughput band as the reference implementation.
- The small wins/losses across runs are close enough that they should be treated as benchmark noise unless repeated with more iterations.
- The stable conclusion is not "always faster"; it is that the end-to-end quantized path is functionally integrated and performs near reference speed on this Apple Silicon workload.

## Full Inference: Quantized vs Dequantized

This is the fairest comparison for the question "does using quantized inference make the model faster?" Both paths use Week 2 KV cache. The dequantized path uses ordinary `linear` on fully dequantized weights; the quantized path uses packed int4 weights and custom `quantized_linear`.

Configuration:

```text
input_len=64
output_len=32
warmup=1
seed=0
flash_attention=False
```

| num_seqs | Dequantized output tok/s | Quantized output tok/s | Output speedup | Dequantized decode tok/s | Quantized decode tok/s | Decode speedup | Dequantized prefill tok/s | Quantized prefill tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.50 | 48.47 | 0.98x | 51.53 | 59.36 | 1.15x | 1430.46 | 464.59 |
| 4 | 48.85 | 48.88 | 1.00x | 50.94 | 59.69 | 1.17x | 1380.58 | 473.58 |
| 8 | 46.59 | 47.00 | 1.01x | 48.45 | 56.99 | 1.18x | 1365.01 | 467.82 |

Interpretation:

- End-to-end output throughput is roughly parity: 0.98x to 1.01x.
- Decode throughput improves consistently with quantized inference: 1.15x to 1.18x.
- Prefill is slower with the current quantized kernel because prefill is a larger-`M` matrix multiplication workload, and the current Metal kernel is a simple teaching kernel rather than a tiled GEMM kernel.
- The honest conclusion is: quantization helps decode speed in the current implementation, but it does not yet improve full end-to-end throughput because prefill is bottlenecked by the non-optimized large-`M` quantized matmul.

## Interpretation

- The custom quantized path helps most when `M` is small, especially for the tied `lm_head` decode case (`M=1`, 3.22x faster than the dequantized baseline).
- For medium `M=16`, the custom quantized path is roughly at parity on the measured layers.
- For larger `M=128`, the current teaching kernel is slower than MLX's dequantized floating-point matmul. This is expected because our Metal kernel uses a simple one-thread-per-output-element design and does not yet implement tiled/threadgroup memory or SIMD-group reduction.
- End-to-end inference is already close to the reference implementation on this workload, at 95.6% of reference output throughput.

## Next Optimization Direction

The measured bottleneck is large-`M` matmul. To improve prefill/batched performance:

- tile the output matrix,
- let a threadgroup cooperate on a tile,
- reuse activation/weight chunks,
- use SIMD-group or threadgroup-level reduction,
- reduce repeated global memory reads.

The current kernel is correct and useful for decode-like small-`M` workloads, but larger prefill workloads need a more GEMM-like implementation.
