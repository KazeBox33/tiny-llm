# Qwen3 Inference Optimization Backlog

This file records performance bottlenecks and future optimization ideas for the Qwen3 inference project.

It is intentionally written as an engineering handoff: after finishing the remaining learning tasks, we can come back here and continue optimization without losing context.

## Current Measured State

Benchmark source:

- `benchmarks/QUANTIZED_LINEAR_BENCHMARK.md`
- `benchmarks/quantized_vs_dequantized_inference_qwen3_0_6b.json`
- `benchmarks/quantized_linear_qwen3_0_6b_metal.json`

Environment:

```text
Date: 2026-06-20
Model: Qwen/Qwen3-0.6B-MLX-4bit
Device: Apple M5 / MLX GPU / Metal
Input length: 64
Output length: 32
FlashAttention: disabled
```

Full inference, quantized vs dequantized:

| num_seqs | Dequantized output tok/s | Quantized output tok/s | Output speedup | Dequantized decode tok/s | Quantized decode tok/s | Decode speedup | Dequantized prefill tok/s | Quantized prefill tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.50 | 48.47 | 0.98x | 51.53 | 59.36 | 1.15x | 1430.46 | 464.59 |
| 4 | 48.85 | 48.88 | 1.00x | 50.94 | 59.69 | 1.17x | 1380.58 | 473.58 |
| 8 | 46.59 | 47.00 | 1.01x | 48.45 | 56.99 | 1.18x | 1365.01 | 467.82 |

Important conclusion:

- Quantized inference improves decode throughput by about `1.15x-1.18x`.
- End-to-end output throughput is roughly parity because the quantized prefill path is currently much slower.
- The bottleneck is not the idea of quantization itself; the bottleneck is the simple current Metal matrix-multiply kernel for larger prefill workloads.

## Main Bottleneck

### Plain-English Explanation

The model has two major generation phases:

- **Prefill**: process the prompt all at once.
- **Decode**: generate one new token at a time.

Our quantized kernel works reasonably well during decode because decode usually has only one active input row:

```text
M = 1
```

But prefill has many active rows:

```text
M = prompt length, for example 64 / 128 / 256
```

The current Metal kernel uses a simple design:

```text
one GPU thread computes one output element
```

That means a single thread does a long dot product mostly by itself. This is easy to understand and correct, but it does not use the GPU efficiently for large matrices.

### What This Means In Practice

Current kernel behavior:

```text
thread A computes out[0, 0] from start to finish
thread B computes out[0, 1] from start to finish
thread C computes out[0, 2] from start to finish
...
```

Better kernel behavior:

```text
a group of threads cooperates on a small block of the output matrix
threads share/reuse loaded data
partial sums are combined efficiently
```

## Does The Reference Implementation Have This Bottleneck?

Mostly yes.

The reference `quantized_matmul.metal` also uses the same high-level strategy:

```text
Each thread processes an element in the output matrix
```

The reference implementation has a small local optimization: it manually unrolls the eight 4-bit values packed inside one `uint32`.

Our current version extracts the eight 4-bit values with a loop. This is easier to read, but it has a little more loop overhead.

So the current limitation is not a single accidental bug in our implementation. It is mainly that both our implementation and the teaching reference use a simple educational kernel rather than a production-grade tiled GEMM kernel.

## Optimization Plan

### 1. Low-Risk: Manually Unroll int4 Unpacking

Current idea:

```text
for pack_idx in 0..7:
    extract one 4-bit value
    dequantize it
    multiply by activation
```

Optimization:

```text
write out the 8 unpack/multiply steps directly
```

Why it helps:

- removes a tiny inner loop,
- gives the compiler a simpler pattern,
- matches the reference implementation more closely.

Expected impact:

- small speedup,
- low correctness risk,
- good first optimization after finishing the learning tasks.

Validation:

```bash
DEBUG=0 pdm run test --week 2 --day 2 -- -k task_3
pdm run python scripts/bench_quantized_linear.py \
  --model qwen3-0.6b \
  --layers q_proj up_proj down_proj lm_head \
  --rows 1 16 128 \
  --warmup 10 \
  --iters 50 \
  --seed 0 \
  --output-json benchmarks/quantized_linear_qwen3_0_6b_metal_after_unroll.json
```

### 2. Medium: Multiple Threads Cooperate On One Dot Product

Current idea:

```text
one thread computes all N multiplications for one output element
```

Optimization:

```text
split the dot product across multiple threads
each thread computes part of the sum
combine partial sums at the end
```

Example:

If one output element needs 1024 multiply-add operations:

```text
thread 0 handles 0..255
thread 1 handles 256..511
thread 2 handles 512..767
thread 3 handles 768..1023
then combine the four partial sums
```

Why it helps:

- reduces the amount of work each single thread has to do,
- improves parallelism for long dot products,
- targets the prefill bottleneck directly.

Expected impact:

- should help large `M` and large hidden dimensions more than decode `M=1`,
- more complex than unrolling,
- requires careful numerical and performance testing.

Validation:

- compare correctness against dequantized linear,
- compare rows `M=1`, `M=16`, `M=128`,
- inspect whether large `M=128` gets closer to or faster than dequantized matmul.

### 3. High-Impact: Tile The Matrix Multiply

Current idea:

```text
each thread repeatedly reads data from global memory
```

Optimization:

```text
divide the output matrix into small blocks
a threadgroup cooperates on one block
reuse chunks of activation/weight data inside the group
```

Plain-English version:

```text
Instead of everyone repeatedly going far away to fetch the same data,
bring a small chunk of data close to the workers,
let the workers reuse it,
then move to the next chunk.
```

Why it helps:

- reduces repeated memory reads,
- better matches how GPUs are designed to run matrix multiplication,
- should improve prefill and batched inference.

Expected impact:

- largest potential speedup,
- highest implementation complexity,
- most resume value if done well.

Validation:

```bash
pdm run python scripts/bench_quantized_vs_dequantized_inference.py \
  --model qwen3-0.6b \
  --num-seqs 1 4 8 \
  --input-len 64 \
  --output-len 32 \
  --warmup 1 \
  --seed 0 \
  --output-json benchmarks/quantized_vs_dequantized_inference_after_tiling.json
```

Success criteria:

- keep decode speedup,
- improve prefill throughput,
- make full output throughput clearly faster than dequantized inference.

### 4. Compare Against MLX Built-In Quantized Matmul

We should also compare our custom kernel against MLX's built-in quantized operations when available.

Why it matters:

- tells us how far our implementation is from a mature runtime,
- gives a more realistic performance target,
- helps avoid optimizing in the wrong direction.

Validation idea:

- add a benchmark mode that calls MLX's native quantized matmul / quantized linear path if exposed by the current MLX API,
- compare:
  - dequantized linear,
  - our custom quantized linear,
  - MLX native quantized linear.

### 5. Add More Stable Benchmark Protocol

Current benchmark is useful, but short.

Future benchmark improvements:

- run more warmup iterations,
- run multiple repeats and report median,
- record memory usage if possible,
- test more input lengths: `64`, `256`, `1024`,
- test more output lengths: `32`, `128`,
- separate prefill and decode reports more clearly.

Why it matters:

- reduces noise,
- makes performance claims safer,
- gives cleaner resume and interview data.

## Resume-Friendly Future Claim After Optimization

Current honest claim:

```text
Implemented end-to-end 4-bit quantized inference for Qwen3-0.6B and measured 1.15x-1.18x decode throughput improvement over dequantized inference, while identifying large-M prefill quantized matmul as the next bottleneck.
```

Potential future claim after the tiled kernel:

```text
Optimized the custom 4-bit Metal quantized matmul kernel from a one-thread-per-output implementation to a cooperative tiled GEMM-style implementation, improving both decode and prefill throughput for Qwen3 inference.
```

Do not use the future claim until it is measured.
