# Qwen3 Inference From Scratch - Resume Notes

This file records resume-ready engineering work, technical highlights, and measurable results from the tiny-llm / Qwen3 inference project.

## How To Use This File

- Keep this file updated after every meaningful task, benchmark, or optimization.
- For each entry, record:
  - what was implemented,
  - why it matters in LLM inference,
  - what systems / performance concept it demonstrates,
  - correctness tests,
  - benchmark data if available,
  - a concise resume bullet.

## Project Positioning

Built a Qwen3 inference engine from scratch on Apple Silicon using MLX, C++, and Metal, progressively implementing model components, KV cache, quantized inference, and custom kernels.

This project is useful for resume positioning around:

- LLM inference systems
- Transformer internals
- MLX / Metal acceleration
- Quantized model serving
- KV cache and decoding optimization
- Custom C++ / GPU kernel development
- AI infrastructure learning and performance debugging

## Implemented So Far

### Week 1: Qwen3 Model Components

Implemented the core Qwen3 inference path in Python / MLX:

- scaled dot-product attention
- grouped-query attention
- RoPE positional encoding
- RMSNorm
- SiLU / gated MLP
- transformer block
- embedding / LM head path
- autoregressive generation
- temperature, top-k, and top-p sampling

Technical value:

- Rebuilt a modern decoder-only Transformer forward pass from first principles.
- Understood tensor shapes across batch, sequence length, heads, KV heads, and head dimension.
- Compared CS336 PyTorch implementation with MLX implementation on Apple Silicon.

Resume bullet draft:

- Implemented Qwen3 decoder inference from scratch with MLX, including GQA attention, RoPE, RMSNorm, gated MLP, token embedding, LM head, and sampling strategies.

### Week 2 Day 1: KV Cache

Implemented KV cache support for incremental decoding:

- full KV cache update and fetch
- attention path with cached keys / values
- transformer block and model wrapper with per-layer cache
- generation loop using prefill + decode

Why it matters:

- Without KV cache, every generated token recomputes attention over the whole prefix.
- With KV cache, prefill computes the prompt once, and decode only computes one new token at a time while reusing previous K/V states.

Correctness checks:

- KV cache smoke test:
  - first update shape: `(1, 2, 3, 4)`
  - second update shape: `(1, 2, 4, 4)`
- Qwen3ModelWeek2 smoke test:
  - prefill logits shape: `(1, 3, 151936)`
  - decode logits shape: `(1, 1, 151936)`
- Official Week 2 Day 1 tests passed.
- Generation check produced a valid response for prompt `Say hi.`

Resume bullet draft:

- Added KV-cache based incremental decoding for Qwen3 inference, reducing repeated prefill computation and enabling efficient token-by-token generation.

### Week 2 Day 2-3: Quantized Inference Preparation

Implemented Python-side quantized layer wrappers:

- `QuantizedWeights` metadata container
- `QuantizedEmbedding`
- quantized embedding lookup with selective row dequantization
- quantized linear wrapper calling custom `quantized_matmul`

Why it matters:

- Qwen3 MLX 4-bit weights are stored as packed int4 values.
- Efficient inference should avoid fully dequantizing the entire weight matrix.
- The next step is fusing dequantization and matrix multiplication inside a C++ / Metal kernel.

Correctness checks:

- Quantized embedding lookup matched MLX reference behavior.
- Quantized linear wrapper passed a monkeypatch smoke test and correctly forwarded packed weight, scales, biases, group size, bits, and transpose flag.

Resume bullet draft:

- Built quantized embedding and linear wrappers for 4-bit Qwen3 weights, preparing a fused dequantization + matmul path for memory-bandwidth-efficient inference.

### Week 2 Day 2: CPU Quantized Matmul Primitive

Implemented the CPU version of the custom MLX `quantized_matmul` primitive:

- added a C++ CPU implementation for fused int4 unpacking, dequantization, and matrix multiplication,
- wired the Python `quantized_matmul` wrapper to the C++ extension,
- flattened batched inputs to 2D before calling the extension and reshaped outputs back,
- supported both `float16` and `bfloat16` output paths,
- accumulated products in `float` before casting back to the output dtype.

Why it matters:

- This avoids materializing the full dequantized weight matrix in Python.
- The CPU implementation establishes a correctness baseline for the later Metal/GPU kernel.
- The same packed-weight math will be reused in the GPU implementation, where the memory-bandwidth benefit matters most.

Correctness checks:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 2 -- -k task_2
```

Result:

```text
4 passed, 4 deselected
```

Resume bullet draft:

- Implemented a custom MLX C++ quantized matmul primitive for Qwen3 4-bit weights, fusing int4 unpacking, group-wise dequantization, and CPU matrix multiplication with fp16/bf16 support.

### Week 2 Day 3: Metal GPU Quantized Matmul Kernel

Implemented the GPU version of the custom `quantized_matmul` primitive:

- added a Metal kernel for Qwen3 4-bit packed weights,
- used one GPU thread per output matrix element,
- unpacked eight int4 weights from each packed `uint32`,
- applied group-wise `scale` and `bias` inside the kernel,
- accumulated products in `float` and cast back to `float16` / `bfloat16`,
- wired `QuantizedMatmul::eval_gpu` to dispatch the Metal kernel through MLX's command encoder,
- registered the new `.metal` file in the extension build system.

Why it matters:

- This keeps weights compressed while computing, preserving the memory-bandwidth advantage of 4-bit inference.
- The CPU implementation was a correctness baseline; the Metal implementation moves the same fused dequantization + matmul work onto Apple GPU.
- This is the first project step that connects Python model code, C++ MLX primitive dispatch, and a custom GPU kernel.

Correctness checks:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 2 -- -k task_3
```

Result:

```text
4 passed, 4 deselected
```

Resume bullet draft:

- Implemented a custom Metal kernel for 4-bit Qwen3 quantized matmul, fusing int4 unpacking, group-wise dequantization, and fp16/bf16 GPU accumulation through an MLX C++ primitive.

### Week 2 Day 3: End-to-End Quantized Qwen3 Integration

Integrated the custom quantized matmul path into the Week 2 Qwen3 model:

- kept attention projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`) as `QuantizedWeights`,
- kept MLP projection weights (`gate_proj`, `up_proj`, `down_proj`) as `QuantizedWeights`,
- replaced ordinary `linear(...)` calls with `quantized_linear(...)`,
- switched token embedding to `QuantizedEmbedding`,
- kept `lm_head` as a quantized projection and dispatched it through `quantized_linear`,
- removed the Week 2 model path's full-weight `dequantize_linear(...)` usage.

Why it matters:

- This changes the model from a "load quantized weights, then immediately dequantize them" path to an actual compressed-weight inference path.
- The custom Metal quantized matmul is now used by real Qwen3 inference layers rather than only standalone unit tests.
- It preserves the memory-bandwidth advantage of packed 4-bit weights across attention, MLP, embedding, and logits projection paths.

Correctness checks:

```bash
DEBUG=0 pdm run test --week 2 --day 2
pdm run main --solution tiny_llm --loader week2 --model qwen3-0.6b --prompt "Say hi."
```

Result:

```text
8 passed
Hello! How can I assist you today? 😊
```

Resume bullet draft:

- Integrated a fused 4-bit quantized matmul kernel end-to-end into Qwen3 inference, replacing full-weight dequantization in attention, MLP, embedding, and LM head paths with packed-weight quantized execution.

### Local C++ / Metal Extension Environment

Configured local development environment for MLX custom extensions:

- Xcode 26.5
- Apple clang
- Metal compiler / metallib
- Homebrew in user directory
- CMake
- Ninja
- PDM virtual environment

Verification:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run build-ext-test
```

Result:

```text
c shape: (3, 4)
c dtype: mlx.core.float32
c correct: True
```

Resume bullet draft:

- Set up and verified MLX C++/Metal extension toolchain on Apple Silicon, compiling and running a custom primitive test successfully.

## Concepts Learned And Explainable In Interviews

### MLX Primitive

An MLX primitive is a low-level operation node in MLX's lazy computation graph.

In this project, custom primitives are used to register operations such as quantized matmul so MLX can schedule them on CPU or GPU.

CUDA comparison:

- MLX primitive is similar to a PyTorch custom op or runtime-registered CUDA extension.
- `eval_cpu` is the CPU implementation.
- `eval_gpu` dispatches the Metal kernel.
- A CUDA equivalent would expose a C++ binding and call a `__global__` CUDA kernel.

### Quantized Matmul

4-bit weights reduce memory bandwidth, but efficient inference requires avoiding full-weight dequantization.

The desired implementation:

```text
packed int4 weight + scale/bias + activation
        ↓
fused dequantization and matmul
        ↓
output activation
```

CUDA comparison:

- Packed int4 weights are analogous to compressed global memory loads.
- Fused dequant + matmul reduces memory traffic.
- GPU kernel design will involve mapping output elements/tiles to threads, similar to CUDA blocks and threads.

### KV Cache

KV cache stores previous key/value tensors during generation.

Instead of recomputing all prompt tokens for every next token, the model only computes Q/K/V for the new token and attends over cached K/V.

Performance impact:

- Reduces decode complexity per step.
- Improves long-context generation latency.
- Introduces cache memory management problems that become important in batched serving.

## Benchmark / Data

Detailed benchmark report:

```text
project-notes/benchmarks/QUANTIZED_LINEAR_BENCHMARK.md
```

Raw reproducible outputs:

- `benchmarks/quantized_linear_qwen3_0_6b_metal.json`
- `benchmarks/quantized_vs_dequantized_inference_qwen3_0_6b.json`
- `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b.txt`
- `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b.txt`
- `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b_numseq1.txt`
- `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b_numseq1.txt`
- `benchmarks/end_to_end_tiny_llm_week2_qwen3_0_6b_numseq8.txt`
- `benchmarks/end_to_end_tiny_llm_ref_week2_qwen3_0_6b_numseq8.txt`

Environment:

```text
Date: 2026-06-20
Platform: macOS-26.5.1-arm64-arm-64bit
CPU: Apple M5
Device: MLX GPU / Metal
Model: Qwen/Qwen3-0.6B-MLX-4bit
```

### Isolated Quantized Linear Benchmark

Command:

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

Baseline means dequantizing the 4-bit weight once and running ordinary `linear`.
Quantized means keeping packed 4-bit weights and calling the custom C++ / Metal `quantized_matmul`.

| Layer | Rows M | Baseline median ms | Quantized median ms | Speedup | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `q_proj` | 1 | 0.227 | 0.219 | 1.04x | decode-like small M |
| `up_proj` | 1 | 0.246 | 0.226 | 1.09x | decode-like small M |
| `down_proj` | 16 | 0.361 | 0.341 | 1.06x | medium M |
| tied `lm_head` | 1 | 3.768 | 1.171 | 3.22x | largest small-M win |
| `q_proj` | 128 | 0.345 | 0.796 | 0.43x | large-M prefill is slower |
| `up_proj` | 128 | 0.425 | 1.127 | 0.38x | needs tiled GEMM-style kernel |
| tied `lm_head` | 128 | 13.607 | 60.783 | 0.22x | simple kernel bottleneck |

Takeaway:

- The custom quantized path is useful for decode-like small `M`, especially tied `lm_head` at `M=1`, where it measured 3.22x faster than the dequantized baseline.
- For large `M`, the current one-thread-per-output-element teaching kernel is slower than MLX's dequantized matmul.
- Next performance work should target tiled/threadgroup-memory/SIMD-reduction matmul for large prefill batches.

### End-to-End Week 2 Benchmark

Command:

```bash
pdm bench --solution tiny_llm --loader week2 --model qwen3-0.6b \
  --num-seqs 4 \
  --min-input-len 64 --max-input-len 64 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 \
  --seed 0
```

Reference comparison:

```bash
DEBUG=0 pdm run build-ext-ref

pdm bench --solution tiny_llm_ref --loader week2 --model qwen3-0.6b \
  --num-seqs 4 \
  --min-input-len 64 --max-input-len 64 \
  --min-output-len 32 --max-output-len 32 \
  --warmup 1 \
  --seed 0
```

| Implementation | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| `tiny_llm` | 47.09 | 141.26 | 491.71 | 56.45 |
| `tiny_llm_ref` | 49.24 | 147.73 | 488.10 | 59.80 |

The current implementation reaches:

```text
47.09 / 49.24 = 95.6% of reference output throughput
56.45 / 59.80 = 94.4% of reference decode throughput
```

### End-to-End Pressure Matrix

All runs used `input_len=64`, `output_len=32`, `warmup=1`, `seed=0`, and `flash_attention=False`.

| Implementation | num_seqs | Output tok/s | Total tok/s | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tiny_llm` | 1 | 49.25 | 147.76 | 470.13 | 60.40 |
| `tiny_llm_ref` | 1 | 46.85 | 140.56 | 486.59 | 56.26 |
| `tiny_llm` | 4 | 47.09 | 141.26 | 491.71 | 56.45 |
| `tiny_llm_ref` | 4 | 49.24 | 147.73 | 488.10 | 59.80 |
| `tiny_llm` | 8 | 47.77 | 143.32 | 481.90 | 57.76 |
| `tiny_llm_ref` | 8 | 47.61 | 142.84 | 493.38 | 57.19 |

Pressure takeaway:

- Across `num_seqs=1/4/8`, the current implementation remains in the same throughput band as the reference implementation.
- The stable claim is near-reference end-to-end throughput, not a universal speedup.
- The isolated linear benchmark shows where the current custom kernel helps most: decode-like small `M`, especially tied `lm_head`.

### Full Inference: Quantized vs Dequantized

This comparison keeps the Week 2 KV-cache inference path fixed. The only intended difference is:

- Dequantized path: fully dequantized weights + ordinary `linear`.
- Quantized path: packed int4 weights + custom `quantized_linear`.

Command:

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

| num_seqs | Dequantized output tok/s | Quantized output tok/s | Output speedup | Dequantized decode tok/s | Quantized decode tok/s | Decode speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.50 | 48.47 | 0.98x | 51.53 | 59.36 | 1.15x |
| 4 | 48.85 | 48.88 | 1.00x | 50.94 | 59.69 | 1.17x |
| 8 | 46.59 | 47.00 | 1.01x | 48.45 | 56.99 | 1.18x |

Honest conclusion:

- Full output throughput is roughly parity, from 0.98x to 1.01x.
- Decode throughput improves with quantized inference, from 1.15x to 1.18x.
- Prefill is slower in the current quantized path because the Metal kernel is not yet a tiled GEMM-style implementation for large `M`.

## Resume Bullets To Refine Later

- Implemented a Qwen3 inference engine from scratch on Apple Silicon using MLX, covering attention, RoPE, RMSNorm, MLP, KV cache, sampling, and quantized model loading.
- Built KV-cache based autoregressive decoding to avoid repeated prefix computation and improve generation efficiency.
- Developed and integrated 4-bit quantized weight wrappers plus a fused C++ / Metal dequantization + matrix multiplication path for memory-bandwidth-aware Qwen3 inference.
- Configured and verified a custom MLX C++/Metal extension toolchain, enabling low-level kernel development for LLM inference acceleration.
- Benchmarked custom 4-bit quantized inference against dequantized inference on Qwen3-0.6B, measuring 1.15x-1.18x decode throughput improvement and identifying large-`M` prefill as the next kernel optimization bottleneck.
- Studied and implemented AI infrastructure concepts including packed weights, custom primitives, lazy execution, KV cache, and GPU kernel dispatch.

## Open Items

- Continue performance work from `project-notes/optimization/OPTIMIZATION_BACKLOG.md`.
- Optimize large-`M` quantized matmul with a cooperative/tiled Metal kernel.
- Benchmark more prompt/output length combinations and batch sizes.
- Continue Week 2 FlashAttention and batching tasks.
- Add Week 3 paged attention / MoE notes after implementation.
