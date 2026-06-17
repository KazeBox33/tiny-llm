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

## Benchmark / Data TODO

Add concrete benchmark data here as we implement more kernels:

| Stage | Command | Model | Device | Prompt / Tokens | Latency | Tokens/s | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Week 1 baseline | TODO | Qwen3-0.6B-MLX-4bit | Apple Silicon | TODO | TODO | TODO | no KV cache |
| Week 2 KV cache | TODO | Qwen3-0.6B-MLX-4bit | Apple Silicon | TODO | TODO | TODO | with KV cache |
| Quantized matmul CPU | `DEBUG=0 pdm run test --week 2 --day 2 -- -k task_2` | Qwen3-0.6B-MLX-4bit | Apple Silicon CPU | unit test tensors | n/a | n/a | 4 official CPU tests passed |
| Quantized matmul Metal | TODO | Qwen3-0.6B-MLX-4bit | Apple Silicon | TODO | TODO | TODO | fused GPU kernel |

## Resume Bullets To Refine Later

- Implemented a Qwen3 inference engine from scratch on Apple Silicon using MLX, covering attention, RoPE, RMSNorm, MLP, KV cache, sampling, and quantized model loading.
- Built KV-cache based autoregressive decoding to avoid repeated prefix computation and improve generation efficiency.
- Developed quantized 4-bit weight wrappers and prepared fused dequantization + matrix multiplication for memory-bandwidth-efficient inference.
- Configured and verified a custom MLX C++/Metal extension toolchain, enabling low-level kernel development for LLM inference acceleration.
- Studied and implemented AI infrastructure concepts including packed weights, custom primitives, lazy execution, KV cache, and GPU kernel dispatch.

## Open Items

- Implement CPU `quantized_matmul` primitive.
- Implement Metal `quantized_matmul` kernel.
- Benchmark Python dequantized path vs C++ CPU path vs Metal path.
- Add quantitative speedup data.
- Continue Week 2 FlashAttention and batching tasks.
- Add Week 3 paged attention / MoE notes after implementation.
