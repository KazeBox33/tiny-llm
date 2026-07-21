# Current Learning Progress

Last updated: 2026-07-21

## Project Goal

Build and understand a Qwen3 inference stack from tensor operations through serving-oriented infrastructure. The target scope is the complete tiny-llm guide through Week 3, including KV cache, quantized inference, Flash Attention, batching, paged attention, and MoE. Correctness comes first; performance claims must be backed by reproducible benchmarks.

## Primary Development Repository

- Repository: `KazeBox33/tiny-llm`
- Branch: `main`
- Upstream course: `skyzh/tiny-llm`
- Resume-facing mirror: `KazeBox33/qwen3-inference-from-scratch`

Continue development in `KazeBox33/tiny-llm`. The resume-facing repository has a separate history and is synchronized after completed tasks.

## Guide Progress

| Guide section | Status | Evidence |
| --- | --- | --- |
| Week 1: model fundamentals | Complete | Attention, RoPE, GQA, RMSNorm, MLP, model loading, generation, and sampling implemented and tested |
| Week 2 Day 1: KV cache | Complete | KV-cache decoding and Week 2 model path implemented and tested |
| Week 2 Day 2-3: quantized matmul | Complete | CPU primitive, Metal kernel, model integration, and benchmark reports committed |
| Week 2 Day 4-5 Task 1: Flash Attention wrapper | Complete | Commit `59ddd81`; Python shape/dtype adapter implemented |
| Week 2 Day 4-5 Task 2: CPU Flash Attention | Complete | Commit `83a768e`; 9 targeted tests pass |
| Week 2 Day 4-5 Task 3: GPU Flash Attention | Next | Metal kernel, GPU dispatch, and metallib registration not implemented yet |
| Week 2 Day 4-5 Task 4: model integration | Pending | `use_flash_attention` exists, but attention execution still always uses the grouped fallback |
| Week 2 Day 6-7: prefill and batching | Pending | Start only after Flash Attention Task 4 passes |
| Week 3: paged attention and MoE | Pending | Start after Week 2 is complete |

## Current Flash Attention Checkpoint

The CPU implementation in `src/extensions/src/flash_attention.cpp` is complete and understood. Topics already studied:

- tiled attention with `Br = Bc = 32`;
- `Tr` and `Tc` tile counts;
- flattened batch/head dimension `N`;
- GQA query-head to KV-head mapping;
- causal alignment when `L != S`;
- fully masked tile skipping and fully valid tile optimization;
- online-softmax state `m_i`, `l_i`, and `o_i`;
- rescaling prior softmax/output state when a later tile introduces a larger maximum;
- MLX lazy primitives, command encoders, lambda captures, and `unsafe_weak_copy`;
- contiguous memory, raw pointers, and manual flattened indexing.

Verified on 2026-07-19:

```text
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_2

9 passed, 11 deselected
```

## Immediate Next Task

Follow `book/src/week2-04-flash-attention.md`, Task 3, in this order:

1. Learn the Metal kernel argument list and memory address spaces: `device`, `constant`, `threadgroup`, and register-local values.
2. Map CPU loops to GPU execution: `(n, i)` to the threadgroup grid, query row `a` to `simd_gid`, and KV column `b` to `simd_lid`.
3. Implement `src/extensions/src/flash_attention.metal` with one threadgroup per `(head, query tile)`.
4. Keep Q and partial O in threadgroup memory; keep online-softmax scalars in registers.
5. Use `simd_max` and `simd_sum` for row reductions while iterating KV tiles inside the threadgroup.
6. Implement `FlashAttention::eval_gpu(...)` in `src/extensions/src/flash_attention.cpp`.
7. Add `src/flash_attention.metal` to `mlx_build_metallib(...)` in `src/extensions/CMakeLists.txt`.
8. Rebuild and run Task 3 tests:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_3
```

Do not start Task 4 model integration until Task 3 correctness and causal-mask performance tests pass.

## New Mac Setup

This project requires an Apple Silicon Mac for MLX/Metal execution.

Clone the primary development repository:

```bash
git clone https://github.com/KazeBox33/tiny-llm.git
cd tiny-llm
```

Install PDM if it is not already available:

```bash
brew install pdm
```

Install Xcode from the App Store, select its developer directory if needed, and install the Metal toolchain:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

Install the Python environment and verify MLX:

```bash
pdm install -v
pdm run check-installation
```

Rebuild the custom C++/Metal extension and verify the current checkpoint:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_2
```

The Task 3 extension tests do not require model weights. Download `Qwen/Qwen3-0.6B-MLX-4bit` only when moving on to Task 4 end-to-end generation and benchmarking.

For VS Code, install the Python/Pylance extensions and clangd. Open the repository root so `.clangd` and generated compile commands can be discovered.

## Working Rules

- Read the matching guide chapter before each task.
- Teach the systems concept, data flow, shapes, and syntax before implementing AI-infrastructure code.
- Compare against `src/tiny_llm_ref` and `src/extensions_ref` for correctness checks.
- Complete one guide task, run its smallest targeted tests, then commit and push.
- Keep performance measurements under `benchmarks/` and interpretations under `project-notes/benchmarks/`.
- Keep resume-ready evidence in `project-notes/resume/RESUME_NOTES.md`.
