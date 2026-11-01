# Week 2 Day 2 Task 3: first cooperative W4 matvec

Date: 2026-11-01. Starting fork commit: `69fd7c3`.

This is an incremental Task 3 implementation, not completion of the revised
Day 2 chapter. The source reading is `week2-02-quantize-model.md` at upstream
`17a8cc6`, especially Stage 1 / Stage 2 and GPU Dispatch.

## Ownership and data flow

Keep the existing CPU implementation intentionally. The updated course's
GPU-only boundary is a known difference, not a mathematical requirement.
The old one-thread-per-output Metal kernel also remains available as a control.

The Python wrapper flattens leading activation dimensions into M and makes
the four arrays contiguous. Packed weights remain uint32 [K, N/8]; scale and
bias have shape [K, N/128]. Activations and outputs are BF16 or optional FP16.
The kernel reconstructs weights and accumulates in FP32 in registers; it does
not expand the full matrix, allocate shared memory, or modify the KV cache.

GPU dispatch uses the cooperative kernel for 1 <= M <= 8. Larger matrices
continue to use the original matrix kernel. `use_simd=False` forces that
control even for decode-shaped inputs. The Python `quantized_matmul_vanilla`
entrypoint preserves leading dimensions and uses this flag. The binding adds
the flag as a keyword-only argument, preserving existing positional calls.
The primitive stores the flag and includes it in `is_equivalent` so different
execution modes are not treated as identical nodes by graph optimizations.
The CPU implementation ignores the GPU scheduling choice.

## Thread mapping

- 32 lanes per SIMD group; validate the pipeline execution width.
- Two SIMD groups / 64 threads per threadgroup.
- Each SIMD group computes two output columns for one activation row.
- Each threadgroup therefore covers four output columns, not eight.
- Grid: [ceil(K/4), M, 1]; group y selects activation row i.
- First output column: k0 = group_x * 4 + simd_id * 2; k1 = k0 + 1.
- Lane l processes packed words p = l, l+32, l+64, ... below N/8.
- Each word represents eight input features, so its quantization group is p/16.

For N=256 there are 32 packed words: each lane reads one word for each
output column, and reads eight activations reused across the two columns.
For N=128 only 16 lanes load words; the other lanes retain zero partial sums
and still participate in both `simd_sum` reductions. For N=384, the first
16 lanes process a second word, exercising the packed-word loop tail.

All lanes execute both reductions. Only lane zero stores the two reduced
results. Guard each output-column load and store independently for odd K
and partial threadgroups. No inter-SIMD reduction or threadgroup barrier is
needed because each SIMD group owns separate outputs.

## Verification

First added a regression for the explicit vanilla control. Before the binding
change it failed with TypeError for the unsupported `use_simd` keyword.

Rebuilt the learner extension from `src/extensions`:

```bash
DEBUG=0 CMAKE_ARGS="-U*Python* -DPython_EXECUTABLE=$VIRTUAL_ENV/bin/python" python build.py
```

The command requires VIRTUAL_ENV to name this checkout's environment. The
actual local build used its absolute Python path and completed successfully.
Existing SDK search-path and nanobind literal warnings were non-fatal.

New tests:

```bash
.venv/bin/python -m pytest tests/test_week_2_simd_matvec.py -q -W error::RuntimeWarning
```

Result: 14 passed. Cases cover BF16/FP16, N=128/256/384/1024, odd output
counts, M=1/2/8/9/128, the explicit vanilla wrapper, and preserved CPU execution.
Both GPU paths are compared with independently constructed codes and signed
scale/bias parameters, using NumPy FP32 arithmetic and output-dtype rounding.
The reference uses explicit `einsum(..., optimize=False)`: the initial NumPy
matmul path emitted runtime warnings even though the assertions passed. The
revised tests assert finite expected/actual values and pass with runtime
warnings treated as errors, without suppressing them.

Focused model/cache/operator regression:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q -rs \
  tests/test_week_1_completion.py tests/test_week_2_cache.py \
  tests/test_week_2_generation.py tests/test_week_2_model.py \
  tests/test_week_2_numerics.py tests/test_week_2_quantized_embedding.py \
  tests/test_week_2_quantized_linear.py tests/test_week_2_simd_matvec.py
```

Result: 133 passed. This is not the entire revised course suite.

An offline GPU smoke with Qwen/Qwen3-4B-MLX-4bit used our cached generation,
greedy decoding, no thinking, and a 16-token generation limit. It answered
the Chinese 1+1 question with `1+1 equals 2` in Chinese. At the Python-to-native
projection boundary, input M=25 occurred 253 times and M=1 occurred 1518 times.
These are shape observations; they are not a GPU profiler trace or a timing
comparison. The C++ dispatcher selects matrix / SIMD schedules respectively.

## Remaining work

- This readable two-output schedule is the first SIMD step only.
- Next study four outputs per group, two adjacent packed words per lane,
  and the affine rearrangement from the current textbook.
- Add the chapter's explicit custom-matvec entrypoint and remaining checkpoint
  wiring as later small steps, without duplicating native bindings.
- Benchmark the same projection and complete-model workloads against the
  vanilla control before making any speedup claim.
- Dense Day 1 baseline / revised benchmark CLI alignment remain separate gaps.

No throughput or latency improvement is claimed by this increment.
