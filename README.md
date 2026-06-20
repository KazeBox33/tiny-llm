# Qwen3 Inference from Scratch

This repository records my implementation of a Qwen3-style LLM inference stack from low-level tensor operations. The goal is to understand how modern decoder-only LLM inference works by building the core components directly instead of only calling high-level model APIs.

The project is based on the excellent [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) course, with my own step-by-step implementation, notes, tests, and future experiments.

## What This Project Covers

Implemented so far:

- Basic matrix APIs used by the model path
- Scaled dot-product attention
- Multi-head attention
- Rotary positional encoding, including Qwen3 non-traditional RoPE
- Grouped Query Attention
- Causal attention masking for both training-style and KV-cache-style shapes
- Qwen3 attention block with Q/K RMSNorm and RoPE
- RMSNorm with float32 accumulation
- Numerically stable SiLU
- Qwen3 SwiGLU MLP

Planned next:

- Full Qwen3 transformer block
- Token embedding and LM head path
- Loading quantized Qwen3 MLX weights into the custom model
- Text generation and sampling
- KV cache, paged attention, continuous batching, and serving-oriented optimizations

## Why This Project

I am using this project to learn the internals of LLM inference:

- How Q/K/V projections become attention heads
- Why GQA reduces KV memory and bandwidth
- How RoPE injects token position into Q/K vectors
- Why causal masks differ between full prefill and cached decoding
- Where low precision is safe and where float32 accumulation is useful
- How Qwen3-style blocks combine RMSNorm, attention, residuals, and SwiGLU MLPs

This is intended as a learning-oriented implementation that can grow into a compact inference engine.

## Current Test Status

The completed Week 1 tasks pass locally:

```bash
pdm run test --week 1 --day 1
pdm run test --week 1 --day 2
pdm run test --week 1 --day 3
pdm run test --week 1 --day 4
```

Recent local results:

```text
Week 1 Day 3: 60 passed
Week 1 Day 4: 22 passed
```

## Setup

Install dependencies:

```bash
pdm install
```

Check the environment:

```bash
pdm run check-installation
```

Run tests for a specific chapter:

```bash
pdm run test --week 1 --day 4
```

## Repository Layout

```text
src/tiny_llm/
  attention.py              attention, GQA, causal mask
  positional_encoding.py    RoPE
  layer_norm.py             RMSNorm
  basics.py                 linear, softmax, SiLU
  qwen3_week1.py            Qwen3 attention, MLP, transformer/model path
```

Reference implementations and tests from the course are kept in:

```text
src/tiny_llm_ref/
tests_refsol/
book/
```

Project notes written during implementation are organized in:

```text
project-notes/
  resume/          resume-ready summaries and measurable results
  benchmarks/      benchmark reports and performance interpretation
  optimization/    future optimization backlog and bottleneck analysis
```

## Attribution

This project is built while following [tiny-llm - LLM Serving in a Week](https://skyzh.github.io/tiny-llm/) by skyzh. The original course repository is [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm).

My work in this repository focuses on implementing, understanding, testing, and documenting the inference components step by step.
