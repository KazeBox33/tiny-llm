# AGENTS.md

## Scope

- This file applies to the entire repository.
- Use this as the default test-running policy for coding agents.

## Objective

- Run and verify tests in a way that matches the book workflow (`book/src/*.md`).
- Prefer `pdm` entrypoints defined in `pyproject.toml`.
- Support this repository as a learning project, not just a code-completion task.

## Learning Collaboration Notes

- The learner is coming from CS336 Assignment 1 and already understands the basics of Transformer blocks, causal attention, RoPE, RMSNorm, MLP/SwiGLU, tokenization, generation, training loops, checkpoints, and experiment logging.
- When starting a new tiny-llm chapter, first read the corresponding `book/src/week*-*.md` file and explain:
  - what the chapter is trying to build,
  - why the component exists in LLM inference/serving,
  - how it connects to the learner's CS336 implementation,
  - which files and tests are involved.
- Do not jump straight to a full implementation unless explicitly asked. Prefer the CS336 learning style:
  1. translate and summarize the task,
  2. explain the math/system idea,
  3. explain relevant Python/MLX syntax from basics,
  4. show the next small code block or patch location,
  5. let the learner write or inspect it,
  6. run the smallest matching test.
- When code is shown, explain each important line's purpose, tensor shape expectations, dtype/precision behavior, and where data moves between CPU/GPU/MLX arrays.
- For inference-system topics such as KV cache, batching, quantization, FlashAttention, paged attention, and MoE, emphasize the engineering tradeoff: memory layout, compute reuse, latency, throughput, and correctness.
- If a task needs implementation by the agent, keep edits scoped to the chapter/task files and preserve the course's intended structure. Avoid writing real implementations inside test adapters or reference-solution files.
- Record meaningful experiments, performance observations, debugging lessons, and optimization points in a durable Markdown file when they are useful for later review or resume writing.
- Prefer comparing with the CS336 codebase when helpful, but avoid mixing CS336 training code into tiny-llm unless the learner explicitly asks for a cross-project note.

## Environment Requirements

- macOS on Apple Silicon is expected by the project.
- Install dependencies first:

```bash
pdm install -v
pdm run check-installation
```

- Optional baseline check from the setup chapter (reference solution, Week 1):

```bash
pdm run test-refsol -- -- -k week_1
```

## Agent Test Workflow

1. Start with the smallest relevant scope (`--week` + `--day`).
2. Use pytest filters via `-- -k ...` to isolate failing tasks.
3. Run broader suites only after targeted tests pass.
4. If extension code changed, rebuild extensions before testing.

## Canonical Commands

Run all tests:

```bash
pdm run test
```

Run a specific chapter/day:

```bash
pdm run test --week <WEEK> --day <DAY>
```

Run with pytest filters:

```bash
pdm run test --week 1 --day 3 -- -k task_2
pdm run test --week 2 --day 2 -- -k cpu
pdm run test --week 2 --day 2 -- -k gpu
```

Run reference-solution tests:

```bash
pdm run test-refsol
pdm run test-refsol --week 2 --day 2 -- -k cpu
```

## Extension Rebuild Rule

Rebuild before tests if these changed:

- `src/extensions/src/*`

Commands:

```bash
pdm run build-ext
```

## Guardrails

- Use `--` before pytest args (`-k`, `-q`, `--collect-only`, etc.).
- `pdm run test --week X --day Y` auto-copies `tests_refsol/test_week_X_day_Y.py` into `tests/`.
- Model-dependent tests (0.5B/1.5B/7B) skip when models are not downloaded locally.
