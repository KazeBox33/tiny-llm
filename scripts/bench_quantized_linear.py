import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import mlx.core as mx
from mlx_lm import load

from tiny_llm.basics import linear
from tiny_llm.models import shortcut_name_to_full_name
from tiny_llm.quantize import QuantizedWeights, dequantize_linear, quantized_linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dequantized linear vs custom quantized_linear."
    )
    parser.add_argument("--model", type=str, default="qwen3-0.6b")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["q_proj", "up_proj", "down_proj", "lm_head"],
        help="Layer names: q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj/lm_head/embed_tokens",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[1, 16, 128],
        help="Number of activation rows M to benchmark. M=1 approximates decode; larger M approximates prefill.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
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


def get_mlx_layer(mlx_model: Any, layer_index: int, layer_name: str) -> tuple[Any, str]:
    if layer_name == "embed_tokens":
        return mlx_model.model.embed_tokens, "embed_tokens"
    if layer_name == "lm_head":
        try:
            return mlx_model.lm_head, "lm_head"
        except AttributeError:
            return mlx_model.model.embed_tokens, "embed_tokens(tied_lm_head)"

    layer = mlx_model.model.layers[layer_index]
    if layer_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, layer_name), f"layers.{layer_index}.self_attn.{layer_name}"
    if layer_name in {"gate_proj", "up_proj", "down_proj"}:
        return getattr(layer.mlp, layer_name), f"layers.{layer_index}.mlp.{layer_name}"
    raise ValueError(f"Unsupported layer name: {layer_name}")


def time_call(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        y = fn()
        mx.eval(y)

    times = []
    for _ in range(iters):
        t0 = perf_counter()
        y = fn()
        mx.eval(y)
        times.append(perf_counter() - t0)
    return times


def summarize(times: list[float]) -> dict[str, float]:
    sorted_times = sorted(times)
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "min_ms": min(times) * 1000,
        "p95_ms": sorted_times[int(0.95 * (len(sorted_times) - 1))] * 1000,
    }


def benchmark_layer(
    mlx_layer: Any,
    layer_name: str,
    source_layer: str,
    rows: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    quantized_weights = QuantizedWeights.from_mlx_layer(mlx_layer)
    dequantized_weight = dequantize_linear(mlx_layer)
    mx.eval(dequantized_weight)

    packs_per_item = 32 // quantized_weights.bits
    input_dim = quantized_weights.weight.shape[1] * packs_per_item
    output_dim = quantized_weights.weight.shape[0]
    dtype = quantized_weights.scales.dtype

    mx.random.seed(seed)
    x = mx.random.normal((rows, input_dim)).astype(dtype)
    mx.eval(x)

    baseline = lambda: linear(x, dequantized_weight)
    quantized = lambda: quantized_linear(x, quantized_weights)

    baseline_out = baseline()
    quantized_out = quantized()
    mx.eval(baseline_out, quantized_out)
    max_abs_diff = mx.max(mx.abs(baseline_out - quantized_out)).item()
    mean_abs_diff = mx.mean(mx.abs(baseline_out - quantized_out)).item()

    baseline_times = time_call(baseline, warmup, iters)
    quantized_times = time_call(quantized, warmup, iters)
    baseline_stats = summarize(baseline_times)
    quantized_stats = summarize(quantized_times)

    speedup = baseline_stats["median_ms"] / quantized_stats["median_ms"]

    return {
        "layer": layer_name,
        "source_layer": source_layer,
        "rows": rows,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "dtype": str(dtype),
        "baseline": baseline_stats,
        "quantized": quantized_stats,
        "speedup_median": speedup,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def main() -> None:
    args = parse_args()
    model_name = shortcut_name_to_full_name(args.model)
    mlx_model, _ = load(model_name)

    results = {
        "benchmark": "quantized_linear_vs_dequantized_linear",
        "model": model_name,
        "layer_index": args.layer_index,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "system": system_info(),
        "results": [],
    }

    with mx.stream(mx.gpu):
        for layer_name in args.layers:
            mlx_layer, source_layer = get_mlx_layer(
                mlx_model, args.layer_index, layer_name
            )
            for rows in args.rows:
                row = benchmark_layer(
                    mlx_layer,
                    layer_name,
                    source_layer,
                    rows,
                    args.warmup,
                    args.iters,
                    args.seed,
                )
                results["results"].append(row)
                print(
                    f"{layer_name:10s} rows={rows:<4d} "
                    f"baseline={row['baseline']['median_ms']:.3f}ms "
                    f"quantized={row['quantized']['median_ms']:.3f}ms "
                    f"speedup={row['speedup_median']:.2f}x "
                    f"max_abs_diff={row['max_abs_diff']:.4f}"
                )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
