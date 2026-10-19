"""CPU single-layer attention: compare prefix recomputation with a growing KV cache."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch


def project(x, weights):
    wq, wk, wv = weights
    return x @ wq, x @ wk, x @ wv


def causal_attention(q, k, v):
    scores = q @ k.T / math.sqrt(q.shape[1])
    scores.masked_fill_(torch.ones_like(scores, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def latest_query_attention(q, cached_k, cached_v):
    # The cache contains only previous/current keys: every cached key is legal.
    # A 1xT upper-left causal mask would incorrectly keep only key zero.
    return torch.softmax(q @ cached_k.T / math.sqrt(q.shape[1]), dim=-1) @ cached_v


@torch.no_grad()
def run_case(prompt=8, steps=4, model_dim=16, head_dim=8, value_dim=8, dtype=torch.float64):
    if any(not isinstance(v, int) for v in (prompt, steps, model_dim, head_dim, value_dim)):
        raise ValueError("All dimensions and counts must be integers")
    if not 1 <= prompt <= 128 or not 0 <= steps <= 128:
        raise ValueError("Use 1 <= prompt <= 128 and 0 <= steps <= 128")
    if any(not 1 <= d <= 128 for d in (model_dim, head_dim, value_dim)):
        raise ValueError("Feature dimensions must be between 1 and 128")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64")
    generator = torch.Generator(device="cpu").manual_seed(29)
    x = torch.randn(prompt + steps, model_dim, generator=generator, dtype=dtype)
    weights = tuple(torch.randn(model_dim, d, generator=generator, dtype=dtype) / math.sqrt(model_dim)
                    for d in (head_dim, head_dim, value_dim))
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12

    q, cached_k, cached_v = project(x[:prompt], weights)
    prefill = causal_attention(q, cached_k, cached_v)
    cached_outputs = [prefill[-1:]]
    cache_elements = [cached_k.numel() + cached_v.numel()]
    for position in range(prompt, prompt + steps):
        query, key, value = project(x[position:position + 1], weights)
        cached_k = torch.cat((cached_k, key), dim=0)
        cached_v = torch.cat((cached_v, value), dim=0)
        cached_outputs.append(latest_query_attention(query, cached_k, cached_v))
        cache_elements.append(cached_k.numel() + cached_v.numel())

    rows = []
    reference_rows = 0
    reference_pairs = 0
    cached_pairs = prompt * prompt
    for index, length in enumerate(range(prompt, prompt + steps + 1)):
        q, k, v = project(x[:length], weights)
        reference = causal_attention(q, k, v)[-1:]
        cached = cached_outputs[index]
        torch.testing.assert_close(cached, reference, rtol=tolerance, atol=tolerance)
        reference_rows += length
        reference_pairs += length * length
        if index:
            cached_pairs += length
        expected_elements = length * (head_dim + value_dim)
        if cache_elements[index] != expected_elements:
            raise AssertionError("Cache size differs from one stored K/V vector per token")
        rows.append({
            "phase": "prefill" if index == 0 else "decode",
            "prefix_length": length, "max_abs_error": (cached - reference).abs().max().item(),
            "kv_elements": cache_elements[index],
            "kv_bytes": cache_elements[index] * x.element_size(),
            "reference_score_pairs": length * length,
            "cached_score_pairs": length * length if index == 0 else length,
        })
    return {
        "prompt": prompt, "steps": steps, "model_dim": model_dim,
        "head_dim": head_dim, "value_dim": value_dim, "dtype": str(dtype),
        "all_outputs_close": True, "max_abs_error": max(row["max_abs_error"] for row in rows),
        "projection_token_rows": {"recompute": reference_rows, "cached": prompt + steps},
        "attention_score_pairs": {"recompute": reference_pairs, "cached": cached_pairs},
        "final_kv_bytes": rows[-1]["kv_bytes"],
        "baseline": "Recompute projections AND complete causal attention for each prefix, then take last row",
        "count_scope": "Algorithmic token rows and score pairs; not FLOPs, elapsed time, or peak memory",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/29_kv_cache.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        example = run_case(args.prompt, args.steps, dtype=getattr(torch, args.dtype))
    except ValueError as error:
        parser.error(str(error))
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": "CPU, fixed input representations, one attention layer, forward only",
        "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "architecture": platform.machine(), "device": "cpu", "torch_threads": 1},
        "example": example,
        "edge_cases": [run_case(prompt, steps, 8, 4, 3) for prompt, steps in ((1, 0), (1, 1), (3, 5), (17, 2))],
        "limitations": ["Not token generation or a trained language model", "No positional encoding or full decoder stack",
                        "torch.cat cache copies are not an optimized serving implementation", "No GPU timing"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU KV cache experiment; GPU execution skipped.")
    print("Outputs match every prefix. Max error:", example["max_abs_error"])
    print("Q/K/V projected token rows:", example["projection_token_rows"])
    print("Attention score pairs:", example["attention_score_pairs"])
    print("Final KV cache bytes:", example["final_kv_bytes"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
