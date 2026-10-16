"""CPU attention lab: dense attention vs tiled online softmax, forward only."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def validate_inputs(q, k, v):
    tensors = (q, k, v)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("Use single-head 2D tensors Q[N,D], K[N,D], V[N,Dv]")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("This lab explicitly uses CPU tensors only")
    if q.dtype not in (torch.float32, torch.float64) or any(t.dtype != q.dtype for t in tensors):
        raise ValueError("Q/K/V must share float32 or float64 dtype")
    if min(*q.shape, *k.shape, *v.shape) <= 0:
        raise ValueError("Empty sequences or feature dimensions are not supported")
    if q.shape != k.shape or q.shape[0] != v.shape[0]:
        raise ValueError("Self-attention requires equal Q/K shapes and matching V sequence length")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("Q/K/V must contain finite values")


@torch.no_grad()
def dense_attention(q, k, v, causal=False):
    validate_inputs(q, k, v)
    scores = (q @ k.T) / math.sqrt(q.shape[1])
    if causal:
        future = torch.ones_like(scores, dtype=torch.bool).triu(diagonal=1)
        scores.masked_fill_(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


@torch.no_grad()
def tiled_attention(q, k, v, query_block=16, key_block=16, causal=False):
    validate_inputs(q, k, v)
    if any(not isinstance(size, int) or size < 1 for size in (query_block, key_block)):
        raise ValueError("query_block and key_block must be positive integers")
    n, d = q.shape
    output = torch.empty(n, v.shape[1], dtype=q.dtype)
    largest_tile = 0
    tile_count = 0
    score_elements_computed = 0
    for q_start in range(0, n, query_block):
        q_tile = q[q_start:q_start + query_block]
        rows = q_tile.shape[0]
        row_max = torch.full((rows, 1), float("-inf"), dtype=q.dtype)
        denominator = torch.zeros(rows, 1, dtype=q.dtype)
        numerator = torch.zeros(rows, v.shape[1], dtype=q.dtype)
        for k_start in range(0, n, key_block):
            k_tile = k[k_start:k_start + key_block]
            scores = (q_tile @ k_tile.T) / math.sqrt(d)
            if causal:
                query_positions = torch.arange(q_start, q_start + rows)[:, None]
                key_positions = torch.arange(k_start, k_start + k_tile.shape[0])[None, :]
                scores.masked_fill_(key_positions > query_positions, float("-inf"))

            # Every row sees key 0 in the first tile, even with the causal mask.
            # Later fully masked tiles add zeros; no -inf - -inf is needed.
            next_max = torch.maximum(row_max, scores.max(dim=-1, keepdim=True).values)
            correction = torch.exp(row_max - next_max)
            block_weights = torch.exp(scores - next_max)
            denominator = correction * denominator + block_weights.sum(dim=-1, keepdim=True)
            numerator = correction * numerator + block_weights @ v[k_start:k_start + key_block]
            row_max = next_max
            largest_tile = max(largest_tile, scores.numel())
            score_elements_computed += scores.numel()
            tile_count += 1
        output[q_start:q_start + rows] = numerator / denominator
    if not torch.isfinite(output).all().item():
        raise ArithmeticError("Nonfinite attention output; check input scale and dtype")
    return output, {
        "largest_score_tile_elements": largest_tile,
        "score_elements_computed": score_elements_computed,
        "tile_count": tile_count,
        "materializes_full_score_matrix": largest_tile == n * n,
    }


@torch.no_grad()
def sdpa_math_reference(q, k, v, causal=False):
    validate_inputs(q, k, v)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None],
            dropout_p=0.0, is_causal=causal,
        )[0, 0]


def make_inputs(n, d, value_dim, dtype, seed=28):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(n, d, generator=generator, dtype=dtype),
        torch.randn(n, d, generator=generator, dtype=dtype),
        torch.randn(n, value_dim, generator=generator, dtype=dtype),
    )


def run_case(n, d, value_dim, query_block, key_block, causal, dtype=torch.float64):
    q, k, v = make_inputs(n, d, value_dim, dtype)
    dense, weights = dense_attention(q, k, v, causal)
    tiled, counts = tiled_attention(q, k, v, query_block, key_block, causal)
    reference = sdpa_math_reference(q, k, v, causal)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    for actual in (dense, tiled):
        torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(n, dtype=dtype),
                               rtol=tolerance, atol=tolerance)
    if causal and torch.count_nonzero(weights.triu(diagonal=1)).item() != 0:
        raise AssertionError("Causal attention leaked future weights")
    if counts["score_elements_computed"] != n * n:
        raise AssertionError("This model must still compute all N*N query/key pairs")
    bytes_per_element = q.element_size()
    return {
        "n": n, "d": d, "value_dim": value_dim, "dtype": str(dtype), "causal": causal,
        "query_block": query_block, "key_block": key_block,
        "output_shape": list(tiled.shape), "reference_backend": "PyTorch SDPA MATH on CPU",
        "max_abs_error_dense_vs_sdpa": (dense - reference).abs().max().item(),
        "max_abs_error_tiled_vs_sdpa": (tiled - reference).abs().max().item(),
        "outputs_close": True, "attention_rows_sum_to_one": True,
        "score_storage": {
            "dense_one_score_matrix_elements": n * n,
            "dense_one_score_matrix_bytes": n * n * bytes_per_element,
            **counts,
            "largest_one_score_tile_bytes": counts["largest_score_tile_elements"] * bytes_per_element,
            "scope": "Individual score tensor sizes, NOT total live memory, allocator peak, or hardware traffic",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--value-dim", type=int, default=8)
    parser.add_argument("--query-block", type=int, default=16)
    parser.add_argument("--key-block", type=int, default=16)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/28_attention_online_softmax.json",
    )
    args = parser.parse_args()
    if not 1 <= args.n <= 512 or not 1 <= args.d <= 128 or not 1 <= args.value_dim <= 128:
        parser.error("Use 1 <= n <= 512 and 1 <= d,value-dim <= 128 for this CPU lab")
    if not 1 <= args.query_block <= 512 or not 1 <= args.key_block <= 512:
        parser.error("Block sizes must be between 1 and 512")
    torch.set_num_threads(1)
    dtype = getattr(torch, args.dtype)
    example = run_case(args.n, args.d, args.value_dim, args.query_block, args.key_block, args.causal, dtype)
    edge_cases = [
        run_case(n, d, dv, qb, kb, causal, dtype)
        for n, d, dv, qb, kb in ((1, 1, 1, 8, 8), (7, 3, 5, 3, 4), (17, 8, 4, 8, 5))
        for causal in (False, True)
    ]
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": "CPU single-head forward-only algorithm; not a FlashAttention GPU kernel",
        "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
        "environment": {
            "python": platform.python_version(), "architecture": platform.machine(),
            "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
        },
        "example": example, "edge_cases": edge_cases,
        "sequence_sweep": [run_case(n, 16, 8, 16, 16, False, dtype) for n in (16, 64, 128)],
        "limitations": ["no timing", "no training/backward/dropout", "no GPU IO measurement",
                        "no padding mask or fully masked rows", "not a full Transformer"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU attention only. GPU execution skipped; no timing or peak-memory claims.")
    print(f"Output shape: {example['output_shape']}; causal={args.causal}; dtype={dtype}")
    print(f"Dense vs SDPA max error: {example['max_abs_error_dense_vs_sdpa']:.3e}")
    print(f"Tiled vs SDPA max error: {example['max_abs_error_tiled_vs_sdpa']:.3e}")
    print("N | dense score elements | largest score tile | score pairs computed")
    for case in report["sequence_sweep"]:
        counts = case["score_storage"]
        print(f"{case['n']:3} | {counts['dense_one_score_matrix_elements']:20} | "
              f"{counts['largest_score_tile_elements']:18} | {counts['score_elements_computed']:20}")
    print(f"Six edge cases passed. Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
