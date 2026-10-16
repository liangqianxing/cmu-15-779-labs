"""CPU model of tiled matmul: count logical loads, not GPU time or DRAM traffic."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch


EDGE_CASES = (
    (0, 5, 3, 4), (3, 5, 0, 4), (3, 0, 5, 4),
    (1, 1, 1, 4), (4, 4, 4, 2), (3, 5, 7, 4),
    (7, 3, 5, 2), (17, 9, 13, 8),
)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def naive_matmul(a, b, m, k, n):
    output = [[float("nan")] * n for _ in range(m)]
    counts = {"a_loads": 0, "b_loads": 0, "output_stores": 0, "multiply_adds": 0}
    for row in range(m):
        for col in range(n):
            total = 0.0
            for inner in range(k):
                total += a[row][inner] * b[inner][col]
                counts["a_loads"] += 1
                counts["b_loads"] += 1
                counts["multiply_adds"] += 1
            output[row][col] = total
            counts["output_stores"] += 1
    return output, counts


def tiled_matmul(a, b, m, k, n, tile):
    output = [[float("nan")] * n for _ in range(m)]
    visits = [[0] * n for _ in range(m)]
    counts = {
        "a_loads": 0, "b_loads": 0, "output_stores": 0,
        "multiply_adds": 0, "zero_fills": 0, "block_barrier_phases": 0,
    }
    for block_y in range(ceil_div(m, tile)):
        for block_x in range(ceil_div(n, tile)):
            accum = [[0.0] * tile for _ in range(tile)]
            for base_k in range(0, k, tile):
                shared_a = [[0.0] * tile for _ in range(tile)]
                shared_b = [[0.0] * tile for _ in range(tile)]
                # Phase 1: every modeled thread cooperatively loads two slots.
                for ty in range(tile):
                    for tx in range(tile):
                        row, col = block_y * tile + ty, block_x * tile + tx
                        if row < m and base_k + tx < k:
                            shared_a[ty][tx] = a[row][base_k + tx]
                            counts["a_loads"] += 1
                        else:
                            counts["zero_fills"] += 1
                        if base_k + ty < k and col < n:
                            shared_b[ty][tx] = b[base_k + ty][col]
                            counts["b_loads"] += 1
                        else:
                            counts["zero_fills"] += 1
                counts["block_barrier_phases"] += 1

                # Phase 2: all loads precede use; this is not a GPU scheduler.
                for ty in range(tile):
                    for tx in range(tile):
                        for inner in range(tile):
                            accum[ty][tx] += shared_a[ty][inner] * shared_b[inner][tx]
                            counts["multiply_adds"] += 1
                # Finish using this tile before the next phase overwrites it.
                counts["block_barrier_phases"] += 1

            for ty in range(tile):
                for tx in range(tile):
                    row, col = block_y * tile + ty, block_x * tile + tx
                    if row < m and col < n:
                        output[row][col] = accum[ty][tx]
                        visits[row][col] += 1
                        counts["output_stores"] += 1
    if any(count != 1 for row in visits for count in row):
        raise AssertionError("Each output must have exactly one writer")
    return output, counts


def run_case(m, k, n, tile):
    if any(not isinstance(size, int) or not 0 <= size <= 64 for size in (m, k, n)):
        raise ValueError("m, k, n must be integers between 0 and 64")
    if not isinstance(tile, int) or not 1 <= tile <= 16:
        raise ValueError("tile must be an integer between 1 and 16")

    # Binary fractions make these fixtures exact in both float32 and float64.
    a = ((torch.arange(m * k, dtype=torch.float64) % 13 - 6) / 4).reshape(m, k)
    b = ((torch.arange(k * n, dtype=torch.float64) % 11 - 5) / 8).reshape(k, n)
    reference = a @ b
    naive, naive_counts = naive_matmul(a.tolist(), b.tolist(), m, k, n)
    tiled, tiled_counts = tiled_matmul(a.tolist(), b.tolist(), m, k, n, tile)
    for result in (naive, tiled):
        actual = torch.tensor(result, dtype=torch.float64).reshape(m, n)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    blocks = ceil_div(m, tile) * ceil_div(n, tile)
    phases = blocks * ceil_div(k, tile)
    expected_a = m * k * ceil_div(n, tile)
    expected_b = k * n * ceil_div(m, tile)
    expected_naive = {
        "a_loads": m * k * n, "b_loads": m * k * n,
        "output_stores": m * n, "multiply_adds": m * k * n,
    }
    expected_tiled = {
        "a_loads": expected_a, "b_loads": expected_b,
        "output_stores": m * n, "multiply_adds": phases * tile ** 3,
        "zero_fills": phases * 2 * tile ** 2 - expected_a - expected_b,
        "block_barrier_phases": phases * 2,
    }
    if naive_counts != expected_naive or tiled_counts != expected_tiled:
        raise AssertionError("Instrumented counts disagree with analytical counts")
    naive_loads = naive_counts["a_loads"] + naive_counts["b_loads"]
    tiled_loads = tiled_counts["a_loads"] + tiled_counts["b_loads"]
    return {
        "shape": {"m": m, "k": k, "n": n}, "tile": tile,
        "output_blocks": blocks, "launched_threads": blocks * tile ** 2,
        "useful_multiply_adds": m * k * n,
        "naive": {**naive_counts, "global_input_loads": naive_loads},
        "tiled": {**tiled_counts, "global_input_loads": tiled_loads},
        "logical_load_reduction_factor": naive_loads / tiled_loads if tiled_loads else None,
        "matches_torch_exactly": True, "every_output_written_once": True,
        "counts_match_formulas": True,
        "output_preview": [row[:8] for row in tiled[:8]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--tile", type=int, default=2)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/26_matmul_tiling.json",
    )
    args = parser.parse_args()
    try:
        example = run_case(args.m, args.k, args.n, args.tile)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parent
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_source_sha256": hashlib.sha256((root / "cuda/26_matmul_tiling.cu").read_bytes()).hexdigest(),
        "execution": "CPU algorithm model; no CUDA compilation, execution, or timing",
        "counting_scope": "Logical scalar A/B reads; excludes C stores, caches, coalescing, DRAM transactions",
        "cuda_execution_verified": False,
        "environment": {
            "architecture": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "torch_cuda_build": torch.version.cuda,
            "mps_available": torch.backends.mps.is_available(),
        },
        "example": example,
        "tile_sweep": [run_case(8, 8, 8, tile) for tile in (1, 2, 4, 8)],
        "edge_cases": [run_case(*case) for case in EDGE_CASES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU model only: logical scalar loads, NOT DRAM traffic or GPU speed.")
    print(f"A[{args.m},{args.k}] @ B[{args.k},{args.n}], tile={args.tile}")
    for mode in ("naive", "tiled"):
        counts = example[mode]
        print(f"{mode:>5}: A/B loads={counts['global_input_loads']}, "
              f"C stores={counts['output_stores']}, multiply-adds={counts['multiply_adds']}")
    print("8x8 tile sweep:")
    for row in report["tile_sweep"]:
        print(f"  tile={row['tile']}: {row['tiled']['global_input_loads']} input loads, "
              f"{row['logical_load_reduction_factor']:.1f}x fewer than naive")
    print(f"Checks: exact outputs, one writer, count formulas, {len(EDGE_CASES)} edge cases passed.")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
