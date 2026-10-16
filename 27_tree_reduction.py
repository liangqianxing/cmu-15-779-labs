"""CPU reduction model: work and dependency depth, not GPU execution or speed."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from zoneinfo import ZoneInfo


def serial_sum(values):
    if not values:
        return 0.0
    total = values[0]
    for value in values[1:]:
        total += value
    return total


def tree_reduce(values, keep_levels=False):
    current = list(values)
    levels = [current.copy()] if keep_levels else []
    additions = 0
    depth = 0
    while len(current) > 1:
        following = []
        for index in range(0, len(current), 2):
            if index + 1 < len(current):
                following.append(current[index] + current[index + 1])
                additions += 1
            else:
                following.append(current[index])
        current = following
        depth += 1
        if keep_levels:
            levels.append(current.copy())
    return {
        "sum": current[0] if current else 0.0,
        "additions": additions, "dependency_depth": depth, "levels": levels,
    }


def staged_reduce(values, block_size):
    if not isinstance(block_size, int) or block_size < 2:
        raise ValueError("block_size must be an integer >= 2")
    current = list(values)
    rounds = []
    additions = 0
    while len(current) > 1:
        partials = []
        max_depth = 0
        for start in range(0, len(current), block_size):
            block = tree_reduce(current[start:start + block_size])
            partials.append(block["sum"])
            additions += block["additions"]
            max_depth = max(max_depth, block["dependency_depth"])
        rounds.append({
            "input_elements": len(current), "output_partials": len(partials),
            "max_block_dependency_depth": max_depth,
            "partial_preview": partials[:16],
        })
        current = partials
    return {
        "sum": current[0] if current else 0.0,
        "additions": additions, "logical_stages": len(rounds), "rounds": rounds,
    }


def run_case(n, block_size):
    if not isinstance(n, int) or not 0 <= n <= 65536:
        raise ValueError("n must be an integer between 0 and 65536")
    values = [float(index % 17 - 8) / 4 for index in range(n)]
    tree = tree_reduce(values)
    staged = staged_reduce(values, block_size)
    reference = math.fsum(values)
    expected_additions = max(0, n - 1)
    expected_depth = (n - 1).bit_length() if n > 1 else 0
    if not serial_sum(values) == tree["sum"] == staged["sum"] == reference:
        raise AssertionError("Reduction differs from the exact binary-fraction fixture sum")
    if tree["additions"] != expected_additions or staged["additions"] != expected_additions:
        raise AssertionError("Reduction must perform N-1 useful additions")
    if tree["dependency_depth"] != expected_depth:
        raise AssertionError("Tree depth must be ceil(log2(N)) for N > 0")
    return {
        "n": n, "block_size": block_size, "sum": reference,
        "serial_additions": expected_additions, "serial_dependency_depth": expected_additions,
        "tree": tree, "staged": staged, "fixture_outputs_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=257)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/27_tree_reduction.json",
    )
    args = parser.parse_args()
    try:
        example = run_case(args.n, args.block_size)
    except ValueError as error:
        parser.error(str(error))
    cancellation = [1e16, 1.0, -1e16, 1.0]
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "architecture": platform.machine()},
        "execution": "Serial CPU simulation; dependency counts are not timings",
        "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
        "example": example,
        "lecture_example": tree_reduce([3, 1, 7, 0, 4, 1, 6, 3], keep_levels=True),
        "edge_cases": [run_case(n, size) for n, size in ((0, 4), (1, 4), (3, 2), (8, 4), (9, 4), (17, 32))],
        "floating_point_order_example": {
            "input": cancellation, "serial": serial_sum(cancellation),
            "tree": tree_reduce(cancellation)["sum"], "math_fsum": math.fsum(cancellation),
            "note": "Changing addition order can change floating-point results; exact fixtures are not a general guarantee",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU model only. GPU execution skipped; no performance claim.")
    print("Lecture tree:", report["lecture_example"]["levels"])
    print(f"N={args.n}, sum={example['sum']}, additions={example['serial_additions']}")
    print(f"Serial dependency depth={example['serial_dependency_depth']}; "
          f"tree depth={example['tree']['dependency_depth']}")
    print("Partial-reduction stages:", example["staged"]["rounds"])
    print("Floating-point ordering:", report["floating_point_order_example"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
