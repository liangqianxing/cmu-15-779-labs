"""CPU model of CUDA thread indexing; this does not execute a GPU kernel."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch


def simulate_vector_add(n, threads_per_block):
    if n < 0 or threads_per_block <= 0:
        raise ValueError("n must be nonnegative and threads_per_block must be positive")

    a = torch.arange(n, dtype=torch.float32)
    b = torch.full((n,), 10.0)
    output = torch.full((n,), float("nan"))
    visits = torch.zeros(n, dtype=torch.int64)
    blocks = (n + threads_per_block - 1) // threads_per_block
    mapping = []

    # These loops describe the mapping; real CUDA does not schedule threads this way.
    for block_idx in range(blocks):
        for thread_idx in range(threads_per_block):
            i = block_idx * threads_per_block + thread_idx
            active = i < n
            mapping.append({
                "block_idx": block_idx,
                "thread_idx": thread_idx,
                "global_index": i,
                "writes_output": active,
            })
            if active:
                output[i] = a[i] + b[i]
                visits[i] += 1

    # Check both the values and whether each output has exactly one writer.
    torch.testing.assert_close(output, a + b, rtol=0, atol=0)
    if not torch.all(visits == 1).item():
        raise AssertionError("Some output elements were skipped or written more than once")
    return {
        "n": n,
        "threads_per_block": threads_per_block,
        "blocks": blocks,
        "launched_threads": blocks * threads_per_block,
        "guarded_threads": blocks * threads_per_block - n,
        "outputs_exact": True,
        "every_element_written_once": True,
        "output": output.tolist(),
        "mapping": mapping,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/25_cuda_indexing.json",
    )
    args = parser.parse_args()
    if not 0 <= args.n <= 4096 or not 1 <= args.threads <= 1024:
        parser.error("For this small demonstration: 0 <= n <= 4096, 1 <= threads <= 1024")

    result = simulate_vector_add(args.n, args.threads)
    edge_cases = []
    for n, threads in ((0, 4), (1, 4), (8, 4), (10, 4), (17, 32), (257, 128)):
        case = simulate_vector_add(n, threads)
        edge_cases.append({key: value for key, value in case.items() if key not in ("mapping", "output")})

    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": "CPU indexing model; no CUDA kernel executed",
        "environment": {
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_build": torch.version.cuda,
            "mps_available": torch.backends.mps.is_available(),
        },
        "cuda_execution_verified": False,
        "example": result,
        "edge_cases": edge_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("CPU indexing model, NOT a CUDA performance measurement.")
    print(f"N={args.n}, threads/block={args.threads}, blocks={result['blocks']}")
    print(f"{'blockIdx.x':>11} {'threadIdx.x':>12} {'i':>5} {'action':>10}")
    for item in result["mapping"][:40]:
        action = "write" if item["writes_output"] else "skip"
        print(f"{item['block_idx']:>11} {item['thread_idx']:>12} {item['global_index']:>5} {action:>10}")
    if len(result["mapping"]) > 40:
        print("Only the first 40 threads are displayed; the JSON contains the full mapping.")
    print("Output (first 20):", result["output"][:20])
    print("Checks: exact output, one writer per element, six edge cases passed.")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
