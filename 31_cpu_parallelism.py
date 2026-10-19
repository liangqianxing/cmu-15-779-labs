"""CPU parallelism lab: weighted gradient aggregation, tensor partitions, optional Gloo."""

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
from zoneinfo import ZoneInfo

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp


def make_problem():
    torch.manual_seed(31)
    model = nn.Linear(3, 2, dtype=torch.float64)
    x = torch.randn(7, 3, dtype=torch.float64)
    target = torch.randn(7, 2, dtype=torch.float64)
    return model, x, target


def model_copy(state):
    model = nn.Linear(3, 2, dtype=torch.float64)
    model.load_state_dict(state)
    return model


def reference_gradients(state, x, target):
    model = model_copy(state)
    nn.functional.mse_loss(model(x), target, reduction="mean").backward()
    return [parameter.grad.clone() for parameter in model.parameters()]


def aggregate_gradients(state, x, target, world_size=2, weighted=True):
    if not isinstance(world_size, int) or not 1 <= world_size <= len(x):
        raise ValueError("world_size must be between 1 and sample count")
    accumulated = [torch.zeros_like(parameter) for parameter in state.values()]
    sizes = []
    for shard_x, shard_y in zip(torch.tensor_split(x, world_size), torch.tensor_split(target, world_size)):
        local = reference_gradients(state, shard_x, shard_y)
        sizes.append(len(shard_x))
        factor = len(shard_x) / len(x) if weighted else 1.0 / world_size
        for total, gradient in zip(accumulated, local):
            total.add_(gradient, alpha=factor)
    return accumulated, sizes


def tensor_partitions(a, b, parts=2):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible 2D matrices")
    if not isinstance(parts, int) or parts < 1:
        raise ValueError("parts must be a positive integer")
    column = torch.cat([a @ block for block in torch.tensor_split(b, parts, dim=1)], dim=1)
    row = sum((x @ w for x, w in zip(torch.tensor_split(a, parts, dim=1),
                                    torch.tensor_split(b, parts, dim=0))),
              torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device))
    return column, row


def zero_storage_model(parameters, world_size):
    if any(not isinstance(value, int) or value < 1 for value in (parameters, world_size)):
        raise ValueError("parameters and world_size must be positive integers")
    shard = math.ceil(parameters / world_size)
    # Assumed mixed-precision Adam: params=2, grads=2, master+moments=12 B/parameter.
    return {
        "parameters": parameters, "world_size": world_size,
        "per_rank_bytes": {"replicated": 16 * parameters,
                           "zero1": 4 * parameters + 12 * shard,
                           "zero2": 2 * parameters + 14 * shard,
                           "zero3": 16 * shard},
        "scope": "Analytical persistent model states with padded equal shards; excludes activations, temporary gathers, communication buffers, allocator and offload",
        "bytes_per_parameter_assumption": {"parameters": 2, "gradients": 2, "optimizer_master_and_moments": 12},
    }


def gloo_worker(rank, world_size, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=30))
    try:
        model, x, target = make_problem()
        state = model.state_dict()
        expected = reference_gradients(state, x, target)
        shard_x = torch.tensor_split(x, world_size)[rank]
        shard_y = torch.tensor_split(target, world_size)[rank]
        local_loss = nn.functional.mse_loss(model(shard_x), shard_y, reduction="mean")
        (local_loss * len(shard_x) / len(x)).backward()
        errors = []
        updated = []
        for parameter, reference in zip(model.parameters(), expected):
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(parameter.grad, reference, atol=1e-12, rtol=1e-12)
            errors.append((parameter.grad - reference).abs().max().item())
            with torch.no_grad():
                parameter.add_(parameter.grad, alpha=-0.1)
            updated.append(parameter.detach().clone())
        # state_dict shares model storage, so compare against a fresh initial model.
        fresh, _, _ = make_problem()
        for actual, initial, reference in zip(updated, fresh.parameters(), expected):
            torch.testing.assert_close(actual, initial - 0.1 * reference, atol=1e-12, rtol=1e-12)
        record = {"rank": rank, "samples": len(shard_x), "backend": "gloo", "device": "cpu",
                  "gradients_close": True, "sgd_update_close": True, "max_abs_error": max(errors)}
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(record) + "\n")
    finally:
        dist.destroy_process_group()


def run_gloo():
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("This run requested Gloo, but this PyTorch build lacks the backend")
    with tempfile.TemporaryDirectory(prefix="cmu15779-gloo-") as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        mp.spawn(gloo_worker, args=(2, rendezvous, directory), nprocs=2, join=True)
        records = [json.loads((Path(directory) / f"rank-{rank}.json").read_text()) for rank in range(2)]
    return {"status": "executed", "world_size": 2, "scope": "Two real CPU processes on one host; no multi-host or GPU communication measured", "ranks": records}


def run_local():
    model, x, target = make_problem()
    expected = reference_gradients(model.state_dict(), x, target)
    weighted, sizes = aggregate_gradients(model.state_dict(), x, target, 2)
    unweighted, _ = aggregate_gradients(model.state_dict(), x, target, 2, weighted=False)
    for actual, reference in zip(weighted, expected):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    unequal_error = max((actual - reference).abs().max().item() for actual, reference in zip(unweighted, expected))
    if unequal_error <= 1e-10:
        raise AssertionError("Unequal-shard example must expose naive rank averaging")
    generator = torch.Generator().manual_seed(3101)
    a = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    b = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    column, row = tensor_partitions(a, b, 2)
    reference = a @ b
    for actual in (column, row):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    return {"data_parallel": {"shard_samples": sizes, "weighted_gradients_close": True,
                              "weighted_max_abs_error": max((a - b).abs().max().item() for a, b in zip(weighted, expected)),
                              "naive_rank_average_max_abs_error": unequal_error},
            "tensor_parallel": {"shape": [7, 5, 9], "parts": 2,
                                "column_concat_max_abs_error": (column - reference).abs().max().item(),
                                "row_sum_max_abs_error": (row - reference).abs().max().item(),
                                "execution": "Serial CPU partition model; not parallel communication"},
            "zero_storage": zero_storage_model(1_000_000, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gloo", action="store_true", help="Run two real CPU processes and all-reduce")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/31_cpu_parallelism.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "architecture": platform.machine(), "device": "cpu"},
              "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
              **run_local(), "gloo": run_gloo() if args.gloo else {"status": "not_requested"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU data/tensor parallelism experiment; GPU execution skipped.")
    print("Data parallel gradients:", report["data_parallel"])
    print("Tensor partitions:", report["tensor_parallel"])
    print("Analytical per-rank state bytes:", report["zero_storage"]["per_rank_bytes"])
    print("Gloo:", report["gloo"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
