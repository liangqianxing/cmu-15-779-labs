"""CPU compiler lab: legal rewrites, measured schedule search, optional TorchInductor."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
from time import perf_counter
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.utils.benchmark import Timer


def fold_linear_batchnorm(linear, batchnorm):
    if batchnorm.training or not batchnorm.track_running_stats:
        raise ValueError("Folding requires eval mode with fixed running statistics")
    if linear.out_features != batchnorm.num_features:
        raise ValueError("Linear outputs and BatchNorm features must match")
    folded = nn.Linear(linear.in_features, linear.out_features, bias=True,
                       dtype=linear.weight.dtype, device=linear.weight.device)
    with torch.no_grad():
        gamma = batchnorm.weight if batchnorm.affine else torch.ones_like(batchnorm.running_mean)
        beta = batchnorm.bias if batchnorm.affine else torch.zeros_like(batchnorm.running_mean)
        bias = linear.bias if linear.bias is not None else torch.zeros_like(batchnorm.running_mean)
        scale = gamma / torch.sqrt(batchnorm.running_var + batchnorm.eps)
        folded.weight.copy_(scale[:, None] * linear.weight)
        folded.bias.copy_((bias - batchnorm.running_mean) * scale + beta)
    return folded.eval()


@torch.no_grad()
def blocked_matmul(a, b, tile):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible 2D matrices")
    if not isinstance(tile, int) or tile <= 0:
        raise ValueError("tile must be a positive integer")
    m, k = a.shape
    n = b.shape[1]
    output = torch.zeros(m, n, dtype=a.dtype, device=a.device)
    for row in range(0, m, tile):
        for col in range(0, n, tile):
            for inner in range(0, k, tile):
                output[row:row + tile, col:col + tile].add_(
                    a[row:row + tile, inner:inner + tile] @ b[inner:inner + tile, col:col + tile])
    return output


def rmsnorm_linear(x, weight, gamma, eps=1e-6):
    inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return ((x * inverse_rms) * gamma) @ weight


def moved_rmsnorm_linear(x, preweighted, eps=1e-6):
    inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return (x @ preweighted) * inverse_rms


def measure(function):
    result = Timer(stmt="function()", globals={"function": function}, num_threads=1).blocked_autorange(min_run_time=0.04)
    return {"median_ms": result.median * 1000, "iqr_ms": result.iqr * 1000,
            "measurements": len(result.raw_times), "number_per_run": result.number_per_run}


@torch.no_grad()
def run_rewrites():
    torch.manual_seed(30)
    linear = nn.Linear(8, 6, dtype=torch.float64).eval()
    bn = nn.BatchNorm1d(6, dtype=torch.float64).eval()
    bn.running_mean.copy_(torch.randn(6, dtype=torch.float64))
    bn.running_var.copy_(torch.rand(6, dtype=torch.float64) + 0.1)
    bn.weight.copy_(torch.randn(6, dtype=torch.float64))
    bn.bias.copy_(torch.randn(6, dtype=torch.float64))
    x = torch.randn(17, 8, dtype=torch.float64)
    original = bn(linear(x))
    folded = fold_linear_batchnorm(linear, bn)(x)
    torch.testing.assert_close(folded, original, atol=1e-12, rtol=1e-12)

    weight = torch.randn(8, 5, dtype=torch.float64)
    gamma = torch.randn(8, dtype=torch.float64)
    reference = rmsnorm_linear(x, weight, gamma)
    rearranged = moved_rmsnorm_linear(x, gamma[:, None] * weight)
    torch.testing.assert_close(rearranged, reference, atol=1e-12, rtol=1e-12)
    bad_x = torch.tensor([[-1.0, 1.0]])
    bad_weight = torch.tensor([[-1.0], [1.0]])
    legal = torch.relu(bad_x @ bad_weight)
    illegal = torch.relu(bad_x) @ bad_weight
    if torch.allclose(legal, illegal):
        raise AssertionError("The intentionally invalid rewrite needs a counterexample")
    return {
        "linear_batchnorm_eval": {"max_abs_error": (folded - original).abs().max().item(),
                                  "module_operations_before": 2, "module_operations_after": 1},
        "rmsnorm_linear": {"max_abs_error": (rearranged - reference).abs().max().item(),
                           "precondition": "Fixed gamma/weight; per-row denominator moved across linear map, no intervening nonlinearity"},
        "rejected_relu_rewrite": {"relu_after_linear": legal.item(), "relu_before_linear": illegal.item()},
        "verification_scope": "Numerical fixtures plus explicit rule preconditions; not a formal equivalence proof",
    }


@torch.no_grad()
def run_schedule_search():
    generator = torch.Generator().manual_seed(3001)
    a = torch.randn(97, 65, generator=generator, dtype=torch.float64)
    b = torch.randn(65, 81, generator=generator, dtype=torch.float64)
    reference = a @ b
    tiles = [16, 32, 64, 128]
    random.Random(30).shuffle(tiles)
    records = []
    for tile in tiles:
        actual = blocked_matmul(a, b, tile)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        timing = measure(lambda: blocked_matmul(a, b, tile))
        records.append({"tile": tile, "outputs_close": True,
                        "max_abs_error": (actual - reference).abs().max().item(), **timing})
    winner = min(records, key=lambda record: record["median_ms"])
    return {"shape": {"m": 97, "k": 65, "n": 81}, "dtype": "float64",
            "candidate_execution": "Python blocked loops invoking PyTorch CPU matmul, not generated Triton/TVM kernels",
            "candidate_measurement_order": tiles, "candidates": records,
            "selected_tile": winner["tile"], "torch_matmul_baseline": measure(lambda: a @ b),
            "interpretation": "Winner applies only to this shape, CPU, threading and candidate set; no global optimality claim"}


def elementwise(x, bias):
    return torch.relu(x + bias) * 0.5


@torch.no_grad()
def run_inductor():
    generator = torch.Generator().manual_seed(3002)
    x = torch.randn(1024, 64, generator=generator)
    bias = torch.randn(64, generator=generator)
    compiled = torch.compile(elementwise, backend="inductor", fullgraph=True)
    start = perf_counter()
    first = compiled(x, bias)
    compile_and_first_call_ms = (perf_counter() - start) * 1000
    torch.testing.assert_close(first, elementwise(x, bias), rtol=1e-6, atol=1e-6)
    other = torch.randn(1024, 64, generator=generator)
    torch.testing.assert_close(compiled(other, bias), elementwise(other, bias), rtol=1e-6, atol=1e-6)
    return {"status": "executed", "backend": "inductor", "device": "cpu", "fullgraph": True,
            "compile_and_first_call_ms": compile_and_first_call_ms,
            "outputs_close_on_two_inputs": True, "eager": measure(lambda: elementwise(x, bias)),
            "compiled_warm": measure(lambda: compiled(x, bias)),
            "timing_scope": "Warm CPU calls measured separately; cold first call includes compilation/cache lookup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Also execute TorchInductor on CPU")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/30_cpu_compiler.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "architecture": platform.machine(), "device": "cpu", "threads": 1},
        "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
        "rewrites": run_rewrites(), "schedule_search": run_schedule_search(),
        "inductor": run_inductor() if args.compile else {"status": "not_requested"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("CPU compiler experiments. GPU compilation/execution skipped.")
    print("Legal rewrite errors:", report["rewrites"])
    print("Measured candidate winner:", report["schedule_search"]["selected_tile"])
    print("TorchInductor:", report["inductor"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
