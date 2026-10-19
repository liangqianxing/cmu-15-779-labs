"""Actual CUDA/Triton correctness and event timing; requires a CUDA host."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl


@triton.jit
def add_kernel(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A+offsets, mask=mask, other=0)
    b = tl.load(B+offsets, mask=mask, other=0)
    tl.store(C+offsets, a+b, mask=mask)


@triton.jit
def reduce_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X+offsets, mask=offsets < N, other=0)
    total = tl.sum(values, axis=0)
    tl.store(Y+tl.program_id(0), total)


@triton.jit
def matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(tl.cdiv(K, BK)):
        ks = base*BK+inner
        a = tl.load(A+rows[:, None]*K+ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B+ks[:, None]*N+cols[None, :], (ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(C+rows[:, None]*N+cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def triton_add(a, b, block=256):
    output = torch.empty_like(a)
    if a.numel():
        add_kernel[(triton.cdiv(a.numel(), block),)](a, b, output, a.numel(), block)
    return output


def triton_sum(x, block=1024):
    if not x.numel():
        return torch.zeros((), device=x.device, dtype=x.dtype)
    count = triton.cdiv(x.numel(), block)
    partial = torch.empty(count, device=x.device, dtype=x.dtype)
    reduce_kernel[(count,)](x, partial, x.numel(), block)
    output = torch.empty(1, device=x.device, dtype=x.dtype)
    reduce_kernel[(1,)](partial, output, count, triton.next_power_of_2(count))
    return output[0]


def triton_mm(a, b, config):
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=torch.float32)
    bm, bn, bk = config
    matmul_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](a, b, output, m, n, k, bm, bn, bk)
    return output


def event_samples(fn, repeats=15, inner=10):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/inner)
    return {"event_ms_samples": samples, "median_event_ms": statistics.median(samples),
            "inner_calls": inner, "warmup_calls": 5,
            "scope": "CUDA stream elapsed time per call, includes submitted GPU work; allocations/Python may cause GPU idle gaps. Excludes initial JIT and host/device transfers. Not end-to-end service latency."}


def run():
    torch.manual_seed(38)
    torch.cuda.manual_seed_all(38)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    adds, sums, matrix, attention = [], [], [], []
    for n in (0, 1, 17, 257, 1048576):
        a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        for block in (256, 1024):
            actual = triton_add(a, b, block)
            torch.testing.assert_close(actual, a+b, atol=0, rtol=0)
            record = {"n": n, "block": block, "correct": True}
            if n == 1048576:
                record.update(event_samples(lambda: triton_add(a, b, block)))
            adds.append(record)
        actual = triton_sum(a)
        expected = a.double().sum()
        torch.testing.assert_close(actual.double(), expected, atol=1e-3, rtol=3e-5)
        record = {"n": n, "actual": actual.item(), "float64_reference": expected.item(),
                  "max_abs_error": abs(actual.item()-expected.item()), "correct": True}
        if n == 1048576:
            record["triton"] = event_samples(lambda: triton_sum(a))
            record["torch"] = event_samples(lambda: a.sum())
        sums.append(record)
    for m, k, n in ((1, 1, 1), (97, 65, 129), (512, 512, 512), (1024, 1024, 1024)):
        a, b = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
        expected = a @ b
        for config in ((32, 32, 32), (64, 64, 32)):
            actual = triton_mm(a, b, config)
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
            record = {"m": m, "k": k, "n": n, "config": config,
                      "input_precision": "float32 IEEE; TF32 disabled for torch reference",
                      "max_abs_error": (actual-expected).abs().max().item(), "correct": True}
            if m >= 512:
                record["triton"] = event_samples(lambda: triton_mm(a, b, config), inner=5)
                record["torch"] = event_samples(lambda: a @ b, inner=5)
            matrix.append(record)
    for n in (64, 129, 512):
        q, k, v = [torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
        for causal in (False, True):
            scores = (q.float() @ k.float().transpose(-2, -1))/8
            if causal:
                scores.masked_fill_(torch.ones(n, n, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
            expected = torch.softmax(scores, -1) @ v.float()
            outputs = {}
            for name, backend in (("math", SDPBackend.MATH), ("flash", SDPBackend.FLASH_ATTENTION)):
                with sdpa_kernel(backend):
                    fn = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=causal)
                    actual = fn()
                    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=3e-3)
                    outputs[name] = {"max_abs_error_vs_float32_dense": (actual.float()-expected).abs().max().item(),
                                     **event_samples(fn)}
            attention.append({"shape": [2, 4, n, 64], "causal": causal, "dtype": "float16",
                              "explicit_backends": outputs, "correct": True})
    return {"vector_add": adds, "reduction": sums, "matmul": matrix, "attention": attention}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/38_gpu_kernels.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: GPU execution was requested, so this is not a passing skip")
    properties = torch.cuda.get_device_properties(0)
    started = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
                              "cuda_runtime": torch.version.cuda, "gpu": properties.name,
                              "compute_capability": [properties.major, properties.minor], "memory_bytes": properties.total_memory},
              "gpu_execution": "executed", "wall_seconds_including_jit": time.perf_counter()-started,
              "limits": "One GPU, synthetic tensors; no training convergence, multi-GPU, full LLM serving, hand-written FlashAttention or megakernel execution", **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PASS: Triton vector add/reduction/matmul and explicit CUDA SDPA MATH/FLASH", flush=True)
    print(report["environment"], flush=True)


if __name__ == "__main__":
    main()
