"""Lecture 2 companion: vary batch size while keeping total work fixed."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128, 256, 1000]
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results/24_batch_size_sweep.json",
    )
    args = parser.parse_args()
    if args.samples <= 0 or any(size <= 0 for size in args.batch_sizes):
        parser.error("samples and batch sizes must be positive integers")
    args.batch_sizes = sorted(set(args.batch_sizes))
    return args


def main():
    args = parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = MyModel().eval()
    x = torch.linspace(-1.0, 1.0, args.samples).reshape(-1, 1)

    def predict_in_batches(batch_size):
        outputs = []
        for start in range(0, args.samples, batch_size):
            # The final slice may contain fewer than batch_size samples.
            outputs.append(model(x[start:start + batch_size]))
        # Every batch size uses the same output assembly procedure.
        return torch.cat(outputs, dim=0)

    rows = []
    with torch.no_grad():
        reference = model(x)
        for batch_size in args.batch_sizes:
            output = predict_in_batches(batch_size)
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
            max_error = (output - reference).abs().max().item()

            # Benchmark independently of the profiler, with one CPU thread.
            timing = Timer(
                stmt="predict_in_batches(batch_size)",
                globals={
                    "predict_in_batches": predict_in_batches,
                    "batch_size": batch_size,
                },
                num_threads=1,
            ).blocked_autorange(min_run_time=0.2)

            with profile(activities=[ProfilerActivity.CPU]) as prof:
                profiled_output = predict_in_batches(batch_size)
            torch.testing.assert_close(profiled_output, reference, rtol=1e-5, atol=1e-6)
            calls = {event.key: event.count for event in prof.key_averages()}
            batches = (args.samples + batch_size - 1) // batch_size
            expected_calls = {
                "aten::linear": 2 * batches,
                "aten::addmm": 2 * batches,
                "aten::relu": batches,
                "aten::slice": batches,
                "aten::cat": 1,
            }
            for operator, expected_count in expected_calls.items():
                actual = calls.get(operator, 0)
                if actual != expected_count:
                    raise RuntimeError(
                        f"batch_size={batch_size}, {operator}: "
                        f"expected {expected_count} calls, got {actual}"
                    )

            rows.append({
                "batch_size": batch_size,
                "batches": batches,
                "last_batch_size": args.samples - (batches - 1) * batch_size,
                "samples_processed": output.shape[0],
                "outputs_close": True,
                "max_abs_error": max_error,
                "operator_calls": {name: calls[name] for name in expected_calls},
                "median_total_ms": timing.median * 1000,
                "iqr_total_ms": timing.iqr * 1000,
                "samples_per_second": args.samples / timing.median,
                "timing_blocks": len(timing.raw_times),
                "runs_per_block": timing.number_per_run,
            })

    report = {
        "course": "CMU 15-779, Lecture 2 companion exercise",
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(x.device),
            "dtype": str(x.dtype),
            "cpu_threads": torch.get_num_threads(),
        },
        "setup": {
            "seed": 0,
            "samples": args.samples,
            "batch_sizes": args.batch_sizes,
            "model": "Linear(1, 8) -> ReLU -> Linear(8, 1)",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "grad_enabled_during_measurement": False,
            "timing_includes": "Python batching, slicing, forward passes, one cat",
            "timing_excludes": "input/model creation, profiler, output validation, file I/O",
            "measurement_unit": "one complete pass over all input samples",
            "min_timing_seconds_per_batch_size": 0.2,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Samples: {args.samples}, device: {x.device}, CPU threads: 1")
    print("All outputs and operator counts passed validation.")
    print(f"{'batch':>7} {'batches':>8} {'linear':>8} {'total ms':>12} {'samples/s':>14}")
    for row in rows:
        print(
            f"{row['batch_size']:>7} {row['batches']:>8} "
            f"{row['operator_calls']['aten::linear']:>8} "
            f"{row['median_total_ms']:>12.4f} {row['samples_per_second']:>14.0f}"
        )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
