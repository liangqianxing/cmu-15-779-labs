"""Compile existing CUDA teaching kernels and preserve functional/sanitizer logs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cuda_native_checks.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build/cuda"
    build.mkdir(parents=True, exist_ok=True)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    sanitizer = shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
    records = []
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "gpu_execution": "requested", "steps": records,
              "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "cuda").glob("*.cu")}}
    failed = False
    for command in (["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], [nvcc, "--version"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        records.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        failed |= result.returncode != 0
    for stem, checks in (("25_vector_add", ("memcheck",)), ("26_matmul_tiling", ("memcheck", "racecheck", "synccheck"))):
        source, binary = ROOT / "cuda" / (stem+".cu"), build / stem
        commands = [[nvcc, "-O2", "-std=c++17", "-lineinfo", str(source), "-o", str(binary)], [str(binary)]]
        commands += [[sanitizer, "--tool", tool, "--error-exitcode", "1", str(binary)] for tool in checks]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            records.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            print(stem, command[0], result.returncode, flush=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            if result.returncode:
                failed = True
                break
    report["status"] = "failed" if failed else "passed"
    report["scope"] = "Actual native CUDA correctness and named sanitizer checks; no native kernel timing"
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
