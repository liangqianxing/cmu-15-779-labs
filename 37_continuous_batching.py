"""Decode-only fixed-cohort vs continuous batching under a unit-step cost model."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def simulate(requests, batch_size=2, policy="continuous"):
    if policy not in ("fixed", "continuous") or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Expected fixed/continuous and positive batch_size")
    if any(not isinstance(r["arrival"], int) or r["arrival"] < 0 or not isinstance(r["tokens"], int) or r["tokens"] < 1 for r in requests):
        raise ValueError("Requests need integer arrival >=0 and tokens >0")
    pending = sorted([dict(r, request_id=i, remaining=r["tokens"]) for i, r in enumerate(requests)], key=lambda r: (r["arrival"], r["request_id"]))
    active, completed, trace = [], [], []
    time = 0
    while pending or active:
        can_admit = policy == "continuous" or not active
        if can_admit:
            while pending and pending[0]["arrival"] <= time and len(active) < batch_size:
                request = pending.pop(0)
                request["admitted"] = time
                active.append(request)
        if not active:
            time = pending[0]["arrival"]
            continue
        trace.append({"time": time, "request_ids": [r["request_id"] for r in active]})
        for request in active:
            request.setdefault("first_token", time+1)
            request["remaining"] -= 1
        time += 1
        survivors = []
        for request in active:
            if request["remaining"] == 0:
                request["finished"] = time
                request["latency"] = time-request["arrival"]
                request["time_to_first_token"] = request["first_token"]-request["arrival"]
                completed.append(request)
            else:
                survivors.append(request)
        active = survivors
    completed.sort(key=lambda r: r["request_id"])
    return {"policy": policy, "makespan_units": time, "requests": completed, "trace": trace,
            "decode_tokens": sum(r["tokens"] for r in requests),
            "scope": "Synthetic one unit per decode iteration independent of batch/context size; FIFO, no prefill, no KV capacity or measured hardware latency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/37_continuous_batching.json")
    args = parser.parse_args()
    requests = [dict(arrival=0, tokens=n) for n in (1, 4, 1, 4)]
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gpu_status": "skipped_by_user_request", "input_requests": requests,
              "schedules": [simulate(requests, 2, p) for p in ("fixed", "continuous")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    for result in report["schedules"]:
        print(result["policy"], result["makespan_units"], "units; TTFT:", [r["time_to_first_token"] for r in result["requests"]])


if __name__ == "__main__":
    main()
