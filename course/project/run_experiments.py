"""Reproducible CPU study of KV block granularity, sharing and cache operations.

Uses synthetic, fixed K/V and Q tensors. Times the complete cache setup, appends,
gather and attention path; excludes input generation and correctness assertions.
"""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("paged", ROOT / "33_paged_kv_cache.py")
paged = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paged)

SCENARIOS = [
    {"name": "independent", "common": 0, "prompt_base": 31},
    {"name": "shared_3", "common": 3, "prompt_base": 3},
    {"name": "shared_31", "common": 31, "prompt_base": 31},
    {"name": "shared_32", "common": 32, "prompt_base": 32},
    {"name": "shared_63", "common": 63, "prompt_base": 63},
]


def make_workload(scenario, seed, requests=8, width=16):
    generator = torch.Generator().manual_seed(seed)
    common = scenario["common"]
    prefix_k = torch.randn(common, width, generator=generator, dtype=torch.float64)
    prefix_v = torch.randn(common, width, generator=generator, dtype=torch.float64)
    sequences = []
    for request in range(requests):
        prompt = scenario["prompt_base"] + [0, 1, 3, 7][request % 4]
        decode = [1, 2, 4, 8][request % 4]
        unique = prompt + decode-common
        keys = torch.cat((prefix_k, torch.randn(unique, width, generator=generator, dtype=torch.float64)))
        values = torch.cat((prefix_v, torch.randn(unique, width, generator=generator, dtype=torch.float64)))
        queries = torch.randn(decode, width, generator=generator, dtype=torch.float64)
        sequences.append({"prompt": prompt, "decode": decode, "keys": keys, "values": values, "queries": queries})
    return {"name": scenario["name"], "common": common, "width": width, "sequences": sequences}


def dense_attention(query, keys, values):
    return torch.softmax(query @ keys.T/keys.shape[-1]**0.5, -1) @ values


def expected_outputs(workload):
    result = []
    for step in range(max(s["decode"] for s in workload["sequences"])):
        for sequence in workload["sequences"]:
            if step < sequence["decode"]:
                end = sequence["prompt"]+step+1
                result.append(dense_attention(sequence["queries"][step], sequence["keys"][:end], sequence["values"][:end]))
    return torch.stack(result)


def execute(workload, policy, block_size=4):
    sequences, common = workload["sequences"], workload["common"]
    slot_bytes = 2*workload["width"]*8
    outputs = []
    if policy == "static":
        maximum = max(s["prompt"]+s["decode"] for s in sequences)
        keys = torch.empty(len(sequences), maximum, workload["width"], dtype=torch.float64)
        values = torch.empty_like(keys)
        for i, s in enumerate(sequences):
            keys[i, :s["prompt"]] = s["keys"][:s["prompt"]]
            values[i, :s["prompt"]] = s["values"][:s["prompt"]]
        peak_slots = len(sequences)*maximum
        copied = 0
        for step in range(max(s["decode"] for s in sequences)):
            for i, s in enumerate(sequences):
                if step < s["decode"]:
                    index = s["prompt"]+step
                    keys[i, index] = s["keys"][index]
                    values[i, index] = s["values"][index]
                    outputs.append(dense_attention(s["queries"][step], keys[i, :index+1], values[i, :index+1]))
        metrics = dict(active_slot_bytes=peak_slots*slot_bytes, pool_tensor_bytes=peak_slots*slot_bytes,
                       copied_old_token_slots=copied, page_count=None, metadata_page_entries=0)
    elif policy == "dynamic":
        keys = [s["keys"][:s["prompt"]].clone() for s in sequences]
        values = [s["values"][:s["prompt"]].clone() for s in sequences]
        copied = 0
        for step in range(max(s["decode"] for s in sequences)):
            for i, s in enumerate(sequences):
                if step < s["decode"]:
                    index = s["prompt"]+step
                    copied += len(keys[i])  # Old K/V paired token slots moved by cat.
                    keys[i] = torch.cat((keys[i], s["keys"][index:index+1]))
                    values[i] = torch.cat((values[i], s["values"][index:index+1]))
                    outputs.append(dense_attention(s["queries"][step], keys[i], values[i]))
        slots = sum(len(k) for k in keys)
        metrics = dict(active_slot_bytes=slots*slot_bytes, pool_tensor_bytes=slots*slot_bytes,
                       copied_old_token_slots=copied, page_count=None, metadata_page_entries=0)
    elif policy in ("paged", "shared"):
        capacity = sum(math.ceil((s["prompt"]+s["decode"])/block_size) for s in sequences)
        cache = paged.PagedKVCache(capacity, block_size, workload["width"], workload["width"])
        for i, s in enumerate(sequences):
            if policy == "shared" and i and common:
                cache.fork("0", str(i), common)
                start = common
            else:
                cache.start(str(i))
                start = 0
            for index in range(start, s["prompt"]):
                cache.append(str(i), s["keys"][index], s["values"][index])
        for step in range(max(s["decode"] for s in sequences)):
            for i, s in enumerate(sequences):
                if step < s["decode"]:
                    index = s["prompt"]+step
                    cache.append(str(i), s["keys"][index], s["values"][index])
                    outputs.append(cache.attend(str(i), s["queries"][step]))
        stats = cache.stats()
        metrics = dict(active_slot_bytes=stats["active_page_bytes"], pool_tensor_bytes=stats["pool_tensor_bytes"],
                       copied_old_token_slots=stats["copied_token_slots"], page_count=stats["active_pages"],
                       metadata_page_entries=sum(len(t) for t in cache.tables.values()))
    else:
        raise ValueError("Unknown policy")
    # All sequences remain retained until the end. Page count is monotone: COW
    # of a shared page keeps that old page live elsewhere, or allocates a new one.
    metrics["logical_token_references"] = sum(s["prompt"]+s["decode"] for s in sequences)
    return torch.stack(outputs), metrics


def run(seeds, repeats, blocks):
    policies = [("static", None), ("dynamic", None)] + [(p, b) for p in ("paged", "shared") for b in blocks]
    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            workload = make_workload(scenario, seed)
            expected = expected_outputs(workload)
            records = {}
            for policy, block in policies:
                actual, metrics = execute(workload, policy, block)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                records[policy, block] = {"policy": policy, "block_size": block,
                    "scenario": scenario["name"], "seed": seed, "common_tokens": scenario["common"],
                    "prompt_lengths": [s["prompt"] for s in workload["sequences"]],
                    "decode_lengths": [s["decode"] for s in workload["sequences"]],
                    "max_abs_error": (actual-expected).abs().max().item(), "elapsed_ms": [], **metrics}
            order = policies.copy()
            rng = random.Random(seed)
            for _ in range(repeats):
                rng.shuffle(order)
                for policy, block in order:
                    start = time.perf_counter_ns()
                    actual, metrics = execute(workload, policy, block)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    assert metrics["active_slot_bytes"] == records[policy, block]["active_slot_bytes"]
                    records[policy, block]["elapsed_ms"].append(elapsed)
            for record in records.values():
                record["median_ms"] = statistics.median(record["elapsed_ms"])
                rows.append(record)
            print(scenario["name"], "seed", seed, "correct:", len(policies), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results/project_kv_granularity.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    seeds, blocks = [41, 42, 43], [2, 4, 8, 16, 32, 64]
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "machine": platform.machine(), "cpu_threads": torch.get_num_threads(), "device": "cpu"},
              "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "33_paged_kv_cache.py")},
              "gpu_status": "skipped_by_user_request", "dtype": "float64", "key_width": 16, "value_width": 16,
              "seeds": seeds, "blocks": blocks, "repeats": args.repeats, "warmup_runs_per_configuration": 1,
              "timing_scope": "Wall time including allocation, prompt insertion, fork, append, gather, attention, output stack and metrics; excludes input generation and assertions. Policies shuffled each repetition.",
              "storage_scope": "Retained KV tensor slots; excludes Q/input tensors, transient gather/cat/attention tensors, Python metadata and allocator. No RSS/peak-memory measurement. All requests retained until end.",
              "workload_scope": "Synthetic fixed single-head K/V/Q, eight requests, supplied identical prefix where applicable; no full LLM, tokenization, radix tree, online arrival/admission or throughput evidence",
              "rows": run(seeds, args.repeats, blocks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Saved", args.output, "rows", len(report["rows"]))


if __name__ == "__main__":
    main()
