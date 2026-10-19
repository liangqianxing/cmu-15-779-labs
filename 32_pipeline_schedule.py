"""Deterministic CPU dependency model of synchronous pipeline schedules and partition search."""

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


def stage_order(stage, stages, microbatches, policy):
    if policy == "gpipe":
        return [("F", m) for m in range(microbatches)] + [("B", m) for m in reversed(range(microbatches))]
    if policy != "1f1b":
        raise ValueError("policy must be gpipe or 1f1b")
    warmup = min(stages - stage - 1, microbatches)
    order = [("F", m) for m in range(warmup)]
    remaining = microbatches - warmup
    for m in range(remaining):
        order.extend([("F", warmup + m), ("B", m)])
    order.extend(("B", m) for m in range(remaining, microbatches))
    return order


def schedule_pipeline(stages=4, microbatches=8, forward=None, backward=None, policy="gpipe"):
    if any(not isinstance(n, int) or n < 1 for n in (stages, microbatches)):
        raise ValueError("stages and microbatches must be positive integers")
    forward = list(forward) if forward is not None else [1.0] * stages
    backward = list(backward) if backward is not None else [2.0] * stages
    if any(len(costs) != stages or any(not math.isfinite(c) or c <= 0 for c in costs)
           for costs in (forward, backward)):
        raise ValueError("Every stage needs a finite positive forward/backward cost")
    dependencies = {}
    for stage in range(stages):
        previous = None
        for phase, m in stage_order(stage, stages, microbatches, policy):
            task = (stage, phase, m)
            deps = set()
            if previous is not None:
                deps.add(previous)
            if phase == "F" and stage > 0:
                deps.add((stage - 1, "F", m))
            if phase == "B":
                deps.add((stage, "F", m))
                if stage < stages - 1:
                    deps.add((stage + 1, "B", m))
            dependencies[task] = deps
            previous = task
    finished = {}
    pending = dict(dependencies)
    tasks = []
    while pending:
        ready = sorted(task for task, deps in pending.items() if deps.issubset(finished))
        if not ready:
            raise AssertionError("Pipeline ordering introduced a dependency cycle")
        for task in ready:
            stage, phase, m = task
            start = max((finished[dep] for dep in dependencies[task]), default=0.0)
            finish = start + (forward if phase == "F" else backward)[stage]
            finished[task] = finish
            tasks.append({"stage": stage, "phase": phase, "microbatch": m,
                          "start": start, "finish": finish,
                          "dependencies": [list(dep) for dep in sorted(dependencies[task])]})
            del pending[task]
    makespan = max(finished.values())
    peak_activations = []
    for stage in range(stages):
        events = sorted((task["finish"], 1 if task["phase"] == "F" else -1)
                        for task in tasks if task["stage"] == stage)
        live = peak = 0
        for _, change in events:
            live += change
            if live < 0:
                raise AssertionError("Backward consumed an activation before forward produced it")
            peak = max(peak, live)
        if live != 0:
            raise AssertionError("Some activations were not consumed")
        peak_activations.append(peak)
    return {"policy": policy, "stages": stages, "microbatches": microbatches,
            "forward_costs": forward, "backward_costs": backward,
            "makespan_model_units": makespan,
            "stage_utilization": [microbatches * (f + b) / makespan for f, b in zip(forward, backward)],
            "peak_live_activation_units_per_stage": peak_activations,
            "weight_version": "All tasks use one fixed version; synchronous update only after the full schedule",
            "activation_scope": "One equal-size unit from forward completion through backward completion per microbatch; excludes backward temporary memory",
            "tasks": sorted(tasks, key=lambda task: (task["start"], task["stage"], task["phase"]))}


def partition_search(layer_costs, stages):
    if not layer_costs or any(not math.isfinite(v) or v <= 0 for v in layer_costs):
        raise ValueError("Layer costs must be finite positive numbers")
    if not isinstance(stages, int) or not 1 <= stages <= len(layer_costs):
        raise ValueError("stages must be between one and number of layers")
    candidates = []
    for cuts in itertools.combinations(range(1, len(layer_costs)), stages - 1):
        boundaries = (0, *cuts, len(layer_costs))
        costs = [sum(layer_costs[left:right]) for left, right in zip(boundaries, boundaries[1:])]
        candidates.append({"cuts": list(cuts), "stage_costs": costs, "max_stage_cost": max(costs)})
    candidates.sort(key=lambda c: (c["max_stage_cost"], c["cuts"]))
    return {"layer_costs": layer_costs, "stages": stages, "candidates_evaluated": len(candidates),
            "selected": candidates[0], "all_candidates": candidates,
            "scope": "Exhaustive contiguous partition search under supplied additive costs; ignores communication and memory limits, not Alpa/FlexFlow execution"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results/32_pipeline_schedule.json")
    args = parser.parse_args()
    if not 1 <= args.stages <= 16 or not 1 <= args.microbatches <= 64:
        parser.error("Use 1 <= stages <= 16 and 1 <= microbatches <= 64")
    schedules = [schedule_pipeline(args.stages, args.microbatches, policy=policy) for policy in ("gpipe", "1f1b")]
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "execution": "Serial CPU dependency and supplied-cost model; no model training or hardware timing",
              "gpu_status": "skipped_by_user_request", "cuda_execution_verified": False,
              "assumptions": ["Dedicated resource for each stage", "No communication delay", "No recomputation",
                              "Equal-sized activation units", "Synchronous optimizer update after all microbatches"],
              "schedules": schedules,
              "partition_search": partition_search([1, 2, 1, 4, 1, 3, 1, 2], 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Pipeline dependency model only; all costs are synthetic units, not milliseconds.")
    for schedule in schedules:
        print(schedule["policy"], "makespan:", schedule["makespan_model_units"],
              "peak activation units:", schedule["peak_live_activation_units_per_stage"])
    print("Best contiguous partition:", report["partition_search"]["selected"])
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
