"""CPU top-k MoE dispatch/combine, token accounting and capacity overflow."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


def routes(logits, top_k):
    if logits.ndim != 2 or not isinstance(top_k, int) or not 1 <= top_k <= logits.shape[1]:
        raise ValueError("Expected [tokens, experts] logits and valid top_k")
    selected, indices = torch.topk(logits, top_k, dim=-1)
    return indices, torch.softmax(selected, dim=-1)


def expert_forward(x, up, down):
    return torch.relu(x @ up) @ down


def moe_dispatch(x, logits, up, down, top_k=2, capacity=None):
    if x.ndim != 2 or logits.shape[0] != len(x) or up.ndim != 3 or down.ndim != 3 or up.shape[0] != logits.shape[1] or up.shape[1] != x.shape[1] or down.shape != (up.shape[0], up.shape[2], x.shape[1]):
        raise ValueError("Incompatible input/router/expert dimensions")
    if capacity is not None and (not isinstance(capacity, int) or capacity < 1):
        raise ValueError("capacity must be a positive integer or None")
    indices, gates = routes(logits, top_k)
    output = torch.zeros_like(x)
    requested, processed = [], []
    for expert in range(up.shape[0]):
        token_ids, slots = torch.where(indices == expert)
        requested.append(len(token_ids))
        if capacity is not None:
            token_ids, slots = token_ids[:capacity], slots[:capacity]
        processed.append(len(token_ids))
        expert_output = expert_forward(x[token_ids], up[expert], down[expert])
        output.index_add_(0, token_ids, expert_output*gates[token_ids, slots, None])
    return output, {"requested_assignments_per_expert": requested,
                    "processed_assignments_per_expert": processed,
                    "dropped_assignments": sum(requested)-sum(processed),
                    "capacity": capacity,
                    "overflow_policy": "Keep earliest token assignments per expert, drop overflow, retain original gates without renormalization"}


def dense_reference(x, logits, up, down, top_k=2):
    indices, gates = routes(logits, top_k)
    all_outputs = torch.stack([expert_forward(x, a, b) for a, b in zip(up, down)], dim=1)
    chosen = all_outputs.gather(1, indices[:, :, None].expand(-1, -1, x.shape[1]))
    return (chosen*gates[:, :, None]).sum(dim=1)


def run():
    generator = torch.Generator().manual_seed(36)
    n, d, hidden, experts, k = 13, 8, 12, 4, 2
    x = torch.randn(n, d, generator=generator, dtype=torch.float64)
    up = torch.randn(experts, d, hidden, generator=generator, dtype=torch.float64)
    down = torch.randn(experts, hidden, d, generator=generator, dtype=torch.float64)
    logits = torch.randn(n, experts, generator=generator, dtype=torch.float64)
    actual, record = moe_dispatch(x, logits, up, down, k)
    expected = dense_reference(x, logits, up, down, k)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    skewed = torch.tensor([[9., 8., -1., -2.]], dtype=torch.float64).expand(n, -1)
    capacity = math.ceil(n*k/experts)
    capped, capped_record = moe_dispatch(x, skewed, up, down, k, capacity)
    uncapped, skewed_record = moe_dispatch(x, skewed, up, down, k)
    return {"tokens": n, "experts": experts, "top_k": k, "routing": record,
            "dense_reference_max_abs_error": (actual-expected).abs().max().item(),
            "expert_token_rows_dense_all": n*experts, "expert_token_rows_sparse": n*k,
            "skewed_routing": skewed_record, "capacity_limited": capped_record,
            "capacity_output_max_abs_difference": (capped-uncapped).abs().max().item(),
            "scope": "Real CPU dispatch/combine with serial expert execution; no expert-parallel all-to-all, GPU grouped GEMM, trained router or measured acceleration"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/36_moe_dispatch.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "torch": torch.__version__, "gpu_status": "skipped_by_user_request", **run()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
