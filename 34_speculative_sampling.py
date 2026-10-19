"""Exact speculative sampling on tiny CPU probability models; no speed benchmark."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Expected a finite nonnegative probability vector summing to one")
    return values / values.sum()


def residual_distribution(p, q):
    p, q = distribution(p), distribution(q)
    if p.shape != q.shape:
        raise ValueError("Target and draft vocabularies must match")
    residual = np.maximum(p-q, 0)
    if residual.sum() == 0:
        raise ValueError("Equal distributions have no rejection branch")
    return residual / residual.sum()


def corrected_marginal(p, q):
    """Enumerate every one-step draft/accept/reject branch, without simulation."""
    p, q = distribution(p), distribution(q)
    if p.shape != q.shape:
        raise ValueError("Target and draft vocabularies must match")
    accepted = np.minimum(p, q)
    rejection_mass = np.maximum(q-p, 0).sum()
    result = accepted.copy()
    if rejection_mass > 0:
        result += rejection_mass * residual_distribution(p, q)
    return result


def speculative_round(prefix, target, draft, lookahead, rng):
    if not isinstance(lookahead, int) or lookahead < 1:
        raise ValueError("lookahead must be positive")
    original = tuple(prefix)
    context = original
    proposals, qs = [], []
    for _ in range(lookahead):
        q = distribution(draft(context))
        token = int(rng.choice(len(q), p=q))
        proposals.append(token)
        qs.append(q)
        context += (token,)
    # An actual autoregressive target can verify these known positions in one
    # masked forward pass. This toy calls a CPU probability function per context.
    ps = [distribution(target(original + tuple(proposals[:i]))) for i in range(lookahead+1)]
    emitted = []
    for i, token in enumerate(proposals):
        p, q = ps[i], qs[i]
        if p.shape != q.shape:
            raise ValueError("Target and draft vocabularies must match")
        if rng.random() < min(1.0, p[token] / q[token]):
            emitted.append(token)
        else:
            corrected = residual_distribution(p, q)
            emitted.append(int(rng.choice(len(corrected), p=corrected)))
            return emitted, {"drafted": lookahead, "accepted": i, "rejected": True,
                             "target_probability_evaluations": lookahead+1}
    emitted.append(int(rng.choice(len(ps[-1]), p=ps[-1])))
    return emitted, {"drafted": lookahead, "accepted": lookahead, "rejected": False,
                     "target_probability_evaluations": lookahead+1}


def target_model(context):
    last = context[-1] if context else 0
    return np.asarray([[0.65, 0.25, 0.10], [0.10, 0.55, 0.35], [0.30, 0.15, 0.55]][last])


def draft_model(context):
    return 0.75*target_model(context) + 0.25/3


def run(samples=20000):
    rng = np.random.default_rng(34)
    p, q = target_model(()), draft_model(())
    marginal = corrected_marginal(p, q)
    np.testing.assert_allclose(marginal, p, atol=1e-14, rtol=0)
    counts = np.zeros(3, dtype=int)
    accepted = emitted = rejected = 0
    for _ in range(samples):
        tokens, record = speculative_round((), target_model, draft_model, 4, rng)
        counts[tokens[0]] += 1
        accepted += record["accepted"]
        emitted += len(tokens)
        rejected += record["rejected"]
    return {"samples": samples, "target_first_token": p.tolist(), "draft_first_token": q.tolist(),
            "exact_corrected_marginal": marginal.tolist(), "sampled_first_token_frequencies": (counts/samples).tolist(),
            "mean_accepted_draft_tokens": accepted/samples, "mean_emitted_tokens_per_round": emitted/samples,
            "rejected_rounds": rejected,
            "scope": "Exact branch-mass calculation plus seeded Monte Carlo of context-dependent toy models; no real LLM, parallel target pass, KV rollback or latency evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/34_speculative_sampling.json")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "gpu_status": "skipped_by_user_request", **run(args.samples)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
