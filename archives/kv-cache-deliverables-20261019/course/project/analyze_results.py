"""Generate report figures and tables from the preserved project measurements."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
report = json.loads((ROOT / "results/project_kv_granularity.json").read_text())
figures = HERE / "figures"
figures.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
names = list(dict.fromkeys(r["scenario"] for r in report["rows"]))
blocks = report["blocks"]


def select(name, policy, block=None):
    return [r for r in report["rows"] if r["scenario"] == name and r["policy"] == policy and r["block_size"] == block]


def metric(name, policy, block, field):
    return statistics.median(r[field] for r in select(name, policy, block))


fig, axes_grid = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True)
axes = axes_grid.flatten()
for ax, name in zip(axes, names):
    for policy, label, color in (("paged", "Paged independent", "#8b9bae"), ("shared", "Paged shared", "#147d92")):
        ax.plot(blocks, [metric(name, policy, b, "active_slot_bytes")/1024 for b in blocks], marker="o", label=label, color=color)
    for policy, label, color in (("static", "Static reservation", "#9a6324"), ("dynamic", "Dynamic retained", "#6b4fa2")):
        ax.axhline(metric(name, policy, None, "active_slot_bytes")/1024, linestyle="--", color=color, label=label)
    ax.set(title=name.replace("_", " "), xlabel="Block size (tokens)", xscale="log", xticks=[2, 8, 32, 64], xticklabels=[2, 8, 32, 64])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
axes[0].set_ylabel("Active K/V slots (KiB)")
axes[3].set_ylabel("Active K/V slots (KiB)")
axes[-1].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.82, .15), fontsize=8)
fig.tight_layout()
fig.savefig(figures / "storage.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, name in zip(axes, ("shared_31", "shared_32")):
    for policy, label in (("paged", "Independent"), ("shared", "Shared")):
        ax.plot(blocks, [metric(name, policy, b, "active_slot_bytes")/1024 for b in blocks], marker="o", label=label)
    ax.axhline(metric(name, "static", None, "active_slot_bytes")/1024, linestyle="--", color="#9a6324", label="Static")
    ax.set(title=name.replace("_", " "), xlabel="Block size (tokens)", ylabel="Active K/V (KiB)", xscale="log", xticks=[2, 8, 32, 64], xticklabels=[2, 8, 32, 64])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(figures / "video_storage.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
for ax, name in zip(axes, ("shared_31", "shared_32")):
    for policy, label in (("paged", "Paged independent"), ("shared", "Paged shared")):
        ax.plot(blocks, [metric(name, policy, b, "median_ms") for b in blocks], marker="o", label=label)
    for policy, color in (("static", "#9a6324"), ("dynamic", "#6b4fa2")):
        ax.axhline(metric(name, policy, None, "median_ms"), linestyle="--", label=policy, color=color)
    ax.set(title=name.replace("_", " "), xlabel="Block size (tokens)", ylabel="Complete CPU path (ms)", xscale="log", xticks=[2, 8, 32, 64], xticklabels=[2, 8, 32, 64])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(.5, .01))
fig.tight_layout(rect=(0, .14, 1, 1))
fig.savefig(figures / "latency.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 2.7))
ax.set(xlim=(0, 10), ylim=(0, 4))
ax.axis("off")
for x, y, w, h, text in ((.1, 1.3, 2, 1.3, "Fixed K/V/Q\nshared prefix + suffix\n5 workloads, 3 seeds"),
                         (3, 2.25, 3, 1.1, "Static / dynamic baseline"),
                         (3, .5, 3, 1.1, "Paged + optional prefix fork\npage tables / refs / COW"),
                         (7, 1.3, 2.8, 1.3, "Gather + same attention\ncorrectness / slot bytes\nCPU path wall time")):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor="#edf4f6", edgecolor="#316577"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
for a, b in (((2.2, 2), (2.9, 2.8)), ((2.2, 1.9), (2.9, 1)), ((6.1, 2.8), (6.9, 2.1)), ((6.1, 1), (6.9, 1.7))):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#316577", "lw": 1.5})
fig.tight_layout()
fig.savefig(figures / "architecture.png", dpi=200, bbox_inches="tight")
plt.close(fig)

lines = []
summary = []
for name in names:
    row = {"scenario": name}
    values = []
    for policy, block in (("static", None), ("dynamic", None), ("shared", 4), ("shared", 32), ("shared", 64)):
        size = metric(name, policy, block, "active_slot_bytes")/1024
        values.append(f"{size:.0f}")
        row[f"{policy}_{block}_KiB"] = size
    lines.append(name.replace("_", r"\_")+" & "+" & ".join(values)+r" \\")
    summary.append(row)
(HERE / "results_table.tex").write_text(
    "\\begin{tabular}{lrrrrr}\n\\toprule\n"
    "Workload & Static & Dynamic & Shared B=4 & Shared B=32 & Shared B=64\\\\\n"
    "\\midrule\n"+"\n".join(lines)+"\n\\bottomrule\n\\end{tabular}\n")
(HERE / "summary.json").write_text(json.dumps({"storage_KiB": summary,
    "max_error": max(r["max_abs_error"] for r in report["rows"]),
    "configuration_rows": len(report["rows"]), "timed_runs": len(report["rows"])*report["repeats"]}, indent=2)+"\n")
print(json.dumps(summary, indent=2))
