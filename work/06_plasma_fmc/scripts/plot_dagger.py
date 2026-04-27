"""Plot DAgger convergence: tracking error and quench rate vs iteration."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    with open(RESULTS_DIR / "dagger_history.json") as f:
        h = json.load(f)
    with open(RESULTS_DIR / "milestone_6_benchmark.json") as f:
        b = json.load(f)

    iters = [r["iter"] for r in h["history"]]
    n_samples = [r["n_samples"] for r in h["history"]]
    mean_err = [r["mean_err"] for r in h["history"]]
    quenches = [r["quench_count"] for r in h["history"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) Tracking error vs iter
    ax = axes[0, 0]
    ax.semilogy(iters, mean_err, "o-", linewidth=2, markersize=10, color="#2980b9")
    fmc_err = b["tracking"]["fmc"]["mean_err"]
    ax.axhline(fmc_err, color="r", ls="--", label=f"FMC ground truth ({fmc_err:.2f})")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("mean shape error (log)")
    ax.set_title("Tracking quality vs DAgger iteration")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    for i, (it, e) in enumerate(zip(iters, mean_err)):
        ax.annotate(f"{e:.2f}", (it, e), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)

    # 2) Quench count vs iter
    ax = axes[0, 1]
    ax.bar(iters, quenches, color="#e74c3c", alpha=0.8)
    for it, q in zip(iters, quenches):
        ax.text(it, q, f"{q}/10", ha="center", va="bottom", fontsize=11)
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("quenches / 10 episodes")
    ax.set_title("Plasma quenches vs DAgger iteration")
    ax.set_xticks(iters)
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)

    # 3) Dataset growth
    ax = axes[1, 0]
    ax.plot(iters, n_samples, "s-", linewidth=2, markersize=10, color="#27ae60")
    for it, n in zip(iters, n_samples):
        ax.annotate(f"{n}", (it, n), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("dataset size |D_k|")
    ax.set_title("Dataset growth")
    ax.set_xticks(iters)
    ax.grid(alpha=0.3)

    # 4) Final comparison bar chart
    ax = axes[1, 1]
    methods = ["BC (M5)", "DAgger (M6)", "FMC online"]
    errs = [b["tracking"]["bc"]["mean_err"], b["tracking"]["dagger"]["mean_err"],
            b["tracking"]["fmc"]["mean_err"]]
    lats = [b["latency_us"]["bc"], b["latency_us"]["dagger"], b["latency_us"]["fmc"]]
    colors = ["#f39c12", "#27ae60", "#e74c3c"]

    ax2 = ax.twinx()
    bars1 = ax.bar([i - 0.2 for i in range(3)], errs, width=0.4,
                   color=colors, edgecolor="black", label="tracking err")
    bars2 = ax2.bar([i + 0.2 for i in range(3)], lats, width=0.4,
                    color=colors, edgecolor="black", alpha=0.5, hatch="//",
                    label="latency [µs]")
    ax2.set_yscale("log")

    ax.set_xticks(range(3)); ax.set_xticklabels(methods)
    ax.set_ylabel("tracking err (lower better)")
    ax2.set_ylabel("latency µs (log, lower better)")
    ax.set_title("Final comparison: BC vs DAgger vs FMC")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    for b1, e in zip(bars1, errs):
        ax.text(b1.get_x() + b1.get_width()/2, e, f"{e:.2f}",
                ha="center", va="bottom", fontsize=9)
    for b2, l in zip(bars2, lats):
        ax2.text(b2.get_x() + b2.get_width()/2, l, f"{l:.0f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Milestone 6 — DAgger closes BC quality gap "
        "(10× tracking improvement, same latency)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_6_dagger.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
