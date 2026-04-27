"""Plot M8 extended DAgger convergence + final comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    with open(RESULTS_DIR / "dagger_jax_history.json") as f:
        h = json.load(f)
    with open(RESULTS_DIR / "dagger_history.json") as f:
        h6 = json.load(f)
    with open(RESULTS_DIR / "milestone_8_benchmark.json") as f:
        b = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) Convergence curve M8 vs M6 vs FMC reference
    ax = axes[0, 0]
    its8 = [r["iter"] for r in h["history"]]
    err8 = [r["mean_err"] for r in h["history"]]
    its6 = [r["iter"] for r in h6["history"]]
    err6 = [r["mean_err"] for r in h6["history"]]
    fmc_err = b["tracking"]["fmc"]["mean_err"]

    ax.semilogy(its6, err6, "o-", linewidth=2, markersize=8, label="M6 DAgger×3 (Python FMC, M=32 H=8)")
    ax.semilogy(its8, err8, "s-", linewidth=2, markersize=6, label=f"M8 DAgger×{max(its8)} (JIT FMC)")
    ax.axhline(fmc_err, color="r", ls="--", linewidth=2,
                label=f"FMC online (M=200 H=20): {fmc_err:.2f}")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("mean shape error (log)")
    ax.set_title("DAgger convergence — M8 plateau matches/beats FMC")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # 2) Per-iter wall-clock M8
    ax = axes[0, 1]
    label_t = [r.get("t_label_s", 0) for r in h["history"][1:]]  # skip iter 0
    train_t = [r.get("t_train_s", 0) for r in h["history"][1:]]
    rollout_t = [r.get("t_rollout_s", 0) for r in h["history"][1:]]
    its = [r["iter"] for r in h["history"][1:]]
    width = 0.7
    ax.bar(its, label_t, width, label="JIT FMC labeling", color="#3498db")
    ax.bar(its, train_t, width, bottom=label_t, label="NN training", color="#27ae60")
    ax.bar(its, rollout_t, width, bottom=np.array(label_t) + np.array(train_t),
           label="rollout collection", color="#e74c3c")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("wall-clock [s]")
    ax.set_title(f"Per-iter cost — total {h['total_wall_s']:.1f}s for {max(its)} iter")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 3) Dataset growth M8
    ax = axes[1, 0]
    ns = [r["n_samples"] for r in h["history"]]
    ax.plot(its8, ns, "s-", linewidth=2, markersize=8, color="#27ae60")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("|D_k|")
    ax.set_title(f"Dataset growth — final {ns[-1]} samples")
    ax.grid(alpha=0.3)

    # 4) Final comparison bar chart (4-way)
    ax = axes[1, 1]
    methods = ["M5 BC", "M6 DAgger×3", "M8 DAgger×N", "FMC online"]
    errs = [b["tracking"]["bc"]["mean_err"], b["tracking"]["dagger3"]["mean_err"],
            b["tracking"]["dagger_jax"]["mean_err"], b["tracking"]["fmc"]["mean_err"]]
    lats = [b["latency_us"]["bc"], b["latency_us"]["dagger3"],
            b["latency_us"]["dagger_jax"], b["latency_us"]["fmc"]]
    colors = ["#f39c12", "#27ae60", "#2980b9", "#e74c3c"]

    ax2 = ax.twinx()
    bars1 = ax.bar([i - 0.2 for i in range(4)], errs, width=0.4, color=colors,
                   edgecolor="black", label="tracking err")
    bars2 = ax2.bar([i + 0.2 for i in range(4)], lats, width=0.4, color=colors,
                    edgecolor="black", alpha=0.5, hatch="//", label="latency [µs]")
    ax2.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("tracking err (lower better)")
    ax2.set_ylabel("latency µs (log, lower better)")
    ax.set_title("Final 4-way comparison")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    for b1, e in zip(bars1, errs):
        ax.text(b1.get_x() + b1.get_width()/2, e, f"{e:.2f}",
                ha="center", va="bottom", fontsize=9)
    for b2, l in zip(bars2, lats):
        ax2.text(b2.get_x() + b2.get_width()/2, l, f"{l:.0f}",
                 ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Milestone 8 — extended DAgger (JIT FMC backbone, "
        f"{ns[-1]}/{h['total_wall_s']:.0f}s) reaches FMC-online quality",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "milestone_8_extended.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
