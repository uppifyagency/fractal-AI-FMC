"""Visualize M14 oracle eval results: truth-err + physicality rate."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    with open(RESULTS_DIR / "milestone_14_oracle_eval.json") as f:
        d14 = json.load(f)
    with open(RESULTS_DIR / "milestone_13_oracle_eval.json") as f:
        d13 = json.load(f)

    s14 = d14["summary"]
    s13 = d13["summary"]

    labels = ["M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"]
    truth_14 = [s14[l]["mean_err_truth"] for l in labels]
    truth_13 = [s13[l]["mean_err_truth"] for l in labels]
    self_14 = [s14[l]["mean_err_self"] for l in labels]
    physicality = [
        s14[l]["n_freegs"] / max(1, s14[l]["n_freegs"] + s14[l]["n_nn_fallback"])
        for l in labels
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: truth-err comparison M13 (NN proxy) vs M14 (real freegs)
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, truth_13, w, color="steelblue", alpha=0.7,
           label="M13 NN-proxy")
    ax.bar(x + w/2, truth_14, w, color="crimson", alpha=0.7,
           label="M14 real freegs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean truth-err")
    ax.set_title("Truth-err: NN proxy vs real FreeGS")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for i, (v13, v14) in enumerate(zip(truth_13, truth_14)):
        ax.text(i - w/2, v13 + 1, f"{v13:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, v14 + 1, f"{v14:.1f}", ha="center", fontsize=8,
                fontweight="bold")

    # Panel 2: physicality rate (% trajectories yielding valid GS equilibrium)
    ax = axes[1]
    colors = ["lightcoral" if p < 0.5 else "mediumseagreen" for p in physicality]
    ax.bar(x, [100 * p for p in physicality], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Physicality rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Physicality rate (freegs converges)")
    ax.grid(alpha=0.3, axis="y")
    for i, p in enumerate(physicality):
        ax.text(i, 100 * p + 2, f"{100*p:.0f}%", ha="center", fontsize=10,
                fontweight="bold")

    # Panel 3: scatter — self-err vs truth-err M14
    ax = axes[2]
    sizes = [200 + 1500 * p for p in physicality]
    sc = ax.scatter(self_14, truth_14, s=sizes, c=physicality,
                    cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolors="black", linewidths=1.5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (self_14[i], truth_14[i]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.plot([0, max(max(self_14), max(truth_14))],
            [0, max(max(self_14), max(truth_14))], "k--", alpha=0.3,
            label="self-err = truth-err")
    ax.set_xlabel("self-err (sim-internal)")
    ax.set_ylabel("truth-err (FreeGS oracle)")
    ax.set_title("Self vs truth (size ∝ physicality)")
    ax.legend()
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax, label="physicality")

    fig.suptitle(
        "Milestone 14 — Oracle eval with REAL FreeGS truth\n"
        "Reveals what M13 (NN proxy) hid: actual ranking is "
        "M6 > M12 ≈ FMC ≫ M10 ≈ M5",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "milestone_14_oracle_eval.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()
