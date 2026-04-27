"""Plot Milestone 13 oracle eval results: truth-err vs self-err bar chart."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    with open(RESULTS_DIR / "milestone_13_oracle_eval.json") as f:
        d = json.load(f)
    summary = d["summary"]

    labels = ["M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"]
    truth = [summary[l]["mean_err_truth"] for l in labels]
    self_e = [summary[l]["mean_err_self"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, self_e, width,
                    label="Self-eval (sim used for training)",
                    color="#3498db", edgecolor="black")
    bars2 = ax.bar(x + width/2, truth, width,
                    label="Truth-eval (NN_shape proxy of FreeGS)",
                    color="#e74c3c", edgecolor="black")

    for b, v in zip(bars1, self_e):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)
    for b, v in zip(bars2, truth):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean shape error (lower better)")
    ax.set_title(
        f"Milestone 13 — Self-eval vs Truth-eval (10 scenarios × 15 ticks)\n"
        f"Self-eval can be misleadingly low for policies trained on linearized sims",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_13_oracle.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
