"""Plot M15 published-target benchmark results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    with open(RESULTS_DIR / "milestone_15_published_eval.json") as f:
        d = json.load(f)

    scenarios = list(d["scenarios"].keys())
    policies = ["M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"]

    # Build matrix: scenarios × policies → truth-err
    truth_mat = np.array([
        [d["scenarios"][s]["policies"][p]["mean_truth_err"]
         for p in policies]
        for s in scenarios
    ])
    phys_mat = np.array([
        [d["scenarios"][s]["policies"][p]["physicality"]
         for p in policies]
        for s in scenarios
    ])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: heatmap of truth-err per (scenario, policy)
    ax = axes[0, 0]
    im = ax.imshow(truth_mat, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=80)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("Truth-err per scenario × policy")
    for i in range(len(scenarios)):
        for j in range(len(policies)):
            color = "white" if truth_mat[i, j] > 40 else "black"
            ax.text(j, i, f"{truth_mat[i,j]:.1f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, label="truth-err (lower = better)")

    # Panel 2: heatmap of physicality
    ax = axes[0, 1]
    im2 = ax.imshow(100 * phys_mat, cmap="RdYlGn", aspect="auto",
                     vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("Physicality rate per scenario × policy (%)")
    for i in range(len(scenarios)):
        for j in range(len(policies)):
            ax.text(j, i, f"{100*phys_mat[i,j]:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    plt.colorbar(im2, ax=ax, label="% steps with valid LCFS")

    # Panel 3: bar chart of aggregate truth-err
    ax = axes[1, 0]
    agg = d["aggregate"]
    means = [agg[p]["mean_truth_across_scenarios"] for p in policies]
    mins = [agg[p]["min_truth"] for p in policies]
    maxs = [agg[p]["max_truth"] for p in policies]
    colors = ["lightcoral" if m > 30 else
              ("yellow" if m > 8 else "mediumseagreen") for m in means]
    x = np.arange(len(policies))
    ax.bar(x, means, color=colors, edgecolor="black")
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(mins),
                      np.array(maxs) - np.array(means)],
                fmt="none", color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("Mean truth-err across scenarios")
    ax.set_title("Aggregate ranking with min-max range")
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(means):
        ax.text(i, m + 1.5, f"{m:.2f}", ha="center", fontsize=10,
                fontweight="bold")

    # Panel 4: scatter — physicality vs mean truth (per policy)
    ax = axes[1, 1]
    phys_means = [agg[p]["mean_physicality"] for p in policies]
    truth_means = [agg[p]["mean_truth_across_scenarios"] for p in policies]
    cmap = plt.cm.viridis
    ax.scatter(100 * np.array(phys_means), truth_means,
               s=200, c=np.arange(len(policies)), cmap=cmap, edgecolor="black",
               linewidths=1.5)
    for i, p in enumerate(policies):
        ax.annotate(p, (100 * phys_means[i], truth_means[i]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Mean physicality (%)")
    ax.set_ylabel("Mean truth-err")
    ax.set_title("Physicality vs accuracy trade-off")
    ax.set_xlim(0, 105)
    ax.grid(alpha=0.3)
    ax.axhline(8, color="green", linestyle="--", alpha=0.5,
               label="Deployable threshold (~8 truth-err)")
    ax.axvline(80, color="orange", linestyle="--", alpha=0.5,
               label="Acceptable physicality (80%)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Milestone 15 — TCV published-target benchmark\n"
        "Source: Degrave 2022 Nat / Reimerdes 2022 / Anand 2021 — "
        "shapes proven on real TCV hardware",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "milestone_15_published_eval.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out}")


if __name__ == "__main__":
    main()
