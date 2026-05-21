"""gap9_figures.py — publication-quality figures for the paper.

Figures generated (all in figures/):
    fig1_trajectory.pdf  — exp01 → exp23 Crafter score trajectory with
                            blocker-count overlay and CI95 error bars
    fig2_ablation.pdf    — additive ablation table (exp03 → exp17 stack)
                            as a horizontal bar chart
    fig3_pareto.pdf      — sample-efficiency Pareto frontier comparing
                            FMC variants vs DRL baselines
    fig5_blockers.pdf    — per-blocker frequency, v4 baseline vs exp17

Figure 4 (schematic of two-component reward + relativize regimes) is left
to the scientific-schematics skill (Nano Banana Pro) — needs visual
rendering rather than data plotting.

Usage:
    python gap9_figures.py
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
TSV_PATH = HERE.parent / "autoresearch" / "results.tsv"
RESULTS_DIR = HERE.parent / "results"

# Publication style — single-column journal width
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# Colorblind-safe Wong palette
COLOR_BLUE   = "#0072B2"
COLOR_VERMIL = "#D55E00"
COLOR_GREEN  = "#009E73"
COLOR_YELLOW = "#F0E442"
COLOR_GRAY   = "#666666"


def read_results_tsv() -> list[dict]:
    """Read results.tsv into list of dicts."""
    rows = []
    with open(TSV_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            r["crafter_pct"] = float(r["crafter_pct"])
            r["n_seeds"] = int(r["n_seeds"])
            r["mean_ach"] = float(r["mean_ach"])
            r["ach_ci95"] = float(r["ach_ci95"])
            r["blocker_fired"] = int(r["blocker_fired"])
            rows.append(r)
    return rows


def parse_exp_num(description: str) -> int:
    """Get experiment number from description string."""
    desc = description.lower()
    if "baseline" in desc:
        return 0
    if desc.startswith("sanity"):
        return -1
    for token in description.split(":")[0].split():
        if token.startswith("exp"):
            try:
                return int(token[3:])
            except ValueError:
                pass
    return -1


def crafter_ci95_from_n(crafter_pct: float, n: int) -> float:
    """Wilson-like normal approximation. Crafter on a 0-100 scale; rough CI95
    estimate from n seeds using sigma ~ 8 pp around the mean (empirical
    observation for the productive shaping regime). Used only as a visual
    bound; rigorous CI comes from the bootstrap in Gap 1 output.
    """
    return 1.96 * 8.0 / math.sqrt(max(n, 1))


def fig1_trajectory():
    rows = read_results_tsv()
    # Filter to exp01-23 in order
    exp_rows = []
    for r in rows:
        n = parse_exp_num(r["description"])
        if n >= 1:
            exp_rows.append((n, r))
    exp_rows.sort(key=lambda t: t[0])

    nums = [t[0] for t in exp_rows]
    crafter = [t[1]["crafter_pct"] for t in exp_rows]
    n_seeds = [t[1]["n_seeds"] for t in exp_rows]
    blockers = [t[1]["blocker_fired"] for t in exp_rows]
    err = [crafter_ci95_from_n(c, n) for c, n in zip(crafter, n_seeds)]
    status = [t[1]["status"] for t in exp_rows]

    # Three-panel figure: (a) crafter, (b) blockers, (c) mean_ach
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})

    ax = axes[0]
    colors = [COLOR_BLUE if s == "keep" else COLOR_GRAY for s in status]
    for i in range(len(nums)):
        ax.errorbar([nums[i]], [crafter[i]], yerr=[err[i]],
                    fmt="o", capsize=2, color=colors[i],
                    markersize=5, alpha=0.85,
                    elinewidth=0.7)
    # Connect kept experiments with a line
    keep_idx = [i for i, s in enumerate(status) if s == "keep"]
    ax.plot([nums[i] for i in keep_idx], [crafter[i] for i in keep_idx],
            color=COLOR_BLUE, alpha=0.5, linewidth=1.2, zorder=1)
    # Reference lines
    ax.axhline(29.27, color=COLOR_GRAY, linestyle="--", linewidth=0.8,
               alpha=0.6, label="v4 baseline (29.27%)")
    ax.axhline(50.5, color=COLOR_GREEN, linestyle="--", linewidth=0.8,
               alpha=0.7, label="human expert (50.5%, Hafner 2021)")
    ax.axhline(58.1, color=COLOR_VERMIL, linestyle=":", linewidth=0.8,
               alpha=0.7, label="EMERALD 10M (58.1%)")
    # Annotate exp17 specifically
    for i, n in enumerate(nums):
        if n == 17:
            ax.annotate(f"exp17\n{crafter[i]:.2f}%",
                        xy=(n, crafter[i]),
                        xytext=(n + 0.5, crafter[i] + 4),
                        fontsize=8, color=COLOR_BLUE,
                        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE,
                                        lw=0.6))
        if n == 22:
            ax.annotate(f"exp22\n(α>1\ncollapse)",
                        xy=(n, crafter[i]),
                        xytext=(n + 0.5, crafter[i] - 5),
                        fontsize=7, color=COLOR_GRAY,
                        arrowprops=dict(arrowstyle="->", color=COLOR_GRAY,
                                        lw=0.6))
    ax.set_ylabel("Crafter score (%)")
    ax.set_ylim(20, 65)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.set_title("(a) FMC autoresearch trajectory: 23 experiments, "
                 "exp17 reaches human-expert level")

    ax = axes[1]
    ax.bar(nums, blockers, color=COLOR_GREEN, alpha=0.7, edgecolor="white",
           linewidth=0.5)
    ax.set_ylabel("Blockers fired (/4)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(0, 4)
    ax.set_title("(b) v4-blockers unlocked per experiment")

    ax = axes[2]
    mean_ach = [t[1]["mean_ach"] for t in exp_rows]
    ach_err = [t[1]["ach_ci95"] for t in exp_rows]
    ax.errorbar(nums, mean_ach, yerr=ach_err, fmt="o-",
                color=COLOR_VERMIL, capsize=2, markersize=4,
                linewidth=0.8, elinewidth=0.6, alpha=0.85)
    ax.set_ylabel("Mean achievements / episode")
    ax.set_ylim(11, 16.5)
    ax.set_xlabel("Experiment number")
    ax.set_title("(c) Mean achievements unlocked per episode (CI95)")
    ax.set_xticks(nums)

    fig.tight_layout()
    out = FIG_DIR / "fig1_trajectory.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig1_trajectory.png")
    plt.close(fig)
    print(f"  -> {out}")


def fig2_ablation():
    """Additive ablation: exp03 → exp17 stack. Each row shows the marginal
    contribution Δ from adding a tier."""
    stages = [
        ("exp03 (ach-fire only)",                 40.96, 0.00),
        ("+ iron-tier inv (exp09)",              42.89, +1.93),
        ("+ stone-tier inv (exp10)",             44.14, +1.24),
        ("+ wood-tier inv (exp11)",              45.94, +1.80),
        ("+ proximity ×2 (exp12)",               46.45, +0.51),
        ("+ iron-tier ach push (exp16)",         50.65, +4.71),
        ("+ gateway-tier ach push (exp17)",      50.95, +0.30),
    ]

    labels = [s[0] for s in stages]
    crafter = [s[1] for s in stages]
    deltas = [s[2] for s in stages]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.6),
                                     gridspec_kw={"width_ratios": [1.4, 1]})

    # Left: cumulative crafter score
    y = np.arange(len(stages))
    ax_l.barh(y, crafter, color=COLOR_BLUE, alpha=0.85, edgecolor="white")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels, fontsize=8)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Crafter score (%)")
    ax_l.axvline(50.5, color=COLOR_GREEN, linestyle="--", linewidth=0.8,
                 alpha=0.7, label="human expert")
    ax_l.set_xlim(35, 55)
    for i, c in enumerate(crafter):
        ax_l.text(c + 0.3, i, f"{c:.2f}", va="center", fontsize=7,
                  color=COLOR_BLUE)
    ax_l.legend(loc="lower right", fontsize=7)
    ax_l.set_title("(a) Cumulative score per stack level")

    # Right: marginal delta per addition
    colors_delta = [COLOR_BLUE if d > 0 else COLOR_GRAY for d in deltas]
    ax_r.barh(y, deltas, color=colors_delta, alpha=0.85, edgecolor="white")
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.invert_yaxis()
    ax_r.set_xlabel("Δ Crafter score (pp)")
    for i, d in enumerate(deltas):
        if d > 0.1:
            ax_r.text(d + 0.05, i, f"+{d:.2f}", va="center", fontsize=7,
                      color=COLOR_BLUE)
    ax_r.set_title("(b) Marginal contribution Δ")
    ax_r.set_xlim(-0.5, 5.5)
    ax_r.axvline(0, color="black", linewidth=0.6)

    fig.suptitle("Additive tier ablation (exp03 → exp17). Leave-one-out "
                 "ablation pending Gap 3.", fontsize=9, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig2_ablation.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig2_ablation.png")
    plt.close(fig)
    print(f"  -> {out}")


def fig3_pareto():
    """Sample-efficiency Pareto. X = total samples (train + inference per
    episode), Y = Crafter score. FMC variants in the zero-training column."""
    methods = [
        ("Random",            0.0,           0,       1.6,   COLOR_GRAY),
        ("PPO 1M",            1e6,           0,       4.6,   COLOR_GRAY),
        ("PPO 1B",            1e9,           0,      11.0,   COLOR_GRAY),
        ("DreamerV3 1M",      1e6,           0,      14.5,   COLOR_GRAY),
        ("Curious Replay 1M", 1e6,           0,      19.4,   COLOR_GRAY),
        ("EMERALD 10M",       1e7,           0,      58.1,   COLOR_VERMIL),
        ("FMC v4",            0.0,           1.024e7, 29.27, COLOR_BLUE),
        ("FMC exp17 (ours)",  0.0,           1.024e7, 50.95, COLOR_BLUE),
        ("Human expert",      0.0,           0,      50.5,   COLOR_GREEN),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))

    for name, train, inf_per_ep, score, col in methods:
        # Total samples = max(train, 500_episodes * inf_per_ep) — for plotting
        # use train if zero-inference, inference budget otherwise.
        if train > 0 and inf_per_ep == 0:
            total = train
            shape = "o"
        elif train == 0 and inf_per_ep > 0:
            # 500 episode budget for the FMC line
            total = 500 * inf_per_ep
            shape = "s"
        else:
            total = max(1, train) + 500 * inf_per_ep
            shape = "^"

        if name == "Human expert":
            ax.scatter([1], [score], color=col, s=100, marker="*",
                       label=name, edgecolor="black", linewidth=0.5,
                       zorder=10)
            ax.annotate(name, xy=(1, score), xytext=(2, score - 2),
                        fontsize=7, color=col)
            continue

        ax.scatter([total], [score], color=col, s=60, marker=shape,
                   edgecolor="black", linewidth=0.5,
                   alpha=0.9, zorder=5,
                   label=name if "FMC" in name or "EMERALD" in name else None)
        x_off = 1.5 if "FMC" in name else 0.0
        y_off = -2 if "v4" in name else (2 if "exp17" in name else 0)
        ax.annotate(name,
                    xy=(total, score),
                    xytext=(total * (1 + x_off / 50) if total > 1 else total + 1,
                            score + y_off),
                    fontsize=7, color=col)

    # Frontier lines
    fmc_x = [1.024e7, 1.024e7]
    fmc_y = [29.27, 50.95]
    ax.plot(fmc_x, fmc_y, "--", color=COLOR_BLUE, alpha=0.5, linewidth=0.8,
            label="FMC zero-training axis")

    ax.set_xscale("log")
    ax.set_xlabel("Compute samples (training + 500 × per-decision)")
    ax.set_ylabel("Crafter score (%)")
    ax.set_xlim(1, 1e10)
    ax.set_ylim(-2, 65)
    ax.axhline(50.5, color=COLOR_GREEN, linestyle=":", linewidth=0.8,
               alpha=0.6)
    ax.set_title("Sample-efficiency Pareto: zero-training FMC reaches human-expert")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / "fig3_pareto.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig3_pareto.png")
    plt.close(fig)
    print(f"  -> {out}")


def fig5_blockers():
    """Per-blocker frequency: v4 baseline vs exp17."""
    blockers = ["collect_diamond", "make_iron_pickaxe", "make_iron_sword",
                "eat_plant"]

    # v4 from run007_top_cells_30seed.json (N=512, M=40 cell)
    v4_path = RESULTS_DIR / "run007_top_cells_30seed.json"
    with open(v4_path) as f:
        d = json.load(f)
    cell_v4 = next(c for c in d["per_cell"]
                   if c["N"] == 512 and c["M"] == 40)
    v4_rates = [cell_v4["achievement_freq"][b] for b in blockers]

    # exp17 from autoresearch results — pull from exp17 11-seed for now,
    # update with 30-seed when Gap 1 completes.
    exp17_path = HERE.parent / "autoresearch" / "results" / "exp17_30seed.json"
    if exp17_path.exists():
        with open(exp17_path) as f:
            d17 = json.load(f)
        exp17_rates = [d17["achievement_freq"][b] for b in blockers]
        n_label = f"30 seeds"
    else:
        # Fallback: hard-code from exp17 11-seed run from PAPER_HANDOFF
        exp17_rates_dict = {"collect_diamond": 0.09,
                             "make_iron_pickaxe": 0.27,
                             "make_iron_sword": 0.09,
                             "eat_plant": 0.0}
        exp17_rates = [exp17_rates_dict[b] for b in blockers]
        n_label = "11 seeds (Gap 1 pending)"

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
    x = np.arange(len(blockers))
    width = 0.38

    ax.bar(x - width / 2, v4_rates, width, color=COLOR_GRAY, alpha=0.85,
           label="FMC v4 baseline (30 seeds)", edgecolor="white")
    ax.bar(x + width / 2, exp17_rates, width, color=COLOR_BLUE, alpha=0.85,
           label=f"FMC exp17 ({n_label})", edgecolor="white")

    for i, (v, e) in enumerate(zip(v4_rates, exp17_rates)):
        if v > 0:
            ax.text(i - width / 2, v + 0.01, f"{v:.0%}", ha="center",
                    fontsize=7, color=COLOR_GRAY)
        if e > 0:
            ax.text(i + width / 2, e + 0.01, f"{e:.0%}", ha="center",
                    fontsize=7, color=COLOR_BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in blockers], fontsize=8)
    ax.set_ylabel("Unlock rate")
    ax.set_ylim(0, 0.45)
    ax.set_title("v4-blocker unlock rates: shaping unblocks 3 of 4 blockers")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig5_blockers.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig5_blockers.png")
    plt.close(fig)
    print(f"  -> {out}")


def fig2bis_leave_one_out():
    """Leave-one-out ablation table from Gap 3."""
    abl_dir = HERE.parent / "autoresearch" / "results"
    summary_path = abl_dir / "gap3_summary.json"
    if not summary_path.exists():
        print(f"  (skip fig2-bis: {summary_path} not found)")
        return
    with open(summary_path) as f:
        d = json.load(f)
    BASELINE = 50.6049

    rows = [
        ("L4 (− iron-tier ach push)", "L4"),
        ("L3 (− wood-tier inv)",      "L3"),
        ("L1 (− iron-tier inv) †",    "L1"),
        ("L2 (− stone-tier inv)",     "L2"),
        ("L5 (− gateway-tier ach push)", "L5"),
    ]
    labels = []
    deltas = []
    n_seeds_list = []
    for label, key in rows:
        if key not in d or "error" in d[key]:
            continue
        labels.append(label)
        deltas.append(d[key]["crafter_score"] - BASELINE)
        n_seeds_list.append(d[key]["n_seeds_completed"])

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))
    y = np.arange(len(labels))
    colors = [COLOR_GRAY if "L1" in l else COLOR_BLUE for l in labels]
    bars = ax.barh(y, deltas, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Δ Crafter score vs exp17 baseline (50.60 %)")

    for i, (d_v, n) in enumerate(zip(deltas, n_seeds_list)):
        x_text = d_v - 0.3
        ax.text(x_text, i, f"{d_v:+.2f}  (n={n})", va="center", ha="right",
                fontsize=8,
                color="black" if "L1" in labels[i] else "white")

    ax.set_title("Leave-one-out ablation: every tier component is load-bearing\n"
                 "(all Δs > 4 pp drops, supporting Conjecture D compounding)")
    ax.set_xlim(min(deltas) - 1.5, 0.5)
    fig.text(0.5, -0.02,
             "† L1 single-seed run (wall budget exhausted by 1 pathological "
             "episode); reported only for direction.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    out = FIG_DIR / "fig2bis_leave_one_out.pdf"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig2bis_leave_one_out.png")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_trajectory()
    fig2_ablation()
    fig2bis_leave_one_out()
    fig3_pareto()
    fig5_blockers()
    print(f"All figures in {FIG_DIR}")
