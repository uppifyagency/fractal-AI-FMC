"""E1-LLM-curve — figures. Reads results/e1_llm_curve.json (Phase A band) and
results/e1_llm_curve_analysis.json (the classified LLM points) and renders:

  e1_llm_curve_band.png      per-layout death-vs-f_abs: random-ablation band
                             (isotonic + 5-95% ribbon) with LLM world-models
                             overlaid — shows the LLM points sitting off-band.
  e1_llm_curve_fidelity.png  death vs each of the three fidelity axes — f_abs
                             is near-flat, done-persistence is what tracks
                             death. The visual core of "f_abs not sufficient".

No experiment is re-run. Run:  python work/12_conjecture_e/e1_llm_curve_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"

LAYOUTS = ["gauntlet", "lake", "scatter", "island", "spur", "archipelago"]
MSHORT = {"llama-3.2-1b-instruct": "L1", "llama-3.2-3b-instruct": "L3",
          "llama-3.1-8b-instruct": "L8", "llama-3.3-70b-instruct": "L70"}
MCOL = {"L3": "#d62728", "L8": "#ff7f0e", "L70": "#1f77b4"}
PMARK = {"P0": "o", "P1": "s", "P2": "^"}


def main():
    band = json.loads((_RES / "e1_llm_curve.json").read_text())["band"]["layouts"]
    pts = json.loads((_RES / "e1_llm_curve_analysis.json").read_text())["points"]
    for p in pts:
        p["m"] = MSHORT.get(p["model"].split("/")[-1], p["model"])

    # --- per-layout isotonic band --------------------------------------------
    iso, ribbon = {}, {}
    for L in LAYOUTS:
        fa = np.array([d["f_abs"] for d in band[L]["draws"]])
        dr = np.array([d["death_rate"] for d in band[L]["draws"]])
        g = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(fa, dr)
        iso[L] = (g, fa, dr)
        res = dr - g.predict(fa)
        ribbon[L] = (np.percentile(res, 5), np.percentile(res, 95))

    # === Fig 1: per-layout band + LLM points =================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, L in zip(axes.flat, LAYOUTS):
        g, fa, dr = iso[L]
        ax.scatter(fa, dr * 100, s=13, c="#cfcfcf", zorder=1)
        gx = np.linspace(0.5, 1.0, 120)
        gy = g.predict(gx)
        lo, hi = ribbon[L]
        ax.fill_between(gx, (gy + lo) * 100, (gy + hi) * 100, color="k",
                        alpha=0.12, zorder=2)
        ax.plot(gx, gy * 100, "k-", lw=2, zorder=3)
        rnd = band[L]["baselines"]["random"]["death_rate"]
        ax.axhline(rnd * 100, ls=":", c="#777777", lw=1, zorder=2)
        for p in [q for q in pts if q["layout"] == L]:
            ax.scatter(p["f_abs"], p["death"] * 100, s=78, c=MCOL[p["m"]],
                       marker=PMARK[p["prompt"]], edgecolors="black",
                       linewidths=0.6, zorder=5)
        ax.set_title(f"{L}   (random {rnd*100:.0f}%)", fontsize=11)
        ax.set_xlabel("f_abs  —  absorbing-flag fidelity")
        ax.set_ylabel("death rate  (%)")
        ax.set_xlim(0.45, 1.03)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    handles = (
        [plt.Line2D([], [], marker="o", ls="", mfc=MCOL[m], mec="k", label=m)
         for m in ("L3", "L8", "L70")]
        + [plt.Line2D([], [], marker=PMARK[p], ls="", mfc="w", mec="k", label=p)
           for p in ("P0", "P1", "P2")]
        + [plt.Line2D([], [], marker="o", ls="", mfc="#cfcfcf", mec="none",
                      label="random-ablation band"),
           plt.Line2D([], [], color="k", lw=2, label="isotonic fit")])
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False,
               fontsize=10)
    fig.suptitle("E1-LLM-curve — LLM world-models vs the random-ablation "
                 "tolerance band  (alpha=0, per layout)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(_RES / "e1_llm_curve_band.png", dpi=110)
    plt.close(fig)

    # === Fig 2: death vs the three fidelity axes =============================
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axes_spec = [("f_abs", "f_abs  (entry-detection)"),
                 ("move_fid", "move-fidelity  (movement)"),
                 ("done_pers", "done-persistence  (absorbing stays)")]
    for ax, (key, lab) in zip(axs, axes_spec):
        for p in pts:
            ax.scatter(p[key], p["death"] * 100, s=58, c=MCOL[p["m"]],
                       marker=PMARK[p["prompt"]], edgecolors="black",
                       linewidths=0.5, alpha=0.85)
        rho, pv = spearmanr([p[key] for p in pts], [p["death"] for p in pts])
        ax.set_title(f"death vs {key}\nSpearman rho = {rho:+.2f}  (p={pv:.1e})",
                     fontsize=11)
        ax.set_xlabel(lab)
        ax.set_ylabel("death rate  (%)")
        ax.set_xlim(-0.03, 1.05)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    fig.suptitle("E1-LLM-curve — death vs the three fidelity axes: f_abs alone "
                 "leaves death widely spread, no single axis is sufficient  "
                 "(156 LLM points)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(_RES / "e1_llm_curve_fidelity.png", dpi=110)
    plt.close(fig)

    print("wrote results/e1_llm_curve_band.png")
    print("wrote results/e1_llm_curve_fidelity.png")
    for key, _ in axes_spec:
        rho, _ = spearmanr([p[key] for p in pts], [p["death"] for p in pts])
        print(f"  Spearman death vs {key:11s} = {rho:+.3f}")


if __name__ == "__main__":
    main()
