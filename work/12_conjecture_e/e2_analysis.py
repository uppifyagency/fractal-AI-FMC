"""Conjecture E — test E2: analysis pipeline.

Executes the PRE-REGISTERED analysis plan of E2_DESIGN.md on results/e2_raw.csv:
  1. descriptive proportions + Wilson 95% CIs
  2. Cochran-Armitage trend tests (H1-H4, separation-robust)  -> primary
  3. monotonicity check (per-level proportions)
  4. Holm-Bonferroni over the primary family
  5. eta-squared two-way decomposition (H5 functional separation) + logistic GLM
  6. Pareto frontier (survive vs goal)
  8. per-layout consistency

Run (after e2_sweep.py finishes):  python work/12_conjecture_e/e2_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
RES = _HERE / "results"
Z95 = 1.959964

# significance level (NB: distinct from the FMC exponent alpha)
SIG = 0.05


# --- statistical primitives ---------------------------------------------------

def wilson_ci(k, n, z=Z95):
    """Wilson score interval for a binomial proportion. Robust to k=0 or k=n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cochran_armitage(scores, k, n):
    """Cochran-Armitage linear-trend test for a proportion across ordered groups.

    A score test — valid (no separation pathology) even when some groups have
    0 or n successes. Returns (z, two-sided p).
    """
    t = np.asarray(scores, float)
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    N, X = n.sum(), k.sum()
    if N == 0 or X == 0 or X == N:
        return 0.0, 1.0
    pbar = X / N
    T = np.sum(t * (k - n * pbar))
    var = pbar * (1 - pbar) * (np.sum(n * t * t) - (np.sum(n * t)) ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = T / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def holm(pvals, labels):
    """Holm-Bonferroni step-down adjustment. Returns {label: adjusted p}."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return {labels[i]: float(adj[i]) for i in range(m)}


def two_way_eta2(table):
    """Balanced two-way SS decomposition on a rows x cols matrix of cell rates.

    Returns (eta2_rows, eta2_cols, eta2_interaction). Descriptive, separation-immune.
    """
    table = np.asarray(table, float)
    nr, nc = table.shape
    g = table.mean()
    ss_tot = float(((table - g) ** 2).sum())
    if ss_tot == 0:
        return 0.0, 0.0, 0.0
    ss_row = nc * float(((table.mean(axis=1) - g) ** 2).sum())
    ss_col = nr * float(((table.mean(axis=0) - g) ** 2).sum())
    ss_int = max(0.0, ss_tot - ss_row - ss_col)
    return ss_row / ss_tot, ss_col / ss_tot, ss_int / ss_tot


# --- analysis -----------------------------------------------------------------

def ca_over(df, factor, outcome):
    """CA trend test of `outcome` across ordered levels of `factor` (equal scores)."""
    levels = sorted(df[factor].unique())
    scores = list(range(len(levels)))
    k = [int(df.loc[df[factor] == lv, outcome].sum()) for lv in levels]
    n = [int((df[factor] == lv).sum()) for lv in levels]
    return cochran_armitage(scores, k, n)


def main():
    raw = RES / "e2_raw.csv"
    if not raw.exists():
        sys.exit(f"missing {raw} — run e2_sweep.py first")
    df = pd.read_csv(raw)
    alphas = sorted(df.alpha.unique())
    betas = sorted(df.beta.unique())
    layouts = sorted(df.layout.unique())
    print(f"loaded {len(df)} episodes — {len(alphas)} alpha x {len(betas)} beta "
          f"x {len(layouts)} layouts")

    # --- 1. descriptive: per (alpha,beta) cell, pooled over layout ------------
    cell = (df.groupby(["alpha", "beta"])
              .agg(n=("died", "size"), died=("died", "sum"), goal=("goal", "sum"))
              .reset_index())
    cell["death_rate"] = cell.died / cell.n
    cell["goal_rate"] = cell.goal / cell.n
    cell["survive_rate"] = 1 - cell["death_rate"]
    dci = cell.apply(lambda r: wilson_ci(r.died, r.n), axis=1)
    gci = cell.apply(lambda r: wilson_ci(r.goal, r.n), axis=1)
    cell["death_lo"], cell["death_hi"] = zip(*dci)
    cell["goal_lo"], cell["goal_hi"] = zip(*gci)

    # --- 2. CA trend tests — primary hypotheses H1-H4 -------------------------
    z1, p1 = ca_over(df, "alpha", "died")
    z2, p2 = ca_over(df, "alpha", "goal")
    z3, p3 = ca_over(df, "beta", "died")
    z4, p4 = ca_over(df, "beta", "goal")
    trend = {
        "H1_died_vs_alpha": {"z": z1, "p": p1, "expect": "z>0 (rises)"},
        "H2_goal_vs_alpha": {"z": z2, "p": p2, "expect": "z>0 (rises)"},
        "H3_died_vs_beta": {"z": z3, "p": p3, "expect": "z<0 (falls)"},
        "H4_goal_vs_beta": {"z": z4, "p": p4, "expect": "z<0 (falls)"},
    }
    holm_adj = holm([p1, p2, p3, p4], list(trend.keys()))
    for h, ap in holm_adj.items():
        trend[h]["p_holm"] = ap

    # --- 3. monotonicity check (per-level proportions) ------------------------
    by_alpha = df.groupby("alpha")[["died", "goal"]].mean()
    by_beta = df.groupby("beta")[["died", "goal"]].mean()

    def shape(series):
        v = series.values
        if all(np.diff(v) >= -1e-9):
            return "monotone_up"
        if all(np.diff(v) <= 1e-9):
            return "monotone_down"
        return "non_monotone"

    monotonicity = {
        "goal_vs_alpha": shape(by_alpha["goal"]),
        "died_vs_alpha": shape(by_alpha["died"]),
        "died_vs_beta": shape(by_beta["died"]),
        "goal_vs_beta": shape(by_beta["goal"]),
    }

    # --- 5. H5: eta-squared two-way decomposition -----------------------------
    death_tab = cell.pivot(index="alpha", columns="beta", values="death_rate").values
    goal_tab = cell.pivot(index="alpha", columns="beta", values="goal_rate").values
    surv_tab = 1.0 - death_tab
    ea_g, eb_g, ei_g = two_way_eta2(goal_tab)
    ea_s, eb_s, ei_s = two_way_eta2(surv_tab)
    eta = {
        "goal": {"eta2_alpha": ea_g, "eta2_beta": eb_g, "eta2_interaction": ei_g},
        "survive": {"eta2_alpha": ea_s, "eta2_beta": eb_s, "eta2_interaction": ei_s},
    }

    # --- 5b. logistic GLM (supplementary effect sizes) ------------------------
    glm = {}
    try:
        import statsmodels.formula.api as smf
        for outcome in ["died", "goal"]:
            try:
                m = smf.logit(f"{outcome} ~ alpha + beta + C(layout)",
                              data=df).fit(disp=0, maxiter=200)
                ci = m.conf_int()
                glm[outcome] = {
                    "converged": bool(m.mle_retvals.get("converged", False)),
                    "or_alpha": float(np.exp(m.params["alpha"])),
                    "or_alpha_ci95": [float(np.exp(ci.loc["alpha", 0])),
                                      float(np.exp(ci.loc["alpha", 1]))],
                    "or_beta": float(np.exp(m.params["beta"])),
                    "or_beta_ci95": [float(np.exp(ci.loc["beta", 0])),
                                     float(np.exp(ci.loc["beta", 1]))],
                    "pseudo_r2": float(m.prsquared),
                }
            except Exception as ex:  # separation or convergence failure
                glm[outcome] = {"error": f"{type(ex).__name__}: {ex}"}
    except ImportError:
        glm = {"error": "statsmodels unavailable"}

    # --- 6. Pareto frontier (pooled over layout) ------------------------------
    pts = cell[["alpha", "beta", "survive_rate", "goal_rate"]].to_dict("records")
    pareto = []
    for a in pts:
        dominated = any(
            (b["survive_rate"] >= a["survive_rate"]) and
            (b["goal_rate"] >= a["goal_rate"]) and
            ((b["survive_rate"] > a["survive_rate"]) or
             (b["goal_rate"] > a["goal_rate"]))
            for b in pts)
        if not dominated:
            pareto.append(a)
    pareto.sort(key=lambda r: r["survive_rate"])

    # --- 8. per-layout consistency --------------------------------------------
    per_layout = {}
    for lay in layouts:
        d = df[df.layout == lay]
        per_layout[lay] = {
            "H1_died_vs_alpha_z": ca_over(d, "alpha", "died")[0],
            "H2_goal_vs_alpha_z": ca_over(d, "alpha", "goal")[0],
            "H3_died_vs_beta_z": ca_over(d, "beta", "died")[0],
            "H4_goal_vs_beta_z": ca_over(d, "beta", "goal")[0],
        }
    sign_consistent = {
        h: len({np.sign(per_layout[L][f"{h}_z"]) for L in layouts}) == 1
        for h in ["H1_died_vs_alpha", "H2_goal_vs_alpha",
                  "H3_died_vs_beta", "H4_goal_vs_beta"]
    }

    # --- effect sizes: extreme-cell risk differences --------------------------
    risk = {
        "death_alpha_max_minus_min": float(
            by_alpha["died"].iloc[-1] - by_alpha["died"].iloc[0]),
        "goal_alpha_max_minus_min": float(
            by_alpha["goal"].iloc[-1] - by_alpha["goal"].iloc[0]),
        "death_beta_max_minus_min": float(
            by_beta["died"].iloc[-1] - by_beta["died"].iloc[0]),
        "goal_beta_max_minus_min": float(
            by_beta["goal"].iloc[-1] - by_beta["goal"].iloc[0]),
    }

    # --- figures --------------------------------------------------------------
    _heatmaps(cell, alphas, betas)
    _pareto_plot(cell, pareto)

    # --- assemble + persist ---------------------------------------------------
    out = {
        "design": {"alphas": alphas, "betas": betas, "layouts": layouts,
                   "n_per_cell": int(cell.n.iloc[0]), "sig_level": SIG},
        "cells": cell.round(4).to_dict("records"),
        "trend_tests": trend,
        "monotonicity": monotonicity,
        "by_alpha": by_alpha.round(4).to_dict(),
        "by_beta": by_beta.round(4).to_dict(),
        "eta2_separation": eta,
        "glm": glm,
        "pareto_frontier": [{k: round(v, 4) if isinstance(v, float) else v
                             for k, v in p.items()} for p in pareto],
        "per_layout_trend_z": per_layout,
        "sign_consistent_across_layouts": sign_consistent,
        "risk_differences": risk,
    }
    (RES / "e2_stats.json").write_text(json.dumps(out, indent=2))
    _report(out)


def _heatmaps(cell, alphas, betas):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, col, title, cmap in [
            (axes[0], "death_rate", "P(died) — terminal lava", "Reds"),
            (axes[1], "goal_rate", "P(reached goal)", "Greens")]:
        tab = cell.pivot(index="alpha", columns="beta", values=col)
        im = ax.imshow(tab.values, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(betas)), betas)
        ax.set_yticks(range(len(alphas)), alphas)
        ax.set_xlabel("beta (preservazione)")
        ax.set_ylabel("alpha (desiderio)")
        ax.set_title(title)
        for i in range(len(alphas)):
            for j in range(len(betas)):
                v = tab.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("E2 — sweep alpha x beta (pooled over 3 layouts, n=60/cell)")
    fig.tight_layout()
    fig.savefig(RES / "e2_heatmaps.png", dpi=130)
    plt.close(fig)


def _pareto_plot(cell, pareto):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sc = ax.scatter(cell.survive_rate, cell.goal_rate, c=cell.alpha,
                    s=40 + 60 * cell.beta, cmap="viridis",
                    edgecolor="k", linewidth=0.4, zorder=3)
    if pareto:
        ax.plot([p["survive_rate"] for p in pareto],
                [p["goal_rate"] for p in pareto],
                "-", color="crimson", lw=1.5, zorder=2, label="frontiera Pareto")
    # E1-base baselines (pooled over 3 layouts).
    try:
        e1 = json.loads((RES / "e1_base.json").read_text())["results"]
        for name, marker in [("random", "x"), ("greedy", "+")]:
            dr = np.mean([e1[L][name]["death_rate"] for L in e1])
            gr = np.mean([e1[L][name]["goal_rate"] for L in e1])
            ax.scatter([1 - dr], [gr], marker=marker, s=120, color="black",
                       zorder=4, label=f"baseline {name}")
    except Exception:
        pass
    ax.set_xlabel("survive rate  (preservazione = 1 - P(died))")
    ax.set_ylabel("goal rate  (desiderio soddisfatto)")
    ax.set_title("E2 — frontiera di Pareto desiderio / preservazione\n"
                 "colore = alpha, dimensione = beta")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="alpha")
    fig.tight_layout()
    fig.savefig(RES / "e2_pareto.png", dpi=130)
    plt.close(fig)


def _report(out):
    t = out["trend_tests"]
    print("\n" + "=" * 74)
    print("E2 RESULTS — alpha x beta functional separation")
    print("=" * 74)
    print(f"\nTrend tests (Cochran-Armitage, primary H1-H4; Holm-corrected):")
    for h, r in t.items():
        sig = "SIG" if r["p_holm"] < SIG else "ns"
        print(f"  {h:22s} z={r['z']:+7.2f}  p={r['p']:.2e}  "
              f"p_holm={r['p_holm']:.2e}  [{sig}]  expect {r['expect']}")
    print(f"\nMonotonicity (per-level shape): {out['monotonicity']}")
    print(f"\nH5 functional separation — eta-squared decomposition:")
    for oc, e in out["eta2_separation"].items():
        print(f"  {oc:8s}: eta2_alpha={e['eta2_alpha']:.3f}  "
              f"eta2_beta={e['eta2_beta']:.3f}  "
              f"eta2_interaction={e['eta2_interaction']:.3f}")
    print(f"\nLogistic GLM (supplementary odds ratios):")
    for oc, g in out["glm"].items():
        if "error" in g:
            print(f"  {oc}: {g['error']}")
        else:
            print(f"  {oc:5s}: OR_alpha={g['or_alpha']:.3f} "
                  f"{g['or_alpha_ci95']}  OR_beta={g['or_beta']:.3f} "
                  f"{g['or_beta_ci95']}  (pseudoR2={g['pseudo_r2']:.3f})")
    print(f"\nPer-layout sign consistency: {out['sign_consistent_across_layouts']}")
    print(f"\nPareto-optimal (alpha,beta) cells [survive, goal]:")
    for p in out["pareto_frontier"]:
        print(f"  alpha={p['alpha']:<5} beta={p['beta']:<5} "
              f"survive={p['survive_rate']:.2f}  goal={p['goal_rate']:.2f}")
    print(f"\nrisk differences: {out['risk_differences']}")
    print(f"\nwrote {RES/'e2_stats.json'}, e2_heatmaps.png, e2_pareto.png")


if __name__ == "__main__":
    main()
