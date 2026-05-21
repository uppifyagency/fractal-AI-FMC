"""Conjecture E / P13 — hP13-0 redo with a fine noise grid.

P13_RESULT.md flagged hP13-0 — the keystone claim, "a VR-rank-preserving
surrogate preserves the FMC decision" — as NOT TESTED: the original eta grid
jumped from perfect rank (eta=0, Spearman 1.00) straight to destroyed rank
(eta>=1, Spearman ~0.15). No arm landed in the partial-but-high-rank regime
where the keystone is decided.

This fills the gap. S1 abs-preserved at a FINE eta grid, measuring jointly:
  - VR-rank Spearman vs FULL  (rank preservation — the hP13-0 antecedent)
  - decision-agreement vs FULL (decision preservation — the hP13-0 consequent)
  - death rate                 (sanity: self-preservation should hold)

hP13-0 is supported if decision-agreement rises with VR-rank Spearman and, in
the high-Spearman regime, reaches the pre-registered >=0.85. It is falsified if
agreement stays low even where rank is well preserved.

alpha=0 (purest Common Sense / E1 regime), beta=1, N=64, M=20, 3 layouts, n=30.
Kernel fmc-core unchanged — reuses p13_proxy.py machinery verbatim.

Run:  python work/12_conjecture_e/p13_hp13_0.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from p13_proxy import (  # noqa: E402  (p13_proxy sets up the fmc-core path)
    free_probes, full_policy, run_episode_agree, s1_policy, vr_rank_diag,
)
from e1_robustness import LAYOUTS, wilson_ci  # noqa: E402
from gridworld_terminal import parse_layout  # noqa: E402

ALPHA = 0.0
ETAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]   # fine grid (was [0,1,2,4])
N_EP = 30
AGREE_TARGET = 0.85          # pre-registered hP13-0 threshold (P13_DESIGN 6.4)
SPEARMAN_HIGH = 0.80         # "rank well preserved" cutoff for the verdict


def main():
    t0 = time.time()
    results = {}
    for li, (lname, ltext) in enumerate(LAYOUTS.items()):
        env, start = parse_layout(ltext)
        probes = free_probes(env, 12, seed=5150 + li)
        ref = full_policy(ALPHA)
        results[lname] = {}
        for ei, eta in enumerate(ETAS):
            pol = s1_policy(ALPHA, eta, True)         # S1, abs-preserved
            outcomes, agree = [], []
            for e in range(N_EP):
                base = 11_000_000 + 100_000 * li + 1_000 * ei + e
                oc, ag = run_episode_agree(env, start, pol, ref, base)
                outcomes.append(oc)
                agree.extend(ag)
            vr = vr_rank_diag(env, probes, ALPHA, eta)
            results[lname][f"eta{eta}"] = {
                "n": N_EP, "deaths": outcomes.count("lava"),
                "death_rate": outcomes.count("lava") / N_EP,
                "agreement": float(np.mean(agree)) if agree else 1.0,
                "vr_spearman_mean": None if vr is None else vr["mean"],
                "vr_spearman_worst": None if vr is None else vr["worst_tick"],
            }
        print(f"  [{time.time()-t0:6.1f}s] {lname:12s} done")

    # --- pooled report --------------------------------------------------------
    print("\n" + "=" * 78)
    print("hP13-0 REDO — VR-rank vs decision-agreement (fine eta grid, alpha=0)")
    print("=" * 78)
    print(f"{'eta':>6s} {'VR-Spearman':>13s} {'agreement':>11s} {'death%':>8s}")
    pooled = {}
    for eta in ETAS:
        k = f"eta{eta}"
        d = sum(results[L][k]["deaths"] for L in results)
        n = sum(results[L][k]["n"] for L in results)
        ag = float(np.mean([results[L][k]["agreement"] for L in results]))
        sp = [results[L][k]["vr_spearman_mean"] for L in results
              if results[L][k]["vr_spearman_mean"] is not None]
        spm = float(np.mean(sp)) if sp else float("nan")
        pooled[eta] = {"spearman": spm, "agreement": ag,
                       "death_rate": d / n, "deaths": d, "n": n}
        print(f"{eta:6.2f} {spm:13.2f} {ag:11.2f} {d/n*100:7.1f}%")

    # --- pre-registered hP13-0 verdict ---------------------------------------
    print("\n" + "-" * 78)
    # monotonicity: does agreement rise with Spearman across the grid?
    order = sorted(pooled.items(), key=lambda kv: kv[1]["spearman"])
    ags = [v["agreement"] for _, v in order]
    monotone = all(ags[i] <= ags[i + 1] + 0.05 for i in range(len(ags) - 1))
    # high-rank regime: any *genuinely noisy* eta (eta>0 -- eta=0 is the
    # degenerate true-kernel case) with Spearman >= 0.80 reaching agreement >= 0.85?
    high = {e: v for e, v in pooled.items()
            if v["spearman"] >= SPEARMAN_HIGH and e > 0.0}
    high_ok = any(v["agreement"] >= AGREE_TARGET for v in high.values())
    print(f"hP13-0: agreement rises with VR-rank (monotone, tol .05)  : {monotone}")
    print(f"hP13-0: a high-rank eta (Spearman>={SPEARMAN_HIGH}) reaches "
          f"agreement>={AGREE_TARGET} : {high_ok}  "
          f"(high-rank etas tested: {sorted(high) or 'none'})")
    if high_ok and monotone:
        print("  => hP13-0 SUPPORTED — rank preservation does carry the decision.")
    elif not high:
        print("  => hP13-0 INCONCLUSIVE — no eta achieved Spearman>=0.80; the fine "
              "grid still did not reach the high-rank regime. Finer grid needed.")
    elif monotone and not high_ok:
        print("  => hP13-0 WEAKENED — agreement tracks rank but never reaches 0.85 "
              "even at high rank: rank preservation is necessary, not sufficient.")
    else:
        print("  => hP13-0 FALSIFIED — agreement does not track VR-rank.")

    out = _HERE / "results" / "p13_hp13_0.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"alpha": ALPHA, "etas": ETAS, "n_episodes": N_EP,
                   "agree_target": AGREE_TARGET, "spearman_high": SPEARMAN_HIGH},
        "results": results,
        "pooled": {str(e): v for e, v in pooled.items()},
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
