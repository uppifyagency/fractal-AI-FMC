"""Conjecture E / P13 — hP13-0, finally tested: the phi rank-inversion knob.

P13_HP13_0_RESULT.md diagnosed why the keystone stayed untested. Additive noise
on reward/observation is all-or-nothing for VR rank: the clustered swarm makes VR
densely tied, so any finite noise reorders the whole cluster at once -- Spearman
jumps 1.00 -> 0.44 with no point between. The keystone -- "a VR-rank-preserving
surrogate preserves the FMC decision" -- needs surrogates with Spearman inside
(0.5, 1.0), which additive noise cannot produce on this task.

The fix (P13_HP13_0_RESULT.md section 3): degrade VR rank with a DIRECT knob --
a controlled fraction phi of pairwise rank inversions on the VR vector. Here:
random disjoint transpositions of VR values among round(phi*N) walkers. This
  - preserves the VR multiset EXACTLY (magnitudes untouched; ties stay ties, so
    no all-or-nothing pathology) -- only the walker<->VR assignment is permuted;
  - gives Spearman ~ 1 - phi by construction, smooth across the whole (0,1) range.

The world model is the TRUE kernel (FullSchema): only the VR vector is corrupted,
nothing else. This isolates hP13-0 exactly -- "VR rank corrupted by phi -> is the
FMC decision preserved?" -- with absorption fully modelled (so survival, which
P13 showed depends on absorbing structure, should hold: death ~ 0%).

alpha=0 (purest Common Sense / E1 regime), beta=1, N=64, M=20, 3 layouts, n=30.
Kernel fmc-core unchanged -- reuses p13_proxy machinery; proxy_plan gained one
optional vr_hook param (default None -> bit-identical to fmc.core.plan).

Run:  python work/12_conjecture_e/p13_hp13_0_phi.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from p13_proxy import (  # noqa: E402  (p13_proxy sets up the fmc-core path)
    BETA, HORIZON_M, N_WALKERS, FullSchema, full_policy, proxy_plan,
    run_episode_agree, spearmanr,
)
from e1_robustness import LAYOUTS  # noqa: E402
from gridworld_terminal import parse_layout  # noqa: E402

ALPHA = 0.0
PHIS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0]
N_EP = 30
AGREE_TARGET = 0.85          # pre-registered hP13-0 threshold (P13_DESIGN 6.4)
SPEARMAN_HIGH = 0.80         # "rank well preserved" cutoff for the verdict


class RankCorruptor:
    """Degrades VR rank by phi: round(phi*N) walkers paired at random, VR values
    swapped within each disjoint pair. Preserves the VR multiset exactly
    (magnitudes untouched, ties stay ties -- no all-or-nothing pathology); only
    the walker<->VR assignment is permuted. Logs the achieved per-tick Spearman."""

    def __init__(self, phi: float):
        self.phi = phi
        self.spearmans: list[float] = []

    def __call__(self, vr, rng):
        n = len(vr)
        out = vr.copy()
        m = int(round(self.phi * n))
        m -= m % 2                                  # even -> disjoint pairs
        if m >= 2:
            idx = rng.permutation(n)[:m]
            for k in range(0, m, 2):
                i, j = idx[k], idx[k + 1]
                out[i], out[j] = vr[j], vr[i]
        if spearmanr is not None and np.std(vr) > 1e-12 and np.std(out) > 1e-12:
            rho = float(spearmanr(vr, out)[0])
            if not np.isnan(rho):
                self.spearmans.append(rho)
        elif self.phi == 0.0:
            self.spearmans.append(1.0)              # identity: rank intact
        return out


def rankcorrupt_policy(alpha, corruptor):
    """Policy on the TRUE kernel, with VR rank degraded by `corruptor` each tick."""
    return lambda env, s, seed: proxy_plan(
        env, s, N_WALKERS, HORIZON_M, alpha, BETA, seed, FullSchema(), corruptor)[0]


def main():
    t0 = time.time()
    ref = full_policy(ALPHA)                        # pure kernel control
    envs = {ln: parse_layout(lt) for ln, lt in LAYOUTS.items()}

    results = {}
    for pi, phi in enumerate(PHIS):
        corruptor = RankCorruptor(phi)
        pol = rankcorrupt_policy(ALPHA, corruptor)
        per_layout = {}
        for li, (lname, (env, start)) in enumerate(envs.items()):
            outcomes, agree = [], []
            for e in range(N_EP):
                base = 12_000_000 + 100_000 * pi + 1_000 * li + e
                oc, ag = run_episode_agree(env, start, pol, ref, base)
                outcomes.append(oc)
                agree.extend(ag)
            per_layout[lname] = {
                "n": N_EP, "deaths": outcomes.count("lava"),
                "goals": outcomes.count("goal"),
                "agreement": float(np.mean(agree)) if agree else 1.0,
            }
        sp = corruptor.spearmans
        results[f"phi{phi}"] = {
            "phi": phi,
            "vr_spearman_mean": float(np.mean(sp)) if sp else float("nan"),
            "vr_spearman_std": float(np.std(sp)) if sp else float("nan"),
            "spearman_samples": len(sp),
            "layouts": per_layout,
        }
        print(f"  [{time.time()-t0:6.1f}s] phi={phi:4.2f} done")

    # --- pooled report --------------------------------------------------------
    print("\n" + "=" * 78)
    print("hP13-0 (phi rank-inversion knob) -- VR-rank vs decision-agreement, alpha=0")
    print("=" * 78)
    print(f"{'phi':>5s} {'VR-Spearman':>13s} {'agreement':>11s} {'death%':>8s} "
          f"{'goal%':>7s}")
    pooled = {}
    for phi in PHIS:
        r = results[f"phi{phi}"]
        L = r["layouts"]
        d = sum(L[x]["deaths"] for x in L)
        g = sum(L[x]["goals"] for x in L)
        n = sum(L[x]["n"] for x in L)
        ag = float(np.mean([L[x]["agreement"] for x in L]))
        pooled[phi] = {"spearman": r["vr_spearman_mean"], "agreement": ag,
                       "death_rate": d / n, "goal_rate": g / n, "deaths": d, "n": n}
        print(f"{phi:5.2f} {r['vr_spearman_mean']:13.2f} {ag:11.2f} "
              f"{d/n*100:7.1f}% {g/n*100:6.1f}%")

    # --- pre-registered hP13-0 verdict (P13_DESIGN section 6.4) ---------------
    print("\n" + "-" * 78)
    order = sorted(pooled.items(), key=lambda kv: kv[1]["spearman"])
    ags = [v["agreement"] for _, v in order]
    monotone = all(ags[i] <= ags[i + 1] + 0.05 for i in range(len(ags) - 1))
    # high-rank regime: a genuinely corrupted arm (phi>0) whose rank is still
    # well preserved (Spearman >= 0.80). Does it keep agreement >= 0.85?
    high = {p: v for p, v in pooled.items()
            if v["spearman"] >= SPEARMAN_HIGH and p > 0.0}
    high_ok = any(v["agreement"] >= AGREE_TARGET for v in high.values())
    # rank-irrelevance check: does agreement stay high even where rank is destroyed?
    low = min(pooled.values(), key=lambda v: v["spearman"])
    rank_irrelevant = low["agreement"] >= AGREE_TARGET
    # near-perfect-rank falsifier: VR rank near-perfectly preserved (Spearman
    # >= 0.95) yet the decision is not -> the keystone's SUFFICIENCY claim is
    # falsified outright, not merely weakened.
    nearperfect = {p: v for p, v in pooled.items()
                   if v["spearman"] >= 0.95 and p > 0.0}
    keystone_falsified = bool(nearperfect) and all(
        v["agreement"] < AGREE_TARGET for v in nearperfect.values())

    print(f"hP13-0: agreement rises with VR-rank (monotone, tol .05)   : {monotone}")
    print(f"hP13-0: a high-rank phi>0 (Spearman>={SPEARMAN_HIGH}) reaches "
          f"agreement>={AGREE_TARGET} : {high_ok}  "
          f"(high-rank phis: {sorted(high) or 'none'})")
    print(f"hP13-0: agreement at lowest rank (Spearman={low['spearman']:.2f}) "
          f"= {low['agreement']:.2f}")

    if not high:
        verdict = "INCONCLUSIVE"
        print("  => hP13-0 INCONCLUSIVE -- no phi>0 reached Spearman>=0.80.")
    elif rank_irrelevant:
        verdict = "RANK-ROBUST"
        print("  => hP13-0 SUPERSEDED (RANK-ROBUST) -- the FMC decision survives "
              "even destroyed VR rank; rank preservation is not needed. Strong "
              "positive for E1-LLM feasibility.")
    elif keystone_falsified:
        verdict = "FALSIFIED"
        print("  => hP13-0 FALSIFIED -- VR rank near-perfectly preserved "
              "(Spearman>=0.95) yet decision-agreement <0.85: rank preservation "
              "does NOT carry the FMC decision. Keystone sufficiency claim dead.")
    elif high_ok and monotone:
        verdict = "SUPPORTED"
        print("  => hP13-0 SUPPORTED -- agreement tracks VR-rank and stays high "
              "where rank is well preserved. The keystone holds.")
    elif monotone and not high_ok:
        verdict = "WEAKENED"
        print("  => hP13-0 WEAKENED -- agreement tracks rank but never reaches "
              "0.85 even at high rank: rank preservation necessary, not sufficient.")
    else:
        verdict = "FALSIFIED"
        print("  => hP13-0 FALSIFIED -- agreement does not track VR-rank.")

    out = _HERE / "results" / "p13_hp13_0_phi.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"alpha": ALPHA, "phis": PHIS, "n_episodes": N_EP,
                   "N": N_WALKERS, "M": HORIZON_M, "beta": BETA,
                   "agree_target": AGREE_TARGET, "spearman_high": SPEARMAN_HIGH},
        "verdict": verdict,
        "results": results,
        "pooled": {str(p): v for p, v in pooled.items()},
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
