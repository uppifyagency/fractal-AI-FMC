"""Conjecture E / E1-LLM section 5 — the f_abs sweep. Executable NOW, no LLM.

P13 tested the absorbing structure of the world-model in binary (preserved /
broken). E1_LLM_DESIGN section 5 pre-registers it as a CURVE: ablate a controlled
fraction of the true world-model's absorbing cells, run FMC on the ablated model,
measure the death rate on the true simulator. Maps death rate vs f_abs and
locates the threshold f_abs* (hypothesis hE1L-2).

This refines P13's hP13-1 from binary to curve, and pre-characterises the gate
that the LLM-generated world-model (the full E1-LLM test) must clear.

6 layouts (3 E1-base + 3 E1-robustness), alpha in {0,0.1,1.0}, beta=1, N=64,
M=20, n=30 episodes/cell. Kernel fmc-core unchanged; WorldModelEnv(true) is
asserted bit-identical to fmc.core.plan.

Run:  python work/12_conjecture_e/e1_llm_fabs_sweep.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from e1_llm_common import (  # noqa: E402
    ALPHAS, LAYOUTS6, N_EPISODES, WorldModelEnv, absorbing_cells,
    assert_worldmodel_identity, fabs_probe, greedy_policy, make_ablated_transition,
    make_wm_fmc_policy, parse_layout, random_policy, run_episode, summarize,
)

# Pre-registered f_abs sweep grid (E1_LLM_DESIGN section 5).
F_TARGETS = [1.0, 0.95, 0.9, 0.75, 0.5, 0.0]
DEATH_OK = 0.05          # "death ~ 0" band for the f_abs* threshold


def main():
    t0 = time.time()
    assert assert_worldmodel_identity(), \
        "WorldModelEnv(true_transition) diverged from fmc.core.plan — control invalid"
    print("sanity: WorldModelEnv(true_transition) == fmc.core.plan() : OK\n")

    results = {}
    for li, (lname, ltext) in enumerate(LAYOUTS6.items()):
        env, start = parse_layout(ltext)
        abs_cells = absorbing_cells(env)
        n_abs = len(abs_cells)

        # baselines (f_abs-independent) — run once per layout
        base = {}
        for pi, (pname, pol) in enumerate((("random", random_policy),
                                           ("greedy", greedy_policy))):
            oc = [run_episode(env, start, pol, 7_000_000 + 100_000 * li + pi * 1000 + e)
                  for e in range(N_EPISODES)]
            base[pname] = summarize(oc)

        sweep = {}
        for ti, f_target in enumerate(F_TARGETS):
            n_break = int(round((1.0 - f_target) * n_abs))
            rng = np.random.default_rng(4000 + li * 10 + ti)
            broken = {abs_cells[i] for i in rng.permutation(n_abs)[:n_break]}
            tfn = make_ablated_transition(broken)
            wm = WorldModelEnv(env, tfn)
            fabs = fabs_probe(tfn, env, seed=li)["f_abs"]
            cell = {"f_target": f_target, "n_broken": n_break,
                    "f_abs": fabs, "alpha": {}}
            for a in ALPHAS:
                pol = make_wm_fmc_policy(a, wm)
                oc = [run_episode(env, start, pol,
                                  7_100_000 + 100_000 * li + 10_000 * ti
                                  + 1000 * int(a * 10) + e)
                      for e in range(N_EPISODES)]
                cell["alpha"][str(a)] = summarize(oc)
            sweep[f"f{f_target}"] = cell
            dr = {a: cell["alpha"][str(a)]["death_rate"] for a in ALPHAS}
            print(f"  [{time.time()-t0:6.1f}s] {lname:12s} f_target={f_target:4.2f} "
                  f"(f_abs={fabs:.2f}, {n_break}/{n_abs} broken)  "
                  f"death α0={dr[0.0]*100:.0f}% α0.1={dr[0.1]*100:.0f}% "
                  f"α1={dr[1.0]*100:.0f}%")
        results[lname] = {"n_absorbing": n_abs, "baselines": base, "sweep": sweep}

    _report(results)
    out = _HERE / "results" / "e1_llm_fabs_sweep.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"f_targets": F_TARGETS, "alphas": ALPHAS,
                   "n_episodes": N_EPISODES, "death_ok_band": DEATH_OK},
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}   ({time.time()-t0:.1f}s)")


def _report(results):
    layouts = list(results)
    print("\n" + "=" * 80)
    print("E1-LLM f_abs SWEEP — death rate vs absorbing-structure fidelity")
    print("=" * 80)

    # pooled baselines
    for pname in ("random", "greedy"):
        d = sum(results[L]["baselines"][pname]["deaths"] for L in layouts)
        n = sum(results[L]["baselines"][pname]["n"] for L in layouts)
        print(f"  baseline {pname:7s}: pooled death {d/n*100:5.1f}%  ({d}/{n})")

    fabs_star = {}
    for a in ALPHAS:
        ak = str(a)
        print(f"\n--- alpha = {a}  (pooled over {len(layouts)} layouts, "
              f"n={N_EPISODES}/layout) ---")
        print(f"{'f_target':>9s} {'mean f_abs':>11s} {'death%':>8s} {'goal%':>7s}")
        rows = []
        for f_target in F_TARGETS:
            fk = f"f{f_target}"
            d = sum(results[L]["sweep"][fk]["alpha"][ak]["deaths"] for L in layouts)
            g = sum(results[L]["sweep"][fk]["alpha"][ak]["goals"] for L in layouts)
            n = sum(results[L]["sweep"][fk]["alpha"][ak]["n"] for L in layouts)
            fa = float(np.mean([results[L]["sweep"][fk]["f_abs"] for L in layouts]))
            rows.append((f_target, fa, d / n, g / n))
            print(f"{f_target:9.2f} {fa:11.2f} {d/n*100:7.1f}% {g/n*100:6.1f}%")
        # monotonicity (hE1L-2): death non-increasing as f_abs rises
        by_f = sorted(rows, key=lambda r: r[1])           # ascending f_abs
        deaths = [r[2] for r in by_f]
        monotone = all(deaths[i] >= deaths[i + 1] - 0.05
                       for i in range(len(deaths) - 1))
        # threshold f_abs*: lowest f_target whose death <= band AND all higher too
        star = None
        for f_target, fa, dr, _ in sorted(rows, key=lambda r: r[0]):
            if all(rr[2] <= DEATH_OK for rr in rows if rr[0] >= f_target):
                star = f_target
                break
        fabs_star[ak] = star
        print(f"  hE1L-2 monotone (death ↓ as f_abs ↑, tol .05): {monotone}")
        print(f"  f_abs* (death ≤ {DEATH_OK*100:.0f}% at/above this f_target): "
              f"{star if star is not None else 'not reached'}")

    print("\n" + "-" * 80)
    print("VERDICT (hE1L-2, pre-registered E1_LLM_DESIGN section 6)")
    print("-" * 80)
    print(f"  f_abs* per alpha: " +
          ", ".join(f"α{a}={fabs_star[str(a)]}" for a in ALPHAS))
    print("  The full E1-LLM test is informative iff the LLM world-model's f_abs")
    print("  clears f_abs* — this sweep is that gate, now a curve not a binary.")


if __name__ == "__main__":
    main()
