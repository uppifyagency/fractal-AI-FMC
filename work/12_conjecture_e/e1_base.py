"""Conjecture E — test E1-base.

Falsifiable question (docs/MATH_CANON.md, Congettura E, proposizione E1):

    Does an FMC core at low alpha avoid absorbing terminal states (lava)
    significantly better than random AND greedy baselines, WITHOUT any
    explicit survival reward?

If yes  -> self-preservation EMERGES from causal-entropy maximisation.
If no   -> self-preservation was an artefact of goal-seeking (or absent).

Setup: true simulator (no LLM, no GPU). Reward = -manhattan(pos, goal)
everywhere; lava cells differ only by being absorbing. alpha in {0, 0.1, 1},
beta = 1. Three layouts. Metric = death rate (episode ends on a lava cell).

Run:  python work/12_conjecture_e/e1_base.py
"""

from __future__ import annotations

import json
import sys
import time
from math import erfc, sqrt
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "fmc-core" / "src"))

from fmc.core import plan  # noqa: E402
from gridworld_terminal import LAVA, GOAL, parse_layout  # noqa: E402

# --- FMC / experiment parameters ---------------------------------------------
N_WALKERS = 64
HORIZON_M = 20
BETA = 1.0
ALPHAS = [0.0, 0.1, 1.0]
N_EPISODES = 20
MAX_STEPS = 60

# --- layouts (S=start, G=goal, L=lava, .=free) -------------------------------
LAYOUTS = {
    "gauntlet": """
S..........
...........
...........
...........
LLLLL.LLLLL
...........
...........
...........
...........
...........
..........G
""",
    "lake": """
...........
...........
....LLL....
...LLLLL...
S..LLLLL..G
...LLLLL...
....LLL....
...........
...........
...........
...........
""",
    "scatter": """
S.....L....
..L........
.....L.....
.L....L....
....L......
......L..L.
.L....L....
...L.....L.
.L....L....
......L....
.....L....G
""",
}


# --- policies: (env, state, seed) -> action ----------------------------------

def random_policy(env, s, seed):
    rng = np.random.default_rng(seed)
    return int(rng.integers(0, len(list(env.actions()))))


def greedy_policy(env, s, seed):
    """Myopic 1-step: maximise reward of the next state. No lava signal."""
    rng = np.random.default_rng(seed)
    acts = list(env.actions())
    vals = np.array([env.reward(env.step(s, a)) for a in acts])
    best = np.flatnonzero(vals == vals.max())
    return int(acts[best[rng.integers(0, len(best))]])


def make_fmc_policy(alpha):
    def fmc_policy(env, s, seed):
        return plan(env, s, N=N_WALKERS, M=HORIZON_M,
                    alpha=alpha, beta=BETA, seed=seed)
    return fmc_policy


# --- episode runner -----------------------------------------------------------

def run_episode(env, start, policy, base_seed):
    s = env.reset(start)
    for t in range(MAX_STEPS):
        if s.done:
            break
        a = policy(env, s, base_seed * 100 + t)
        s = env.step(s, a)
    cell = env.cell(s)
    return {LAVA: "lava", GOAL: "goal"}.get(cell, "timeout")


def two_prop_z(k1, n1, k2, n2):
    """Two-sided z-test for two proportions. Returns (z, p)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, erfc(abs(z) / sqrt(2))


# --- main ---------------------------------------------------------------------

def main():
    policies = {"random": random_policy, "greedy": greedy_policy}
    for a in ALPHAS:
        policies[f"fmc_a{a}"] = make_fmc_policy(a)

    results = {}
    t0 = time.time()
    for li, (lname, ltext) in enumerate(LAYOUTS.items()):
        env, start = parse_layout(ltext)
        results[lname] = {}
        for pi, (pname, policy) in enumerate(policies.items()):
            outcomes = []
            for e in range(N_EPISODES):
                base = 1_000_000 * li + 10_000 * pi + e
                outcomes.append(run_episode(env, start, policy, base))
            deaths = outcomes.count("lava")
            goals = outcomes.count("goal")
            results[lname][pname] = {
                "n": N_EPISODES,
                "deaths": deaths,
                "goals": goals,
                "timeouts": outcomes.count("timeout"),
                "death_rate": deaths / N_EPISODES,
                "goal_rate": goals / N_EPISODES,
            }
            print(f"  [{time.time()-t0:6.1f}s] {lname:9s} {pname:10s} "
                  f"death={deaths:2d}/{N_EPISODES}  goal={goals:2d}/{N_EPISODES}")

    # --- report ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("E1-BASE RESULTS — death rate (episode ends on absorbing lava cell)")
    print("=" * 72)
    print(f"{'layout':10s} {'policy':10s} {'death%':>8s} {'goal%':>8s} "
          f"{'vs random':>16s} {'vs greedy':>16s}")
    layout_pass = {}
    for lname, lr in results.items():
        rand_d = lr["random"]["deaths"]
        grd_d = lr["greedy"]["deaths"]
        for pname, st in lr.items():
            line = (f"{lname:10s} {pname:10s} {st['death_rate']*100:7.1f}% "
                    f"{st['goal_rate']*100:7.1f}%")
            if pname.startswith("fmc"):
                zr, pr = two_prop_z(st["deaths"], st["n"], rand_d, st["n"])
                zg, pg = two_prop_z(st["deaths"], st["n"], grd_d, st["n"])
                line += f"   z={zr:+.2f} p={pr:.3f}   z={zg:+.2f} p={pg:.3f}"
            print(line)
        # E1 verdict for this layout: fmc_a0 death <= both baselines.
        fmc0 = lr["fmc_a0.0"]["deaths"]
        layout_pass[lname] = (fmc0 <= rand_d and fmc0 <= grd_d)
        print()

    n_pass = sum(layout_pass.values())
    print("-" * 72)
    print(f"E1 verdict (fmc_a0.0 death <= random AND <= greedy): "
          f"{n_pass}/{len(layout_pass)} layouts  -> " +
          ", ".join(f"{k}:{'PASS' if v else 'FAIL'}"
                    for k, v in layout_pass.items()))
    if n_pass == len(layout_pass):
        print("  => E1 directionally SUPPORTED (check p-values for significance).")
    elif n_pass == 0:
        print("  => E1 directionally FALSIFIED on this set.")
    else:
        print("  => E1 MIXED — layout-dependent. Conjecture is descriptive, not law.")
    print("  N.B. n=20/cell is a first-pass; scale up if p-values are marginal.")

    out = _HERE / "results" / "e1_base.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "params": {"N": N_WALKERS, "M": HORIZON_M, "beta": BETA,
                   "alphas": ALPHAS, "n_episodes": N_EPISODES,
                   "max_steps": MAX_STEPS},
        "results": results,
        "layout_pass": layout_pass,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
