"""Conjecture E — test E2: alpha x beta factorial sweep.

Pre-registered design: see E2_DESIGN.md.
Emits results/e2_raw.csv — one row per episode. Analysis is e2_analysis.py.

Run:  python work/12_conjecture_e/e2_sweep.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "fmc-core" / "src"))

from fmc.core import plan  # noqa: E402
from gridworld_terminal import LAVA, GOAL, parse_layout  # noqa: E402
from e1_base import LAYOUTS  # reuse the 3 pre-registered layouts  # noqa: E402

# --- factorial grid (pre-registered, E2_DESIGN.md) ---------------------------
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
BETAS = [0.0, 0.5, 1.0, 2.0]
N_EPISODES = 60

# FMC kernel config — identical to E1-base.
N_WALKERS, HORIZON_M, MAX_STEPS = 64, 20, 60


def run_episode(env, start, alpha, beta, base_seed):
    s = env.reset(start)
    for t in range(MAX_STEPS):
        if s.done:
            break
        a = plan(env, s, N=N_WALKERS, M=HORIZON_M,
                 alpha=alpha, beta=beta, seed=base_seed * 100 + t)
        s = env.step(s, a)
    cell = env.cell(s)
    return {LAVA: "lava", GOAL: "goal"}.get(cell, "timeout")


def main():
    out = _HERE / "results" / "e2_raw.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    total = len(LAYOUTS) * len(ALPHAS) * len(BETAS)
    done = 0
    for li, (lname, ltext) in enumerate(LAYOUTS.items()):
        env, start = parse_layout(ltext)
        for ai, alpha in enumerate(ALPHAS):
            for bi, beta in enumerate(BETAS):
                deaths = goals = 0
                for e in range(N_EPISODES):
                    base = 1_000_000 * li + 10_000 * ai + 100 * bi + e
                    oc = run_episode(env, start, alpha, beta, base)
                    rows.append((lname, alpha, beta, e, oc,
                                 int(oc == "lava"), int(oc == "goal")))
                    deaths += oc == "lava"
                    goals += oc == "goal"
                done += 1
                print(f"[{time.time()-t0:6.0f}s] {done:2d}/{total} "
                      f"{lname:9s} a={alpha:<5} b={beta:<5} "
                      f"death={deaths:2d}/{N_EPISODES} goal={goals:2d}/{N_EPISODES}",
                      flush=True)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "alpha", "beta", "episode",
                    "outcome", "died", "goal"])
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} episodes, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
