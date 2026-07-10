#!/usr/bin/env python3
"""
W8B -- pin down FMC's genuine edge: SAMPLE EFFICIENCY at low planning budget on a
dynamically-divergent deceptive task. This is the D2 story (FMC >> MCTS at low
sample budget) reproduced in a controlled planning env, against matched-budget
MPC baselines.

Two axes, cleanly separated:
  (1) E2 is a property of the ENV, measured at a FIXED reference horizon M_ref=30
      (not the planning budget). The deceptive-nav env is strongly E2-fit.
  (2) Planning BUDGET is swept independently. FMC's SMC resampling concentrates
      few samples; random-shooting / CEM waste them. So FMC's advantage should be
      largest at LOW budget and erode as budget grows.

High n (paired planner seeds on the fixed deceptive env) + a two-proportion
z-test of FMC vs the best baseline per budget.
"""

import sys, os
import numpy as np
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w8_deceptive_nav import DeceptiveNav, PLANNERS, run_episode   # noqa: E402
from w34_e2_smoke import e2_divergence                            # noqa: E402


def two_prop_z(k1, n1, k2, n2):
    """z and p (two-sided) for difference of proportions p1 - p2."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # normal two-sided p via erfc
    from math import erfc
    return z, erfc(abs(z) / sqrt(2))


def run(offset=1.5, instances=40, H=70,
        budgets=((12, 6), (16, 8), (24, 10), (36, 11), (48, 12))):
    env_ref = DeceptiveNav(offset=offset, reward_mode="dense")
    e2ref = e2_divergence(env_ref, env_ref.reset(), N=64, M=30, seeds=(0, 1, 2, 3))
    print("=" * 100)
    print(f"W8B -- sample-efficiency sweep | DeceptiveNav offset={offset} (dense deceptive), "
          f"n={instances}")
    print(f"ENV E2 (reference M=30): disp_ratio={e2ref['disp_ratio']:.2f} -> "
          f"{'DIVERGE (fit)' if e2ref['disp_ratio']>=3 else 'collapse'}")
    print("=" * 100)
    print(f"{'B_dec':>6} {'N':>3} {'M':>3} | "
          + " ".join(f"{k:>10}" for k in PLANNERS)
          + f" | {'FMC-best_base':>14} {'z':>6} {'p':>8}")
    print("-" * 100)
    rows = []
    for (N, M) in budgets:
        succ = {k: 0 for k in PLANNERS}
        for inst in range(instances):
            for k, fn in PLANNERS.items():
                rng = np.random.default_rng(90210 + inst * 31 + N * 7)
                env = DeceptiveNav(offset=offset, reward_mode="dense")
                succ[k] += int(run_episode(env, fn, N, M, H, rng)["success"])
        rates = {k: succ[k] / instances for k in PLANNERS}
        base_best = max(k for k in PLANNERS if k != "FMC"),
        bb = max(rates[k] for k in PLANNERS if k != "FMC")
        bb_name = [k for k in PLANNERS if k != "FMC" and rates[k] == bb][0]
        z, p = two_prop_z(succ["FMC"], instances,
                          int(round(bb * instances)), instances)
        print(f"{N*M:>6} {N:>3} {M:>3} | "
              + " ".join(f"{rates[k]:>10.2f}" for k in PLANNERS)
              + f" | {rates['FMC']-bb:>+9.2f}({bb_name[:4]}) {z:>+6.2f} {p:>8.3f}")
        rows.append((N * M, rates, z, p))
    print("-" * 100)
    print("FMC-best_base = FMC success minus the strongest non-FMC baseline at that budget.")
    print("Positive at low budget (sample efficiency), eroding as budget grows = the claim.")
    print("=" * 100)
    return rows


if __name__ == "__main__":
    for off in (1.0, 1.5):
        run(offset=off, instances=40)
        print()
