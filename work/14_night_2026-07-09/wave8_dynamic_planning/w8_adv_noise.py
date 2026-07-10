#!/usr/bin/env python3
"""ADVERSARIAL probe (D): is the n=12->n=40 reversal real, or a seeding artifact?
Reproduce offset=1.5 low budget at n in {12,20,40,60} for greedy/rand/CEM/FMC
using the EXACT w8b seed scheme, so n=40 must match the paper table. Then check
whether the 'FMC wins at low budget' pilot survives at larger n."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w8_deceptive_nav import DeceptiveNav, PLANNERS, run_episode  # noqa: E402


def rates_at(N, M, offset, n, H=70):
    succ = {k: 0 for k in PLANNERS}
    for inst in range(n):
        for k, fn in PLANNERS.items():
            rng = np.random.default_rng(90210 + inst * 31 + N * 7)  # w8b scheme
            env = DeceptiveNav(offset=offset, reward_mode="dense")
            succ[k] += int(run_episode(env, fn, N, M, H, rng)["success"])
    return {k: succ[k] / n for k in PLANNERS}


print("=" * 90)
print("NOISE/REVERSAL (D): offset=1.5, low+mid budget, growing n (w8b seed scheme, H=70)")
print("=" * 90)
for (N, M) in ((12, 6), (16, 8), (24, 10)):
    print(f"\n--- B={N*M} (N={N}, M={M}) ---")
    print(f"{'n':>4} | " + " ".join(f"{k:>11}" for k in PLANNERS) + "  | FMC-bestbase")
    for n in (12, 20, 40, 60):
        r = rates_at(N, M, 1.5, n)
        bb = max(r[k] for k in PLANNERS if k != "FMC")
        print(f"{n:>4} | " + " ".join(f"{r[k]:>11.3f}" for k in PLANNERS)
              + f"  | {r['FMC']-bb:>+8.3f}")
print("\n" + "=" * 90)
print("If FMC-bestbase flips sign as n grows -> low-n success rates are noise (paper's point).")
