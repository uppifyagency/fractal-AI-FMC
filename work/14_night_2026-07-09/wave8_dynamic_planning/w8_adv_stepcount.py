#!/usr/bin/env python3
"""ADVERSARIAL probe (C): count actual env.step calls per DECISION for each
planner at the study budgets, to confirm matched-budget parity (nobody is
secretly using more sim calls than N*M). Wraps DeceptiveNav to count step()."""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w8_deceptive_nav import DeceptiveNav, PLANNERS  # noqa: E402


class CountingNav(DeceptiveNav):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_step = 0

    def step(self, s, a):
        self.n_step += 1
        return super().step(s, a)


BUDGETS = ((12, 6), (16, 8), (24, 10), (36, 11), (48, 12))

print("=" * 78)
print("FAIRNESS (C): env.step() calls per SINGLE decision vs matched budget N*M")
print("=" * 78)
print(f"{'B=N*M':>7} {'N':>3} {'M':>3} | " + " ".join(f"{k:>11}" for k in PLANNERS))
print("-" * 78)
for (N, M) in BUDGETS:
    counts = {}
    for k, fn in PLANNERS.items():
        env = CountingNav(offset=1.5, reward_mode="dense")
        s = env.reset()
        rng = np.random.default_rng(12345)
        env.n_step = 0
        fn(env, s, N, M, rng)
        counts[k] = env.n_step
    print(f"{N*M:>7} {N:>3} {M:>3} | " + " ".join(f"{counts[k]:>11}" for k in PLANNERS))
print("-" * 78)
print("greedy uses K=9 (1-step floor). rand-shoot/CEM/FMC should ~= N*M.")
print("If CEM/rand > N*M -> baseline inflated. If FMC < N*M -> FMC handicapped.")
