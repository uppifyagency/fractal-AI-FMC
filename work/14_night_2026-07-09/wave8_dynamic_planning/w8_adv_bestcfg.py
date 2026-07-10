#!/usr/bin/env python3
"""ADVERSARIAL: take the best FMC config(s) found by the grid and test them
across ALL study budgets at offset=1.5, vs random-shooting & CEM on identical
paired seeds. Decides whether the best-tuned FMC overturns the negative at any
budget. Pass configs as CLI: --cfgs "0.3:2:argmax,0.5:4:majority"
"""
import sys, os
import numpy as np
from math import sqrt, erfc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w8_deceptive_nav import plan_random_shooting, plan_cem  # noqa: E402
from w8_adv_tune import make_fmc_planner, eval_planner  # noqa: E402


def two_prop_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, erfc(abs(z) / sqrt(2))


BUDGETS = ((12, 6), (16, 8), (24, 10), (36, 11), (48, 12))


def run(cfgs, offset=1.5, n=40, H=70):
    print("=" * 108)
    print(f"BEST-CFG CROSS-BUDGET | offset={offset} n={n} H={H} | cfgs={cfgs}")
    print("=" * 108)
    hdr = f"{'B':>5} {'N':>3} {'M':>3} | {'rand':>6} {'CEM':>6} {'base*':>6}"
    for (a, b, d) in cfgs:
        hdr += f" | {'a'+str(a)+'b'+str(b)+d[:3]:>14} {'z':>6}"
    print(hdr)
    print("-" * 108)
    for (N, M) in BUDGETS:
        rk = eval_planner(plan_random_shooting, N, M, offset, n, H)
        ck = eval_planner(plan_cem, N, M, offset, n, H)
        bb = max(rk, ck)
        line = f"{N*M:>5} {N:>3} {M:>3} | {rk/n:>6.3f} {ck/n:>6.3f} {bb/n:>6.3f}"
        for (a, b, d) in cfgs:
            fk = eval_planner(make_fmc_planner(a, b, d), N, M, offset, n, H)
            z, p = two_prop_z(fk, n, bb, n)
            line += f" | {fk/n:>14.3f} {z:>+6.2f}"
        print(line)
    print("-" * 108)
    print("z = FMC-cfg vs best baseline (rand/CEM). Positive & |z|>1.96 => FMC significantly WINS.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="0.3:2.0:argmax,0.1:4.0:majority")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--offset", type=float, default=1.5)
    args = ap.parse_args()
    cfgs = []
    for tok in args.cfgs.split(","):
        a, b, d = tok.split(":")
        cfgs.append((float(a), float(b), d))
    run(cfgs, offset=args.offset, n=args.n)
