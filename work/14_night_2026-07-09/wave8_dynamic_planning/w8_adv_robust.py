#!/usr/bin/env python3
"""ADVERSARIAL self-check: is the FMC argmax-decode WIN robust, or overfit to the
40 study seeds? Re-evaluate top configs at B=396 across 3 independent seed bases
and at n=80, vs baselines on the SAME seeds each time."""
import sys, os
import numpy as np
from math import sqrt, erfc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w8_deceptive_nav import plan_random_shooting, plan_cem  # noqa: E402
from w8_adv_tune import make_fmc_planner, eval_planner  # noqa: E402


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, erfc(abs(z) / sqrt(2))


N, M, offset = 36, 11, 1.5
CFGS = [(0.5, 2.0, "argmax"), (1.0, 2.0, "argmax"), (1.0, 1.0, "argmax")]
print("=" * 96)
print(f"ROBUSTNESS | B={N*M} offset={offset} | fresh seed bases + n=80")
print("=" * 96)
print(f"{'seed_base':>10} {'n':>4} | {'rand':>6} {'CEM':>6} {'base*':>6}"
      + "".join(f" | {'a%sb%s'%(a,b):>9} {'z':>6}" for a, b, d in CFGS))
print("-" * 96)
for seed_base, n in ((90210, 40), (777, 40), (424242, 40), (90210, 80)):
    rk = eval_planner(plan_random_shooting, N, M, offset, n, 70, seed_base)
    ck = eval_planner(plan_cem, N, M, offset, n, 70, seed_base)
    bb = max(rk, ck)
    line = f"{seed_base:>10} {n:>4} | {rk/n:>6.3f} {ck/n:>6.3f} {bb/n:>6.3f}"
    for a, b, d in CFGS:
        fk = eval_planner(make_fmc_planner(a, b, d), N, M, offset, n, 70, seed_base)
        z, p = two_prop_z(fk, n, bb, n)
        line += f" | {fk/n:>9.3f} {z:>+6.2f}"
    print(line)
print("-" * 96)
print("z = best-argmax-FMC vs best baseline. If win holds across seed bases => robust overturn.")
