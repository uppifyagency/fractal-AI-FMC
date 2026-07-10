#!/usr/bin/env python3
"""ADVERSARIAL probe 4 -- OUT-OF-SAMPLE validation of the sweep 'winners'.

A grid sweep over 24 cells x 2 variants x 2 K with only 3 instances WILL surface
some cells that beat greedy by chance (winner's curse). The honest test: take the
apparent winners and re-run them on a LARGER, DIFFERENT seed set (fresh NK
landscapes), paired against greedy on the SAME instances, and report the paired
mean difference with its standard error. If the edge survives, it's real; if it
evaporates into noise, the negative claim stands.

Runs under BOTH budget metrics (unique-eval = charitable to FMC; total-eval =
authors' protocol) so we can say exactly under which the claim would flip.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

import w7_nk_fmc as W7
from w7_nk_fmc import NK, NKEnv, Budget, opt_greedy_restart
from w7b_nk_hard import vectorised_optimum, opt_fmc_ea
from w7_nk_fmc import opt_fmc
from w7_adv_budget import UniqueBudget

# apparent winners from w7_adv_sweep (instances=3) + the authors' default:
CANDIDATES = [
    ("fmc", 2.0, 1.0, 64, 15),   # best K=12 in sweep (+0.029)
    ("fmc", 2.0, 0.5, 64, 15),   # K=12 (+0.017)
    ("fmc", 1.0, 1.0, 64, 15),   # authors' default
    ("ea",  1.0, 0.5, 64, 15),   # best K=8 in sweep (+0.007)
    ("ea",  10.0, 2.0, 64, 15),  # K=12 (+0.020)
]


def run_one(variant, N, B, K, seed, nw, m, alpha, beta, BudCls):
    W7.Budget = BudCls
    nk = NK(N, K, seed)
    env = NKEnv(nk)
    opt = vectorised_optimum(nk)
    bud = BudCls(nk.fitness, B)
    if variant == "ea":
        r = opt_fmc_ea(N, B, nw, m, alpha, beta, np.random.default_rng(seed + 5), bud)
    else:
        r = opt_fmc(env, N, B, nw, m, alpha, beta, np.random.default_rng(seed + 4), bud)
    W7.Budget = Budget
    return r / opt


def greedy_one(N, B, K, seed, BudCls):
    W7.Budget = BudCls
    nk = NK(N, K, seed)
    opt = vectorised_optimum(nk)
    r = opt_greedy_restart(N, B, nk.fitness, np.random.default_rng(seed + 2)) / opt
    W7.Budget = Budget
    return r


def validate(N=20, B=9600, Ks=(8, 12), instances=16, seed_base=40260711):
    print("=" * 104)
    print(f"OUT-OF-SAMPLE VALIDATION | N={N} B={B} instances={instances} "
          f"(FRESH seeds base={seed_base})")
    print("PAIRED vs greedy on identical instances. margin = mean(FMC - greedy); "
          "se = std/sqrt(n).")
    print("=" * 104)
    for budget_name, BudCls in (("UNIQUE-eval", UniqueBudget), ("TOTAL-eval", Budget)):
        print(f"\n---- {budget_name} budget " + "-" * 80)
        print(f"{'K':>3} {'variant':>7} {'a':>4} {'b':>4} | {'greedy':>7} {'FMC':>7} "
              f"{'margin':>8} {'se':>7} {'t=m/se':>7} {'win%':>6} | verdict")
        print("-" * 100)
        for K in Ks:
            seeds = [seed_base + 1000 * K + i for i in range(instances)]
            g = np.array([greedy_one(N, B, K, s, BudCls) for s in seeds])
            for (variant, alpha, beta, nw, m) in CANDIDATES:
                f = np.array([run_one(variant, N, B, K, s, nw, m, alpha, beta, BudCls)
                              for s in seeds])
                diff = f - g
                margin = diff.mean()
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                t = margin / se if se > 0 else 0.0
                winpct = 100.0 * np.mean(diff > 0)
                if margin > 0 and t > 2.0:
                    verdict = "FMC WINS (sig)"
                elif margin > 0:
                    verdict = "FMC up, NOT sig"
                else:
                    verdict = "greedy wins"
                print(f"{K:>3} {variant:>7} {alpha:>4} {beta:>4} | {g.mean():>7.4f} "
                      f"{f.mean():>7.4f} {margin:>+8.4f} {se:>7.4f} {t:>7.2f} "
                      f"{winpct:>5.0f}% | {verdict}")
            print("-" * 100)
    print("=" * 104)


if __name__ == "__main__":
    validate()
