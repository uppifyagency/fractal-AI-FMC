#!/usr/bin/env python3
"""ADVERSARIAL probe 3 -- TRY HARD to make FMC beat greedy-restart.

Sweeps alpha in {0.1,0.5,1,2,5,10}, beta in {0,0.5,1,2}, and (N_walk,M)
swarm/horizon trade-offs at FIXED eval budget, for both FMC (planner) and
FMC-EA (generational). Focus on the interesting rugged-but-not-random regime
K in {8,12} at N=20. Runs under the UNIQUE-eval budget (the FAIR, charitable
metric that removes FMC's clone-reeval waste) so FMC gets its best shot.

Prints the BEST FMC config found and its margin vs greedy-restart.
"""
import sys, os, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

import w7_nk_fmc as W7
from w7_nk_fmc import NK, NKEnv, Budget, opt_greedy_restart, opt_sim_anneal, opt_fmc
from w7b_nk_hard import vectorised_optimum, opt_fmc_ea
from w7_adv_budget import UniqueBudget


def greedy_baseline(N, B, Ks, instances, unique):
    W7.Budget = UniqueBudget if unique else Budget
    out = {}
    for K in Ks:
        vals = []
        for inst in range(instances):
            seed = 20260711 + 1000 * K + inst
            nk = NK(N, K, seed)
            opt = vectorised_optimum(nk)
            vals.append(opt_greedy_restart(N, B, nk.fitness,
                                           np.random.default_rng(seed + 2)) / opt)
        out[K] = float(np.mean(vals))
    W7.Budget = Budget
    return out


def fmc_config(N, B, K, instances, N_walk, M, alpha, beta, variant, unique):
    W7.Budget = UniqueBudget if unique else Budget
    BudCls = UniqueBudget if unique else Budget
    fn = opt_fmc_ea if variant == "ea" else opt_fmc
    vals = []
    for inst in range(instances):
        seed = 20260711 + 1000 * K + inst
        nk = NK(N, K, seed)
        env = NKEnv(nk)
        opt = vectorised_optimum(nk)
        bud = BudCls(nk.fitness, B)
        if variant == "ea":
            r = fn(N, B, N_walk, M, alpha, beta, np.random.default_rng(seed + 5), bud)
        else:
            r = fn(env, N, B, N_walk, M, alpha, beta, np.random.default_rng(seed + 4), bud)
        vals.append(r / opt)
    W7.Budget = Budget
    return float(np.mean(vals))


def run(N=20, B=9600, Ks=(8, 12), instances=5, unique=True):
    tag = "UNIQUE-eval" if unique else "TOTAL-eval"
    print("=" * 100)
    print(f"FMC HYPERPARAMETER SWEEP vs greedy-restart | N={N} B={B} ({tag} budget) "
          f"instances={instances}")
    print("=" * 100)
    grd = greedy_baseline(N, B, Ks, instances, unique)
    for K in Ks:
        print(f"  greedy-restart baseline  K={K}: {grd[K]:.4f}")
    print("-" * 100)

    best = {K: (-9, None) for K in Ks}
    # Phase 1: alpha x beta at default swarm (N_walk=64, M=15)
    print("\nPHASE 1: alpha x beta grid at (N_walk=64, M=15)")
    print(f"{'variant':>8} {'alpha':>6} {'beta':>6} | " +
          " ".join(f"K={K:<2}(dvsG)" for K in Ks))
    print("-" * 100)
    for variant in ("fmc", "ea"):
        for alpha in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            for beta in (0.0, 0.5, 1.0, 2.0):
                cells = []
                for K in Ks:
                    v = fmc_config(N, B, K, instances, 64, 15, alpha, beta, variant, unique)
                    dv = v - grd[K]
                    cells.append(f"{v:.4f}({dv:+.3f})")
                    if v > best[K][0]:
                        best[K] = (v, (variant, alpha, beta, 64, 15))
                print(f"{variant:>8} {alpha:>6} {beta:>6} | " + "  ".join(cells))
    print("-" * 100)

    # Phase 2: swarm/horizon trade-off at the phase-1 best (alpha,beta) per variant
    print("\nPHASE 2: (N_walk, M) trade-off at fixed budget, best phase-1 (alpha,beta)")
    swarms = [(32, 30), (48, 20), (64, 15), (96, 10), (128, 8), (256, 4)]
    for K in Ks:
        _, cfg = best[K]
        variant, alpha, beta = cfg[0], cfg[1], cfg[2]
        print(f"  K={K}  best phase-1: variant={variant} alpha={alpha} beta={beta} "
              f"(greedy={grd[K]:.4f})")
        for (nw, m) in swarms:
            v = fmc_config(N, B, K, instances, nw, m, alpha, beta, variant, unique)
            dv = v - grd[K]
            flag = "  <== BEATS GREEDY" if dv > 0 else ""
            print(f"     N_walk={nw:>3} M={m:>2}: {v:.4f} (dvsG {dv:+.4f}){flag}")
            if v > best[K][0]:
                best[K] = (v, (variant, alpha, beta, nw, m))
    print("-" * 100)

    print("\nBEST FMC CONFIG FOUND (vs greedy-restart):")
    for K in Ks:
        v, cfg = best[K]
        dv = v - grd[K]
        verdict = "OVERTURNS negative claim" if dv > 0 else "greedy still wins"
        print(f"  K={K}: best={v:.4f} greedy={grd[K]:.4f}  margin={dv:+.4f}  "
              f"cfg={cfg}  -> {verdict}")
    print("=" * 100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", action="store_true", help="use total-eval budget")
    ap.add_argument("--instances", type=int, default=5)
    args = ap.parse_args()
    run(instances=args.instances, unique=not args.total)
