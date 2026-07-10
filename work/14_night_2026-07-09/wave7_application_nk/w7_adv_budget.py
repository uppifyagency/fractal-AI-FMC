#!/usr/bin/env python3
"""ADVERSARIAL probe 2 -- BUDGET FAIRNESS.

The authors count EVERY fitness call against the shared budget, including
cache-hit re-evaluations. FMC clones walkers, so many of its N_walk per-tick
evals hit IDENTICAL states -- wasted budget greedy never spends. This probe:

  (1) INSTRUMENT: at the authors' standard config, measure UNIQUE vs TOTAL
      fitness evaluations for each method. If FMC's unique/total is far below
      greedy's, FMC is structurally disadvantaged by the total-eval budget.

  (2) RE-RUN under a UNIQUE-eval budget (cache hits are FREE for everyone --
      the standard black-box-optimisation budget metric). Does FMC catch up or
      win? If it still loses, the negative claim is robust to this objection.
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

import w7_nk_fmc as W7
from w7_nk_fmc import NK, NKEnv, Budget, opt_random, opt_greedy_restart, opt_sim_anneal, opt_fmc
from w7b_nk_hard import vectorised_optimum, opt_fmc_ea

# --------------------------------------------------------------------------- #
# Instrumented total-eval budget: tracks the set of distinct states charged.  #
# --------------------------------------------------------------------------- #
_REG = []


class InstrBudget(Budget):
    def __init__(self, f, B):
        super().__init__(f, B)
        self.seen = set()
        _REG.append(self)

    def __call__(self, x):
        self.seen.add(x.tobytes())
        return super().__call__(x)


# --------------------------------------------------------------------------- #
# Unique-eval budget: cache hits are FREE; only distinct states decrement B.  #
# --------------------------------------------------------------------------- #
class UniqueBudget:
    # Stall guard: if a method makes CAP*B total calls (incl. cache hits) without
    # reaching B UNIQUE states, it has stopped exploring -- declare exhausted so
    # the `while budget.left>0` loops terminate. CAP is generous: normal FMC uses
    # ~3-5x total/unique, so this only fires on genuine collapse-to-seen stalls.
    CAP = 60

    def __init__(self, f, B):
        self.f, self.B, self.used, self.best = f, B, 0, -1.0
        self.cache = {}
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        k = x.tobytes()
        c = self.cache.get(k)
        if c is not None:
            return c
        v = self.f(x)
        self.cache[k] = v
        self.used += 1
        if v > self.best:
            self.best = v
        return v

    @property
    def left(self):
        if self.calls >= self.CAP * self.B:
            return 0
        return self.B - self.used


def instrument(N=20, Ks=(4, 8, 12), instances=6, N_walk=64, M=15, H=10):
    B = H * N_walk * M
    print("=" * 92)
    print(f"(A1) UNIQUE vs TOTAL evals at authors' config | N={N} B={B}")
    print("=" * 92)
    print(f"{'K':>3} | {'method':>8} {'total':>7} {'unique':>7} {'uniq/tot':>9} "
          f"{'uniq/greedy_uniq':>17}")
    print("-" * 92)
    W7.Budget = InstrBudget          # random/greedy/sa build Budget internally
    for K in Ks:
        gu = {}
        acc = {m: {"t": [], "u": []} for m in ("random", "greedy", "SA", "FMC", "FMC-EA")}
        for inst in range(instances):
            seed = 20260711 + 1000 * K + inst
            nk = NK(N, K, seed)
            env = NKEnv(nk)
            def measure(fn):
                _REG.clear()
                fn()
                b = _REG[0]
                return b.used, len(b.seen)
            t, u = measure(lambda: opt_random(N, B, nk.fitness, np.random.default_rng(seed + 1)))
            acc["random"]["t"].append(t); acc["random"]["u"].append(u)
            t, u = measure(lambda: opt_greedy_restart(N, B, nk.fitness, np.random.default_rng(seed + 2)))
            acc["greedy"]["t"].append(t); acc["greedy"]["u"].append(u)
            t, u = measure(lambda: opt_sim_anneal(N, B, nk.fitness, np.random.default_rng(seed + 3)))
            acc["SA"]["t"].append(t); acc["SA"]["u"].append(u)
            # FMC: pass an InstrBudget instance explicitly
            _REG.clear(); fb = InstrBudget(nk.fitness, B)
            opt_fmc(env, N, B, N_walk, M, 1.0, 1.0, np.random.default_rng(seed + 4), fb)
            acc["FMC"]["t"].append(fb.used); acc["FMC"]["u"].append(len(fb.seen))
            _REG.clear(); eb = InstrBudget(nk.fitness, B)
            opt_fmc_ea(N, B, N_walk, M, 1.0, 1.0, np.random.default_rng(seed + 5), eb)
            acc["FMC-EA"]["t"].append(eb.used); acc["FMC-EA"]["u"].append(len(eb.seen))
        greedy_u = np.mean(acc["greedy"]["u"])
        for m in ("random", "greedy", "SA", "FMC", "FMC-EA"):
            t = np.mean(acc[m]["t"]); u = np.mean(acc[m]["u"])
            print(f"{K:>3} | {m:>8} {t:>7.0f} {u:>7.0f} {u/t:>9.3f} {u/greedy_u:>17.3f}")
        print("-" * 92)
    W7.Budget = Budget               # restore


def rerun_unique(N=20, Ks=(4, 8, 12), instances=6, N_walk=64, M=15, H=10):
    B = H * N_walk * M
    print("\n" + "=" * 92)
    print(f"(A2) RE-RUN under UNIQUE-eval budget (cache hits FREE) | N={N} B={B} unique evals")
    print("=" * 92)
    print(f"{'K':>3} | {'random':>8} {'greedy':>8} {'SA':>8} {'FMC':>8} {'FMC-EA':>8} | "
          f"{'FMC-grd':>8} {'EA-grd':>8}")
    print("-" * 92)
    W7.Budget = UniqueBudget         # patch internal-budget optimisers
    rows = []
    for K in Ks:
        agg = {m: [] for m in ("rnd", "grd", "sa", "fmc", "ea")}
        for inst in range(instances):
            seed = 20260711 + 1000 * K + inst
            nk = NK(N, K, seed)
            env = NKEnv(nk)
            opt = vectorised_optimum(nk)
            agg["rnd"].append(opt_random(N, B, nk.fitness, np.random.default_rng(seed + 1)) / opt)
            agg["grd"].append(opt_greedy_restart(N, B, nk.fitness, np.random.default_rng(seed + 2)) / opt)
            agg["sa"].append(opt_sim_anneal(N, B, nk.fitness, np.random.default_rng(seed + 3)) / opt)
            fb = UniqueBudget(nk.fitness, B)
            agg["fmc"].append(opt_fmc(env, N, B, N_walk, M, 1.0, 1.0,
                                      np.random.default_rng(seed + 4), fb) / opt)
            eb = UniqueBudget(nk.fitness, B)
            agg["ea"].append(opt_fmc_ea(N, B, N_walk, M, 1.0, 1.0,
                                        np.random.default_rng(seed + 5), eb) / opt)
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        print(f"{K:>3} | {m['rnd']:>8.4f} {m['grd']:>8.4f} {m['sa']:>8.4f} "
              f"{m['fmc']:>8.4f} {m['ea']:>8.4f} | {m['fmc']-m['grd']:>+8.4f} "
              f"{m['ea']-m['grd']:>+8.4f}")
        rows.append((K, m))
    W7.Budget = Budget
    print("-" * 92)
    print("If FMC/EA are still < greedy here, the total-eval budget was NOT what sank FMC.")
    print("=" * 92)
    return rows


if __name__ == "__main__":
    instrument()
    rerun_unique()
