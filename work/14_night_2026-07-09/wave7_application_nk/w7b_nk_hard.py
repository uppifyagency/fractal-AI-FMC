#!/usr/bin/env python3
"""
W7B -- give FMC its BEST shot and fix the E2 observation.

Two changes vs w7:
  (1) HARDER regime: N=20 (space 2^20 ~ 1.05M), tighter relative budget, so
      greedy/SA cannot nearly cover the space and genuinely get trapped. This
      is where a population method could, in principle, win.
  (2) REWARD-COUPLED E2: w7 showed the raw-configuration disp_ratio is blind to
      ruggedness (constant 2.108 for all K -- random flips diffuse identically).
      Here we also measure E2 with observation = per-gene contribution vector
      (reward-coupled coordinates) to test whether THAT tracks ruggedness.

Vectorised brute-force optimum over all 2^N states (numpy bit ops) for the
approximation-ratio denominator.
"""

import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w7_nk_fmc import (NK, NKEnv, Budget, opt_random, opt_greedy_restart,   # noqa: E402
                       opt_sim_anneal, opt_fmc)
from w34_e2_smoke import e2_divergence                                       # noqa: E402
from fmc.core import virtual_reward, clone_step                              # noqa: E402


def opt_fmc_ea(N, B, N_walk, M, alpha, beta, rng, f_budget):
    """FMC selection as a generational EA -- gives the mechanism its best shot as
    an optimiser: the swarm is NOT collapsed to a single best between plans; the
    whole population persists, each member mutates (single flip) M times per
    generation, cloning (virtual_reward + clone_step) selects, diversity (beta)
    is preserved. Best-visited tracked. Same eval budget as everything else."""
    pop = [(rng.random(N) < 0.5).astype(np.int8) for _ in range(N_walk)]
    while f_budget.left > 0:
        states = [p.copy() for p in pop]
        for _ in range(M):
            if f_budget.left <= 0:
                break
            for i in range(N_walk):
                states[i] = states[i].copy()
                states[i][int(rng.integers(0, N))] ^= 1     # mutation = single flip
            rewards = np.array([f_budget(s) for s in states], dtype=np.float64)
            obs = np.stack([s.astype(np.float64) for s in states])
            partners = rng.permutation(N_walk)
            for i in range(N_walk):
                if partners[i] == i:
                    partners[i] = (i + 1) % N_walk
            vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
            clone_idx = clone_step(vr, rng)
            states = [states[k].copy() for k in clone_idx]
        pop = states                                        # generation persists
    return f_budget.best


def vectorised_optimum(nk):
    """Exact global max over all 2^N states via vectorised bit ops."""
    N = nk.N
    S = 1 << N
    codes = np.arange(S, dtype=np.int64)
    bits = ((codes[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)  # (S, N)
    tot = np.zeros(S, dtype=np.float64)
    for i in range(N):
        nb = nk.neigh[i]
        idx = np.zeros(S, dtype=np.int64)
        for b in nb:                                    # MSB-first, matches NK.fitness
            idx = (idx << 1) | bits[:, b].astype(np.int64)
        tot += nk.tables[i][idx]
    return float(tot.max() / N)


class NKEnvRewardObs(NKEnv):
    """Same env but observation = per-gene contribution vector (reward-coupled)."""

    def observe(self, state):
        nk = self.nk
        out = np.empty(nk.N, dtype=np.float64)
        for i in range(nk.N):
            bits = state[nk.neigh[i]]
            idx = 0
            for b in bits:
                idx = (idx << 1) | int(b)
            out[i] = nk.tables[i][idx]
        return out


def run(N=20, Ks=(0, 2, 4, 8, 12, 19), instances=6,
        N_walk=64, M=15, H=10, alpha=1.0, beta=1.0):
    B = H * N_walk * M
    print("=" * 104)
    print(f"W7B -- HARD regime | N={N} space=2^{N}={1<<N}  budget B={B} "
          f"({100*B/(1<<N):.2f}% of space)")
    print("=" * 104)
    hdr = (f"{'K':>3} {'loc_opt%':>8} | {'disp_cfg':>8} {'disp_rew':>8} {'cv_M':>6} | "
           f"{'random':>7} {'greedy':>7} {'SA':>7} {'FMC':>7} {'FMC-EA':>7} | "
           f"{'EA-grd':>7} {'best-of':>7}")
    print(hdr)
    print("-" * 104)
    rows = []
    for K in Ks:
        agg = {k: [] for k in ["rnd", "grd", "sa", "fmc", "ea", "lopt",
                                "dcfg", "drew", "cv", "bestof"]}
        for inst in range(instances):
            seed = 20260711 + 1000 * K + inst
            nk = NK(N, K, seed)
            env = NKEnv(nk)
            env_r = NKEnvRewardObs(nk)
            opt = vectorised_optimum(nk)
            x0 = np.zeros(N, dtype=np.int8)
            # E2 two ways (golden rule; small budget)
            e2c = e2_divergence(env, x0, N=48, M=M, alpha=alpha, beta=beta, seeds=(0, 1, 2))
            e2r = e2_divergence(env_r, x0, N=48, M=M, alpha=alpha, beta=beta, seeds=(0, 1, 2))
            agg["dcfg"].append(e2c["disp_ratio"])
            agg["drew"].append(e2r["disp_ratio"])
            agg["cv"].append(e2c["reward_cv_M"])
            agg["lopt"].append(nk.count_local_optima(sample=1200,
                               rng=np.random.default_rng(seed + 7)))
            rr = opt_random(N, B, nk.fitness, np.random.default_rng(seed + 1)) / opt
            gr = opt_greedy_restart(N, B, nk.fitness, np.random.default_rng(seed + 2)) / opt
            sa = opt_sim_anneal(N, B, nk.fitness, np.random.default_rng(seed + 3)) / opt
            fb = Budget(nk.fitness, B)
            fm = opt_fmc(env, N, B, N_walk, M, alpha, beta,
                         np.random.default_rng(seed + 4), fb) / opt
            eb = Budget(nk.fitness, B)
            ea = opt_fmc_ea(N, B, N_walk, M, alpha, beta,
                            np.random.default_rng(seed + 5), eb) / opt
            agg["rnd"].append(rr); agg["grd"].append(gr)
            agg["sa"].append(sa); agg["fmc"].append(fm); agg["ea"].append(ea)
            agg["bestof"].append(max(rr, gr, sa, fm, ea))
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        print(f"{K:>3} {m['lopt']*100:>7.1f}% | {m['dcfg']:>8.3f} {m['drew']:>8.3f} "
              f"{m['cv']:>6.3f} | {m['rnd']:>7.4f} {m['grd']:>7.4f} {m['sa']:>7.4f} "
              f"{m['fmc']:>7.4f} {m['ea']:>7.4f} | {m['ea']-m['grd']:>+7.4f} "
              f"{m['bestof']:>7.4f}")
        rows.append((K, m))
    print("-" * 104)
    print("disp_cfg = E2 disp_ratio with config observation; disp_rew = with reward-coupled")
    print("observation. approximation ratios (1.0 = global optimum). Matched eval budget.")
    print("=" * 104)
    return rows


if __name__ == "__main__":
    run()
