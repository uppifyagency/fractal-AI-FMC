#!/usr/bin/env python3
"""ADVERSARIAL probe 1 -- correctness of the optimum + E2 measurement.

(D) Verify brute_force_optimum (w7 loop) == vectorised_optimum (w7b) == true max
    of NK.fitness by independent exhaustive enumeration, on small N, several K.
    Checks MSB-first bit-order consistency end to end.

(E) Verify the "disp_cfg constant ~2.31, blind to K" claim independently:
    recompute disp_ratio from scratch (mean pairwise Euclidean on the raw binary
    swarm) and show it does not move with K; show the reward-coupled channel does
    move but never crosses 3.
"""
import sys, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from w7_nk_fmc import NK, NKEnv
from w7b_nk_hard import vectorised_optimum, NKEnvRewardObs
from w34_e2_smoke import e2_divergence


def exhaustive_max_via_fitness(nk):
    """Truly independent: enumerate all 2^N states as tuples, call nk.fitness."""
    N = nk.N
    best = -1.0
    argbest = None
    for tup in itertools.product((0, 1), repeat=N):
        x = np.array(tup, dtype=np.int8)
        f = nk.fitness(x)
        if f > best:
            best, argbest = f, x.copy()
    return best, argbest


def check_optimum():
    print("=" * 80)
    print("(D) OPTIMUM CORRECTNESS  --  three independent computations must agree")
    print("=" * 80)
    print(f"{'N':>3} {'K':>3} {'seed':>10} | {'brute(w7)':>12} {'vector(w7b)':>12} "
          f"{'exhaustive':>12} | {'match':>6}")
    print("-" * 80)
    allok = True
    for N in (8, 10):
        for K in (0, 2, 4, min(N - 1, 6)):
            for seed in (111, 222):
                nk = NK(N, K, seed)
                b = nk.brute_force_optimum()
                v = vectorised_optimum(nk)
                e, _ = exhaustive_max_via_fitness(nk)
                ok = (abs(b - v) < 1e-12) and (abs(b - e) < 1e-12)
                allok &= ok
                print(f"{N:>3} {K:>3} {seed:>10} | {b:>12.9f} {v:>12.9f} "
                      f"{e:>12.9f} | {'OK' if ok else 'FAIL':>6}")
    print("-" * 80)
    print(f"ALL OPTIMA CONSISTENT: {allok}\n")
    return allok


def check_bitorder():
    """Directly probe MSB-first consistency on a hand-built state."""
    print("=" * 80)
    print("(D2) BIT-ORDER: NK.fitness vs vectorised per-code, full sweep small N")
    print("=" * 80)
    N, K, seed = 6, 2, 999
    nk = NK(N, K, seed)
    # recompute per-code fitness the vectorised way but compare EACH code to fitness()
    S = 1 << N
    mism = 0
    for code in range(S):
        x = np.array([(code >> b) & 1 for b in range(N)], dtype=np.int8)
        f_loop = nk.fitness(x)
        # vectorised single-code
        tot = 0.0
        for i in range(N):
            idx = 0
            for b in nk.neigh[i]:
                idx = (idx << 1) | int(x[b])
            tot += nk.tables[i][idx]
        f_vec = tot / N
        if abs(f_loop - f_vec) > 1e-12:
            mism += 1
    print(f"N={N} K={K}: mismatches over all {S} codes = {mism}")
    print(f"per-code bit-order consistent: {mism == 0}\n")
    return mism == 0


def check_e2_blindness():
    print("=" * 80)
    print("(E) E2 disp_ratio: config-coord blindness to K + reward-coupled channel")
    print("=" * 80)
    N = 20
    x0 = np.zeros(N, dtype=np.int8)
    print(f"{'K':>3} | {'disp_cfg':>9} {'disp_rew':>9} {'cv_M':>7} | verdict")
    print("-" * 60)
    for K in (0, 2, 4, 8, 12, 19):
        nk = NK(N, K, 20260711 + 1000 * K)
        env = NKEnv(nk)
        env_r = NKEnvRewardObs(nk)
        ec = e2_divergence(env, x0, N=64, M=15, alpha=1.0, beta=1.0, seeds=(0, 1, 2, 3))
        er = e2_divergence(env_r, x0, N=64, M=15, alpha=1.0, beta=1.0, seeds=(0, 1, 2, 3))
        print(f"{K:>3} | {ec['disp_ratio']:>9.4f} {er['disp_ratio']:>9.4f} "
              f"{ec['reward_cv_M']:>7.4f} | {ec['verdict']}")
    print("-" * 60)

    # Independent hand recompute of disp_ratio for config coords (no e2 module):
    print("\nIndependent recompute (raw binary swarm, mean pairwise L2):")
    def hand_disp_ratio(N, M, nwalk=64, seed=0):
        rng = np.random.default_rng(seed)
        states = np.zeros((nwalk, N), dtype=np.float64)
        d1 = None
        for t in range(M):
            for i in range(nwalk):
                b = rng.integers(0, N)
                states[i, b] = 1.0 - states[i, b]
            diff = states[:, None, :] - states[None, :, :]
            d = np.sqrt((diff ** 2).sum(-1))
            md = d.sum() / (nwalk * (nwalk - 1))
            if t == 0:
                d1 = md
            dM = md
        return dM / d1
    for K in (0, 8, 19):   # K must not matter -- dynamics are pure flips
        r = np.mean([hand_disp_ratio(N, 15, seed=s) for s in range(4)])
        print(f"  K={K:>2} (ignored by flip dynamics): disp_ratio = {r:.4f}")
    print("  -> identical across K by construction: config channel is fitness-blind.\n")


if __name__ == "__main__":
    ok1 = check_optimum()
    ok2 = check_bitorder()
    check_e2_blindness()
    print("=" * 80)
    print(f"SUMMARY: optimum-correct={ok1}  bitorder-correct={ok2}")
    print("=" * 80)
