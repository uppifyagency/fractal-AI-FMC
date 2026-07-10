#!/usr/bin/env python3
"""
W7 -- FMC on tunable-ruggedness landscapes (Kauffman NK model): does the E2
divergence gate CAUSALLY predict where FMC-base beats greedy / simulated
annealing? Both W4 spikes (quantum routing, logic synthesis) found FMC only
"at par" because their landscapes were flat/monotone (E2 said no-fit and was
right). Here we DIAL ruggedness K from smooth to maximal and watch, at matched
evaluation budget, (a) the E2 signal and (b) the FMC advantage move together.

NK landscape (Kauffman 1989): N binary genes; gene i's fitness contribution
f_i depends on bit i plus K other bits (its "epistatic neighbours"); the total
fitness is the mean of the N contributions. K=0 => single smooth peak (greedy
solves it); K=N-1 => maximally rugged (exponentially many local optima).
Canonical, rigorously-defined; NOT a bespoke toy.

Fairness: every optimiser gets the SAME number of fitness evaluations B, and B
is set by the FMC budget H*N_walkers*M. We report the APPROXIMATION RATIO
found/global-optimum (global optimum found by brute force over 2^N states).

Baselines: random search, steepest-ascent hill climbing with random restarts,
simulated annealing (geometric cooling). FMC used as an optimiser = best state
visited by the swarm over a closed-loop plan (tracks the true FMC mechanism:
relativize -> virtual reward -> pairwise cloning + causal-entropy diversity).
"""

import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "fmc-core", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "wave3_validation"))

from fmc.core import virtual_reward, clone_step          # noqa: E402
from w34_e2_smoke import e2_divergence                    # noqa: E402


# ===========================================================================
# NK landscape
# ===========================================================================
class NK:
    """Kauffman NK landscape. fitness in [0,1], higher is better."""

    def __init__(self, N, K, seed):
        self.N, self.K = N, K
        rng = np.random.default_rng(seed)
        # neighbours[i] = sorted list of the (K+1) bit indices that gene i depends on
        self.neigh = []
        for i in range(N):
            others = [j for j in range(N) if j != i]
            chosen = rng.choice(others, size=K, replace=False) if K > 0 else np.array([], int)
            self.neigh.append(np.sort(np.concatenate([[i], chosen])).astype(int))
        # per-gene random fitness table over the 2^(K+1) local configurations
        self.tables = [rng.random(2 ** (K + 1)) for _ in range(N)]
        self._cache = {}

    def fitness(self, x):
        """x: np.int8 array of length N. Returns mean of per-gene contributions."""
        key = x.tobytes()
        c = self._cache.get(key)
        if c is not None:
            return c
        tot = 0.0
        for i in range(self.N):
            bits = x[self.neigh[i]]
            idx = 0
            for b in bits:
                idx = (idx << 1) | int(b)
            tot += self.tables[i][idx]
        val = tot / self.N
        self._cache[key] = val
        return val

    def brute_force_optimum(self):
        """Exact global max over all 2^N states (feasible for N<=~20)."""
        N = self.N
        best = -1.0
        x = np.zeros(N, dtype=np.int8)
        for code in range(2 ** N):
            for b in range(N):
                x[b] = (code >> b) & 1
            f = self.fitness(x)
            if f > best:
                best = f
        return best

    def count_local_optima(self, sample=4000, rng=None):
        """Fraction of random states that are 1-flip local optima (ruggedness proxy)."""
        rng = rng or np.random.default_rng(0)
        cnt = 0
        for _ in range(sample):
            x = (rng.random(self.N) < 0.5).astype(np.int8)
            f = self.fitness(x)
            is_opt = True
            for b in range(self.N):
                x[b] ^= 1
                if self.fitness(x) > f:
                    is_opt = False
                x[b] ^= 1
                if not is_opt:
                    break
            cnt += is_opt
        return cnt / sample


# ===========================================================================
# FMC-compatible environment (fmc.envs.base.Environment protocol)
# ===========================================================================
class NKEnv:
    def __init__(self, nk):
        self.nk = nk
        self.N = nk.N

    def actions(self):
        return list(range(self.N))          # action a = flip bit a

    def clone_state(self, state):
        return state.copy()

    def step(self, state, action):
        s = state.copy()
        s[action] ^= 1
        return s

    def observe(self, state):
        return state.astype(np.float64)     # Hamming geometry for the diversity term

    def reward(self, state):
        return self.nk.fitness(state)

    def sample_action(self, state, rng):
        return int(rng.integers(0, self.N))


# ===========================================================================
# Optimisers -- all take a fitness callable f(x)->float and a budget B (evals).
# A shared counter enforces the budget; each returns (best_fitness, evals_used).
# ===========================================================================
class Budget:
    def __init__(self, f, B):
        self.f, self.B, self.used = f, B, 0
        self.best = -1.0

    def __call__(self, x):
        self.used += 1
        v = self.f(x)
        if v > self.best:
            self.best = v
        return v

    @property
    def left(self):
        return self.B - self.used


def opt_random(N, B, f, rng):
    bud = Budget(f, B)
    while bud.left > 0:
        bud((rng.random(N) < 0.5).astype(np.int8))
    return bud.best


def opt_greedy_restart(N, B, f, rng):
    """Steepest-ascent hill climbing with random restarts."""
    bud = Budget(f, B)
    while bud.left > 0:
        x = (rng.random(N) < 0.5).astype(np.int8)
        fx = bud(x)
        improved = True
        while improved and bud.left > 0:
            improved = False
            best_b, best_val = -1, fx
            for b in range(N):
                if bud.left <= 0:
                    break
                x[b] ^= 1
                v = bud(x)
                x[b] ^= 1
                if v > best_val:
                    best_val, best_b = v, b
            if best_b >= 0:
                x[best_b] ^= 1
                fx = best_val
                improved = True
    return bud.best


def opt_sim_anneal(N, B, f, rng, T0=0.25, Tend=0.002):
    """Simulated annealing, single chain, geometric cooling over B steps."""
    bud = Budget(f, B)
    x = (rng.random(N) < 0.5).astype(np.int8)
    fx = bud(x)
    steps = max(B - 1, 1)
    ratio = (Tend / T0) ** (1.0 / steps)
    T = T0
    while bud.left > 0:
        b = int(rng.integers(0, N))
        x[b] ^= 1
        fy = bud(x)
        if fy >= fx or rng.random() < np.exp((fy - fx) / max(T, 1e-9)):
            fx = fy                         # accept
        else:
            x[b] ^= 1                       # reject
        T *= ratio
    return bud.best


def opt_fmc(env, N, B, N_walk, M, alpha, beta, rng, f_budget):
    """FMC as an optimiser: closed-loop plan, tracking best state visited by the
    swarm. Uses the SAME primitives as core.plan (relativize/virtual_reward/clone)
    but records max reward seen and runs until the eval budget is spent."""
    x0 = (rng.random(N) < 0.5).astype(np.int8)
    actions = list(env.actions())
    while f_budget.left > 0:
        states = [x0.copy() for _ in range(N_walk)]
        labels = np.array([actions[rng.integers(0, len(actions))] for _ in range(N_walk)],
                          dtype=object)
        for t in range(M):
            if f_budget.left <= 0:
                break
            for i in range(N_walk):
                a = labels[i] if t == 0 else int(rng.integers(0, N))
                states[i] = env.step(states[i], a)
            rewards = np.array([f_budget(s) for s in states], dtype=np.float64)
            obs = np.stack([s.astype(np.float64) for s in states])
            partners = rng.permutation(N_walk)
            for i in range(N_walk):
                if partners[i] == i:
                    partners[i] = (i + 1) % N_walk
            vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
            clone_idx = clone_step(vr, rng)
            states = [states[k].copy() for k in clone_idx]
            labels = labels[clone_idx]
        # advance the real trajectory one greedy-of-swarm step: start next plan
        # from the best current walker (closed-loop optimiser use of FMC)
        best_i = int(np.argmax([f_budget.f(s) for s in states])) if states else 0
        x0 = states[best_i].copy()
    return f_budget.best


# ===========================================================================
# Study
# ===========================================================================
def run(N=16, Ks=(0, 1, 2, 4, 8, 12, 15), instances=8,
        N_walk=48, M=12, H=8, alpha=1.0, beta=1.0):
    B = H * N_walk * M
    print("=" * 96)
    print(f"W7 -- FMC on NK landscapes | N={N}  budget B={B} evals (H*N_walk*M={H}*{N_walk}*{M})")
    print(f"      approximation ratio = found / global_optimum (brute force over 2^{N})")
    print("=" * 96)
    hdr = (f"{'K':>3} {'loc_opt%':>8} {'disp_ratio':>10} {'rew_cv_M':>9} {'E2':>9} | "
           f"{'random':>8} {'greedy':>8} {'SA':>8} {'FMC':>8} | {'FMC-grdy':>9} {'FMC-SA':>8}")
    print(hdr)
    print("-" * 96)
    results = []
    for K in Ks:
        agg = {k: [] for k in ["rnd", "grd", "sa", "fmc", "lopt", "disp", "cv", "e2fit"]}
        for inst in range(instances):
            seed = 20260710 + 1000 * K + inst
            nk = NK(N, K, seed)
            env = NKEnv(nk)
            opt = nk.brute_force_optimum()
            x0 = np.zeros(N, dtype=np.int8)
            # E2 gate FIRST (golden rule) -- reuse wave3 e2_divergence
            e2 = e2_divergence(env, x0, N=48, M=M, alpha=alpha, beta=beta, seeds=(0, 1, 2))
            agg["disp"].append(e2["disp_ratio"])
            agg["cv"].append(e2["reward_cv_M"])
            agg["e2fit"].append(1.0 if e2["disp_ratio"] >= 3.0 else 0.0)
            agg["lopt"].append(nk.count_local_optima(sample=1500,
                               rng=np.random.default_rng(seed + 7)))
            # optimisers, matched budget, approximation ratio
            r_rng = np.random.default_rng(seed + 1)
            agg["rnd"].append(opt_random(N, B, nk.fitness, r_rng) / opt)
            g_rng = np.random.default_rng(seed + 2)
            agg["grd"].append(opt_greedy_restart(N, B, nk.fitness, g_rng) / opt)
            s_rng = np.random.default_rng(seed + 3)
            agg["sa"].append(opt_sim_anneal(N, B, nk.fitness, s_rng) / opt)
            f_rng = np.random.default_rng(seed + 4)
            fb = Budget(nk.fitness, B)
            agg["fmc"].append(opt_fmc(env, N, B, N_walk, M, alpha, beta, f_rng, fb) / opt)
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        verdict = "DIVERGE" if m["disp"] >= 3.0 else "collapse"
        print(f"{K:>3} {m['lopt']*100:>7.1f}% {m['disp']:>10.3f} {m['cv']:>9.3f} "
              f"{verdict:>9} | {m['rnd']:>8.4f} {m['grd']:>8.4f} {m['sa']:>8.4f} "
              f"{m['fmc']:>8.4f} | {m['fmc']-m['grd']:>+9.4f} {m['fmc']-m['sa']:>+8.4f}")
        results.append((K, m))
    print("-" * 96)
    print("Reading: approximation ratios (1.0 = found global optimum). FMC-grdy / FMC-SA")
    print("are the FMC advantage over each baseline at identical eval budget.")
    print("=" * 96)
    return results


if __name__ == "__main__":
    run()
