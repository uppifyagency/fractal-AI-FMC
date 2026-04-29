"""SMC resampling schemes comparison: FMC pairwise cloning vs canonical SMC.

Question: does FMC's pairwise cloning rule (eq.14 of 1803.05049v5) suffer worse
weight degeneracy than canonical SMC resampling (multinomial / residual / stratified)?

Setup: synthetic plan-DAG (n=15 components), random walker actions, virtual reward
weights computed identically to Round-1. Run K=32 walker for T=20 steps. After each
step, apply 4 different resampling rules and measure:
  - Effective Sample Size (ESS)
  - Weight entropy H(w)
  - Final coverage
  - Inter-step weight stability
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np


HERE = Path(__file__).parent
RESULTS = HERE / "results"


# ----------------------------- Reuse from Round-1 ------------------------------


from synthetic_walker import (
    PlanState, available_actions, step, generate_plan_dag, impact_weights,
    swarm_pairwise_distance, virtual_reward, relativize,
)


# ----------------------------- Resampling schemes ------------------------------


def fmc_pairwise_clone(states: list[PlanState], V: np.ndarray,
                       rng: np.random.Generator) -> list[PlanState]:
    """Original FMC eq.14: pair walker_i with random peer j; clone if V_j > V_i with
    probability (V_j - V_i) / V_j."""
    K = len(states)
    perm = rng.permutation(K)
    cloned = list(states)
    for i in range(K):
        j = int(perm[i])
        if i == j:
            continue
        v_i = V[i] + 1e-9
        v_j = V[j]
        if v_j <= v_i:
            continue
        p_clone = (v_j - v_i) / v_j
        if rng.random() < p_clone:
            cloned[i] = states[j]
    return cloned


def multinomial_resample(states: list[PlanState], V: np.ndarray,
                         rng: np.random.Generator) -> list[PlanState]:
    """Standard SMC multinomial resampling: each particle sampled iid from weight dist."""
    K = len(states)
    w = V / V.sum() if V.sum() > 0 else np.ones(K) / K
    idx = rng.choice(K, size=K, replace=True, p=w)
    return [states[i] for i in idx]


def residual_resample(states: list[PlanState], V: np.ndarray,
                      rng: np.random.Generator) -> list[PlanState]:
    """Residual resampling: deterministic floor + multinomial residual.
    Lower variance than multinomial."""
    K = len(states)
    w = V / V.sum() if V.sum() > 0 else np.ones(K) / K
    floor = np.floor(K * w).astype(int)
    residual_w = K * w - floor
    n_residual = K - floor.sum()
    if n_residual > 0:
        residual_w = residual_w / residual_w.sum()
        residual_idx = rng.choice(K, size=int(n_residual), replace=True, p=residual_w)
    else:
        residual_idx = np.array([], dtype=int)
    idx = np.concatenate([np.repeat(np.arange(K), floor), residual_idx])
    rng.shuffle(idx)
    return [states[i] for i in idx[:K]]


def stratified_resample(states: list[PlanState], V: np.ndarray,
                        rng: np.random.Generator) -> list[PlanState]:
    """Stratified resampling: divide [0,1] in K strata, draw one u from each."""
    K = len(states)
    w = V / V.sum() if V.sum() > 0 else np.ones(K) / K
    cum_w = np.cumsum(w)
    u = (rng.random(K) + np.arange(K)) / K
    idx = np.searchsorted(cum_w, u)
    idx = np.clip(idx, 0, K - 1)
    return [states[int(i)] for i in idx]


# ----------------------------- ESS + entropy diagnostics -----------------------


def effective_sample_size(weights: np.ndarray) -> float:
    """Kong's ESS = (Σw)² / Σw²."""
    s = weights.sum()
    if s <= 0:
        return 0.0
    return float(s * s / (weights ** 2).sum())


def weight_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of normalized weights, in nats."""
    s = weights.sum()
    if s <= 0:
        return 0.0
    p = weights / s
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def diversity_post_resample(states: list[PlanState]) -> int:
    """Number of unique state signatures after resampling."""
    return len({s.signature() for s in states})


# ----------------------------- Run loop ----------------------------------------


def run_with_resampler(resampler_name: str, resampler_fn, K: int, T: int,
                       g: nx.DiGraph, weights: dict[int, float], seed: int,
                       alpha: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    states = [PlanState(frozenset(), frozenset()) for _ in range(K)]
    n_total = g.number_of_nodes()

    ess_traj: list[float] = []
    H_traj: list[float] = []
    diversity_traj: list[int] = []
    coverage_traj: list[float] = []

    for t in range(T):
        # Walker step (random action policy, identical for all schemes)
        new_states: list[PlanState] = []
        step_rewards = np.zeros(K)
        cum_risk = np.zeros(K)
        for i, s in enumerate(states):
            actions = available_actions(s, g)
            if not actions:
                new_states.append(s)
                step_rewards[i] = -0.5
                continue
            a = actions[rng.integers(0, len(actions))]
            ns, info = step(s, a, g, weights)
            new_states.append(ns)
            step_rewards[i] = info.reward
            cum_risk[i] = info.risk

        # Compute virtual reward
        distances = swarm_pairwise_distance(new_states)
        V = virtual_reward(step_rewards - 0.3 * cum_risk, distances, beta=1.0, alpha=alpha)
        V = np.clip(V, 1e-9, None)

        # ESS + entropy BEFORE resample
        ess_traj.append(effective_sample_size(V))
        H_traj.append(weight_entropy(V))

        # Resample
        states = resampler_fn(new_states, V, rng)
        diversity_traj.append(diversity_post_resample(states))
        coverage_traj.append(float(np.mean([s.coverage(n_total) for s in states])))

    final_coverages = np.array([s.coverage(n_total) for s in states])
    return {
        "resampler": resampler_name,
        "seed": seed,
        "K": K,
        "T": T,
        "alpha": alpha,
        "ess_trajectory": ess_traj,
        "weight_entropy_trajectory": H_traj,
        "diversity_trajectory": diversity_traj,
        "coverage_trajectory": coverage_traj,
        "final_coverage_mean": float(final_coverages.mean()),
        "final_coverage_std": float(final_coverages.std()),
        "final_unique_plans": int(diversity_post_resample(states)),
        "ess_mean": float(np.mean(ess_traj)),
        "ess_min": float(np.min(ess_traj)),
        "H_mean": float(np.mean(H_traj)),
        "H_min": float(np.min(H_traj)),
        "diversity_mean": float(np.mean(diversity_traj)),
    }


def main() -> None:
    K = 32
    T = 20
    n_components = 15
    n_seeds = 5
    alpha = 1.0

    schemes = {
        "fmc_pairwise": fmc_pairwise_clone,
        "multinomial": multinomial_resample,
        "residual": residual_resample,
        "stratified": stratified_resample,
    }

    summary: dict[str, dict] = {}
    for name, fn in schemes.items():
        seed_results = []
        for seed in range(n_seeds):
            g = generate_plan_dag(n_components, 2.0, seed=20000 + seed)
            weights = impact_weights(g)
            r = run_with_resampler(name, fn, K, T, g, weights,
                                  seed=21000 + seed * 13, alpha=alpha)
            seed_results.append(r)
        # Aggregate
        coverages = np.array([r["final_coverage_mean"] for r in seed_results])
        ess_mean = np.mean([r["ess_mean"] for r in seed_results])
        ess_min = np.mean([r["ess_min"] for r in seed_results])
        H_mean = np.mean([r["H_mean"] for r in seed_results])
        H_min = np.mean([r["H_min"] for r in seed_results])
        div_mean = np.mean([r["diversity_mean"] for r in seed_results])
        summary[name] = {
            "n_seeds": n_seeds,
            "final_coverage_mean": float(coverages.mean()),
            "final_coverage_std": float(coverages.std()),
            "ess_mean_avg": float(ess_mean),
            "ess_min_avg": float(ess_min),
            "weight_entropy_mean_avg": float(H_mean),
            "weight_entropy_min_avg": float(H_min),
            "diversity_post_resample_mean": float(div_mean),
            "ess_trajectory_avg": [float(x) for x in
                                   np.mean([r["ess_trajectory"] for r in seed_results], axis=0)],
            "H_trajectory_avg": [float(x) for x in
                                 np.mean([r["weight_entropy_trajectory"] for r in seed_results], axis=0)],
        }

    (RESULTS / "09_smc_resampling_comparison.json").write_text(json.dumps(summary, indent=2))

    # Summary table
    print(f"{'Scheme':<15} {'Cov':>8} {'ESS_avg':>8} {'ESS_min':>8} {'H_avg':>8} {'H_min':>8} {'Div_post':>10}")
    print("-" * 75)
    for name, s in summary.items():
        print(f"{name:<15} {s['final_coverage_mean']:>8.3f} "
              f"{s['ess_mean_avg']:>8.2f} {s['ess_min_avg']:>8.2f} "
              f"{s['weight_entropy_mean_avg']:>8.2f} {s['weight_entropy_min_avg']:>8.2f} "
              f"{s['diversity_post_resample_mean']:>10.1f}")
    # max ESS = K = 32, min reasonable >= K/2 = 16

    print(f"\n(K={K}, T={T}, n_components={n_components}, n_seeds={n_seeds})")
    print(f"ESS max possible = {K} (all weights uniform)")
    print(f"ESS = 1 means single-particle dominates (degeneracy)")
    print(f"H max = log({K}) = {math.log(K):.3f} (uniform)")


if __name__ == "__main__":
    main()
