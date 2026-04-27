"""K dependence test for Conjecture A.

Hypothesis: if Sergio's b_eff* ~ 6 is universal, it should NOT scale linearly
with K. Run navigation2D at the Sergio-config (alpha=0.1, beta=0) for K=9
and K=16, compare b_eff*.

If b_eff*(K=9) ~ b_eff*(K=16) ~ 6, the constant is universal.
If b_eff*(K=16) ~ 12 (~ K * b_eff*(K=9) / K_9), it scales with K -> artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fmc.core import virtual_reward, clone_step, effective_branching_factor
from fmc.envs.navigation2d import Navigation2D
from fmc.envs.navigation2d_k16 import Navigation2DK16
from bench.runner import run_cell, write_jsonl


def _measure_b_eff_generic(seed, alpha, beta, N, M, env_factory):
    rng = np.random.default_rng(seed)
    env = env_factory()
    x0 = env.reset()
    actions = list(env.actions())

    states = [env.clone_state(x0) for _ in range(N)]
    labels = np.array(
        [actions[rng.integers(0, len(actions))] for _ in range(N)],
        dtype=np.int64,
    )

    for t in range(M):
        for i in range(N):
            a = labels[i] if t == 0 else env.sample_action(states[i], rng)
            states[i] = env.step(states[i], int(a))
        rewards = np.array([env.reward(s) for s in states], dtype=np.float64)
        obs = np.stack([np.asarray(env.observe(s), dtype=np.float64).ravel() for s in states])
        partners = rng.permutation(N)
        for i in range(N):
            if partners[i] == i:
                partners[i] = (i + 1) % N
        vr = virtual_reward(rewards, obs, partners, alpha=alpha, beta=beta)
        idx = clone_step(vr, rng)
        states = [env.clone_state(states[k]) for k in idx]
        labels = labels[idx]

    return effective_branching_factor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results" / "k_dependence.jsonl"),
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    N, M = 32, 15

    cases = [
        ("navigation2d_K9", Navigation2D, 9),
        ("navigation2d_K16", Navigation2DK16, 16),
    ]

    results = []
    # Sergio's config
    alpha, beta = 0.1, 0.0
    for env_name, factory, K in cases:
        params = {"alpha": alpha, "beta": beta, "N": N, "M": M, "K": K}
        result = run_cell(
            benchmark="k_dependence",
            env_name=env_name,
            params=params,
            metric="b_eff",
            sample_fn=lambda s, f=factory: _measure_b_eff_generic(s, alpha, beta, N, M, f),
            seeds=seeds,
            notes=f"K={K} at Sergio config (alpha=0.1, beta=0). If universal, b_eff ~ 6 regardless of K.",
        )
        results.append(result)
        max_possible = K
        ratio = result.mean / max_possible
        print(
            f"K={K} -> b_eff = {result.mean:.2f} "
            f"[{result.ci95_low:.2f}, {result.ci95_high:.2f}] "
            f"  (max possible = K = {K}, ratio = {ratio:.2f})"
        )

    # Comparison verdict.
    b_eff_k9 = results[0].mean
    b_eff_k16 = results[1].mean
    if b_eff_k16 < 8.0:
        print(f"\nVERDETTO: b_eff(K=16)={b_eff_k16:.2f} < 8 -> 'magic 6' is NOT scaling with K (universal)")
    else:
        print(f"\nVERDETTO: b_eff(K=16)={b_eff_k16:.2f} >= 8 -> 'magic 6' may scale with K (artifact)")

    write_jsonl(results, args.output)
    print(f"\nWrote {len(results)} cells to {args.output}")


if __name__ == "__main__":
    main()
