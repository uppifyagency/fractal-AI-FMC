"""N-dependence test: how does b_eff* scale with walker count?

If FMC dynamics is Wright-Fisher-like under weak selection, the time scale
for label fixation should be O(N) — meaning at fixed M, larger N preserves
more diversity (slower fixation).

Sergio config: alpha=0.1, beta=0.
K=9, M=15 fixed.
N sweep: {8, 16, 32, 64, 128, 256, 512}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fmc.core import virtual_reward, clone_step, effective_branching_factor
from fmc.envs.navigation2d_kN import Navigation2DKN
from bench.runner import run_cell, write_jsonl


def _measure_b_eff(seed, alpha, beta, N, M, K):
    rng = np.random.default_rng(seed)
    env = Navigation2DKN(K=K)
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
    parser.add_argument("--ns", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--K", type=int, default=9)
    parser.add_argument("--M", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results" / "N_dependence.jsonl"),
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    alpha, beta = 0.1, 0.0
    K, M = args.K, args.M

    print(f"N-dependence at Sergio config (α={alpha}, β={beta}, K={K}, M={M})")
    print(f"N values: {args.ns}, seeds={args.seeds}\n")

    results = []
    for N in args.ns:
        params = {"alpha": alpha, "beta": beta, "N": N, "M": M, "K": K}
        result = run_cell(
            benchmark="N_dependence",
            env_name=f"navigation2d_K{K}",
            params=params,
            metric="b_eff",
            sample_fn=lambda s, n=N: _measure_b_eff(s, alpha, beta, n, M, K),
            seeds=seeds,
            notes=f"N={N} K={K} M={M} at Sergio config",
        )
        results.append(result)
        print(
            f"N={N:>4d} → b_eff = {result.mean:.2f} "
            f"[{result.ci95_low:.2f}, {result.ci95_high:.2f}]   "
            f"({result.duration_seconds:.1f}s)"
        )

    write_jsonl(results, args.output)
    print(f"\nWrote {len(results)} cells to {args.output}")


if __name__ == "__main__":
    main()
