"""Navigation2D alpha x beta sweep — second task for Conjecture A (Bet 3).

Same sweep as rocket_sweep but on a different physics + reward landscape.
If Sergio's b_eff* ~ 6 is universal, this sweep should show a sweet spot
in the same band on this task too.

Usage:
    python -m bench.navigation2d_sweep
    python -m bench.navigation2d_sweep --quick
    python -m bench.navigation2d_sweep --full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fmc.core import virtual_reward, clone_step, effective_branching_factor
from fmc.envs.navigation2d import Navigation2D
from bench.runner import run_cell, write_jsonl


def _measure_b_eff(seed: int, alpha: float, beta: float, N: int, M: int) -> float:
    rng = np.random.default_rng(seed)
    env = Navigation2D()
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results" / "navigation2d_sweep.jsonl"),
    )
    args = parser.parse_args()

    if args.quick:
        n_seeds = 3
        cells = [(0.0, 0.0), (0.1, 0.0), (1.0, 1.0)]
    elif args.full:
        n_seeds = 20
        alphas = [0.0, 0.1, 0.5, 1.0]
        betas = [0.0, 0.5, 1.0]
        cells = [(a, b) for a in alphas for b in betas]
    else:
        n_seeds = 8
        cells = [(0.0, 0.0), (0.1, 0.0), (0.5, 1.0), (1.0, 1.0)]

    N, M = 32, 15
    seeds = list(range(n_seeds))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for alpha, beta in cells:
        params = {"alpha": alpha, "beta": beta, "N": N, "M": M}
        notes = f"navigation2d alpha={alpha} beta={beta} (Bet 3 second task for Conjecture A)"
        result = run_cell(
            benchmark="navigation2d_alpha_beta_sweep",
            env_name="navigation2d_K9",
            params=params,
            metric="b_eff",
            sample_fn=lambda s, a=alpha, b=beta: _measure_b_eff(s, a, b, N, M),
            seeds=seeds,
            notes=notes,
        )
        results.append(result)
        print(
            f"alpha={alpha:.2f} beta={beta:.2f} -> "
            f"b_eff = {result.mean:.2f} "
            f"[{result.ci95_low:.2f}, {result.ci95_high:.2f}] "
            f"(n={n_seeds}, {result.duration_seconds:.1f}s)"
        )

    write_jsonl(results, args.output)
    print(f"\nWrote {len(results)} cells to {args.output}")


if __name__ == "__main__":
    main()
