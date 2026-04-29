"""Run a full control episode with a planner (FMC or MCTS-UCT).

A "step" of the outer control loop calls planner(env, state, B) to pick an
action, applies it once, and accumulates reward. We log per-step samples,
so the comparison is at fixed *samples-per-action* budget B.

Usage
-----
python -m scripts.run_episode --algo fmc  --env cartpole --B 1000 --seed 0
python -m scripts.run_episode --algo mcts --env cartpole --B 1000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Resolve project paths regardless of cwd.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fmc-core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fmc.envs.cartpole import CartPole       # noqa: E402
from fmc.envs.pendulum import Pendulum       # noqa: E402
import fmc.core as fmc_core                   # noqa: E402

import mcts_uct                                # noqa: E402


ENV_REGISTRY = {
    "cartpole": (CartPole, 500),    # max episode length
    "pendulum": (Pendulum, 200),
}


def _fmc_factor_budget(B: int, M_default: int = 30) -> tuple[int, int]:
    """Pick (N, M) such that N * M ~= B and M ~= M_default."""
    M = min(M_default, max(5, int(math.sqrt(B))))
    N = max(2, B // M)
    return N, M


def run(algo: str, env_name: str, B: int, seed: int, max_steps: int | None = None) -> dict:
    EnvCls, default_max_steps = ENV_REGISTRY[env_name]
    env = EnvCls()
    state = env.reset(seed=seed)
    max_steps = max_steps or default_max_steps

    total_reward = 0.0
    actions_log = []
    t0 = time.perf_counter()

    for step_i in range(max_steps):
        if not getattr(state, "alive", True):
            break
        if algo == "fmc":
            N, M = _fmc_factor_budget(B)
            a = fmc_core.plan(env, state, N=N, M=M, seed=seed * 1000 + step_i)
        elif algo == "mcts":
            a = mcts_uct.plan(env, state, sample_budget=B, seed=seed * 1000 + step_i)
        else:
            raise ValueError(f"unknown algo {algo}")

        state = env.step(state, a)
        total_reward += env.reward(state)
        actions_log.append(a)

    wall = time.perf_counter() - t0
    return {
        "algo": algo,
        "env": env_name,
        "B": B,
        "seed": seed,
        "total_reward": total_reward,
        "steps": len(actions_log),
        "wall_s": round(wall, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["fmc", "mcts"], required=True)
    ap.add_argument("--env", choices=list(ENV_REGISTRY), default="cartpole")
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = run(args.algo, args.env, args.B, args.seed, args.max_steps)
    print(json.dumps(result))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
