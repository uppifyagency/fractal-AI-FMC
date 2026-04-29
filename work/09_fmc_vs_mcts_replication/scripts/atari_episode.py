"""Run one full Atari episode with FMC or MCTS-UCT at fixed B per action.

Usage
-----
python -m scripts.atari_episode --algo fmc  --game Boxing --B 80 --seed 0
python -m scripts.atari_episode --algo mcts --game Boxing --B 80 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fmc-core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fmc.envs.atari import AtariEnv  # noqa: E402
import fmc.core as fmc_core            # noqa: E402
import mcts_uct                        # noqa: E402


def _fmc_factor_budget(B: int, M_default: int = 15) -> tuple[int, int]:
    M = min(M_default, max(5, int(math.sqrt(B))))
    N = max(2, B // M)
    return N, M


def run(
    algo: str,
    game: str,
    B: int,
    seed: int,
    obs_type: str = "ram",
    max_actions: int = 600,
    rollout_depth: int = 15,
) -> dict:
    env = AtariEnv(name=f"ALE/{game}-v5", obs_type=obs_type, frame_skip=4)
    state = env.reset(seed=seed)
    t0 = time.perf_counter()
    n = 0
    while not state.done and n < max_actions:
        if algo == "fmc":
            N, M = _fmc_factor_budget(B)
            a = fmc_core.plan(env, state, N=N, M=M, seed=seed * 1000 + n)
        elif algo == "mcts":
            a = mcts_uct.plan(
                env, state,
                sample_budget=B,
                rollout_depth=rollout_depth,
                seed=seed * 1000 + n,
            )
        else:
            raise ValueError(algo)
        state = env.step(state, a)
        n += 1

    wall = time.perf_counter() - t0
    return {
        "algo": algo,
        "game": game,
        "obs_type": obs_type,
        "B": B,
        "seed": seed,
        "cum_reward": state.cum_reward,
        "actions": n,
        "done": state.done,
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["fmc", "mcts"], required=True)
    ap.add_argument("--game", default="Boxing")
    ap.add_argument("--obs_type", default="ram", choices=["ram", "rgb"])
    ap.add_argument("--B", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_actions", type=int, default=600)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rec = run(args.algo, args.game, args.B, args.seed,
              obs_type=args.obs_type, max_actions=args.max_actions)
    print(json.dumps(rec))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
