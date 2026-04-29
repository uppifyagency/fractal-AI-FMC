"""P3 — RAM vs IMG ablation driver across (game, N, M) cells.

Operationalizes
docs/bibliography/protocols/P3_ram_vs_img_ablation_protocol.md.

For each (game, obs_type, N, M, seed) cell, runs one full FMC episode
and emits a JSONL row. The aggregator then computes the RAM/IMG ratio
surface in (N, M) for each game.

Cells: 8 games × 2 obs × 4 N × 4 M × 5 seeds = 1280 (matches protocol).

Usage
-----
python -m scripts.ram_img_sweep \\
    --games Boxing QBert MsPacman \\
    --N 30 60 120 \\
    --M 10 15 30 \\
    --seeds 3 \\
    --max_actions 300 \\
    --out runs/p3_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fmc-core" / "src"))

from fmc.envs.atari import AtariEnv  # noqa: E402
import fmc.core as fmc_core            # noqa: E402


def run_cell(game: str, obs_type: str, N: int, M: int, seed: int,
             max_actions: int = 300) -> dict:
    env = AtariEnv(name=f"ALE/{game}-v5", obs_type=obs_type, frame_skip=4)
    state = env.reset(seed=seed)
    t0 = time.perf_counter()
    n = 0
    while not state.done and n < max_actions:
        a = fmc_core.plan(env, state, N=N, M=M, seed=seed * 1000 + n)
        state = env.step(state, a)
        n += 1
    return {
        "game": game,
        "obs_type": obs_type,
        "N": N, "M": M,
        "seed": seed,
        "max_actions": max_actions,
        "cum_reward": state.cum_reward,
        "actions": n,
        "done": state.done,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[30, 60, 120, 240])
    ap.add_argument("--M", type=int, nargs="+", default=[10, 15, 30, 60])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_actions", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # truncate

    total = len(args.games) * 2 * len(args.N) * len(args.M) * args.seeds
    done = 0
    for game in args.games:
        for obs in ("ram", "rgb"):
            for N in args.N:
                for M in args.M:
                    for seed in range(args.seeds):
                        rec = run_cell(game, obs, N, M, seed,
                                       max_actions=args.max_actions)
                        done += 1
                        with out.open("a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"[{done}/{total}] {json.dumps(rec)}")


if __name__ == "__main__":
    main()
