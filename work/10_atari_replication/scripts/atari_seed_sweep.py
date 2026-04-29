"""P1a — multi-seed Atari replication runner with bootstrap CI95.

Drives the per-game seed sweep specified in
docs/bibliography/protocols/P1a_atari_replication_protocol.md.

For each game in the configured list, runs n_seeds full episodes with
the paper §5.1.3.3 base parameters and emits a per-cell JSONL row plus
an aggregated CSV with mean, std, and bootstrap CI95.

Usage
-----
python -m scripts.atari_seed_sweep \\
    --games Boxing QBert MsPacman \\
    --seeds 10 \\
    --N 30 --M 15 \\
    --out_runs runs/atari_seeds.jsonl \\
    --out_summary runs/atari_seeds_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fmc-core" / "src"))

from fmc.envs.atari import AtariEnv  # noqa: E402
import fmc.core as fmc_core            # noqa: E402


def _bootstrap_ci(xs: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=np.float64)
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_episode(
    game: str,
    seed: int,
    N: int,
    M: int,
    obs_type: str = "ram",
    max_actions: int = 600,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict:
    env = AtariEnv(name=f"ALE/{game}-v5", obs_type=obs_type, frame_skip=4)
    state = env.reset(seed=seed)
    t0 = time.perf_counter()
    n = 0
    while not state.done and n < max_actions:
        a = fmc_core.plan(env, state, N=N, M=M, alpha=alpha, beta=beta,
                          seed=seed * 1000 + n)
        state = env.step(state, a)
        n += 1
    return {
        "game": game,
        "seed": seed,
        "N": N,
        "M": M,
        "obs_type": obs_type,
        "alpha": alpha,
        "beta": beta,
        "max_actions": max_actions,
        "cum_reward": state.cum_reward,
        "actions": n,
        "done": state.done,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--N", type=int, default=30,
                    help="walker count (paper default 30)")
    ap.add_argument("--M", type=int, default=15,
                    help="time horizon (paper default 15)")
    ap.add_argument("--obs_type", default="ram", choices=["ram", "rgb"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max_actions", type=int, default=600)
    ap.add_argument("--out_runs", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    Path(args.out_runs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_runs).write_text("")  # truncate

    summary_rows = []
    for game in args.games:
        scores = []
        for seed in range(args.seeds):
            rec = run_episode(
                game, seed,
                N=args.N, M=args.M,
                obs_type=args.obs_type,
                alpha=args.alpha, beta=args.beta,
                max_actions=args.max_actions,
            )
            with open(args.out_runs, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))
            scores.append(rec["cum_reward"])

        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(scores, seed=42)
        summary_rows.append({
            "game": game, "n_seeds": args.seeds,
            "N": args.N, "M": args.M, "obs_type": args.obs_type,
            "mean": round(mean, 2), "std": round(std, 2),
            "ci95_lo": round(ci_lo, 2), "ci95_hi": round(ci_hi, 2),
            "min": min(scores), "max": max(scores),
        })

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("\n=== summary ===")
    for r in summary_rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
