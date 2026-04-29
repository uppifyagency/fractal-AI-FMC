"""End-to-end smoke run: FMC drives RocketHookEnv from start to delivery.

This is the canonical "does it work?" demo. Run from the rocket_hook
directory with:

    cd work/08_simulators/rocket_hook
    python3 run_fmc.py

For a richer CLI with hydraclick + fragile.FractalTree (heavyweight stack
with torch/panel/holoviews), see run_fmc_fragile.py — that's the path you
take when you want a live shaolin dashboard. This script keeps the
dependency surface to NumPy + plangym for fast iteration and CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `env` and `fmc_swarm` importable when invoked as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from env import RocketHookEnv  # type: ignore[import-not-found]
from fmc_swarm import FMCConfig, run_episode  # type: ignore[import-not-found]


def main() -> int:
    parser = argparse.ArgumentParser(description="FMC smoke run on RocketHookEnv")
    parser.add_argument("--n_walkers", type=int, default=80)
    parser.add_argument("--time_horizon", type=int, default=20)
    parser.add_argument("--balance", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None, help="reward exponent (default = balance)")
    parser.add_argument("--beta", type=float, default=None, help="distance exponent (default = balance)")
    parser.add_argument("--dt", type=int, default=2)
    parser.add_argument("--max_decisions", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_x", type=float, default=8.0)
    parser.add_argument("--target_y", type=float, default=1.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    env = RocketHookEnv(
        seed=args.seed,
        max_steps=args.max_steps,
        target_xy=(args.target_x, args.target_y),
    )
    cfg = FMCConfig(
        n_walkers=args.n_walkers,
        time_horizon=args.time_horizon,
        balance=args.balance,
        alpha=args.alpha,
        beta=args.beta,
        dt=args.dt,
    )

    if not args.quiet:
        print(f"== RocketHook FMC smoke run ==")
        print(f"   walkers={cfg.n_walkers} horizon={cfg.time_horizon} dt={cfg.dt} balance={cfg.balance}")
        print(f"   target=({args.target_x}, {args.target_y}) seed={args.seed}")
        print()

    t0 = time.time()
    result = run_episode(
        env=env,
        config=cfg,
        seed=args.seed,
        max_decisions=args.max_decisions,
        verbose=not args.quiet,
    )
    wall = time.time() - t0
    result["wall_time_s"] = round(wall, 2)
    result["config"] = vars(args)

    print()
    print("== result ==")
    print(json.dumps(result, indent=2))

    # Exit code 0 if any reward was earned; 1 if zero (likely env broken).
    return 0 if result["cum_reward"] > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
