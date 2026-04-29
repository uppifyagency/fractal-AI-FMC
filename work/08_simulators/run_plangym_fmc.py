"""Run FMC on a plangym built-in env and save a GIF.

This is the *canonical* path: plangym provides the env (CartPole, LunarLander,
Atari, dm_control), our minimal NumPy FMC swarm in
work/08_simulators/rocket_hook/fmc_swarm.py drives it.

Usage:
    python3 run_plangym_fmc.py                       # default: CartPole-v1
    python3 run_plangym_fmc.py --env LunarLander-v3
    python3 run_plangym_fmc.py --env ALE/Boxing-v5 --n_walkers 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Suppress duplicate-SDL warnings on macOS (cv2 + pygame both ship libSDL).
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

_HERE = Path(__file__).resolve().parent
_ROCKET_DIR = _HERE / "rocket_hook"
if str(_ROCKET_DIR) not in sys.path:
    sys.path.insert(0, str(_ROCKET_DIR))

import numpy as np
import plangym

from fmc_swarm import FMCConfig, FMCSwarm  # type: ignore[import-not-found]


def main() -> int:
    parser = argparse.ArgumentParser(description="FMC on a plangym built-in env")
    parser.add_argument("--env", default="CartPole-v1", help="plangym env id")
    parser.add_argument("--n_walkers", type=int, default=80)
    parser.add_argument("--time_horizon", type=int, default=20)
    parser.add_argument("--balance", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--dt", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif", type=str, default="run.gif", help="output GIF path (only when --mode=gif)")
    parser.add_argument("--gif_fps", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("live", "gif", "headless"),
        default="live",
        help="live = open the env's native window (gym render_mode='human'), "
             "gif = save frames to GIF, headless = no rendering at all",
    )
    args = parser.parse_args()

    print(f"== FMC on plangym '{args.env}' (mode={args.mode}) ==")
    if args.mode == "live":
        env = plangym.make(name=args.env, render_mode="human")
    elif args.mode == "gif":
        env = plangym.make(name=args.env, render_mode="rgb_array")
    else:
        env = plangym.make(name=args.env)
    print(f"   obs_space={env.observation_space.shape} action_space={env.action_space}")

    cfg = FMCConfig(
        n_walkers=args.n_walkers,
        time_horizon=args.time_horizon,
        balance=args.balance,
        alpha=args.alpha,
        beta=args.beta,
        dt=args.dt,
    )
    swarm = FMCSwarm(env, cfg, rng_seed=args.seed)

    state, obs, info = env.reset(return_state=True)
    cum_reward = 0.0
    frames: list[np.ndarray] = []
    t0 = time.time()
    print("running... close the window or press Ctrl+C to stop")
    try:
        for step in range(args.max_steps):
            root = env.get_state()
            action, info_step = swarm.decide(root)
            env.set_state(root)
            out = env.step(action=action, return_state=True)
            if len(out) == 6:
                state, obs, r, term, trunc, info = out
            else:
                obs, r, term, trunc, info = out
            cum_reward += float(r)

            if args.mode == "live":
                # gymnasium with render_mode='human' renders inside step(),
                # but plangym calls set_state() which doesn't trigger render.
                # Force a render to refresh the window.
                try:
                    env.gym_env.render()
                except Exception:
                    pass
            elif args.mode == "gif":
                try:
                    img = env.get_image()
                    if img is not None and img.size > 0:
                        frames.append(np.asarray(img))
                except Exception as e:  # noqa: BLE001
                    print(f"  [render warn step={step}: {e}]")

            if step % 25 == 0 or term or trunc:
                print(
                    f"  step={step:4d} action={action} r={r:+.2f} cum={cum_reward:+.1f} "
                    f"alive={info_step['alive_walkers']} term={term} trunc={trunc}"
                )
            if term or trunc:
                if args.mode == "live":
                    print("  episode terminated; restarting…")
                    state, obs, info = env.reset(return_state=True)
                    cum_reward = 0.0
                    continue
                break
    except KeyboardInterrupt:
        print("\n  interrupted by user")

    wall = time.time() - t0
    print()
    print(f"== result ==  cum_reward={cum_reward:.2f}  decisions={step + 1}  wall={wall:.1f}s")

    if args.mode == "gif" and frames:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore[no-redef]
        out_path = _HERE / args.gif
        imageio.mimsave(str(out_path), frames, fps=args.gif_fps)
        print(f"GIF saved → {out_path}  ({len(frames)} frames @ {args.gif_fps} fps)")

    if args.mode == "live":
        try:
            env.close()
        except Exception:
            pass
    return 0 if cum_reward > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
