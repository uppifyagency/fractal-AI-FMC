"""random_baseline.py — episode random per fissare il floor.

Per Craftax-Classic-Symbolic-v1 il random scoring atteso è ~1.6% (Hafner 2021).
Per Craftax-Symbolic-v1 (full) random non sblocca nulla.
"""
from __future__ import annotations
import argparse, json, time
import jax
from craftax.craftax_env import make_craftax_env_from_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    env = make_craftax_env_from_name(args.env, auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    rng = jax.random.PRNGKey(args.seed)
    rng, k = jax.random.split(rng)
    obs, state = env.reset(k, params)

    cum, n, done = 0.0, 0, False
    t0 = time.time()
    info = {}
    while not done and n < args.max_steps:
        rng, k1, k2 = jax.random.split(rng, 3)
        a = jax.random.randint(k1, (), 0, n_actions)
        obs, state, r, done, info = env.step(k2, state, int(a), params)
        cum += float(r)
        n += 1

    achievements = sum(1 for k, v in info.items() if k.startswith("Achievements/") and float(v) > 0)
    wall = time.time() - t0
    print(json.dumps({
        "env": args.env, "seed": args.seed, "n_steps": n,
        "reward": cum, "achievements_unlocked": achievements,
        "wall_time_s": wall, "steps_per_sec": n / wall,
    }, indent=2))


if __name__ == "__main__":
    main()
