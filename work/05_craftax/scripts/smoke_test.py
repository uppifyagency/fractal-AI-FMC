"""Smoke test: verifica che Craftax sia installato e giri un episodio random."""
from __future__ import annotations
import time
import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name


def main():
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params

    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)
    obs, state = env.reset(key, params)

    print(f"Action space n: {env.action_space(params).n}")
    print(f"Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

    # Step a few random actions to verify
    n_actions = env.action_space(params).n
    cum = 0.0
    n_steps = 0
    t_start = time.time()
    done = False
    while not done and n_steps < 200:
        rng, k1, k2 = jax.random.split(rng, 3)
        action = jax.random.randint(k1, (), 0, n_actions)
        obs, state, reward, done, info = env.step(k2, state, action, params)
        cum += float(reward)
        n_steps += 1
    wall = time.time() - t_start

    print(f"Random episode: {n_steps} steps, reward={cum:.2f}, "
          f"wall={wall:.2f}s, steps/sec={n_steps/wall:.0f}")
    print(f"Info keys: {list(info.keys())[:10]}")
    print("SMOKE TEST PASSED" if n_steps > 0 else "FAILED")


if __name__ == "__main__":
    main()
