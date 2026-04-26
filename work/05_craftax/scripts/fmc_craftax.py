"""fmc_craftax.py — Fractal Monte Carlo per Craftax (JAX, vmap-vectorized).

Sfrutta lo stato funzionale di Craftax (PyTree immutabile) → niente cloneState/restoreState:
ogni walker tiene la propria copia dello stato come slice di un batch JAX.

Mapping concettuale (paper §4.3):
  - walker.state = stato Craftax replicato N volte
  - walker.init_action = prima azione del walker (decisione candidate)
  - perturbation = azione random a t>0
  - reward = cumulato Craftax (achievement-based)
  - distance = L2 sull'obs (8268-dim symbolic)
  - virtual reward = relativize(R)^α · relativize(D)^β
  - cloning = swap di stato + init_action via jnp.where
  - decisione finale = argmax(bincount(init_action[alive]))

Uso:
    python fmc_craftax.py --n_walkers 16 --time_horizon 8 --max_steps 100 --seed 42

Output: JSON su stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name


def relativize(x: jnp.ndarray) -> jnp.ndarray:
    """Paper §2.2.3 — preserva ordinamento, garantisce > 0."""
    std = x.std()
    safe_std = jnp.where(std > 1e-8, std, 1.0)
    z = (x - x.mean()) / safe_std
    out = jnp.where(z <= 0, jnp.exp(jnp.clip(z, -50, 0)), 1.0 + jnp.log1p(jnp.maximum(z, 0)))
    return out


@dataclass
class FMCConfig:
    n_walkers: int = 16
    time_horizon: int = 8
    alpha: float = 1.0  # esponente reward (0 = Common Sense pure, paper §6.1)
    beta: float = 1.0   # esponente distanza


def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig):
    """Costruisce la funzione FMC decide jitted per la config data."""
    N = cfg.n_walkers
    M = cfg.time_horizon

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))

    def fmc_decide(rng, root_state):
        # Replica root_state N volte (axis 0 batch)
        walker_states = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (N,) + x.shape) if hasattr(x, "shape") else x,
            root_state,
        )

        rng, k_init = jax.random.split(rng)
        init_actions = jax.random.randint(k_init, (N,), 0, n_actions)
        cum_rewards = jnp.zeros(N)
        alive = jnp.ones(N, dtype=jnp.bool_)

        def tick_body(carry, t):
            walker_states, init_actions, cum_rewards, alive, rng = carry

            # Azione: tick 0 = init_action, tick > 0 = random
            rng, k_act = jax.random.split(rng)
            random_actions = jax.random.randint(k_act, (N,), 0, n_actions)
            actions = jnp.where(t == 0, init_actions, random_actions)

            # Step parallelo
            rng, k_step = jax.random.split(rng)
            step_keys = jax.random.split(k_step, N)
            obs, walker_states, rewards, dones, _ = vmapped_step(step_keys, walker_states, actions)

            # Aggiorna cumulati e alive
            new_cum = jnp.where(alive, cum_rewards + rewards, cum_rewards)
            new_alive = alive & ~dones

            # Distanza: ogni walker confronta con un compagno random
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, N)
            # Evita j == i con rotazione di 1 dove serve
            perm = jnp.where(perm == jnp.arange(N), (perm + 1) % N, perm)

            partner_obs = obs[perm]
            distances = jnp.linalg.norm(obs - partner_obs, axis=-1)

            R_norm = relativize(new_cum) * new_alive
            D_norm = relativize(distances) * new_alive

            VR = (R_norm ** cfg.alpha) * (D_norm ** cfg.beta)

            # Cloning: confronta con un secondo partner random
            rng, k_perm2 = jax.random.split(rng)
            perm2 = jax.random.permutation(k_perm2, N)
            perm2 = jnp.where(perm2 == jnp.arange(N), (perm2 + 1) % N, perm2)

            VR_self = VR
            VR_other = VR[perm2]
            denom = jnp.where(VR_self > 1e-8, VR_self, 1e-8)
            clone_prob = jnp.clip((VR_other - VR_self) / denom, 0, 1)
            # Walker morti clonano sempre
            clone_prob = jnp.where(new_alive, clone_prob, 1.0)

            rng, k_draw = jax.random.split(rng)
            draws = jax.random.uniform(k_draw, (N,))
            will_clone = (draws < clone_prob) & new_alive[perm2]  # solo verso vivi

            # Applica clone via tree_map
            def clone_field(field):
                if not hasattr(field, "shape") or len(field.shape) == 0:
                    return field
                # Espandi will_clone alle dim del field
                wc = will_clone.reshape(will_clone.shape + (1,) * (len(field.shape) - 1))
                return jnp.where(wc, field[perm2], field)

            walker_states = jax.tree_util.tree_map(clone_field, walker_states)
            init_actions = jnp.where(will_clone, init_actions[perm2], init_actions)
            new_cum = jnp.where(will_clone, new_cum[perm2], new_cum)
            new_alive = jnp.where(will_clone, new_alive[perm2], new_alive)

            return (walker_states, init_actions, new_cum, new_alive, rng), None

        carry = (walker_states, init_actions, cum_rewards, alive, rng)
        carry, _ = jax.lax.scan(tick_body, carry, jnp.arange(M))
        _, init_actions, _, alive, _ = carry

        # Decisione: argmax bincount sulle init_actions dei vivi
        # Mascheriamo i morti azzerando i loro voti
        votes = jnp.zeros(n_actions)
        # Add 1 a votes[init_actions[i]] se alive[i]
        votes = votes.at[init_actions].add(alive.astype(jnp.float32))
        return jnp.argmax(votes), alive.sum()

    return jax.jit(fmc_decide)


def run_episode(seed: int, cfg: FMCConfig, max_steps: int = 200, verbose: bool = False,
                env_name: str = "Craftax-Symbolic-v1") -> dict:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    fmc_decide = make_fmc_decide(env, params, n_actions, cfg)

    rng = jax.random.PRNGKey(seed)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)

    cum_reward = 0.0
    achievements_unlocked = 0
    t_start = time.time()
    done = False
    n_steps = 0
    n_alive_log = []

    for step in range(max_steps):
        rng, k_dec = jax.random.split(rng)
        action, n_alive = fmc_decide(k_dec, state)
        action = int(action)
        n_alive_log.append(int(n_alive))

        rng, k_step = jax.random.split(rng)
        obs, state, reward, done, info = env.step(k_step, state, action, params)
        cum_reward += float(reward)
        n_steps += 1

        if verbose and (step + 1) % 10 == 0:
            print(f"  step {step+1}: action={action} reward={cum_reward:.3f} "
                  f"alive={int(n_alive)}/{cfg.n_walkers} "
                  f"elapsed={time.time()-t_start:.1f}s", file=sys.stderr)

        if done:
            break

    # Achievement unlocked + lista nomi
    achievements_dict = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if k.startswith("Achievements/"):
                if float(v) > 0:
                    achievements_dict[k.replace("Achievements/", "")] = float(v)
        achievements_unlocked = len(achievements_dict)

    wall = time.time() - t_start
    decisions = max(1, n_steps)
    return {
        "reward": float(cum_reward),
        "n_steps": int(n_steps),
        "achievements_unlocked": int(achievements_unlocked),
        "achievements_list": sorted(achievements_dict.keys()),
        "wall_time_s": float(wall),
        "decisions_per_sec": float(decisions / wall),
        "samples_per_decision": int(cfg.n_walkers * cfg.time_horizon),
        "samples_total": int(cfg.n_walkers * cfg.time_horizon * decisions),
        "n_alive_avg": float(sum(n_alive_log) / max(1, len(n_alive_log))),
        "config": {"n_walkers": cfg.n_walkers, "time_horizon": cfg.time_horizon,
                   "alpha": cfg.alpha, "beta": cfg.beta},
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_walkers", type=int, default=16)
    ap.add_argument("--time_horizon", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=1.0, help="reward exponent (0=Common Sense)")
    ap.add_argument("--beta", type=float, default=1.0, help="distance exponent")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--env", default="Craftax-Symbolic-v1",
                    help="Craftax-Symbolic-v1 (full) | Craftax-Classic-Symbolic-v1 (classic)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = FMCConfig(n_walkers=args.n_walkers, time_horizon=args.time_horizon,
                    alpha=args.alpha, beta=args.beta)
    result = run_episode(args.seed, cfg, args.max_steps, verbose=args.verbose, env_name=args.env)
    result["env"] = args.env
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
