"""fmc_craftax_v2.py — variante con distance metric low-D su state fields.

Differenza vs fmc_craftax.py:
  - distance NON è L2 sull'obs 1345-D (dominata dal map view)
  - distance è L2 su un vettore 18-D estratto dallo state:
      [pos_x, pos_y, health, food, drink, energy,
       wood, stone, coal, iron, diamond, sapling,
       wood_pickaxe, stone_pickaxe, iron_pickaxe,
       wood_sword, stone_sword, iron_sword]

Razionale: l'obs 1345-D è dominata dal map view one-hot (~1300 dim) che varia
per pixel di mapview ma non per "progresso del giocatore". Due stati con
inventario molto diverso ma stesso map view sarebbero "vicini" in obs-distance.
Per FMC a la Sergio, vogliamo distanza che catturi "progresso reale" del walker.

Riferimento: paper §5.1.3.3 — "RAM is more informative than IMG".
La nostra obs full è IMG-like (map one-hot); il low-D vector è il nostro RAM.
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
    std = x.std()
    safe_std = jnp.where(std > 1e-8, std, 1.0)
    z = (x - x.mean()) / safe_std
    out = jnp.where(z <= 0, jnp.exp(jnp.clip(z, -50, 0)), 1.0 + jnp.log1p(jnp.maximum(z, 0)))
    return out


def state_to_lowd(state) -> jnp.ndarray:
    """Estrae 18-D vector significativo dallo state (Craftax-Classic)."""
    inv = state.inventory
    return jnp.stack([
        state.player_position[0].astype(jnp.float32),
        state.player_position[1].astype(jnp.float32),
        state.player_health.astype(jnp.float32),
        state.player_food.astype(jnp.float32),
        state.player_drink.astype(jnp.float32),
        state.player_energy.astype(jnp.float32),
        inv.wood.astype(jnp.float32),
        inv.stone.astype(jnp.float32),
        inv.coal.astype(jnp.float32),
        inv.iron.astype(jnp.float32),
        inv.diamond.astype(jnp.float32),
        inv.sapling.astype(jnp.float32),
        inv.wood_pickaxe.astype(jnp.float32),
        inv.stone_pickaxe.astype(jnp.float32),
        inv.iron_pickaxe.astype(jnp.float32),
        inv.wood_sword.astype(jnp.float32),
        inv.stone_sword.astype(jnp.float32),
        inv.iron_sword.astype(jnp.float32),
    ])


@dataclass
class FMCConfig:
    n_walkers: int = 32
    time_horizon: int = 12
    alpha: float = 1.0
    beta: float = 1.0


def make_fmc_decide_v2(env, params, n_actions: int, cfg: FMCConfig):
    N = cfg.n_walkers
    M = cfg.time_horizon

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))
    vmapped_lowd = jax.vmap(state_to_lowd)

    def fmc_decide(rng, root_state):
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

            rng, k_act = jax.random.split(rng)
            random_actions = jax.random.randint(k_act, (N,), 0, n_actions)
            actions = jnp.where(t == 0, init_actions, random_actions)

            rng, k_step = jax.random.split(rng)
            step_keys = jax.random.split(k_step, N)
            obs, walker_states, rewards, dones, _ = vmapped_step(step_keys, walker_states, actions)

            new_cum = jnp.where(alive, cum_rewards + rewards, cum_rewards)
            new_alive = alive & ~dones

            # Distance: low-D vector L2 (NEW)
            lowd = vmapped_lowd(walker_states)  # shape (N, 18)
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, N)
            perm = jnp.where(perm == jnp.arange(N), (perm + 1) % N, perm)

            partner_lowd = lowd[perm]
            distances = jnp.linalg.norm(lowd - partner_lowd, axis=-1)

            R_norm = relativize(new_cum) * new_alive
            D_norm = relativize(distances) * new_alive

            VR = (R_norm ** cfg.alpha) * (D_norm ** cfg.beta)

            rng, k_perm2 = jax.random.split(rng)
            perm2 = jax.random.permutation(k_perm2, N)
            perm2 = jnp.where(perm2 == jnp.arange(N), (perm2 + 1) % N, perm2)

            VR_self = VR
            VR_other = VR[perm2]
            denom = jnp.where(VR_self > 1e-8, VR_self, 1e-8)
            clone_prob = jnp.clip((VR_other - VR_self) / denom, 0, 1)
            clone_prob = jnp.where(new_alive, clone_prob, 1.0)

            rng, k_draw = jax.random.split(rng)
            draws = jax.random.uniform(k_draw, (N,))
            will_clone = (draws < clone_prob) & new_alive[perm2]

            def clone_field(field):
                if not hasattr(field, "shape") or len(field.shape) == 0:
                    return field
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

        votes = jnp.zeros(n_actions)
        votes = votes.at[init_actions].add(alive.astype(jnp.float32))
        return jnp.argmax(votes), alive.sum()

    return jax.jit(fmc_decide)


def run_episode(seed: int, cfg: FMCConfig, max_steps: int = 500, verbose: bool = False) -> dict:
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    fmc_decide = make_fmc_decide_v2(env, params, n_actions, cfg)

    rng = jax.random.PRNGKey(seed)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)

    cum_reward = 0.0
    t_start = time.time()
    done = False
    n_steps = 0
    info = {}

    for step in range(max_steps):
        rng, k_dec = jax.random.split(rng)
        action, n_alive = fmc_decide(k_dec, state)
        action = int(action)

        rng, k_step = jax.random.split(rng)
        obs, state, reward, done, info = env.step(k_step, state, action, params)
        cum_reward += float(reward)
        n_steps += 1

        if verbose and (step + 1) % 20 == 0:
            print(f"  step {step+1}: action={action} reward={cum_reward:.3f} "
                  f"alive={int(n_alive)}/{cfg.n_walkers}", file=sys.stderr)

        if done:
            break

    achievements_dict = {k.replace("Achievements/", ""): float(v)
                         for k, v in info.items()
                         if k.startswith("Achievements/") and float(v) > 0}
    return {
        "reward": float(cum_reward),
        "n_steps": int(n_steps),
        "achievements_unlocked": len(achievements_dict),
        "achievements_list": sorted(achievements_dict.keys()),
        "wall_time_s": time.time() - t_start,
        "config": {"n_walkers": cfg.n_walkers, "time_horizon": cfg.time_horizon,
                   "alpha": cfg.alpha, "beta": cfg.beta},
        "seed": seed,
        "distance_metric": "lowd_18",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_walkers", type=int, default=32)
    ap.add_argument("--time_horizon", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    cfg = FMCConfig(args.n_walkers, args.time_horizon, args.alpha, args.beta)
    print(json.dumps(run_episode(args.seed, cfg, args.max_steps, args.verbose), indent=2))
