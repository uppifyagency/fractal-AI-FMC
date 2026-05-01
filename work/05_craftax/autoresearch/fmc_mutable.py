"""fmc_mutable.py — autoresearch experiment 19: diamond proximity 4x.

Built on exp17 (50.95% Crafter, 3/4 blockers). exp18 showed diamond ach push
had zero effect (50.95% identical) — bottleneck is walkers REACHING diamond,
not their reward signal once they have it.

Hypothesis (exp19):
  proximity_bonus_single rewards being near diamond ore when has_iron_pickaxe
  & need_diamond. Current weight = 16 (with proximity_alpha=0.2 multiplier
  the per-tick contribution is up to 0.2 * 16 * exp(-d/sigma) = ~3.2 max).
  This is dwarfed by the iron-pickaxe ach reward (200 per tick after firing).

  Boosting diamond proximity weight to 64 (4x) makes navigation toward diamond
  the dominant gradient AFTER iron_pickaxe is acquired. Walkers should now
  actively seek diamond ore once they're chain-ready.

Mutation:
  Override proximity_bonus_single locally with diamond weight 16 -> 64.
  All other proximity weights unchanged. ACH_WEIGHTS, inv_total, CONFIG = exp17.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fmc_craftax_v4 import (
    relativize, FMCConfig,
    TREE_ID, STONE_ID, COAL_ID, IRON_ID, DIAMOND_ID, WATER_ID, RIPE_PLANT_ID,
)


# exp19: override proximity_bonus with diamond weight 4x (16 -> 64)
def proximity_bonus_single(state, sigma: float = 10.0) -> jnp.ndarray:
    px = state.player_position[0]
    py = state.player_position[1]
    map_arr = state.map

    H, W = map_arr.shape
    xs = jnp.arange(H).reshape(H, 1)
    ys = jnp.arange(W).reshape(1, W)
    d_grid = jnp.abs(xs - px) + jnp.abs(ys - py)

    def min_dist(target_id):
        mask = (map_arr == target_id)
        d_masked = jnp.where(mask, d_grid.astype(jnp.float32), jnp.float32(1e6))
        return d_masked.min()

    d_tree = min_dist(TREE_ID)
    d_stone = min_dist(STONE_ID)
    d_coal = min_dist(COAL_ID)
    d_iron = min_dist(IRON_ID)
    d_diamond = min_dist(DIAMOND_ID)
    d_water = min_dist(WATER_ID)
    d_ripe = min_dist(RIPE_PLANT_ID)

    inv = state.inventory
    need_wood = (inv.wood < 1).astype(jnp.float32)
    has_wood_p = (inv.wood_pickaxe > 0).astype(jnp.float32)
    has_stone_p = (inv.stone_pickaxe > 0).astype(jnp.float32)
    has_iron_p = (inv.iron_pickaxe > 0).astype(jnp.float32)
    need_stone = ((inv.stone < 5) | (inv.stone_pickaxe < 1)).astype(jnp.float32)
    need_coal = (inv.coal < 1).astype(jnp.float32)
    need_iron = (inv.iron < 1).astype(jnp.float32)
    need_diamond = (inv.diamond < 1).astype(jnp.float32)
    need_water = (state.player_drink < 5).astype(jnp.float32)
    need_ripe = (inv.sapling > 0).astype(jnp.float32)

    bonus = (
        1.0 * need_wood * jnp.exp(-d_tree / sigma)
        + 2.0 * has_wood_p * need_stone * jnp.exp(-d_stone / sigma)
        + 4.0 * has_stone_p * need_coal * jnp.exp(-d_coal / sigma)
        + 8.0 * has_stone_p * need_iron * jnp.exp(-d_iron / sigma)
        + 64.0 * has_iron_p * need_diamond * jnp.exp(-d_diamond / sigma)  # exp19: 16 -> 64
        + 0.5 * need_water * jnp.exp(-d_water / sigma)
        + 0.5 * need_ripe * jnp.exp(-d_ripe / sigma)
    )
    return bonus


# Inventory weights: exp11 (full tier-stack)
def inventory_total(state) -> jnp.ndarray:
    inv = state.inventory
    return (
        inv.wood.astype(jnp.float32) * 2.0
        + inv.stone.astype(jnp.float32) * 4.0
        + inv.coal.astype(jnp.float32) * 8.0
        + inv.iron.astype(jnp.float32) * 16.0
        + inv.diamond.astype(jnp.float32) * 64.0
        + inv.sapling.astype(jnp.float32) * 0.5
        + inv.wood_pickaxe.astype(jnp.float32) * 6.0
        + inv.stone_pickaxe.astype(jnp.float32) * 12.0
        + inv.iron_pickaxe.astype(jnp.float32) * 24.0
        + inv.wood_sword.astype(jnp.float32) * 6.0
        + inv.stone_sword.astype(jnp.float32) * 12.0
        + inv.iron_sword.astype(jnp.float32) * 24.0
    )


# exp17: exp16 weights + gateway tier push (stone_pickaxe, iron, coal, furnace)
ACH_WEIGHTS_LIST = [
    10.0, 10.0, 30.0, 20.0, 20.0, 20.0,
    80.0,                     # 6: MAKE_STONE_PICKAXE *** exp17: 50 -> 80 ***
    200.0,                    # 7: MAKE_IRON_PICKAXE (exp16)
    20.0, 50.0,
    200.0,                    # 10: MAKE_IRON_SWORD (exp16)
    20.0, 30.0, 30.0, 20.0, 200.0, 50.0,
    120.0,                    # 17: COLLECT_IRON *** exp17: 80 -> 120 ***
    80.0,                     # 18: COLLECT_COAL *** exp17: 50 -> 80 ***
    80.0,                     # 19: PLACE_FURNACE *** exp17: 50 -> 80 ***
    300.0,                    # 20: COLLECT_DIAMOND (back to exp17 since exp18 had zero effect)
    20.0,
]
ACH_WEIGHTS = jnp.array(ACH_WEIGHTS_LIST, dtype=jnp.float32)
assert ACH_WEIGHTS.shape == (22,)


def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig):
    N = cfg.n_walkers
    M = cfg.time_horizon
    K = cfg.action_repeat
    INV_A = cfg.intrinsic_inv_alpha
    PROX_A = cfg.proximity_alpha
    SIGMA = cfg.proximity_sigma
    PROX_DELTA = cfg.proximity_mode == "delta"

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))
    vmapped_inv = jax.vmap(inventory_total)
    vmapped_prox = jax.vmap(lambda s: proximity_bonus_single(s, SIGMA))
    vmapped_ach_bool = jax.vmap(lambda s: s.achievements.astype(jnp.float32))

    def fmc_decide(rng, root_state):
        walker_states = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (N,) + x.shape) if hasattr(x, "shape") else x,
            root_state,
        )

        rng, k_init = jax.random.split(rng)
        init_actions = jax.random.randint(k_init, (N,), 0, n_actions)
        cum_rewards = jnp.zeros(N)
        alive = jnp.ones(N, dtype=jnp.bool_)
        inv_baseline = vmapped_inv(walker_states)
        prox_prev = vmapped_prox(walker_states) if PROX_A > 0.0 else jnp.zeros(N)
        ach_baseline_bool = vmapped_ach_bool(walker_states)

        def tick_body(carry, t):
            walker_states, init_actions, cum_rewards, alive, prox_prev, rng = carry

            rng, k_act = jax.random.split(rng)
            random_actions = jax.random.randint(k_act, (N,), 0, n_actions)
            actions = jnp.where(t == 0, init_actions, random_actions)

            def inner_step(carry_inner, _):
                ws, cr, al, rng_in = carry_inner
                rng_in, k_step = jax.random.split(rng_in)
                step_keys = jax.random.split(k_step, N)
                obs_in, ws, r, d, _ = vmapped_step(step_keys, ws, actions)
                cr = jnp.where(al, cr + r, cr)
                al = al & ~d
                return (ws, cr, al, rng_in), obs_in

            (walker_states, cum_rewards, alive, rng), obs_seq = jax.lax.scan(
                inner_step, (walker_states, cum_rewards, alive, rng), jnp.arange(K)
            )
            obs = obs_seq[-1]

            if INV_A > 0.0:
                cur_inv = vmapped_inv(walker_states)
                inv_delta = jnp.maximum(cur_inv - inv_baseline, 0.0)
                cum_rewards = cum_rewards + INV_A * inv_delta

            if PROX_A > 0.0:
                prox = vmapped_prox(walker_states)
                if PROX_DELTA:
                    delta = jnp.maximum(prox - prox_prev, 0.0)
                    cum_rewards = cum_rewards + PROX_A * delta
                    prox_prev = prox
                else:
                    cum_rewards = cum_rewards + PROX_A * prox

            cur_ach_bool = vmapped_ach_bool(walker_states)
            new_ach_per_walker = jnp.maximum(cur_ach_bool - ach_baseline_bool, 0.0)
            weighted_bonus = jnp.sum(new_ach_per_walker * ACH_WEIGHTS, axis=-1)
            cum_rewards = cum_rewards + weighted_bonus

            new_alive = alive
            new_cum = cum_rewards

            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, N)
            perm = jnp.where(perm == jnp.arange(N), (perm + 1) % N, perm)
            partner_obs = obs[perm]
            distances = jnp.linalg.norm(obs - partner_obs, axis=-1)

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
            prox_prev = jnp.where(will_clone, prox_prev[perm2], prox_prev)

            return (walker_states, init_actions, new_cum, new_alive, prox_prev, rng), None

        carry = (walker_states, init_actions, cum_rewards, alive, prox_prev, rng)
        carry, _ = jax.lax.scan(tick_body, carry, jnp.arange(M))
        _, init_actions, _, alive, _, _ = carry

        votes = jnp.zeros(n_actions)
        votes = votes.at[init_actions].add(alive.astype(jnp.float32))
        return jnp.argmax(votes), alive.sum()

    return jax.jit(fmc_decide)


CONFIG = FMCConfig(
    n_walkers=512, time_horizon=40,
    alpha=1.0, beta=1.0, action_repeat=1,
    intrinsic_inv_alpha=0.5, proximity_alpha=0.2,
    proximity_sigma=10.0, proximity_mode="delta",
)


def run_episode(seed: int, max_steps: int = 500,
                env_name: str = "Craftax-Classic-Symbolic-v1") -> dict:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    fmc_decide = make_fmc_decide(env, params, n_actions, CONFIG)

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
        for _ in range(CONFIG.action_repeat):
            rng, k_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(k_step, state, action, params)
            cum_reward += float(reward)
            if done:
                break
        n_steps += 1
        if done:
            break

    achievements_dict = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if k.startswith("Achievements/") and float(v) > 0:
                achievements_dict[k.replace("Achievements/", "")] = float(v)

    wall = time.time() - t_start
    decisions = max(1, n_steps)
    return {
        "reward": float(cum_reward),
        "n_steps_decisions": int(n_steps),
        "n_steps_env": int(n_steps * CONFIG.action_repeat),
        "achievements_unlocked": int(len(achievements_dict)),
        "achievements_list": sorted(achievements_dict.keys()),
        "wall_time_s": float(wall),
        "decisions_per_sec": float(decisions / wall),
        "samples_per_decision": int(CONFIG.n_walkers * CONFIG.time_horizon * CONFIG.action_repeat),
        "config": {
            "n_walkers": CONFIG.n_walkers, "time_horizon": CONFIG.time_horizon,
            "alpha": CONFIG.alpha, "beta": CONFIG.beta,
            "action_repeat": CONFIG.action_repeat,
            "intrinsic_inv_alpha": CONFIG.intrinsic_inv_alpha,
            "proximity_alpha": CONFIG.proximity_alpha,
            "proximity_sigma": CONFIG.proximity_sigma,
            "proximity_mode": CONFIG.proximity_mode,
            "_mutation": "exp19: diamond proximity weight 16 -> 64 (4x), targeting reach-diamond bottleneck",
        },
        "seed": seed,
    }
