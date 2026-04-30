"""fmc_mutable.py — autoresearch experiment 09: STACK inv-boost on exp03 ach.

Built on exp03 (40.96% Crafter, 3/4 blockers fired) — current best.

Hypothesis (exp09):
  exp01 alone (iron-tier inv-boost) was neutral vs baseline (29.30%, 0 blockers).
  But exp01 was tested BEFORE the ach-fire bonus existed. With exp03's
  tier-weighted ach bonus driving the cloning kernel toward blockers, the
  iron-tier inv-boost may now amplify the IRON-stage gradient specifically
  (collect_iron, collect_coal, make_iron_pickaxe), giving a small structural
  boost to the chain.

  This is Tier 1C from HANDOFF.md: "Stack exp03 weights with iron-tier
  inv-boost (exp01 weights)."

Mutation:
  Override the v4 `inventory_total` with iron-tier-boosted weights.
  Original (v4) -> exp09:
    iron       8 -> 16   (2x)
    coal       4 ->  8   (2x)
    diamond   16 -> 64   (4x)
    iron_p    12 -> 24   (2x)
    iron_s    12 -> 24   (2x)
  Other inventory weights unchanged.

  ACH_WEIGHTS unchanged from exp03 (tier-weighted blockers 150-300).
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
    relativize, proximity_bonus_single, FMCConfig,
)


# exp09 mutation: iron-tier-boosted inventory_total
def inventory_total(state) -> jnp.ndarray:
    inv = state.inventory
    return (
        inv.wood.astype(jnp.float32) * 1.0
        + inv.stone.astype(jnp.float32) * 2.0
        + inv.coal.astype(jnp.float32) * 8.0          # 4 -> 8 (2x)
        + inv.iron.astype(jnp.float32) * 16.0         # 8 -> 16 (2x)
        + inv.diamond.astype(jnp.float32) * 64.0      # 16 -> 64 (4x)
        + inv.sapling.astype(jnp.float32) * 0.5
        + inv.wood_pickaxe.astype(jnp.float32) * 3.0
        + inv.stone_pickaxe.astype(jnp.float32) * 6.0
        + inv.iron_pickaxe.astype(jnp.float32) * 24.0  # 12 -> 24 (2x)
        + inv.wood_sword.astype(jnp.float32) * 3.0
        + inv.stone_sword.astype(jnp.float32) * 6.0
        + inv.iron_sword.astype(jnp.float32) * 24.0    # 12 -> 24 (2x)
    )


# ============================================================================
# MUTATION exp03: tier-weighted achievement-fire bonus
# ============================================================================

# Order matches CRAFTAX_CLASSIC_ACHIEVEMENTS in prepare_craftax.py
# but here we need the order matching state.achievements[22] which is the
# Craftax internal order.
# From craftax/craftax_classic/constants.py Achievement enum:
#  0:COLLECT_WOOD, 1:PLACE_TABLE, 2:EAT_COW, 3:COLLECT_SAPLING, 4:COLLECT_DRINK,
#  5:MAKE_WOOD_PICKAXE, 6:MAKE_STONE_PICKAXE, 7:MAKE_IRON_PICKAXE,
#  8:MAKE_WOOD_SWORD, 9:MAKE_STONE_SWORD, 10:MAKE_IRON_SWORD,
#  11:PLACE_PLANT, 12:DEFEAT_ZOMBIE, 13:COLLECT_STONE, 14:PLACE_STONE,
#  15:EAT_PLANT, 16:DEFEAT_SKELETON, 17:COLLECT_IRON, 18:COLLECT_COAL,
#  19:PLACE_FURNACE, 20:COLLECT_DIAMOND, 21:WAKE_UP

ACH_WEIGHTS_LIST = [
    10.0,   # 0: COLLECT_WOOD (easy)
    10.0,   # 1: PLACE_TABLE (easy)
    30.0,   # 2: EAT_COW (gateway, requires combat)
    20.0,   # 3: COLLECT_SAPLING
    20.0,   # 4: COLLECT_DRINK
    20.0,   # 5: MAKE_WOOD_PICKAXE (gateway)
    50.0,   # 6: MAKE_STONE_PICKAXE (key gateway)
    150.0,  # 7: MAKE_IRON_PICKAXE *** BLOCKER ***
    20.0,   # 8: MAKE_WOOD_SWORD
    50.0,   # 9: MAKE_STONE_SWORD
    150.0,  # 10: MAKE_IRON_SWORD *** BLOCKER ***
    20.0,   # 11: PLACE_PLANT
    30.0,   # 12: DEFEAT_ZOMBIE
    30.0,   # 13: COLLECT_STONE
    20.0,   # 14: PLACE_STONE
    200.0,  # 15: EAT_PLANT *** BLOCKER (time-based) ***
    50.0,   # 16: DEFEAT_SKELETON
    80.0,   # 17: COLLECT_IRON (gateway to iron chain)
    50.0,   # 18: COLLECT_COAL (gateway to iron chain)
    50.0,   # 19: PLACE_FURNACE (gateway to iron chain)
    300.0,  # 20: COLLECT_DIAMOND *** ULTIMATE BLOCKER ***
    20.0,   # 21: WAKE_UP
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

    # Achievement bool[22] per walker; tier-weighted bonus = sum(new * weights)
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
        # Track baseline achievement bitmask at planning root (N x 22 bool/float)
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

            # NEW: tier-weighted achievement-fire bonus
            #   cur_ach[w][a] in {0,1}, baseline same.
            #   new_ach[w][a] = max(cur - baseline, 0) -> 1 only if newly unlocked
            #   bonus[w] = sum_a (new_ach[w][a] * ACH_WEIGHTS[a])
            cur_ach_bool = vmapped_ach_bool(walker_states)
            new_ach_per_walker = jnp.maximum(cur_ach_bool - ach_baseline_bool, 0.0)
            # weighted sum across the 22 achievements per walker
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
            "_mutation": "exp09-stack: exp03 ach + exp01 iron-tier inv-boost (iron/coal 2x, diamond 4x, iron-tools 2x)",
        },
        "seed": seed,
    }
