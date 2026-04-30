"""fmc_mutable.py — autoresearch experiment 02: achievement-fire bonus.

Hypothesis (autoresearch exp02):
  exp01 iron-boost (inv weight amplification) was neutral (29.30% vs 29.27
  baseline). The signal "more iron in inv" is too DENSE — walker gets reward
  every tile mined. The REAL sparse-reward signal in Craftax is
  ACHIEVEMENT UNLOCK: state.achievements[i] flips False -> True at the
  exact moment a goal completes. Targeting this directly biases search
  toward chain progression rather than incremental hoarding.

Mutation:
  - ACHIEVEMENT_BONUS = 50.0 added to cum_reward per newly-unlocked
    achievement during walker simulation (since planning root).
  - Tracks baseline_achievements at root, computes new-since-root each tick.
  - Stacks ON TOP of inv-delta + delta-proximity shaping (additive).

Why bonus = 50:
  - Wood-tier inv = 1 unit, intrinsic_inv_alpha=0.5 -> wood gives +0.5/wood
  - Typical episode collects ~30 wood = +15 inv signal at end of M=40 horizon
  - Single achievement unlock = +50 -> dominates 3x ANY other shaping signal
  - Keeps intrinsic shaping useful as exploration prior, but achievement
    unlock becomes the SUPREME goal.

Risk: if relativize collapses on bimodal cum_reward (many walkers at +0,
few at +50), cloning kernel might over-concentrate. Mitigated by relativize's
sub-exponential asymptote (Def 2 prop 4).
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
    relativize, proximity_bonus_single, inventory_total, FMCConfig,
)


# ============================================================================
# MUTATION exp02: achievement-fire bonus
# ============================================================================

ACHIEVEMENT_BONUS = 50.0   # reward per newly-unlocked achievement (since root)


def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig):
    N = cfg.n_walkers
    M = cfg.time_horizon
    K = cfg.action_repeat
    INV_A = cfg.intrinsic_inv_alpha
    PROX_A = cfg.proximity_alpha
    SIGMA = cfg.proximity_sigma
    PROX_DELTA = cfg.proximity_mode == "delta"
    ACH_BONUS = ACHIEVEMENT_BONUS

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))
    vmapped_inv = jax.vmap(inventory_total)
    vmapped_prox = jax.vmap(lambda s: proximity_bonus_single(s, SIGMA))

    # Achievement count per walker: state.achievements is bool[22]
    vmapped_ach_count = jax.vmap(
        lambda s: jnp.sum(s.achievements.astype(jnp.float32))
    )

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
        # NEW: track baseline achievement count at planning root (per walker, broadcasted)
        ach_baseline = vmapped_ach_count(walker_states)

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

            # NEW: achievement-fire bonus
            cur_ach = vmapped_ach_count(walker_states)
            new_ach_count = jnp.maximum(cur_ach - ach_baseline, 0.0)
            cum_rewards = cum_rewards + ACH_BONUS * new_ach_count

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
            "_mutation": f"exp02-ach-bonus: ACHIEVEMENT_BONUS={ACHIEVEMENT_BONUS}",
        },
        "seed": seed,
    }
