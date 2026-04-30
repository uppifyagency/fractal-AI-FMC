"""fmc_mutable.py — autoresearch experiment 14: multi-pop swarm.

Built on exp12 (46.45% Crafter, 2/4 blockers) — current numerical best.
Falls back to exp11 (45.94, 3/4 blockers) as the structural reference.

Hypothesis (exp14):
  Single pool of 512 walkers all share the same shaping. The cloning kernel
  preferentially clones high-VR walkers, so the population converges around
  whatever shaping rewards. This homogenizes the search.

  Multi-pop swarm = 2 sub-pops of N=256 with DIFFERENT shaping. Cloning is
  WITHIN each pop, never across, so each pop maintains its own gradient.
  At decision time, action votes from both pops are summed.

  Pop A (specialist): exp12 config — boosted inv weights + tier-weighted
    ach (300 for diamond, 150 for iron-tools) + proximity_alpha=0.4. Drives
    the chain end (iron->diamond).

  Pop B (explorer): v4 baseline inv weights + uniform ach +50 (exp02-style)
    + proximity_alpha=0.2. Broader exploration, less greedy on chain end.

  Combined vote bias toward actions that BOTH pops favor — robust consensus.

Mutation:
  Refactored make_fmc_decide to accept (ach_weights, inv_total_fn) params.
  Two JIT'd functions called sequentially per env step. fmc_decide returns
  vote ARRAY (shape n_actions) instead of argmax, so caller can sum.

Cost: 2 * (256 * 40) = 20480 walker-steps per decision = same total compute
as exp11's single 512-pop. ~10s extra JIT compile (one per pop). Memory 2x.
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


# ============================================================================
# Pop A (specialist): exp12 boosted inv + tier-weighted ach
# ============================================================================

def inv_total_boosted(state) -> jnp.ndarray:
    """Pop A inventory weights (exp11 boost: wood 2x, stone 2x, iron-tier 2x, diamond 4x)."""
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


# Pop A: tier-weighted ach (exp03), blockers 150-300, gateway 50-80, easy 10-30
ACH_WEIGHTS_TIER_LIST = [
    10.0, 10.0, 30.0, 20.0, 20.0, 20.0, 50.0, 150.0, 20.0, 50.0, 150.0,
    20.0, 30.0, 30.0, 20.0, 200.0, 50.0, 80.0, 50.0, 50.0, 300.0, 20.0,
]
ACH_WEIGHTS_TIER = jnp.array(ACH_WEIGHTS_TIER_LIST, dtype=jnp.float32)


# ============================================================================
# Pop B (explorer): v4 baseline inv + uniform ach +50 (exp02-style)
# ============================================================================

def inv_total_v4(state) -> jnp.ndarray:
    """Pop B inventory weights (v4 baseline, no boost)."""
    inv = state.inventory
    return (
        inv.wood.astype(jnp.float32) * 1.0
        + inv.stone.astype(jnp.float32) * 2.0
        + inv.coal.astype(jnp.float32) * 4.0
        + inv.iron.astype(jnp.float32) * 8.0
        + inv.diamond.astype(jnp.float32) * 16.0
        + inv.sapling.astype(jnp.float32) * 0.5
        + inv.wood_pickaxe.astype(jnp.float32) * 3.0
        + inv.stone_pickaxe.astype(jnp.float32) * 6.0
        + inv.iron_pickaxe.astype(jnp.float32) * 12.0
        + inv.wood_sword.astype(jnp.float32) * 3.0
        + inv.stone_sword.astype(jnp.float32) * 6.0
        + inv.iron_sword.astype(jnp.float32) * 12.0
    )


# Pop B: uniform +50 per ach unlock (exp02-style)
ACH_WEIGHTS_UNIFORM = jnp.full((22,), 50.0, dtype=jnp.float32)


# Module-level alias used by `inventory_total` consumers / smoke tests
ACH_WEIGHTS = ACH_WEIGHTS_TIER  # backward-compat with prior smoke tests
inventory_total = inv_total_boosted  # backward-compat


# ============================================================================
# Parameterized make_fmc_decide
# ============================================================================

def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig,
                    ach_weights, inv_total_fn):
    """Build a JIT'd fmc_decide returning a VOTE ARRAY of shape (n_actions,)."""
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
    vmapped_inv = jax.vmap(inv_total_fn)
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
            weighted_bonus = jnp.sum(new_ach_per_walker * ach_weights, axis=-1)
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

        # exp14: return votes ARRAY (not argmax) so multiple pops can sum
        votes = jnp.zeros(n_actions)
        votes = votes.at[init_actions].add(alive.astype(jnp.float32))
        return votes, alive.sum()

    return jax.jit(fmc_decide)


# ============================================================================
# Two pop configs (specialist + explorer), each N=256
# ============================================================================

CONFIG_A = FMCConfig(
    n_walkers=256, time_horizon=40,
    alpha=1.0, beta=1.0, action_repeat=1,
    intrinsic_inv_alpha=0.5, proximity_alpha=0.4,
    proximity_sigma=10.0, proximity_mode="delta",
)

CONFIG_B = FMCConfig(
    n_walkers=256, time_horizon=40,
    alpha=1.0, beta=1.0, action_repeat=1,
    intrinsic_inv_alpha=0.5, proximity_alpha=0.2,
    proximity_sigma=10.0, proximity_mode="delta",
)

# Backward-compat single-pop CONFIG (used by some places that read CONFIG.n_walkers)
CONFIG = FMCConfig(
    n_walkers=512, time_horizon=40,  # combined effective N
    alpha=1.0, beta=1.0, action_repeat=1,
    intrinsic_inv_alpha=0.5, proximity_alpha=0.4,
    proximity_sigma=10.0, proximity_mode="delta",
)


def run_episode(seed: int, max_steps: int = 500,
                env_name: str = "Craftax-Classic-Symbolic-v1") -> dict:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    fmc_a = make_fmc_decide(env, params, n_actions, CONFIG_A,
                             ACH_WEIGHTS_TIER, inv_total_boosted)
    fmc_b = make_fmc_decide(env, params, n_actions, CONFIG_B,
                             ACH_WEIGHTS_UNIFORM, inv_total_v4)

    rng = jax.random.PRNGKey(seed)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)

    cum_reward = 0.0
    t_start = time.time()
    done = False
    n_steps = 0
    info = {}

    for step in range(max_steps):
        rng, k_a, k_b = jax.random.split(rng, 3)
        votes_a, _ = fmc_a(k_a, state)
        votes_b, _ = fmc_b(k_b, state)
        combined = votes_a + votes_b
        action = int(jnp.argmax(combined))
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
        "samples_per_decision": int((CONFIG_A.n_walkers + CONFIG_B.n_walkers) * CONFIG.time_horizon * CONFIG.action_repeat),
        "config": {
            "n_walkers_total": CONFIG_A.n_walkers + CONFIG_B.n_walkers,
            "n_walkers_pop_a": CONFIG_A.n_walkers,
            "n_walkers_pop_b": CONFIG_B.n_walkers,
            "time_horizon": CONFIG.time_horizon,
            "alpha": CONFIG.alpha, "beta": CONFIG.beta,
            "action_repeat": CONFIG.action_repeat,
            "intrinsic_inv_alpha": CONFIG.intrinsic_inv_alpha,
            "proximity_alpha_a": CONFIG_A.proximity_alpha,
            "proximity_alpha_b": CONFIG_B.proximity_alpha,
            "proximity_sigma": CONFIG.proximity_sigma,
            "proximity_mode": CONFIG.proximity_mode,
            "_mutation": "exp14: multi-pop swarm (specialist exp12 + explorer baseline-uniform), N=256+256",
        },
        "seed": seed,
    }
