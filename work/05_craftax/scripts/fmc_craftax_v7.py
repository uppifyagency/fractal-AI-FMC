"""fmc_craftax_v7.py — v4 + vitality bonus per allungare la sopravvivenza.

OSSERVAZIONE empirica (2026-04-27 dopo run_004):
  La hypothesis "max_steps lungo unlock eat_plant + iron chain" è FALSA.
  3 seed con max_steps=10000 hanno episodes che terminano sempre a 192-300
  decisioni — l'agente muore (zombie/fame) ben prima del cap.

  Quindi il vero blocker per `eat_plant` (richiede sapling → ripe_plant cycle
  ~30 in-game days) e per la chain iron→diamond (richiede long traversal
  cave) è la **mortalità precoce** del giocatore, non l'orizzonte temporale.

CORREZIONE v7:
  Aggiunge bonus intrinseco alla cum_reward del walker per il delta
  delle "vital stats":
    - player_health (max 9): combat damage / lava
    - player_food   (max 9): hunger
    - player_drink  (max 9): thirst
    - player_energy (max 9): sleep deprivation

  vitality_score = (h + f + d + e)
  Bonus per tick = vitality_alpha · (current - baseline)

  Negative delta → walker meno competitivo (penalizzato in cloning).
  Positive delta → walker che mangia/beve/dorme premiato.

Coerente con paper §6.2: shaping multi-component additivo.
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


TREE_ID, STONE_ID, COAL_ID, IRON_ID, DIAMOND_ID = 5, 4, 8, 9, 10
WATER_ID, RIPE_PLANT_ID = 3, 16


def relativize(x):
    std = x.std()
    safe_std = jnp.where(std > 1e-8, std, 1.0)
    z = (x - x.mean()) / safe_std
    return jnp.where(z <= 0, jnp.exp(jnp.clip(z, -50, 0)), 1.0 + jnp.log1p(jnp.maximum(z, 0)))


def inventory_total(state):
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


def vitality_total(state):
    """Somma scalare delle 4 vital stats (max ~36)."""
    return (
        state.player_health.astype(jnp.float32)
        + state.player_food.astype(jnp.float32)
        + state.player_drink.astype(jnp.float32)
        + state.player_energy.astype(jnp.float32)
    )


def proximity_bonus_single(state, sigma):
    px, py = state.player_position[0], state.player_position[1]
    map_arr = state.map
    H, W = map_arr.shape
    xs = jnp.arange(H).reshape(H, 1)
    ys = jnp.arange(W).reshape(1, W)
    d_grid = jnp.abs(xs - px) + jnp.abs(ys - py)

    def md(tid):
        m = (map_arr == tid)
        return jnp.where(m, d_grid.astype(jnp.float32), jnp.float32(1e6)).min()

    inv = state.inventory
    nw = (inv.wood < 1).astype(jnp.float32)
    hwp = (inv.wood_pickaxe > 0).astype(jnp.float32)
    hsp = (inv.stone_pickaxe > 0).astype(jnp.float32)
    hip = (inv.iron_pickaxe > 0).astype(jnp.float32)
    ns = ((inv.stone < 5) | (inv.stone_pickaxe < 1)).astype(jnp.float32)
    nc = (inv.coal < 1).astype(jnp.float32)
    ni = (inv.iron < 1).astype(jnp.float32)
    nd = (inv.diamond < 1).astype(jnp.float32)
    nwa = (state.player_drink < 5).astype(jnp.float32)
    nrp = (inv.sapling > 0).astype(jnp.float32)

    return (
        1.0 * nw * jnp.exp(-md(TREE_ID) / sigma)
        + 2.0 * hwp * ns * jnp.exp(-md(STONE_ID) / sigma)
        + 4.0 * hsp * nc * jnp.exp(-md(COAL_ID) / sigma)
        + 8.0 * hsp * ni * jnp.exp(-md(IRON_ID) / sigma)
        + 16.0 * hip * nd * jnp.exp(-md(DIAMOND_ID) / sigma)
        + 0.5 * nwa * jnp.exp(-md(WATER_ID) / sigma)
        + 0.5 * nrp * jnp.exp(-md(RIPE_PLANT_ID) / sigma)
    )


@dataclass
class FMCConfig:
    n_walkers: int = 64
    time_horizon: int = 20
    alpha: float = 1.0
    beta: float = 1.0
    action_repeat: int = 1
    intrinsic_inv_alpha: float = 0.5
    proximity_alpha: float = 0.2
    proximity_sigma: float = 10.0
    vitality_alpha: float = 0.0  # bonus per delta vital stats (max 36)


def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig):
    N = cfg.n_walkers
    M = cfg.time_horizon
    K = cfg.action_repeat
    INV_A = cfg.intrinsic_inv_alpha
    PROX_A = cfg.proximity_alpha
    SIGMA = cfg.proximity_sigma
    VIT_A = cfg.vitality_alpha

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))
    vmapped_inv = jax.vmap(inventory_total)
    vmapped_vit = jax.vmap(vitality_total)
    vmapped_prox = jax.vmap(lambda s: proximity_bonus_single(s, SIGMA))

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
        vit_baseline = vmapped_vit(walker_states) if VIT_A > 0.0 else jnp.zeros(N)
        prox_prev = vmapped_prox(walker_states) if PROX_A > 0.0 else jnp.zeros(N)

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

            if VIT_A > 0.0:
                cur_vit = vmapped_vit(walker_states)
                # SIGNED delta: rewards positive change (heal/eat/drink/sleep)
                # AND penalizes negative change (damage/hunger/thirst)
                vit_delta = cur_vit - vit_baseline
                cum_rewards = cum_rewards + VIT_A * vit_delta

            if PROX_A > 0.0:
                prox = vmapped_prox(walker_states)
                delta = jnp.maximum(prox - prox_prev, 0.0)
                cum_rewards = cum_rewards + PROX_A * delta
                prox_prev = prox

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


def run_episode(seed: int, cfg: FMCConfig, max_steps: int = 500, verbose: bool = False,
                env_name: str = "Craftax-Classic-Symbolic-v1") -> dict:
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    params = env.default_params
    n_actions = env.action_space(params).n

    fmc_decide = make_fmc_decide(env, params, n_actions, cfg)

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

        for _ in range(cfg.action_repeat):
            rng, k_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(k_step, state, action, params)
            cum_reward += float(reward)
            if done:
                break
        n_steps += 1

        if verbose and (step + 1) % 50 == 0:
            print(f"  step {step+1}: action={action} reward={cum_reward:.3f} "
                  f"alive={int(n_alive)}/{cfg.n_walkers} "
                  f"hp={int(state.player_health)} food={int(state.player_food)} "
                  f"drink={int(state.player_drink)} en={int(state.player_energy)}",
                  file=sys.stderr)

        if done:
            break

    achievements_dict = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if k.startswith("Achievements/") and float(v) > 0:
                achievements_dict[k.replace("Achievements/", "")] = float(v)

    return {
        "reward": float(cum_reward),
        "n_steps_decisions": int(n_steps),
        "n_steps_env": int(n_steps * cfg.action_repeat),
        "achievements_unlocked": int(len(achievements_dict)),
        "achievements_list": sorted(achievements_dict.keys()),
        "wall_time_s": float(time.time() - t_start),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_walkers", type=int, default=64)
    ap.add_argument("--time_horizon", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--action_repeat", type=int, default=1)
    ap.add_argument("--intrinsic_inv_alpha", type=float, default=0.5)
    ap.add_argument("--proximity_alpha", type=float, default=0.2)
    ap.add_argument("--proximity_sigma", type=float, default=10.0)
    ap.add_argument("--vitality_alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = FMCConfig(
        n_walkers=args.n_walkers, time_horizon=args.time_horizon,
        alpha=args.alpha, beta=args.beta,
        action_repeat=args.action_repeat,
        intrinsic_inv_alpha=args.intrinsic_inv_alpha,
        proximity_alpha=args.proximity_alpha,
        proximity_sigma=args.proximity_sigma,
        vitality_alpha=args.vitality_alpha,
    )
    print(json.dumps(run_episode(args.seed, cfg, args.max_steps, args.verbose, args.env), indent=2))


if __name__ == "__main__":
    main()
