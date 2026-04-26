"""fmc_craftax_v5.py — FMC v4 + Fractal Memory cross-episode action prior.

Aggiunge a v4 una memoria persistente (in-process, cross-episode) che bias
le init_actions dei walker secondo un Wigner-weighted recall.

Design (allineato slide 2020 di Sergio):
  - fingerprint(state) = tupla discretizzata che cattura "progresso del giocatore":
        (has_wood_pick, has_stone_pick, has_iron_pick,
         has_wood, has_stone, has_coal, has_iron, has_diamond,
         has_table, has_furnace, position_bucket_x, position_bucket_y)
  - bank: dict[fp] -> {counts[n_actions], value_sum, visits}
  - update(state, action, value): incrementa counts[action], value_sum
  - recall(state) -> action prior (n_actions,), normalized
  - mix con uniform: prior = mem_weight * memory + (1-mem_weight) * uniform

Senza training: ogni "episode" insegna alle prossime decisioni quale azione
funziona meglio per ciascun progresso. Cumulando 5+ episode, la memoria
diventa una policy euristica appresa zero-shot.

Cross-episode persistence: la memoria vive fra seed diversi (in-process).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name


# Block IDs (Craftax-Classic)
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
    intrinsic_inv_alpha: float = 0.2
    proximity_alpha: float = 1.0
    proximity_sigma: float = 10.0
    mem_weight: float = 0.0          # 0 = no memory; 0.5 = mix; 1.0 = full memory
    mem_min_visits: int = 3          # min visits to trust a fingerprint


# ============================================================================
# Fractal Memory — Python-side, persistent across episodes
# ============================================================================

class FractalMemory:
    """Cross-episode memory bank: fingerprint(state) -> action_distribution.

    Fingerprint cattura il livello di progresso del giocatore.
    counts[a] = quante volte l'azione `a` è stata scelta in stati con questo fp.
    value_sum = somma dei valori (n_alive_post_rollout / N) per stimare quality.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        # fp -> {'counts': np.array(n_actions), 'value_sum': float, 'visits': int}
        self.bank: dict = {}

    @staticmethod
    def fingerprint(state) -> tuple:
        inv = state.inventory
        return (
            int(inv.wood_pickaxe > 0),
            int(inv.stone_pickaxe > 0),
            int(inv.iron_pickaxe > 0),
            int(inv.wood >= 1),
            int(inv.stone >= 1),
            int(inv.coal >= 1),
            int(inv.iron >= 1),
            int(inv.diamond >= 1),
            int(state.player_position[0] // 16),  # 4x4 spatial buckets
            int(state.player_position[1] // 16),
        )

    def update(self, state, action: int, value: float = 1.0):
        fp = self.fingerprint(state)
        if fp not in self.bank:
            self.bank[fp] = {
                "counts": np.zeros(self.n_actions, dtype=np.float64),
                "value_sum": 0.0,
                "visits": 0,
            }
        e = self.bank[fp]
        e["counts"][action] += value
        e["value_sum"] += value
        e["visits"] += 1

    def recall_prior(self, state, min_visits: int) -> np.ndarray:
        fp = self.fingerprint(state)
        if fp not in self.bank or self.bank[fp]["visits"] < min_visits:
            return np.ones(self.n_actions) / self.n_actions
        counts = self.bank[fp]["counts"]
        s = counts.sum()
        if s <= 0:
            return np.ones(self.n_actions) / self.n_actions
        return counts / s

    def stats(self) -> dict:
        return {
            "n_fingerprints": len(self.bank),
            "total_visits": sum(e["visits"] for e in self.bank.values()),
            "max_visits_single_fp": max((e["visits"] for e in self.bank.values()), default=0),
        }


# ============================================================================
# FMC decide — accetta init_actions esterne (così possiamo iniettare il prior)
# ============================================================================

def make_fmc_decide(env, params, n_actions: int, cfg: FMCConfig):
    N = cfg.n_walkers
    M = cfg.time_horizon
    K = cfg.action_repeat
    INV_A = cfg.intrinsic_inv_alpha
    PROX_A = cfg.proximity_alpha
    SIGMA = cfg.proximity_sigma

    def step_walker(rng, state, action):
        return env.step(rng, state, action, params)

    vmapped_step = jax.vmap(step_walker, in_axes=(0, 0, 0))
    vmapped_inv = jax.vmap(inventory_total)
    vmapped_prox = jax.vmap(lambda s: proximity_bonus_single(s, SIGMA))

    def fmc_decide(rng, root_state, init_actions):
        # init_actions: (N,) int32 — sampled outside (uniform o memory-prior)
        walker_states = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (N,) + x.shape) if hasattr(x, "shape") else x,
            root_state,
        )
        cum_rewards = jnp.zeros(N)
        alive = jnp.ones(N, dtype=jnp.bool_)
        inv_baseline = vmapped_inv(walker_states)

        def tick_body(carry, t):
            walker_states, init_actions, cum_rewards, alive, rng = carry

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
                cum_rewards = cum_rewards + PROX_A * prox

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

            return (walker_states, init_actions, new_cum, new_alive, rng), None

        carry = (walker_states, init_actions, cum_rewards, alive, rng)
        carry, _ = jax.lax.scan(tick_body, carry, jnp.arange(M))
        _, init_actions, _, alive, _ = carry

        votes = jnp.zeros(n_actions)
        votes = votes.at[init_actions].add(alive.astype(jnp.float32))
        return jnp.argmax(votes), alive.sum()

    return jax.jit(fmc_decide)


def sample_init_actions(rng, prior_np: np.ndarray, mem_weight: float, n_actions: int, N: int) -> np.ndarray:
    """Mix uniform e memory prior, poi sample N init_actions."""
    uniform = np.ones(n_actions) / n_actions
    mixed = mem_weight * prior_np + (1.0 - mem_weight) * uniform
    mixed = mixed / mixed.sum()
    rng_np, _ = jax.random.split(rng)
    rng_seed = int(rng_np[0]) ^ int(rng_np[1])
    np.random.seed(rng_seed & 0x7fffffff)
    return np.random.choice(n_actions, size=N, p=mixed)


def run_episode(seed: int, cfg: FMCConfig, memory: FractalMemory,
                max_steps: int = 500, verbose: bool = False,
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
    n_mem_hit = 0  # quanti step hanno usato il prior memoria

    for step in range(max_steps):
        # 1) Recall prior dalla memory per lo state corrente
        prior = memory.recall_prior(state, cfg.mem_min_visits)
        # uniformità → no hit, distribuzione → hit
        is_hit = not np.allclose(prior, 1.0 / n_actions)
        if is_hit:
            n_mem_hit += 1

        # 2) Sample init_actions con mix memoria+uniform
        rng, k_sample = jax.random.split(rng)
        init_actions_np = sample_init_actions(
            k_sample, prior, cfg.mem_weight if is_hit else 0.0,
            n_actions, cfg.n_walkers,
        )
        init_actions = jnp.array(init_actions_np, dtype=jnp.int32)

        # 3) FMC decide
        rng, k_dec = jax.random.split(rng)
        action, n_alive = fmc_decide(k_dec, state, init_actions)
        action = int(action)
        value = float(n_alive) / cfg.n_walkers  # confidence proxy

        # 4) Update memory PRIMA di applicare l'azione (state at decision time)
        memory.update(state, action, value=max(value, 0.1))

        # 5) Step env
        for _ in range(cfg.action_repeat):
            rng, k_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(k_step, state, action, params)
            cum_reward += float(reward)
            if done:
                break
        n_steps += 1

        if verbose and (step + 1) % 25 == 0:
            print(f"  step {step+1}: action={action} reward={cum_reward:.2f} "
                  f"alive={int(n_alive)}/{cfg.n_walkers} mem_hit_rate={n_mem_hit/(step+1):.2f}",
                  file=sys.stderr)

        if done:
            break

    achievements_dict = {}
    if isinstance(info, dict):
        for k, v in info.items():
            if k.startswith("Achievements/") and float(v) > 0:
                achievements_dict[k.replace("Achievements/", "")] = float(v)

    wall = time.time() - t_start
    return {
        "reward": float(cum_reward),
        "n_steps_decisions": int(n_steps),
        "n_steps_env": int(n_steps * cfg.action_repeat),
        "achievements_unlocked": int(len(achievements_dict)),
        "achievements_list": sorted(achievements_dict.keys()),
        "wall_time_s": float(wall),
        "mem_hit_rate": float(n_mem_hit / max(1, n_steps)),
        "samples_per_decision": int(cfg.n_walkers * cfg.time_horizon * cfg.action_repeat),
        "config": {
            "n_walkers": cfg.n_walkers, "time_horizon": cfg.time_horizon,
            "alpha": cfg.alpha, "beta": cfg.beta,
            "action_repeat": cfg.action_repeat,
            "intrinsic_inv_alpha": cfg.intrinsic_inv_alpha,
            "proximity_alpha": cfg.proximity_alpha,
            "proximity_sigma": cfg.proximity_sigma,
            "mem_weight": cfg.mem_weight,
            "mem_min_visits": cfg.mem_min_visits,
        },
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_walkers", type=int, default=64)
    ap.add_argument("--time_horizon", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--action_repeat", type=int, default=1)
    ap.add_argument("--intrinsic_inv_alpha", type=float, default=0.2)
    ap.add_argument("--proximity_alpha", type=float, default=1.0)
    ap.add_argument("--proximity_sigma", type=float, default=10.0)
    ap.add_argument("--mem_weight", type=float, default=0.5)
    ap.add_argument("--mem_min_visits", type=int, default=3)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=42)
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
        mem_weight=args.mem_weight,
        mem_min_visits=args.mem_min_visits,
    )

    # Build env once to get n_actions for memory init
    env = make_craftax_env_from_name(args.env, auto_reset=False)
    n_actions = env.action_space(env.default_params).n
    memory = FractalMemory(n_actions=n_actions)

    results = []
    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        print(f"--- seed={s}  (memory: {memory.stats()}) ---", file=sys.stderr)
        r = run_episode(s, cfg, memory, args.max_steps, args.verbose, args.env)
        results.append(r)
        print(f"  ach={r['achievements_unlocked']}  reward={r['reward']:.2f}  "
              f"hit_rate={r['mem_hit_rate']:.2f}  wall={r['wall_time_s']:.1f}s",
              file=sys.stderr)

    print(json.dumps({
        "config": vars(args),
        "memory_final_stats": memory.stats(),
        "per_seed": results,
        "achievements_per_seed": [r["achievements_unlocked"] for r in results],
        "mean_achievements": round(sum(r["achievements_unlocked"] for r in results) / len(results), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
