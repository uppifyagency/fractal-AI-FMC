"""fmc_crafter.py — FMC port to Crafter-original (Hafner 2021).

Pure-NumPy FMC implementation that branches via env deepcopy. Compatible
with crafter==1.8.3, which exposes a deepcopy-able Env (~1.2 ms per copy).

Run with `--shaping=v4` for baseline, `--shaping=exp17` for our recipe.

Usage:
    python fmc_crafter.py --seed 42 --shaping exp17 --N 64 --M 20
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import crafter


CRAFTER_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]
N_ACTIONS = 17
EPS = 1e-8


# Tier weights — copied from work/05_craftax/autoresearch/fmc_mutable.py
INV_TIER_WEIGHTS_EXP17 = {
    "wood": 2.0, "stone": 4.0, "coal": 8.0, "iron": 16.0, "diamond": 64.0,
    "sapling": 0.5,
    "wood_pickaxe": 6.0, "stone_pickaxe": 12.0, "iron_pickaxe": 24.0,
    "wood_sword": 6.0, "stone_sword": 12.0, "iron_sword": 24.0,
}
INV_TIER_WEIGHTS_V4 = {k: 0.0 for k in INV_TIER_WEIGHTS_EXP17}

ACH_WEIGHTS_EXP17 = {
    "collect_wood": 10.0, "place_table": 10.0, "eat_cow": 30.0,
    "collect_sapling": 20.0, "collect_drink": 20.0, "make_wood_pickaxe": 20.0,
    "make_stone_pickaxe": 80.0, "make_iron_pickaxe": 200.0,
    "make_wood_sword": 20.0, "make_stone_sword": 50.0, "make_iron_sword": 200.0,
    "place_plant": 20.0, "defeat_zombie": 30.0, "collect_stone": 30.0,
    "place_stone": 20.0, "eat_plant": 200.0, "defeat_skeleton": 50.0,
    "collect_iron": 120.0, "collect_coal": 80.0, "place_furnace": 80.0,
    "collect_diamond": 300.0, "wake_up": 20.0,
}
ACH_WEIGHTS_V4 = {k: 0.0 for k in ACH_WEIGHTS_EXP17}


def inventory_total(info: dict, weights: dict) -> float:
    inv = info.get("inventory", {})
    return sum(weights.get(k, 0.0) * inv.get(k, 0) for k in weights)


def relativize(rs: np.ndarray) -> np.ndarray:
    """FMC relativize: log regime for z>0, exp regime for z<=0."""
    mu = float(rs.mean())
    sd = float(rs.std())
    if sd < EPS:
        return np.ones_like(rs)
    z = (rs - mu) / sd
    out = np.where(z > 0, 1.0 + np.log1p(np.maximum(z, 0)),
                   np.exp(np.minimum(z, 0)) / np.e)
    return out


class FMCCrafter:
    def __init__(self, root_env, N: int, M: int, alpha: float = 1.0,
                 beta: float = 1.0, inv_alpha: float = 0.5,
                 inv_weights: dict = INV_TIER_WEIGHTS_EXP17,
                 ach_weights: dict = ACH_WEIGHTS_EXP17,
                 rng: np.random.Generator = None):
        self.root_env = root_env
        self.N = N
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.inv_alpha = inv_alpha
        self.inv_weights = inv_weights
        self.ach_weights = ach_weights
        self.rng = rng or np.random.default_rng()

    def decide(self) -> int:
        N, M = self.N, self.M
        # Branch N walker copies of the root state
        walkers = [copy.deepcopy(self.root_env) for _ in range(N)]
        init_actions = self.rng.integers(0, N_ACTIONS, size=N)
        cum_rewards = np.zeros(N)
        alive = np.ones(N, dtype=bool)
        # Baseline inv-value at root (for delta tracking)
        # To get info without stepping, do a no-op step on a throwaway copy:
        probe = copy.deepcopy(self.root_env)
        _, _, _, info0 = probe.step(0)
        inv_baseline = inventory_total(info0, self.inv_weights)
        ach_baseline = set(k for k, v in info0.get("achievements", {}).items()
                           if v > 0)
        # Per-walker last seen for ach-fire bonus
        ach_seen = [set(ach_baseline) for _ in range(N)]

        # Walker observation tensor for distance computation — collected
        # at end of each tick for cloning step
        for t in range(M):
            actions = init_actions if t == 0 else \
                      self.rng.integers(0, N_ACTIONS, size=N)
            for i in range(N):
                if not alive[i]:
                    continue
                obs_i, r, done, info = walkers[i].step(int(actions[i]))
                # Env reward
                cum_rewards[i] += r
                # Ach-fire bonus (relative to root)
                ach_now = set(k for k, v in info.get("achievements", {}).items()
                              if v > 0)
                new_ach = ach_now - ach_baseline - ach_seen[i]
                # Note: subtract ach_seen[i] so we only fire ONCE per
                # walker per achievement, even if it remains True
                for a in new_ach:
                    cum_rewards[i] += self.ach_weights.get(a, 0.0)
                ach_seen[i] |= ach_now
                # Inv-tier delta
                cur_inv = inventory_total(info, self.inv_weights)
                delta_inv = max(0.0, cur_inv - inv_baseline)
                cum_rewards[i] += self.inv_alpha * delta_inv
                if done:
                    alive[i] = False

            # Cloning at end of tick
            R_norm = relativize(cum_rewards) * alive.astype(np.float32)
            # Distance: use random pair distance on cum_reward (no obs
            # tensor available cheaply); a cheap proxy is the diversity
            # induced by the rolling cum_reward gap.
            perm = self.rng.permutation(N)
            perm = np.where(perm == np.arange(N), (perm + 1) % N, perm)
            distances = np.abs(cum_rewards - cum_rewards[perm])
            D_norm = relativize(distances) * alive.astype(np.float32)
            VR = (R_norm ** self.alpha) * (D_norm ** self.beta)

            perm2 = self.rng.permutation(N)
            perm2 = np.where(perm2 == np.arange(N), (perm2 + 1) % N, perm2)
            VR_self = VR
            VR_other = VR[perm2]
            denom = np.where(VR_self > EPS, VR_self, EPS)
            clone_prob = np.clip((VR_other - VR_self) / denom, 0.0, 1.0)
            # Don't clone if self is dead (unless partner alive)
            clone_prob = np.where(alive, clone_prob, 1.0)
            draws = self.rng.uniform(size=N)
            will_clone = (draws < clone_prob) & alive[perm2]

            for i in range(N):
                if will_clone[i]:
                    src = perm2[i]
                    walkers[i] = copy.deepcopy(walkers[src])
                    init_actions[i] = init_actions[src]
                    cum_rewards[i] = cum_rewards[src]
                    alive[i] = alive[src]
                    ach_seen[i] = set(ach_seen[src])

        # Vote
        votes = np.zeros(N_ACTIONS)
        for i in range(N):
            if alive[i]:
                votes[init_actions[i]] += 1
        return int(votes.argmax())


def run_episode(seed: int, max_steps: int = 500, N: int = 64, M: int = 20,
                shaping: str = "exp17") -> dict:
    """Run one FMC-Crafter episode."""
    if shaping == "exp17":
        inv_w = INV_TIER_WEIGHTS_EXP17
        ach_w = ACH_WEIGHTS_EXP17
        inv_a = 0.5
    elif shaping == "v4":
        inv_w = INV_TIER_WEIGHTS_V4
        ach_w = ACH_WEIGHTS_V4
        inv_a = 0.0
    else:
        raise ValueError(f"unknown shaping: {shaping}")

    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    env.reset()
    fmc = FMCCrafter(env, N=N, M=M, inv_weights=inv_w,
                     ach_weights=ach_w, inv_alpha=inv_a, rng=rng)

    cum_reward = 0.0
    achievements_unlocked = set()
    n_steps = 0
    t_start = time.time()

    for step in range(max_steps):
        action = fmc.decide()
        obs, r, done, info = env.step(action)
        cum_reward += r
        for k, v in info.get("achievements", {}).items():
            if v > 0:
                achievements_unlocked.add(k)
        n_steps += 1
        if done:
            break

    wall = time.time() - t_start
    return {
        "seed": seed,
        "shaping": shaping,
        "N": N, "M": M,
        "reward": float(cum_reward),
        "n_steps_decisions": n_steps,
        "achievements_unlocked": len(achievements_unlocked),
        "achievements_list": sorted(achievements_unlocked),
        "wall_time_s": float(wall),
        "decisions_per_sec": float(n_steps / wall) if wall > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42",
                    help="comma-separated seed list, e.g. 42,43,44")
    ap.add_argument("--shaping", choices=["v4", "exp17"], default="exp17")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_runs = []
    print(f"[fmc_crafter] running {len(seeds)} seeds, "
          f"shaping={args.shaping}, N={args.N}, M={args.M}", file=sys.stderr)
    t0 = time.time()
    for s in seeds:
        print(f"[fmc_crafter] seed={s} (t={time.time()-t0:.0f}s)...",
              file=sys.stderr, flush=True)
        try:
            r = run_episode(seed=s, max_steps=args.max_steps,
                            N=args.N, M=args.M, shaping=args.shaping)
            out_runs.append(r)
            print(f"  → ach={r['achievements_unlocked']} "
                  f"reward={r['reward']:.2f} wall={r['wall_time_s']:.1f}s",
                  file=sys.stderr)
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}", file=sys.stderr)
            break

    # Aggregate Crafter score (Hafner formula)
    import math
    rates = {a: 0.0 for a in CRAFTER_ACHIEVEMENTS}
    for run in out_runs:
        for a in run["achievements_list"]:
            if a in rates:
                rates[a] += 1.0 / len(out_runs)
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTER_ACHIEVEMENTS]
    crafter_score = math.exp(sum(log_terms) / len(log_terms)) - 1.0

    out = {
        "crafter_score": round(crafter_score, 4),
        "n_seeds": len(out_runs),
        "shaping": args.shaping,
        "N": args.N, "M": args.M,
        "achievement_freq": {k: round(v, 4) for k, v in rates.items()},
        "raw_runs": out_runs,
        "wall_total_s": round(time.time() - t0, 1),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[fmc_crafter] crafter_score={crafter_score:.2f}% "
          f"n={len(out_runs)} wall={time.time()-t0:.0f}s "
          f"saved={args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
