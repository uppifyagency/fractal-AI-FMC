"""compare_distance.py — compara FMC v1 (obs distance 1345-D) vs v2 (state lowd 18-D).

Usa la stessa config (N, M, alpha, beta) per entrambe le varianti, stessi seed.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter

sys.path.insert(0, "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/work/05_craftax/scripts")

from fmc_craftax import run_episode as run_v1, FMCConfig as Cfg1
from fmc_craftax_v2 import run_episode as run_v2, FMCConfig as Cfg2
from sweep_seeds import CLASSIC_ACHIEVEMENTS, crafter_score


def aggregate(results: list[dict], n_seeds: int) -> dict:
    counter = Counter()
    for r in results:
        for ach in r["achievements_list"]:
            counter[ach] += 1
    rates = {a: c / n_seeds for a, c in counter.items()}
    all_achs = sorted(set(CLASSIC_ACHIEVEMENTS) | set(rates.keys()))
    score = crafter_score(rates, all_achs)
    mean_ach = sum(r["achievements_unlocked"] for r in results) / n_seeds
    std_ach = (sum((r["achievements_unlocked"] - mean_ach) ** 2 for r in results) / n_seeds) ** 0.5
    return {
        "mean_achievements": round(mean_ach, 2),
        "std_achievements": round(std_ach, 2),
        "n_unique_achievements": len(rates),
        "crafter_score_pct": round(score, 4),
        "achievements_per_seed": [r["achievements_unlocked"] for r in results],
        "mean_steps": round(sum(r["n_steps"] for r in results) / n_seeds, 1),
        "mean_reward": round(sum(r["reward"] for r in results) / n_seeds, 3),
        "achievements_seen": sorted(rates.keys()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_walkers", type=int, default=32)
    ap.add_argument("--time_horizon", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    args = ap.parse_args()

    print("=== V1: obs distance (1345-D L2) ===", file=sys.stderr)
    cfg1 = Cfg1(args.n_walkers, args.time_horizon, args.alpha, args.beta)
    v1_results = []
    t0 = time.time()
    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        r = run_v1(s, cfg1, args.max_steps, verbose=False, env_name="Craftax-Classic-Symbolic-v1")
        v1_results.append(r)
        print(f"  seed={s}: ach={r['achievements_unlocked']} steps={r['n_steps']}", file=sys.stderr)
    v1_wall = time.time() - t0

    print("\n=== V2: state lowd distance (18-D L2) ===", file=sys.stderr)
    cfg2 = Cfg2(args.n_walkers, args.time_horizon, args.alpha, args.beta)
    v2_results = []
    t0 = time.time()
    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        r = run_v2(s, cfg2, args.max_steps, verbose=False)
        v2_results.append(r)
        print(f"  seed={s}: ach={r['achievements_unlocked']} steps={r['n_steps']}", file=sys.stderr)
    v2_wall = time.time() - t0

    summary = {
        "config": {"N": args.n_walkers, "M": args.time_horizon,
                   "alpha": args.alpha, "beta": args.beta,
                   "n_seeds": args.n_seeds, "seed_start": args.seed_start},
        "v1_obs_1345d": {**aggregate(v1_results, args.n_seeds), "wall_total_s": round(v1_wall, 1)},
        "v2_state_18d": {**aggregate(v2_results, args.n_seeds), "wall_total_s": round(v2_wall, 1)},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
