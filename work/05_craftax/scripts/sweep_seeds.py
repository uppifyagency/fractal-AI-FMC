"""sweep_seeds.py — multi-seed FMC su Craftax-Classic, raccoglie achievements per seed.

Output: JSON aggregato con score Crafter calcolato.

Crafter score (Hafner 2021):
    S = exp( mean_i [ ln(1 + s_i) ] ) - 1
    dove s_i ∈ [0, 100] è la success rate % dell'achievement i tra tutti gli episode.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import Counter

# Import the run_episode function
sys.path.insert(0, "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/work/05_craftax/scripts")
from fmc_craftax import run_episode, FMCConfig


# 22 Crafter-Classic achievements (Hafner 2021 §3.1)
CLASSIC_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "wake_up",
    # 22nd is "place_table" but original Hafner list may differ
]


def crafter_score(success_rates: dict[str, float], achievement_names: list[str]) -> float:
    """Geometric-mean score à la Hafner."""
    log_terms = []
    for name in achievement_names:
        s = success_rates.get(name, 0.0) * 100.0  # convert to %
        log_terms.append(math.log(1 + s))
    return math.exp(sum(log_terms) / len(log_terms)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--n_walkers", type=int, default=32)
    ap.add_argument("--time_horizon", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=42)
    args = ap.parse_args()

    cfg = FMCConfig(n_walkers=args.n_walkers, time_horizon=args.time_horizon,
                    alpha=args.alpha, beta=args.beta)

    per_seed_results = []
    achievement_counter = Counter()

    for s in range(args.seed_start, args.seed_start + args.n_seeds):
        print(f"  running seed={s}...", file=sys.stderr)
        r = run_episode(s, cfg, args.max_steps, verbose=False, env_name=args.env)
        per_seed_results.append(r)
        for ach in r["achievements_list"]:
            achievement_counter[ach] += 1

    # Compute success rate per achievement
    n_eps = len(per_seed_results)
    success_rates = {ach: count / n_eps for ach, count in achievement_counter.items()}

    # Determine achievement set: use union of seen achievements; warn if not 22
    all_achs = sorted(set(CLASSIC_ACHIEVEMENTS) | set(success_rates.keys()))
    score = crafter_score(success_rates, all_achs)

    summary = {
        "env": args.env,
        "config": {"n_walkers": args.n_walkers, "time_horizon": args.time_horizon,
                   "alpha": args.alpha, "beta": args.beta},
        "n_seeds": n_eps,
        "achievements_per_seed": [r["achievements_unlocked"] for r in per_seed_results],
        "n_steps_per_seed": [r["n_steps"] for r in per_seed_results],
        "rewards_per_seed": [r["reward"] for r in per_seed_results],
        "achievement_success_rates": {k: round(v, 4) for k, v in success_rates.items()},
        "achievements_seen": list(success_rates.keys()),
        "n_unique_achievements": len(success_rates),
        "crafter_score_pct": round(score, 4),
        "mean_achievements": round(sum(r["achievements_unlocked"] for r in per_seed_results) / n_eps, 2),
        "mean_steps": round(sum(r["n_steps"] for r in per_seed_results) / n_eps, 1),
        "mean_reward": round(sum(r["reward"] for r in per_seed_results) / n_eps, 3),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
