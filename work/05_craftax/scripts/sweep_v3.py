"""sweep_v3.py — sweep multi-config FMC v3 (action_repeat × intrinsic_inv_alpha).

Esegue una matrice di config sui seed 42-46, con la config base N=64 M=20.
Calcola Crafter score (Hafner geometric-mean) per ogni config.

Output: JSON aggregato su stdout, log su stderr.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter

sys.path.insert(0, "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/work/05_craftax/scripts")
from fmc_craftax_v3 import run_episode, FMCConfig


CLASSIC_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "wake_up", "place_table",
]


def crafter_score(success_rates: dict, ach_names: list) -> float:
    log_terms = [math.log(1 + success_rates.get(n, 0.0) * 100.0) for n in ach_names]
    return math.exp(sum(log_terms) / len(log_terms)) - 1


def run_config(cfg: FMCConfig, seeds: list[int], max_steps: int, env_name: str, label: str) -> dict:
    t0 = time.time()
    per_seed = []
    counter = Counter()
    print(f"\n[{label}] cfg={cfg}", file=sys.stderr)
    for s in seeds:
        r = run_episode(s, cfg, max_steps, verbose=False, env_name=env_name)
        per_seed.append(r)
        for ach in r["achievements_list"]:
            counter[ach] += 1
        print(f"  seed={s}  ach={r['achievements_unlocked']}  reward={r['reward']:.1f}  "
              f"steps_dec={r['n_steps_decisions']}  steps_env={r['n_steps_env']}  "
              f"wall={r['wall_time_s']:.1f}s", file=sys.stderr)

    n_eps = len(per_seed)
    success = {a: c / n_eps for a, c in counter.items()}
    all_achs = sorted(set(CLASSIC_ACHIEVEMENTS) | set(success.keys()))
    score = crafter_score(success, all_achs)

    return {
        "label": label,
        "config": {
            "n_walkers": cfg.n_walkers, "time_horizon": cfg.time_horizon,
            "alpha": cfg.alpha, "beta": cfg.beta,
            "action_repeat": cfg.action_repeat,
            "intrinsic_inv_alpha": cfg.intrinsic_inv_alpha,
        },
        "n_seeds": n_eps,
        "achievements_per_seed": [r["achievements_unlocked"] for r in per_seed],
        "rewards_per_seed": [r["reward"] for r in per_seed],
        "n_steps_dec_per_seed": [r["n_steps_decisions"] for r in per_seed],
        "achievement_success_rates": {k: round(v, 4) for k, v in success.items()},
        "n_unique_achievements": len(success),
        "crafter_score_pct": round(score, 4),
        "mean_achievements": round(sum(r["achievements_unlocked"] for r in per_seed) / n_eps, 2),
        "std_achievements": round(
            (sum((r["achievements_unlocked"] - sum(rr["achievements_unlocked"] for rr in per_seed) / n_eps) ** 2
                  for r in per_seed) / n_eps) ** 0.5, 2),
        "wall_total_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--n_walkers", type=int, default=64)
    ap.add_argument("--time_horizon", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--configs", default="all",
                    help="csv di config: baseline,ar3,ar5,inv,ar3inv,ar5inv,all")
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    base = dict(n_walkers=args.n_walkers, time_horizon=args.time_horizon,
                alpha=args.alpha, beta=args.beta)

    all_configs = {
        "baseline":  FMCConfig(**base, action_repeat=1, intrinsic_inv_alpha=0.0),
        "ar3":       FMCConfig(**base, action_repeat=3, intrinsic_inv_alpha=0.0),
        "ar5":       FMCConfig(**base, action_repeat=5, intrinsic_inv_alpha=0.0),
        "inv":       FMCConfig(**base, action_repeat=1, intrinsic_inv_alpha=0.2),
        "ar3inv":    FMCConfig(**base, action_repeat=3, intrinsic_inv_alpha=0.2),
        "ar5inv":    FMCConfig(**base, action_repeat=5, intrinsic_inv_alpha=0.2),
    }

    if args.configs == "all":
        chosen = list(all_configs.keys())
    else:
        chosen = [c.strip() for c in args.configs.split(",")]

    results = {}
    for label in chosen:
        if label not in all_configs:
            print(f"[skip] unknown config: {label}", file=sys.stderr)
            continue
        results[label] = run_config(all_configs[label], seeds, args.max_steps, args.env, label)

    # Riassunto top
    summary_table = sorted(
        [(label, r["crafter_score_pct"], r["mean_achievements"], r["n_unique_achievements"], r["wall_total_s"])
         for label, r in results.items()],
        key=lambda x: -x[1],
    )
    print("\n=== Crafter score ranking ===", file=sys.stderr)
    for label, score, mean_a, n_uniq, wall in summary_table:
        print(f"  {label:10s}  score={score:5.2f}%  mean_ach={mean_a:.2f}  uniq={n_uniq:2d}  wall={wall:.0f}s",
              file=sys.stderr)

    print(json.dumps({"sweep": results, "ranking": summary_table}, indent=2))


if __name__ == "__main__":
    main()
