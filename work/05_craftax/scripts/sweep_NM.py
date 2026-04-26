"""sweep_NM.py — sweep iperparametri (N, M) su Craftax-Classic, single-process.

Tutto in un processo Python per riusare il cache JIT tra configurazioni che
condividono shape (cambia solo N e M scalari → JAX retracceia per ognuna,
ma i restartup di Python sono evitati).

Output: tabella JSON + Crafter score per ogni config.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter

sys.path.insert(0, "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/work/05_craftax/scripts")
from fmc_craftax import run_episode, FMCConfig
from sweep_seeds import CLASSIC_ACHIEVEMENTS, crafter_score


def run_config(n_walkers: int, time_horizon: int, alpha: float, beta: float,
               n_seeds: int, seed_start: int, max_steps: int, env_name: str) -> dict:
    cfg = FMCConfig(n_walkers=n_walkers, time_horizon=time_horizon, alpha=alpha, beta=beta)
    counter = Counter()
    per_seed = []
    t0 = time.time()
    for s in range(seed_start, seed_start + n_seeds):
        r = run_episode(s, cfg, max_steps, verbose=False, env_name=env_name)
        per_seed.append(r)
        for ach in r["achievements_list"]:
            counter[ach] += 1
        print(f"    seed={s}: ach={r['achievements_unlocked']} steps={r['n_steps']}", file=sys.stderr)
    wall = time.time() - t0

    rates = {ach: count / n_seeds for ach, count in counter.items()}
    all_achs = sorted(set(CLASSIC_ACHIEVEMENTS) | set(rates.keys()))
    score = crafter_score(rates, all_achs)

    return {
        "config": {"N": n_walkers, "M": time_horizon, "alpha": alpha, "beta": beta},
        "n_seeds": n_seeds,
        "wall_time_total_s": round(wall, 1),
        "wall_time_per_seed_s": round(wall / n_seeds, 1),
        "mean_achievements": round(sum(r["achievements_unlocked"] for r in per_seed) / n_seeds, 2),
        "std_achievements": round(
            (sum((r["achievements_unlocked"] - sum(rr["achievements_unlocked"] for rr in per_seed) / n_seeds) ** 2
                 for r in per_seed) / n_seeds) ** 0.5, 2),
        "mean_reward": round(sum(r["reward"] for r in per_seed) / n_seeds, 3),
        "mean_steps": round(sum(r["n_steps"] for r in per_seed) / n_seeds, 1),
        "n_unique_achievements": len(rates),
        "crafter_score_pct": round(score, 4),
        "achievements_per_seed": [r["achievements_unlocked"] for r in per_seed],
        "samples_per_decision": n_walkers * time_horizon,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Craftax-Classic-Symbolic-v1")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    # Sweep grid
    configs = [
        (32, 12),   # baseline
        (64, 12),   # 2× walkers
        (32, 20),   # longer horizon
        (64, 20),   # both
        (128, 12),  # 4× walkers
    ]

    results = []
    for N, M in configs:
        print(f"\n=== N={N}, M={M}, samples/dec={N*M} ===", file=sys.stderr)
        r = run_config(N, M, args.alpha, args.beta, args.n_seeds, args.seed_start,
                       args.max_steps, args.env)
        print(f"  → Crafter score: {r['crafter_score_pct']:.2f}%, "
              f"mean ach: {r['mean_achievements']}±{r['std_achievements']}, "
              f"wall: {r['wall_time_total_s']:.0f}s", file=sys.stderr)
        results.append(r)

    summary = {
        "env": args.env,
        "alpha": args.alpha, "beta": args.beta,
        "n_seeds": args.n_seeds, "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "configs": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
