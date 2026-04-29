"""merge_run007.py — Combina due batch di seed in un aggregato 5-seed.

Input:
  - run007_strategic.json (seeds 42,43,44)
  - run007_strategic_extended.json (seeds 45,46)

Output:
  - run007_strategic_5seed.json con per_cell aggregato su 5 seed totali
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]

NEVER_UNLOCKED_v4 = ["collect_diamond", "make_iron_pickaxe", "make_iron_sword", "eat_plant"]


def crafter_score(success_rates_dict: dict) -> float:
    rates = []
    for ach in CRAFTAX_CLASSIC_ACHIEVEMENTS:
        s = success_rates_dict.get(ach, 0.0)
        rates.append(math.log(1.0 + 100.0 * s))
    return math.exp(sum(rates) / len(rates)) - 1.0


def aggregate(raw_runs: list[dict]) -> dict:
    by_NM = {}
    for r in raw_runs:
        if "error" in r:
            continue
        key = (r["N"], r["M"])
        by_NM.setdefault(key, []).append(r)

    summary = []
    for (N, M), rs in sorted(by_NM.items()):
        freq = {ach: 0 for ach in CRAFTAX_CLASSIC_ACHIEVEMENTS}
        for r in rs:
            for ach in r["achievements_list"]:
                if ach in freq:
                    freq[ach] += 1
        rates = {ach: freq[ach] / len(rs) for ach in freq}
        score = crafter_score(rates)
        mean_ach = sum(r["achievements_unlocked"] for r in rs) / len(rs)
        mean_rew = sum(r["reward"] for r in rs) / len(rs)
        mean_steps = sum(r["n_steps_decisions"] for r in rs) / len(rs)
        mean_wall = sum(r["wall_total_s"] for r in rs) / len(rs)
        # CI95 di mean_ach via deviazione standard / sqrt(n) * 1.96
        achs = [r["achievements_unlocked"] for r in rs]
        if len(achs) > 1:
            mu = sum(achs) / len(achs)
            var = sum((x - mu) ** 2 for x in achs) / (len(achs) - 1)
            ci95 = 1.96 * math.sqrt(var / len(achs))
        else:
            ci95 = 0.0

        gate_unlocks = {ach: rates[ach] for ach in NEVER_UNLOCKED_v4 if rates[ach] > 0}

        summary.append({
            "N": N, "M": M, "n_seeds": len(rs),
            "seeds_used": sorted(r["seed"] for r in rs),
            "crafter_score_pct": round(score, 2),
            "mean_achievements": round(mean_ach, 2),
            "mean_achievements_ci95": round(ci95, 2),
            "mean_reward": round(mean_rew, 2),
            "mean_n_steps": round(mean_steps, 1),
            "mean_wall_s": round(mean_wall, 1),
            "achievement_freq": rates,
            "gate_unlocks_v4_blockers": gate_unlocks,
        })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+",
                    default=["work/05_craftax/results/run007_strategic.json",
                             "work/05_craftax/results/run007_strategic_extended.json"])
    ap.add_argument("--out", default="work/05_craftax/results/run007_strategic_5seed.json")
    args = ap.parse_args()

    raw_combined = []
    total_wall_input = 0.0
    for i, path in enumerate(args.batches):
        d = json.load(open(path))
        raws = d.get("raw_runs", [])
        raw_combined.extend(raws)
        total_wall_input += d.get("elapsed_s", 0) or 0
        print(f"Batch {i+1} ({path}): {len(raws)} raw runs", file=sys.stderr)
    print(f"Combined: {len(raw_combined)} raw runs", file=sys.stderr)

    summary = aggregate(raw_combined)

    # Total wall e seeds metadata
    total_wall = total_wall_input
    all_seeds = sorted(set(r["seed"] for r in raw_combined))

    out = {
        "per_cell": summary,
        "raw_runs": raw_combined,
        "total_wall_s": total_wall,
        "seeds": all_seeds,
        "n_total_episodes": len(raw_combined),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"Total wall (both batches): {total_wall/60:.1f} min", file=sys.stderr)
    print(f"Episodes: {len(raw_combined)} across seeds {all_seeds}", file=sys.stderr)
    print(f"\nPer-cell 5-seed summary:", file=sys.stderr)
    print(f"{'N':>4} {'M':>4} {'Seeds':>5} {'Crafter%':>8} {'mean_ach':>8} {'CI95':>5}", file=sys.stderr)
    for c in summary:
        print(f"{c['N']:>4} {c['M']:>4} {c['n_seeds']:>5} "
              f"{c['crafter_score_pct']:>8.2f} {c['mean_achievements']:>8.2f} "
              f"+/-{c['mean_achievements_ci95']:.2f}", file=sys.stderr)

    # Decision gate verdict
    any_blocker = any(c["gate_unlocks_v4_blockers"] for c in summary)
    print(f"\nDECISION GATE: {'BLOCKER FIRED' if any_blocker else 'NO BLOCKER (hypothesis FALSIFIED)'}",
          file=sys.stderr)
    if any_blocker:
        for c in summary:
            if c["gate_unlocks_v4_blockers"]:
                print(f"  N={c['N']} M={c['M']}: {c['gate_unlocks_v4_blockers']}", file=sys.stderr)


if __name__ == "__main__":
    main()
