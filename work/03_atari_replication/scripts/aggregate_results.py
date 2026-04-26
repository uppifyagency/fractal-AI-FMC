"""aggregate_results.py — JSONs in results/ → SUMMARY.md con tabella + CI.

Calcola mean ± 95%CI (t-distribution, n piccolo) per ciascun gioco.
Confronta col `target.paper_score` se presente nella config.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def t_critical_95(n: int) -> float:
    """t critico al 95% per n osservazioni (tabella ridotta, n=2..30)."""
    table = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45,
             8: 2.36, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.05}
    if n in table:
        return table[n]
    if n < 2:
        return float("inf")
    if n > 30:
        return 1.96  # asintotico
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= n < keys[i + 1]:
            return table[keys[i]]
    return 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    by_game: dict[str, list[dict]] = defaultdict(list)
    for f in args.input.glob("*.json"):
        data = json.loads(f.read_text())
        by_game[data["game"]].append(data)

    if not by_game:
        print("No results to aggregate")
        return 1

    rows = []
    rows.append("# Phase 1 — Risultati replicazione Atari")
    rows.append("")
    rows.append(f"_Generato il {Path(__file__).stem}, {len(by_game)} giochi, "
                f"{sum(len(v) for v in by_game.values())} run totali_")
    rows.append("")
    rows.append("| Gioco | n | Reward (mean ± 95%CI) | Samples/action | Walltime (s) | Paper score | Δ paper |")
    rows.append("|---|---|---|---|---|---|---|")

    for game, runs in sorted(by_game.items()):
        rewards = [r["result"]["reward"] for r in runs]
        samples = [r["result"]["samples_per_action_avg"] for r in runs]
        walls = [r["result"]["wall_time_s"] for r in runs]
        n = len(rewards)
        mean_r = statistics.mean(rewards)
        if n >= 2:
            std_r = statistics.stdev(rewards)
            ci = t_critical_95(n) * std_r / math.sqrt(n)
        else:
            ci = float("nan")
        mean_s = statistics.mean(samples)
        mean_w = statistics.mean(walls)
        paper = runs[0]["config"].get("target", {}).get("paper_score")
        delta = f"{(mean_r - paper) / paper * 100:+.1f}%" if paper else "—"
        paper_str = f"{paper:,}" if paper else "—"
        rows.append(
            f"| {game} | {n} | {mean_r:,.0f} ± {ci:,.0f} | "
            f"{mean_s:.1f} | {mean_w:.1f} | {paper_str} | {delta} |"
        )

    rows.append("")
    rows.append("## Discussione")
    rows.append("")
    rows.append("- Δ paper > +30% atteso solo se hardware moderno permette N più alto")
    rows.append("- Δ paper < -30% ⇒ probabile bug di setup (controllare ROM, sticky-actions)")
    rows.append("- CI larghi su Montezuma sono attesi (varianza alta su sparse reward)")

    args.output.write_text("\n".join(rows))
    print(f"Saved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
