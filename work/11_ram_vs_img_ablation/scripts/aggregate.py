"""Aggregate P3 ablation runs into a (game, obs, N, M) summary CSV.

Computes mean per cell and the RAM/IMG ratio per (game, N, M).

Usage
-----
python -m scripts.aggregate --runs runs/p3_sweep.jsonl --out runs/p3_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.runs) if l.strip()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["game"], r["obs_type"], r["N"], r["M"])].append(r["cum_reward"])

    summary = []
    for (game, obs, N, M), scores in sorted(cells.items()):
        summary.append({
            "game": game, "obs_type": obs, "N": N, "M": M,
            "n_seeds": len(scores),
            "mean": round(float(np.mean(scores)), 2),
            "std": round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 2),
        })

    # RAM/IMG ratio per (game, N, M).
    by_gnm = defaultdict(dict)
    for r in summary:
        by_gnm[(r["game"], r["N"], r["M"])][r["obs_type"]] = r["mean"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "game", "N", "M", "ram_mean", "rgb_mean", "ram_minus_rgb",
        ])
        w.writeheader()
        for (game, N, M), d in sorted(by_gnm.items()):
            ram = d.get("ram", float("nan"))
            rgb = d.get("rgb", float("nan"))
            w.writerow({
                "game": game, "N": N, "M": M,
                "ram_mean": ram, "rgb_mean": rgb,
                "ram_minus_rgb": (
                    round(ram - rgb, 2)
                    if isinstance(ram, (int, float)) and isinstance(rgb, (int, float))
                    else ""
                ),
            })

    print(f"wrote {args.out}")
    for (game, N, M), d in sorted(by_gnm.items()):
        ram = d.get("ram"); rgb = d.get("rgb")
        if ram is not None and rgb is not None:
            print(f"  {game:>10} N={N:>4} M={M:>3}  RAM={ram:>+6.1f}  RGB={rgb:>+6.1f}  Δ={ram-rgb:>+6.1f}")


if __name__ == "__main__":
    main()
