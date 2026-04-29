"""analyze_run007.py — Decision-gate analyzer per il sweep run007.

Verifica:
  1. Crafter score per ogni cella (N, M)
  2. Achievement frequency map (22 ach x cells)
  3. Decision gate: M=160 sblocca uno dei 4 blocker?
       collect_diamond, make_iron_pickaxe, make_iron_sword, eat_plant
  4. Wall time scaling vs (N, M)
  5. Best cell vs v4_p02_delta historical 21.87%
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]

NEVER_UNLOCKED_v4 = ["collect_diamond", "make_iron_pickaxe", "make_iron_sword", "eat_plant"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="work/05_craftax/results/run007_strategic.json")
    ap.add_argument("--baseline", type=float, default=21.87,
                    help="Historical v4_p02_delta 30-seed score")
    args = ap.parse_args()

    d = json.load(open(args.input))
    cells = d["per_cell"]
    raw = d.get("raw_runs", [])

    print("=" * 78)
    print(f"RUN 007 — Strategic NxM Sweep Analysis")
    print(f"Progress: {d.get('progress', '?')}    "
          f"Elapsed: {d.get('elapsed_s', 0)/60:.1f} min    "
          f"Partial: {d.get('partial', False)}")
    print("=" * 78)

    # --- Tabella principale ---
    print("\n## CELL RESULTS (3 seeds each)")
    print()
    header = f"{'N':>4} {'M':>4} {'Seeds':>5} {'Crafter%':>8} {'mean_ach':>8} {'mean_steps':>10} {'wall_s':>7}"
    print(header)
    print("-" * len(header))
    for c in cells:
        print(f"{c['N']:>4} {c['M']:>4} {c['n_seeds']:>5} "
              f"{c['crafter_score_pct']:>8.2f} {c['mean_achievements']:>8.2f} "
              f"{c['mean_n_steps']:>10.1f} {c['mean_wall_s']:>7.1f}")

    # --- DECISION GATE ---
    print("\n## DECISION GATE — did M increase unlock the 4 v4-blockers?")
    print(f"\nBlockers: {NEVER_UNLOCKED_v4}")
    print()
    header = f"{'N':>4} {'M':>4} | " + " | ".join(f"{a:>22}" for a in NEVER_UNLOCKED_v4)
    print(header)
    print("-" * len(header))
    any_blocker_fired = False
    for c in cells:
        rates = c["achievement_freq"]
        cells_str = []
        for ach in NEVER_UNLOCKED_v4:
            r = rates.get(ach, 0.0)
            if r > 0:
                cells_str.append(f"{r:>22.2f}")
                any_blocker_fired = True
            else:
                cells_str.append(f"{'.':>22}")
        print(f"{c['N']:>4} {c['M']:>4} | " + " | ".join(cells_str))

    if any_blocker_fired:
        print("\n*** AT LEAST ONE BLOCKER FIRED ***")
    else:
        print("\nNO BLOCKER FIRED across all cells — hypothesis FALSIFIED")

    # --- Best cell vs baseline ---
    print(f"\n## VS HISTORICAL BASELINE v4_p02_delta = {args.baseline:.2f}% (30 seeds CI95)")
    print()
    sorted_by_score = sorted(cells, key=lambda c: -c["crafter_score_pct"])
    for c in sorted_by_score[:5]:
        delta = c["crafter_score_pct"] - args.baseline
        marker = "+" if delta > 0 else ""
        print(f"  N={c['N']:>4} M={c['M']:>4}  Crafter={c['crafter_score_pct']:>5.2f}%   "
              f"Delta vs baseline = {marker}{delta:+.2f} pp   ach={c['mean_achievements']:.1f}")

    # --- Achievement frequency heatmap (per achievement, all cells) ---
    print("\n## ACHIEVEMENT FREQUENCY HEATMAP (rate of unlock across 3 seeds)")
    print()
    cell_keys = [(c['N'], c['M']) for c in cells]
    cell_labels = [f"N{N}M{M}" for N, M in cell_keys]
    col_widths = [max(6, len(lbl)) for lbl in cell_labels]
    header = f"{'achievement':>22} | " + " | ".join(
        f"{lbl:>{w}}" for lbl, w in zip(cell_labels, col_widths))
    print(header)
    print("-" * len(header))
    for ach in CRAFTAX_CLASSIC_ACHIEVEMENTS:
        cells_str = []
        for c, w in zip(cells, col_widths):
            r = c["achievement_freq"].get(ach, 0.0)
            if r > 0:
                cells_str.append(f"{r:>{w}.2f}")
            else:
                cells_str.append(f"{'.':>{w}}")
        print(f"{ach:>22} | " + " | ".join(cells_str))

    # --- Cost scaling table ---
    print("\n## COST SCALING (mean wall sec per episode)")
    print()
    by_N = defaultdict(dict)
    for c in cells:
        by_N[c['N']][c['M']] = c['mean_wall_s']
    Ns = sorted(by_N.keys())
    Ms = sorted({c['M'] for c in cells})
    nm_label = "N\\M"
    header = f"{nm_label:>5} | " + " | ".join(f"{m:>6}" for m in Ms)
    print(header)
    print("-" * len(header))
    for N in Ns:
        row = []
        for M in Ms:
            v = by_N[N].get(M)
            row.append(f"{v:>6.1f}" if v is not None else f"{'-':>6}")
        print(f"{N:>5} | " + " | ".join(row))

    # --- Verdict ---
    print("\n## VERDICT")
    if any_blocker_fired:
        print("HYPOTHESIS PARTIALLY/FULLY VALIDATED:")
        print("  Increasing M (or N) unlocks one or more v4-blocker achievements.")
        print("  The bottleneck WAS planning horizon — pivot to scaled-NxM is justified.")
        print(f"  Best cell: see top of vs-baseline table.")
    else:
        print("HYPOTHESIS FALSIFIED:")
        print("  No blocker achievement fired even at M=160.")
        print("  Planning horizon is NOT the limiting factor — pivot to:")
        print("    a) macro-actions / skill primitives (reduces effective horizon)")
        print("    b) hybrid FMC + NN value function")
        print("    c) Badger Level-1 (meta-FMC over reward configs)")


if __name__ == "__main__":
    main()
