"""sweep_run007_NM_GPU.py — Run 007: NxM sweep su GPU/CPU per testare M=160 hypothesis.

Hypothesis (decision-gate):
  Se M=160 sblocca anche solo 1-2 dei 4 achievement mai unlocked
  (collect_diamond, make_iron_pickaxe, make_iron_sword, eat_plant)
  -> il problema era horizon, non algoritmo. Si pivota su scaling N x M.
  Altrimenti FMC vanilla scaled e' davvero saturo -> macro-actions.

Setup:
  Config base = v4_p02_delta (best 30-seed: 21.87% +/- 1.21):
    intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_sigma=10, mode='delta'

  Grid (strategic, 9 cell):
    Asse M @ N=256: M in {20, 40, 80, 160}
    Asse N @ M=20:  N in {128, 256, 512}
    Corner low: (N=128, M=80)
    Corner high: (N=512, M=160)

  3 seed per cell (42, 43, 44) -> stima del Crafter score con CI ridotto.
  La cella di parita' (N=64, M=20, intrinsic ON) gira come reference per
  verificare che il setup riproduca il 21-22% atteso.

Output:
  results/run007_sweep.json — record per (config, seed) con achievements,
    timing, reward, n_steps. Compatibile con analisi pandas.

Run:
  PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
  $PY work/05_craftax/scripts/sweep_run007_NM_GPU.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# JAX_PLATFORMS=cpu di default (Metal non supporta default_memory_space)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmc_craftax_v4 import FMCConfig, run_episode  # noqa: E402


# Lista dei 22 achievement Craftax-Classic per book-keeping completo
CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "collect_iron",
    "collect_coal",
    "place_furnace",
    "collect_diamond",
    "wake_up",
]

NEVER_UNLOCKED_v4 = ["collect_diamond", "make_iron_pickaxe", "make_iron_sword", "eat_plant"]


def crafter_score(success_rates_dict: dict) -> float:
    """Crafter score = exp(mean(log(1 + 100*s_i))) - 1, su tutti i 22 achievement.

    s_i = frequenza di unlock dell'achievement i sui seed.
    """
    rates = []
    for ach in CRAFTAX_CLASSIC_ACHIEVEMENTS:
        s = success_rates_dict.get(ach, 0.0)
        rates.append(math.log(1.0 + 100.0 * s))
    mean_log = sum(rates) / len(rates)
    return math.exp(mean_log) - 1.0


def make_grid(mode: str = "strategic"):
    """Strategic 9-cell grid o full 12-cell (slow)."""
    if mode == "full":
        return [(N, M) for N in [128, 256, 512] for M in [20, 40, 80, 160]]
    if mode == "strategic":
        cells = []
        # Asse M @ N=256
        for M in [20, 40, 80, 160]:
            cells.append((256, M))
        # Asse N @ M=20 (escludo N=256 gia' fatto sopra)
        cells.append((128, 20))
        cells.append((512, 20))
        # Corners
        cells.append((128, 80))
        cells.append((512, 160))
        return cells
    if mode == "smoke":
        # 2 cells solo per debug
        return [(64, 20), (256, 80)]
    if mode == "missing":
        # Le 4 celle del full 12-cell grid non coperte dallo strategic
        return [(128, 40), (128, 160), (512, 40), (512, 80)]
    raise ValueError(f"unknown grid mode {mode}")


def run_one(N: int, M: int, seed: int, max_steps: int) -> dict:
    cfg = FMCConfig(
        n_walkers=N, time_horizon=M, alpha=1.0, beta=1.0,
        action_repeat=1, intrinsic_inv_alpha=0.5,
        proximity_alpha=0.2, proximity_sigma=10.0, proximity_mode="delta",
    )
    t0 = time.time()
    res = run_episode(seed=seed, cfg=cfg, max_steps=max_steps,
                      verbose=False, env_name="Craftax-Classic-Symbolic-v1")
    res["wall_total_s"] = time.time() - t0
    res["N"] = N
    res["M"] = M
    return res


def aggregate(results: list[dict]) -> dict:
    """Compute Crafter score + per-achievement frequency from a list of seed results."""
    by_NM = {}
    for r in results:
        key = (r["N"], r["M"])
        by_NM.setdefault(key, []).append(r)

    summary = []
    for (N, M), rs in sorted(by_NM.items()):
        # Per-achievement frequency
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

        # Decision-gate: did any of the 4 never-unlocked fire?
        gate_unlocks = {ach: rates[ach] for ach in NEVER_UNLOCKED_v4 if rates[ach] > 0}

        summary.append({
            "N": N,
            "M": M,
            "n_seeds": len(rs),
            "crafter_score_pct": round(score, 2),
            "mean_achievements": round(mean_ach, 2),
            "mean_reward": round(mean_rew, 2),
            "mean_n_steps": round(mean_steps, 1),
            "mean_wall_s": round(mean_wall, 1),
            "achievement_freq": rates,
            "gate_unlocks_v4_blockers": gate_unlocks,
        })

    return {"per_cell": summary, "raw_runs": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--grid", choices=["strategic", "full", "smoke", "missing"], default="strategic")
    ap.add_argument("--out", type=str,
                    default="work/05_craftax/results/run007_sweep.json")
    ap.add_argument("--include_baseline", action="store_true",
                    help="Include N=64,M=20 reference (parity check ~21%)")
    args = ap.parse_args()

    grid = make_grid(args.grid)
    if args.include_baseline:
        grid = [(64, 20)] + grid

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {jax.default_backend()} {jax.devices()}", file=sys.stderr)
    print(f"Grid {args.grid}: {len(grid)} cells, {len(args.seeds)} seeds = "
          f"{len(grid)*len(args.seeds)} runs", file=sys.stderr)
    print(f"Cells: {grid}", file=sys.stderr)
    print(f"Seeds: {args.seeds}", file=sys.stderr)
    print(f"Max steps: {args.max_steps}", file=sys.stderr)

    all_results = []
    t_global = time.time()
    total_cells = len(grid) * len(args.seeds)
    done = 0

    for (N, M) in grid:
        for seed in args.seeds:
            done += 1
            print(f"[{done}/{total_cells}] N={N} M={M} seed={seed} ...",
                  file=sys.stderr, flush=True)
            try:
                r = run_one(N, M, seed, args.max_steps)
                print(f"  -> wall={r['wall_total_s']:.1f}s reward={r['reward']:.1f} "
                      f"ach={r['achievements_unlocked']} steps={r['n_steps_decisions']}",
                      file=sys.stderr, flush=True)
                # Mostriamo se uno dei 4 blocker e' fired
                fired_blockers = [a for a in r["achievements_list"]
                                  if a in NEVER_UNLOCKED_v4]
                if fired_blockers:
                    print(f"  *** BLOCKER FIRED: {fired_blockers} ***",
                          file=sys.stderr, flush=True)
                all_results.append(r)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                all_results.append({
                    "N": N, "M": M, "seed": seed,
                    "error": f"{type(e).__name__}: {e}",
                    "achievements_list": [], "achievements_unlocked": 0,
                    "reward": 0.0, "n_steps_decisions": 0, "wall_total_s": 0.0,
                })

            # Save incremental in caso di crash dopo 2h
            with open(out_path, "w") as f:
                summary = aggregate([r for r in all_results if "error" not in r])
                summary["partial"] = (done < total_cells)
                summary["progress"] = f"{done}/{total_cells}"
                summary["elapsed_s"] = time.time() - t_global
                json.dump(summary, f, indent=2)

    summary = aggregate([r for r in all_results if "error" not in r])
    summary["partial"] = False
    summary["progress"] = f"{done}/{total_cells}"
    summary["elapsed_s"] = time.time() - t_global
    summary["seeds"] = args.seeds
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===", file=sys.stderr)
    print(f"Total wall: {summary['elapsed_s']/60:.1f} min", file=sys.stderr)
    for cell in summary["per_cell"]:
        gate = cell["gate_unlocks_v4_blockers"]
        gate_str = ", ".join(f"{k}={v:.2f}" for k, v in gate.items()) if gate else "—"
        print(f"  N={cell['N']:>3} M={cell['M']:>3}: "
              f"Crafter={cell['crafter_score_pct']:>5.2f}% "
              f"ach={cell['mean_achievements']:.1f} "
              f"steps={cell['mean_n_steps']:.0f} "
              f"wall={cell['mean_wall_s']:.0f}s "
              f"BLOCKERS={gate_str}", file=sys.stderr)


if __name__ == "__main__":
    main()
