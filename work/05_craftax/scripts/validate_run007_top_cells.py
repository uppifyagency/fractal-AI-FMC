"""validate_run007_top_cells.py — 30-seed validation dei due Pareto-optimal cells.

Top cells dal run 007 full grid (5 seed each):
  - (N=128, M=20) -> 27.23%
  - (N=512, M=40) -> 28.61%

Aggiungiamo seed 47-71 (25 nuovi seed) sopra i 5 esistenti (42-46) per
arrivare a 30 seed/cella. Questo da' un CI95 stretto, paragonabile al
30-seed baseline storico v4_p02_delta a 21.87%.

Output: results/run007_validation_top_cells.json — singolo file con i 50 raw run
nuovi (25 seed x 2 cell). Da combinare poi col run007_full_grid_5seed.json
per ottenere 30-seed total.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmc_craftax_v4 import FMCConfig, run_episode  # noqa: E402


TOP_CELLS = [(128, 20), (512, 40)]
NEW_SEEDS = list(range(47, 72))  # 25 seed: 47-71


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    ap.add_argument("--cells", type=str, nargs="+", default=None,
                    help="Lista 'N,M' override; default = (128,20) e (512,40)")
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=str,
                    default="work/05_craftax/results/run007_validation_top_cells.json")
    args = ap.parse_args()

    if args.cells:
        cells = [tuple(int(x) for x in s.split(",")) for s in args.cells]
    else:
        cells = TOP_CELLS

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {jax.default_backend()} {jax.devices()}", file=sys.stderr)
    print(f"Cells: {cells}", file=sys.stderr)
    print(f"Seeds (new): {args.seeds}", file=sys.stderr)
    print(f"Total runs: {len(cells)*len(args.seeds)}", file=sys.stderr)

    all_results = []
    total = len(cells) * len(args.seeds)
    done = 0
    t_global = time.time()

    for (N, M) in cells:
        for seed in args.seeds:
            done += 1
            print(f"[{done}/{total}] N={N} M={M} seed={seed} ...",
                  file=sys.stderr, flush=True)
            cfg = FMCConfig(
                n_walkers=N, time_horizon=M, alpha=1.0, beta=1.0,
                action_repeat=1, intrinsic_inv_alpha=0.5,
                proximity_alpha=0.2, proximity_sigma=10.0, proximity_mode="delta",
            )
            try:
                t0 = time.time()
                r = run_episode(seed=seed, cfg=cfg, max_steps=args.max_steps,
                                verbose=False, env_name="Craftax-Classic-Symbolic-v1")
                r["wall_total_s"] = time.time() - t0
                r["N"] = N
                r["M"] = M
                fired = [a for a in r["achievements_list"]
                         if a in {"collect_diamond", "make_iron_pickaxe",
                                  "make_iron_sword", "eat_plant"}]
                if fired:
                    print(f"  *** BLOCKER FIRED: {fired} ***",
                          file=sys.stderr, flush=True)
                print(f"  -> wall={r['wall_total_s']:.1f}s reward={r['reward']:.1f} "
                      f"ach={r['achievements_unlocked']} steps={r['n_steps_decisions']}",
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

            with open(out_path, "w") as f:
                json.dump({
                    "raw_runs": all_results,
                    "elapsed_s": time.time() - t_global,
                    "progress": f"{done}/{total}",
                    "partial": (done < total),
                }, f, indent=2)

    elapsed = time.time() - t_global
    print(f"\nTotal wall: {elapsed/60:.1f} min", file=sys.stderr)
    print(f"Saved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
