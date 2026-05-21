"""evaluate_30seed.py — fixed-N seed driver for Gap 1 (PAPER_HANDOFF).

Calls prepare_craftax.evaluate with max_seeds capping, so the run completes
exactly N seeds (not time-bounded). Writes a per-seed JSON for later stat tests.

Usage:
    python evaluate_30seed.py --out_json results/exp17_30seed.json --n_seeds 30
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_craftax  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--wall_budget_s", type=int, default=7200,
                    help="upper time bound in case per-seed slows; 30 seeds at "
                         "~120s/each fit in 4800s; we budget 7200s safety.")
    ap.add_argument("--description", default="exp17_30seed_validation")
    args = ap.parse_args()

    info = prepare_craftax.sanity_check_environment()
    print(f"[setup] backend={info['jax_backend']} devices={info['jax_devices']}",
          file=sys.stderr, flush=True)
    print(f"[setup] env={info['env_name']} K={info['n_actions']}",
          file=sys.stderr, flush=True)
    print(f"[setup] seeds {args.seed_start}..{args.seed_start + args.n_seeds - 1}"
          f" ({args.n_seeds} total), wall_cap={args.wall_budget_s}s",
          file=sys.stderr, flush=True)

    if "fmc_mutable" in sys.modules:
        importlib.reload(sys.modules["fmc_mutable"])
    import fmc_mutable

    t0 = time.time()
    result = prepare_craftax.evaluate(
        impl_module=fmc_mutable,
        wall_budget_s=args.wall_budget_s,
        seed_start=args.seed_start,
        max_seeds=args.n_seeds,
        verbose=True,
    )
    wall_total = time.time() - t0

    out = result.to_dict()
    out["description"] = args.description
    out["wall_total"] = round(wall_total, 1)
    out["seed_start"] = args.seed_start
    out["n_seeds_target"] = args.n_seeds
    out["raw_runs"] = result.raw_runs

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n=== Gap 1 30-seed validation ===", file=sys.stderr)
    print(f"  crafter_score: {result.crafter_score:.4f}", file=sys.stderr)
    print(f"  n_seeds:       {result.n_seeds_completed}", file=sys.stderr)
    print(f"  mean_ach:      {result.mean_ach:.2f} +/-{result.ach_ci95:.2f}",
          file=sys.stderr)
    print(f"  wall_total:    {wall_total:.1f}s ({wall_total/60:.1f} min)",
          file=sys.stderr)
    print(f"  decisions/sec: {result.decisions_per_sec:.2f}", file=sys.stderr)
    print(f"  saved:         {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
