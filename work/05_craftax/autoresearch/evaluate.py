"""evaluate.py — driver that runs ONE experiment.

Imports `fmc_mutable.py` (the agent's edited file), runs `prepare_craftax.evaluate`
on it for the wall budget, prints a summary, and appends a row to results.tsv.

Usage:
    uv run evaluate.py                 # run experiment, write to results.tsv
    uv run evaluate.py --no-tsv        # skip results.tsv write (for sanity test)
    uv run evaluate.py --description "test new shaping"   # tag the row
    uv run evaluate.py --status keep   # keep|discard|crash for results.tsv

The agent typically calls this with the appropriate --description after
editing fmc_mutable.py and committing. Then reads back results.tsv to know
the score. The KEEP/DISCARD/REVERT decision is the agent's, per program_fmc.md.

Wall budget is fixed in prepare_craftax.WALL_BUDGET_S (20 minutes).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force CPU before any JAX import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Resolve dirs relative to this script
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_craftax  # noqa: E402


def short_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nocommit"


def append_tsv(commit: str, score: float, mem_gb: float, status: str,
               description: str, n_seeds: int, mean_ach: float,
               ach_ci95: float, blocker_fired: int):
    """Append a row to results.tsv.

    Schema (extends Karpathy's 5-col with FMC-specific signals):
      commit  crafter_pct  n_seeds  mean_ach  ach_ci95  blocker_fired  status  description
    """
    tsv_path = HERE / "results.tsv"
    if not tsv_path.exists():
        tsv_path.write_text(
            "commit\tcrafter_pct\tn_seeds\tmean_ach\tach_ci95\t"
            "blocker_fired\tstatus\tdescription\n"
        )
    line = (
        f"{commit}\t{score:.4f}\t{n_seeds}\t{mean_ach:.2f}\t"
        f"{ach_ci95:.2f}\t{blocker_fired}\t{status}\t{description}\n"
    )
    with open(tsv_path, "a") as f:
        f.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", type=str, default="(no description)",
                    help="Short text for results.tsv")
    ap.add_argument("--status", choices=["keep", "discard", "crash", "auto"],
                    default="auto",
                    help="Status for results.tsv. 'auto' = decided by gate "
                         "(score > baseline+1pp -> keep else discard)")
    ap.add_argument("--no-tsv", action="store_true",
                    help="Skip writing results.tsv (for sanity test)")
    ap.add_argument("--wall_budget_s", type=int,
                    default=prepare_craftax.WALL_BUDGET_S,
                    help="Override wall budget (default 1200s = 20 min)")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Write the full EvalResult dict to this JSON file")
    args = ap.parse_args()

    print("=" * 70, file=sys.stderr)
    print("autoresearch-FMC: experiment driver", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Sanity check
    info = prepare_craftax.sanity_check_environment()
    print(f"  JAX backend: {info['jax_backend']} {info['jax_devices']}",
          file=sys.stderr)
    print(f"  Env: {info['env_name']} (K={info['n_actions']}, "
          f"obs_dim={info['obs_shape']})", file=sys.stderr)
    print(f"  Wall budget: {args.wall_budget_s}s "
          f"({args.wall_budget_s/60:.1f} min)", file=sys.stderr)
    print(f"  Baseline target: {info['baseline_crafter']}% Crafter "
          f"({info['baseline_label']})", file=sys.stderr)

    # Import agent's mutable file
    print(f"\n  Loading fmc_mutable.py...", file=sys.stderr)
    if "fmc_mutable" in sys.modules:
        importlib.reload(sys.modules["fmc_mutable"])
    import fmc_mutable

    # Run the wall-budget-bounded eval
    print(f"\n  Starting evaluation...", file=sys.stderr)
    t0 = time.time()
    result = prepare_craftax.evaluate(
        impl_module=fmc_mutable,
        wall_budget_s=args.wall_budget_s,
        verbose=True,
    )
    wall_total = time.time() - t0

    # Print summary (Karpathy-style)
    print(f"\n---", file=sys.stderr)
    print(f"crafter_pct:        {result.crafter_score:.4f}", file=sys.stderr)
    print(f"vs_baseline_pp:     {result.vs_baseline_pp():+.2f}", file=sys.stderr)
    print(f"n_seeds_completed:  {result.n_seeds_completed}", file=sys.stderr)
    print(f"mean_ach:           {result.mean_ach:.2f} +/-{result.ach_ci95:.2f}",
          file=sys.stderr)
    print(f"decisions_per_sec:  {result.decisions_per_sec:.2f}", file=sys.stderr)
    print(f"wall_s:             {result.wall_s:.1f}", file=sys.stderr)

    n_blockers_fired = sum(1 for r in result.blocker_freq.values() if r > 0)
    print(f"blocker_fired:      {n_blockers_fired}/4", file=sys.stderr)
    if n_blockers_fired:
        for k, v in result.blocker_freq.items():
            if v > 0:
                print(f"   {k}: {v:.2f}", file=sys.stderr)

    # Auto-status decision
    status = args.status
    if status == "auto":
        if result.n_seeds_completed == 0:
            status = "crash"
        elif result.is_significant_improvement(min_delta_pp=1.0):
            status = "keep"
        else:
            status = "discard"

    print(f"\nstatus (auto):      {status}", file=sys.stderr)
    print(f"description:        {args.description}", file=sys.stderr)

    # Write results.tsv
    if not args.no_tsv:
        commit = short_commit()
        append_tsv(
            commit=commit,
            score=result.crafter_score,
            mem_gb=0.0,  # not tracked for now; could add resource.getrusage
            status=status,
            description=args.description,
            n_seeds=result.n_seeds_completed,
            mean_ach=result.mean_ach,
            ach_ci95=result.ach_ci95,
            blocker_fired=n_blockers_fired,
        )
        print(f"\n  Logged to results.tsv (commit {commit})", file=sys.stderr)

    # Write JSON if asked
    if args.out_json:
        out = result.to_dict()
        out["description"] = args.description
        out["status"] = status
        out["raw_runs"] = result.raw_runs
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {args.out_json}", file=sys.stderr)

    # Exit code: 0 if completed at least one seed, 1 if crashed before any
    sys.exit(0 if result.n_seeds_completed > 0 else 1)


if __name__ == "__main__":
    main()
