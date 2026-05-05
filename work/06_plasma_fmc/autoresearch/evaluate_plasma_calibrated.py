"""evaluate_plasma.py — Karpathy autoresearch driver for plasma FMC.

Imports fmc_mutable_plasma.py (the agent's edited file), runs
prepare_plasma.evaluate, prints summary, appends row to results.tsv.

Usage:
    python evaluate_plasma.py
    python evaluate_plasma.py --description "exp01: N_walkers 64 -> 128"
    python evaluate_plasma.py --status keep
    python evaluate_plasma.py --wall_budget_s 300
    python evaluate_plasma.py --out_json results/expNN.json

Auto-status (default): keep if score > BASELINE + 1.0 (Δ in score units),
                       discard otherwise. crash if 0 seeds completed.

Score (defined in prepare_plasma.evaluate):
    score = -mean_err + 0.1 * physicality_pct
    higher = better
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

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_plasma_calibrated as prepare_plasma  # noqa: E402


def short_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "nocommit"


def append_tsv(commit: str, score: float, mean_err: float,
               phys: float, n_scen: int, n_seeds: int,
               status: str, description: str):
    tsv_path = HERE / "results_calibrated.tsv"
    if not tsv_path.exists():
        tsv_path.write_text(
            "commit\tscore\tmean_err\tphys_pct\t"
            "n_scenarios\tn_seeds\tstatus\tdescription\n"
        )
    line = (
        f"{commit}\t{score:.4f}\t{mean_err:.4f}\t{phys:.2f}\t"
        f"{n_scen}\t{n_seeds}\t{status}\t{description}\n"
    )
    with open(tsv_path, "a") as f:
        f.write(line)


def read_baseline_score(tsv_path: Path) -> float | None:
    """Return the score of the row with status=baseline, if any."""
    if not tsv_path.exists():
        return None
    with open(tsv_path) as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 7 and parts[6] == "baseline":
                try:
                    return float(parts[1])
                except ValueError:
                    pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--description", type=str, default="(no description)")
    ap.add_argument("--status",
                    choices=["keep", "discard", "crash", "baseline", "auto"],
                    default="auto")
    ap.add_argument("--no-tsv", action="store_true")
    ap.add_argument("--wall_budget_s", type=int,
                    default=prepare_plasma.WALL_BUDGET_S)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70, file=sys.stderr)
    print("autoresearch-plasma: experiment driver", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    info = prepare_plasma.sanity_check_environment()
    for k, v in info.items():
        print(f"  {k}: {v}", file=sys.stderr)

    print(f"\n  Loading fmc_mutable_plasma.py...", file=sys.stderr)
    if "fmc_mutable_plasma" in sys.modules:
        importlib.reload(sys.modules["fmc_mutable_plasma"])
    import fmc_mutable_plasma

    print(f"\n  Starting evaluation...", file=sys.stderr)
    t0 = time.time()
    result = prepare_plasma.evaluate(
        impl_module=fmc_mutable_plasma,
        wall_budget_s=args.wall_budget_s,
        verbose=True,
    )
    wall_total = time.time() - t0

    print(f"\n---", file=sys.stderr)
    print(f"score:               {result.score:.4f}", file=sys.stderr)
    print(f"mean_err:            {result.mean_err:.4f}", file=sys.stderr)
    print(f"physicality_pct:     {result.physicality_pct:.2f}", file=sys.stderr)
    print(f"n_scenarios:         {result.n_scenarios_completed}/"
          f"{info['total_targets']}", file=sys.stderr)
    print(f"n_seeds:             {result.n_seeds_completed}", file=sys.stderr)
    print(f"wall_s:              {result.wall_s:.1f}", file=sys.stderr)
    print(f"decisions_per_sec:   {result.decisions_per_sec:.1f}",
          file=sys.stderr)

    print(f"\n  per scenario:", file=sys.stderr)
    for sname, ps in result.per_scenario.items():
        print(
            f"    {sname:30s}  err={ps['mean_err_mean']:7.2f}±"
            f"{ps['mean_err_std']:5.2f}  phys={ps['physicality_pct_mean']:5.1f}%",
            file=sys.stderr,
        )

    # Auto-status decision
    status = args.status
    tsv_path = HERE / "results_calibrated.tsv"
    baseline_score = read_baseline_score(tsv_path)
    if status == "auto":
        if result.n_seeds_completed == 0:
            status = "crash"
        elif baseline_score is None:
            # First experiment ever — the baseline run itself.
            status = "baseline"
        elif result.score >= baseline_score + 1.0:
            status = "keep"
        else:
            status = "discard"

    print(f"\nstatus (auto):       {status}", file=sys.stderr)
    if baseline_score is not None:
        print(f"baseline_score:      {baseline_score:.4f}", file=sys.stderr)
        print(f"vs_baseline:         "
              f"{result.score - baseline_score:+.4f}", file=sys.stderr)
    print(f"description:         {args.description}", file=sys.stderr)

    if not args.no_tsv:
        commit = short_commit()
        append_tsv(
            commit=commit,
            score=result.score,
            mean_err=result.mean_err,
            phys=result.physicality_pct,
            n_scen=result.n_scenarios_completed,
            n_seeds=result.n_seeds_completed,
            status=status,
            description=args.description,
        )
        print(f"\n  Logged to results.tsv (commit {commit})",
              file=sys.stderr)

    if args.out_json:
        out = result.to_dict()
        out["description"] = args.description
        out["status"] = status
        out["raw_runs"] = result.raw_runs
        out["baseline_score"] = baseline_score
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {args.out_json}", file=sys.stderr)

    sys.exit(0 if result.n_seeds_completed > 0 else 1)


if __name__ == "__main__":
    main()
