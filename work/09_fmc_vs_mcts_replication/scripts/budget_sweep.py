"""Budget sweep: FMC vs MCTS-UCT at varying samples-per-action B.

Outputs JSONL of one record per (env, algo, B, seed) cell. Aggregation
into mean/std/CI95 lives in notebooks/analysis.ipynb (or any tool of
your choice).

This is the "in-session smoke version" of protocol P0. The full P0 run
on Atari requires plangym + cluster GPU; see ../../docs/bibliography/
protocols/P0_fmc_vs_mcts_protocol.md for the full spec.

Usage
-----
python -m scripts.budget_sweep --env cartpole --seeds 5 \
    --budgets 30 100 300 1000 --out runs/cartpole_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_episode import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="cartpole")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", type=int, nargs="+", default=[30, 100, 300, 1000])
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # truncate

    for B in args.budgets:
        for seed in range(args.seeds):
            for algo in ("fmc", "mcts"):
                rec = run(algo, args.env, B, seed, args.max_steps)
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps(rec))


if __name__ == "__main__":
    main()
