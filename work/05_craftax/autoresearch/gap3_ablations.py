"""gap3_ablations.py — Gap 3 (PAPER_HANDOFF) leave-one-out ablation runner.

Generates 5 mutated copies of fmc_mutable.py, each removing one tier
component, runs a 30-seed evaluation per ablation, and writes results to
results/gap3_<L>.json.

Mutations:
  L1 — remove iron-tier inv (coal/iron/diamond + iron pickaxes/swords -> 1.0)
       AND revert diamond proximity 64 -> 16
  L2 — remove stone-tier inv (stone/stone_pickaxe/stone_sword -> 1.0)
  L3 — remove wood-tier inv (wood/wood_pickaxe/wood_sword -> 1.0)
  L4 — remove iron-tier ach push (iron_pickaxe 200 -> 150,
       iron_sword 200 -> 150)
  L5 — remove gateway-tier ach push (stone_pickaxe 80 -> 50,
       collect_iron 120 -> 80, collect_coal 80 -> 50,
       place_furnace 80 -> 50)

Each mutation is applied to a *temporary copy* `fmc_mutable_L<n>.py`,
loaded via importlib for the ablation run. The original fmc_mutable.py
is untouched.

Usage:
    python gap3_ablations.py            # run all five
    python gap3_ablations.py --only L1  # just L1
"""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_craftax  # noqa: E402

ABLATIONS = {
    "L1": {
        "label": "minus iron-tier inv (coal/iron/diamond + iron tools -> 1.0)",
        "subs": [
            ("inv.coal.astype(jnp.float32) * 8.0",      "inv.coal.astype(jnp.float32) * 1.0"),
            ("inv.iron.astype(jnp.float32) * 16.0",     "inv.iron.astype(jnp.float32) * 1.0"),
            ("inv.diamond.astype(jnp.float32) * 64.0",  "inv.diamond.astype(jnp.float32) * 1.0"),
            ("inv.iron_pickaxe.astype(jnp.float32) * 24.0",
             "inv.iron_pickaxe.astype(jnp.float32) * 1.0"),
            ("inv.iron_sword.astype(jnp.float32) * 24.0",
             "inv.iron_sword.astype(jnp.float32) * 1.0"),
            ("+ 64.0 * has_iron_p * need_diamond * jnp.exp(-d_diamond / sigma)  # exp19: 16 -> 64",
             "+ 16.0 * has_iron_p * need_diamond * jnp.exp(-d_diamond / sigma)  # L1 ablation: revert to 16"),
        ],
    },
    "L2": {
        "label": "minus stone-tier inv (stone + stone tools -> 1.0)",
        "subs": [
            ("inv.stone.astype(jnp.float32) * 4.0",     "inv.stone.astype(jnp.float32) * 1.0"),
            ("inv.stone_pickaxe.astype(jnp.float32) * 12.0",
             "inv.stone_pickaxe.astype(jnp.float32) * 1.0"),
            ("inv.stone_sword.astype(jnp.float32) * 12.0",
             "inv.stone_sword.astype(jnp.float32) * 1.0"),
        ],
    },
    "L3": {
        "label": "minus wood-tier inv (wood + wood tools -> 1.0)",
        "subs": [
            ("inv.wood.astype(jnp.float32) * 2.0",      "inv.wood.astype(jnp.float32) * 1.0"),
            ("inv.wood_pickaxe.astype(jnp.float32) * 6.0",
             "inv.wood_pickaxe.astype(jnp.float32) * 1.0"),
            ("inv.wood_sword.astype(jnp.float32) * 6.0",
             "inv.wood_sword.astype(jnp.float32) * 1.0"),
        ],
    },
    "L4": {
        "label": "minus iron-tier ach push (iron_pickaxe/sword 200 -> 150)",
        "subs": [
            ("    200.0,                    # 7: MAKE_IRON_PICKAXE *** BLOCKER (exp17 final) ***",
             "    150.0,                    # 7: MAKE_IRON_PICKAXE -- L4 ablation 200 -> 150"),
            ("    200.0,                    # 10: MAKE_IRON_SWORD (exp16)",
             "    150.0,                    # 10: MAKE_IRON_SWORD -- L4 ablation 200 -> 150"),
        ],
    },
    "L5": {
        "label": "minus gateway-tier ach push",
        "subs": [
            ("    80.0,                     # 6: MAKE_STONE_PICKAXE *** exp17: 50 -> 80 ***",
             "    50.0,                     # 6: MAKE_STONE_PICKAXE -- L5 ablation 80 -> 50"),
            ("    120.0,                    # 17: COLLECT_IRON *** exp17: 80 -> 120 ***",
             "    80.0,                     # 17: COLLECT_IRON -- L5 ablation 120 -> 80"),
            ("    80.0,                     # 18: COLLECT_COAL *** exp17: 50 -> 80 ***",
             "    50.0,                     # 18: COLLECT_COAL -- L5 ablation 80 -> 50"),
            ("    80.0,                     # 19: PLACE_FURNACE *** exp17: 50 -> 80 ***",
             "    50.0,                     # 19: PLACE_FURNACE -- L5 ablation 80 -> 50"),
        ],
    },
}


def make_mutated_module(level: str, src: str) -> str:
    """Apply substitutions for the given ablation; return mutated source."""
    abl = ABLATIONS[level]
    out = src
    for old, new in abl["subs"]:
        if old not in out:
            raise ValueError(
                f"[{level}] substitution miss: {old!r} not found in source. "
                f"Did the source diverge from exp17?"
            )
        out = out.replace(old, new)
    return out


def import_from_path(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(level: str, n_seeds: int, wall_budget_s: int) -> dict:
    src = (HERE / "fmc_mutable.py").read_text()
    mutated_src = make_mutated_module(level, src)

    mut_path = HERE / f"fmc_mutable_{level}.py"
    mut_path.write_text(mutated_src)

    print(f"\n[gap3:{level}] {ABLATIONS[level]['label']}", file=sys.stderr)
    print(f"[gap3:{level}] mutated module written: {mut_path}", file=sys.stderr)
    print(f"[gap3:{level}] running {n_seeds} seeds, wall_cap={wall_budget_s}s",
          file=sys.stderr)

    mod_name = f"fmc_mutable_{level}"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = import_from_path(mut_path, mod_name)

    t0 = time.time()
    result = prepare_craftax.evaluate(
        impl_module=mod,
        wall_budget_s=wall_budget_s,
        seed_start=42,
        max_seeds=n_seeds,
        verbose=True,
    )
    wall = time.time() - t0

    out = result.to_dict()
    out["ablation"] = level
    out["ablation_label"] = ABLATIONS[level]["label"]
    out["wall_total"] = round(wall, 1)
    out["raw_runs"] = result.raw_runs

    out_path = HERE / "results" / f"gap3_{level}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print(f"[gap3:{level}] crafter={result.crafter_score:.4f} "
          f"n={result.n_seeds_completed} wall={wall/60:.1f}min "
          f"saved={out_path}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=list(ABLATIONS),
                    help="run only a single ablation")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--wall_budget_s", type=int, default=7200)
    args = ap.parse_args()

    levels = [args.only] if args.only else list(ABLATIONS)

    results = {}
    for L in levels:
        try:
            results[L] = run_one(L, args.n_seeds, args.wall_budget_s)
        except Exception as e:
            print(f"[gap3:{L}] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            results[L] = {"error": f"{type(e).__name__}: {e}"}

    summary_path = HERE / "results" / "gap3_summary.json"
    summary_path.write_text(json.dumps({
        L: {k: v for k, v in r.items() if k != "raw_runs"}
        for L, r in results.items()
    }, indent=2))
    print(f"\n[gap3] summary saved to {summary_path}", file=sys.stderr)
    print("\n=== Gap 3 leave-one-out summary ===", file=sys.stderr)
    print(f"  exp17 baseline (30 seeds): TBD from results/exp17_30seed.json",
          file=sys.stderr)
    for L, r in results.items():
        if "error" in r:
            print(f"  {L}: ERROR {r['error']}", file=sys.stderr)
        else:
            print(f"  {L}: crafter={r['crafter_score']:.2f}% "
                  f"(Δ vs baseline TBD)", file=sys.stderr)


if __name__ == "__main__":
    main()
