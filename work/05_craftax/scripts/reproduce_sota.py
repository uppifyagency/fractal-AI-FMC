"""reproduce_sota.py — Standalone reproduction del nuovo SOTA zero-training su Craftax-Classic.

Riproduce il risultato 30-seed di run 007:
  - (N=512, M=40) -> 29.27% Crafter score (mean_ach 12.77 +/-1.04 CI95)
  - (N=128, M=20) -> 24.61% Crafter score (mean_ach 11.27 +/-1.18 CI95)

Entrambi sopra il baseline tabular SOTA Curious Replay (19.4%, 1M training).

Usage:
    pip install craftax==1.5.0 jax==0.10.0 jaxlib==0.10.0
    python reproduce_sota.py --config sota          # (N=512, M=40), 30 seeds
    python reproduce_sota.py --config fast          # (N=128, M=20), 30 seeds (5x faster)
    python reproduce_sota.py --config sota --seeds 5    # quick smoke test (5 seeds)
    python reproduce_sota.py --config sota --seeds 30   # full repro (~63 min CPU)

Hardware:
    - CPU sufficient (testato MacBook Apple M1 Pro, ~125s/seed at N=512,M=40)
    - JAX GPU acceleration NON necessaria (env stepping vmap CPU e' gia' efficiente)

Tested on:
    - Python 3.11.7
    - JAX 0.10.0 + jaxlib 0.10.0 (CPU backend)
    - craftax 1.5.0
    - macOS Apple M1 Pro 16GB
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

# Import della implementazione FMC v4 verificata 15/15 unit test theory-code parity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fmc_craftax_v4 import FMCConfig, run_episode  # noqa: E402


# Crafter score: Hafner 2021 = exp(mean(log(1 + 100*s_i))) - 1, s_i in [0,1]
CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]


def crafter_score(success_rates: dict[str, float]) -> float:
    log_terms = [math.log(1.0 + 100.0 * success_rates.get(a, 0.0))
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0


CONFIGS = {
    "sota": {
        "name": "FMC v4 N=512 M=40 (SOTA zero-training, 29.27%)",
        "n_walkers": 512, "time_horizon": 40,
        "expected_crafter": 29.27, "expected_ach_mean": 12.77,
        "expected_ach_ci95": 1.04,
    },
    "fast": {
        "name": "FMC v4 N=128 M=20 (5x faster, 24.61%)",
        "n_walkers": 128, "time_horizon": 20,
        "expected_crafter": 24.61, "expected_ach_mean": 11.27,
        "expected_ach_ci95": 1.18,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS.keys()), default="sota",
                    help="'sota' = (N=512,M=40), 'fast' = (N=128,M=20)")
    ap.add_argument("--seeds", type=int, default=30,
                    help="Number of seeds (default 30 = full repro)")
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--out", type=str, default=None,
                    help="Write JSON results to this file")
    args = ap.parse_args()

    spec = CONFIGS[args.config]
    print(f"Reproducing: {spec['name']}", file=sys.stderr)
    print(f"Seeds: {args.seed_start}..{args.seed_start + args.seeds - 1}",
          file=sys.stderr)
    print(f"JAX backend: {jax.default_backend()} on {jax.devices()}", file=sys.stderr)
    print(file=sys.stderr)

    cfg = FMCConfig(
        n_walkers=spec["n_walkers"], time_horizon=spec["time_horizon"],
        alpha=1.0, beta=1.0, action_repeat=1,
        intrinsic_inv_alpha=0.5,           # v4_p02_delta best historical config
        proximity_alpha=0.2,
        proximity_sigma=10.0,
        proximity_mode="delta",
    )

    results = []
    t0 = time.time()
    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds)):
        ts = time.time()
        r = run_episode(seed=seed, cfg=cfg, max_steps=args.max_steps,
                        verbose=False, env_name="Craftax-Classic-Symbolic-v1")
        dt = time.time() - ts
        results.append(r)
        cum_min = (time.time() - t0) / 60
        print(f"[{i+1}/{args.seeds}] seed={seed} ach={r['achievements_unlocked']}/22 "
              f"reward={r['reward']:.1f} steps={r['n_steps_decisions']} "
              f"wall={dt:.1f}s (cumul {cum_min:.1f} min)", file=sys.stderr, flush=True)

    # Aggregate
    n = len(results)
    achs = [r["achievements_unlocked"] for r in results]
    mu = sum(achs) / n
    if n > 1:
        var = sum((x - mu) ** 2 for x in achs) / (n - 1)
        ci95 = 1.96 * math.sqrt(var / n)
    else:
        ci95 = 0.0

    freq = {a: 0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for r in results:
        for a in r["achievements_list"]:
            if a in freq:
                freq[a] += 1
    rates = {a: freq[a] / n for a in freq}
    score = crafter_score(rates)

    elapsed = (time.time() - t0) / 60

    print(f"\n=== REPRODUCTION RESULT ===", file=sys.stderr)
    print(f"  Crafter score:       {score:.2f}%", file=sys.stderr)
    print(f"  Mean achievements:   {mu:.2f} +/-{ci95:.2f} CI95", file=sys.stderr)
    print(f"  Wall time:           {elapsed:.1f} min CPU", file=sys.stderr)
    print(f"\n  Expected (paper):    {spec['expected_crafter']:.2f}%", file=sys.stderr)
    print(f"                       mean_ach {spec['expected_ach_mean']:.2f} "
          f"+/-{spec['expected_ach_ci95']:.2f}", file=sys.stderr)

    # Sanity check: score within 2x CI95 of expected
    expected_score = spec["expected_crafter"]
    score_ci_proxy = ci95 * 2  # conservative
    if abs(score - expected_score) <= score_ci_proxy + 1.0:
        print(f"\n  [PASS] Result within statistical noise of expected", file=sys.stderr)
    else:
        print(f"\n  [WARN] Result {abs(score-expected_score):.2f}pp off expected — "
              f"check JAX/Craftax versions", file=sys.stderr)

    print(f"\n  Decision gate (4 v4-blockers): ", end="", file=sys.stderr)
    blockers = ["collect_diamond", "make_iron_pickaxe", "make_iron_sword", "eat_plant"]
    fired = [(a, rates[a]) for a in blockers if rates[a] > 0]
    if fired:
        print(f"*** SOMETHING FIRED ***", file=sys.stderr)
        for a, r in fired:
            print(f"    {a}: {r:.2f}", file=sys.stderr)
    else:
        print(f"NONE (consistent with run_007 0/115 finding)", file=sys.stderr)

    if args.out:
        out = {
            "config": args.config,
            "n_walkers": cfg.n_walkers,
            "time_horizon": cfg.time_horizon,
            "n_seeds": n,
            "seeds": list(range(args.seed_start, args.seed_start + n)),
            "crafter_score_pct": round(score, 2),
            "mean_achievements": round(mu, 2),
            "mean_achievements_ci95": round(ci95, 2),
            "achievement_frequencies": {k: round(v, 4) for k, v in rates.items()},
            "elapsed_min": round(elapsed, 1),
            "raw": results,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
