#!/usr/bin/env python3
"""fractal_reward.py — multiplicative composite reward for FMC walkers.

Implementation of the reward function from:
  Hernández-Cerezo & Duran-Ballester, "Fractal AI: A Fragile Theory of Intelligence" (2020), §2.2.2
  + Pareto Frontiers blog post (2016): single-objective composition

Formula:
  R = R_alive × R_tests × (1 + R_lint) × (1 + R_diff) × (1 + R_goal)

Where:
  R_alive  ∈ {0, 1}  — 0 if walker compile_ok=false, 1 otherwise (HARD constraint)
  R_tests  ∈ [0, 1]  — passed/total ratio, or 1 if tests_run=false (soft default)
  R_lint   ∈ [0, 1]  — 1 / (1 + log(1 + warnings)), or 0.5 if no linter
  R_diff   ∈ [0, 1]  — max(0, 1 - lines_changed / 200), penalize huge diffs
  R_goal   ∈ [0, 1]  — provided by fractal-judge; 0.7 default if not provided

Each factor uses (1 + x) for soft constraints to avoid zero-collapse,
and direct multiplication for hard constraints (R_alive, R_tests).

Then applies relativize() across walkers for fair comparison
(paper §2.2.3) and computes virtual_reward = R^alpha * Dist^beta.

Usage:
  python3 fractal_reward.py --walker-jsons '<json1>' '<json2>' '<json3>' [--alpha 1.0] [--beta 1.0]
  python3 fractal_reward.py --walker-jsons-file walkers.json [--alpha 1.0] [--beta 1.0]

Output: JSON with per-walker scores + winner selection.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from typing import Any


# =============================================================================
# Reward components
# =============================================================================

def r_alive(walker: dict) -> float:
    """Hard constraint: 0 if doesn't compile, 1 otherwise."""
    return 1.0 if walker.get("compile_ok", True) else 0.0


def r_tests(walker: dict) -> float:
    """Test pass ratio. Defaults to 1.0 if no tests were run."""
    if not walker.get("tests_run", False):
        return 1.0  # neutral default
    passed = walker.get("tests_passed") or 0
    total = walker.get("tests_total") or 0
    if total == 0:
        return 1.0
    return passed / total


def r_lint(walker: dict) -> float:
    """Lint cleanliness score. -1 → no linter (neutral 0.5)."""
    warnings = walker.get("lint_warnings", -1)
    if warnings < 0:
        return 0.5
    # Smooth decay: 0 warnings → 1.0, 10 warnings → 0.30, 100 → 0.18
    return 1.0 / (1.0 + math.log1p(warnings))


def r_diff(walker: dict, max_acceptable: int = 200) -> float:
    """Diff size penalty. Smaller diffs preferred (less risk)."""
    added = walker.get("lines_added") or 0
    deleted = walker.get("lines_deleted") or 0
    total = added + deleted
    if total == 0:
        return 0.5  # walker did nothing — neutral, suspicious
    return max(0.0, 1.0 - total / max_acceptable)


def r_goal(walker: dict) -> float:
    """Goal alignment from fractal-judge sub-agent. Default 0.7 if not judged."""
    return walker.get("goal_score", 0.7)


def composite_reward(walker: dict) -> dict:
    """Compute multiplicative composite reward + breakdown."""
    alive = r_alive(walker)
    tests = r_tests(walker)
    lint = r_lint(walker)
    diff = r_diff(walker)
    goal = r_goal(walker)

    # Multiplicative: hard factors direct, soft factors (1 + x)
    R = alive * tests * (1.0 + lint) * (1.0 + diff) * (1.0 + goal)

    return {
        "R": R,
        "R_alive": alive,
        "R_tests": tests,
        "R_lint": lint,
        "R_diff": diff,
        "R_goal": goal,
        "explanation": (
            f"R = {alive:.2f} × {tests:.2f} × {1+lint:.2f} × {1+diff:.2f} × {1+goal:.2f} "
            f"= {R:.3f}"
        ),
    }


# =============================================================================
# Relativize (paper §2.2.3) and Virtual Reward (paper §4.2)
# =============================================================================

def relativize(values: list[float]) -> list[float]:
    """Map any real vector to strictly positive values, preserving order.
    Compresses high outliers via log, expands low outliers via exp.
    """
    n = len(values)
    if n == 0:
        return values
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std == 0 or not math.isfinite(std):
        return [1.0] * n
    out = []
    for v in values:
        z = (v - mean) / std
        if z <= 0:
            out.append(math.exp(max(z, -50)))
        else:
            out.append(1.0 + math.log1p(z))
    return out


def file_overlap_distance(w1: dict, w2: dict) -> float:
    """Distance between two walkers' state changes via files-changed Jaccard.
    Walkers that touched the same files are CLOSE; very different file sets are FAR.
    """
    f1 = set(w1.get("files_changed", []) or [])
    f2 = set(w2.get("files_changed", []) or [])
    if not f1 and not f2:
        return 0.0
    intersection = len(f1 & f2)
    union = len(f1 | f2)
    if union == 0:
        return 0.0
    jaccard = intersection / union
    return 1.0 - jaccard  # distance = 1 - similarity


def lines_distance(w1: dict, w2: dict) -> float:
    """Distance via diff size difference (proxy for semantic distance)."""
    s1 = (w1.get("lines_added") or 0) + (w1.get("lines_deleted") or 0)
    s2 = (w2.get("lines_added") or 0) + (w2.get("lines_deleted") or 0)
    return abs(s1 - s2)


def virtual_reward(
    walkers: list[dict],
    rewards: list[float],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[float]:
    """Compute virtual reward per walker: VR_i = R_i^alpha · Dist_i^beta.
    Uses stochastic O(N) distance: each walker compared to one random partner.
    """
    import random

    n = len(walkers)
    if n < 2:
        return rewards[:]

    # Compute distances: each walker vs random partner
    distances = []
    for i in range(n):
        # Pick partner != i
        candidates = [j for j in range(n) if j != i]
        partner = random.choice(candidates)
        # Combine file overlap + lines distance (normalized)
        d = (
            file_overlap_distance(walkers[i], walkers[partner])
            + 0.01 * lines_distance(walkers[i], walkers[partner])
        )
        distances.append(d)

    R_norm = relativize(rewards)
    D_norm = relativize(distances)

    return [
        (R_norm[i] ** alpha) * (D_norm[i] ** beta)
        for i in range(n)
    ]


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walker-jsons", nargs="+", help="JSON strings, one per walker")
    ap.add_argument("--walker-jsons-file", help="File with JSON array of walkers")
    ap.add_argument("--alpha", type=float, default=1.0, help="Exploitation balance")
    ap.add_argument("--beta", type=float, default=1.0, help="Exploration balance")
    ap.add_argument("--max-diff", type=int, default=200, help="Max acceptable diff lines")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)

    # Parse walkers
    walkers = []
    if args.walker_jsons_file:
        with open(args.walker_jsons_file) as f:
            walkers = json.load(f)
    elif args.walker_jsons:
        for s in args.walker_jsons:
            walkers.append(json.loads(s))
    else:
        sys.stderr.write("error: provide --walker-jsons or --walker-jsons-file\n")
        sys.exit(2)

    if not walkers:
        sys.stderr.write("error: no walkers provided\n")
        sys.exit(2)

    # Compute composite rewards
    breakdowns = [composite_reward(w) for w in walkers]
    rewards = [b["R"] for b in breakdowns]

    # Compute virtual rewards (with distance)
    vrs = virtual_reward(walkers, rewards, alpha=args.alpha, beta=args.beta)

    # Determine winner: max virtual_reward
    winner_idx = max(range(len(vrs)), key=lambda i: vrs[i])

    # Compute confidence: top1 / sum(top3)
    sorted_vrs = sorted(vrs, reverse=True)
    top1 = sorted_vrs[0]
    total = sum(sorted_vrs) or 1.0
    confidence = top1 / total

    # Reward variance among walkers (standard deviation)
    if len(rewards) >= 2:
        reward_variance = statistics.stdev(rewards)
    else:
        reward_variance = 0.0

    out = {
        "winner_idx": winner_idx,
        "winner_label": walkers[winner_idx].get("approach_label", f"walker_{winner_idx}"),
        "confidence": confidence,
        "reward_variance": reward_variance,
        "walkers": [
            {
                "idx": i,
                "approach_label": walkers[i].get("approach_label", f"walker_{i}"),
                "R": rewards[i],
                "VR": vrs[i],
                "breakdown": {k: v for k, v in breakdowns[i].items() if k != "explanation"},
                "explanation": breakdowns[i]["explanation"],
            }
            for i in range(len(walkers))
        ],
        "alpha": args.alpha,
        "beta": args.beta,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
