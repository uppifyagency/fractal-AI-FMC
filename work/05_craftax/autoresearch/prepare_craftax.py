"""prepare_craftax.py — FROZEN evaluation harness for autoresearch-FMC.

DO NOT MODIFY THIS FILE. It defines:
  - The Craftax-Classic-Symbolic-v1 environment (the official benchmark target)
  - The wall-clock-bounded evaluation procedure (analogous to Karpathy's
    5-minute training budget; here 20 minutes per experiment)
  - The Crafter score formula (Hafner 2021, geometric mean of log success rates)
  - The list of 22 achievements (immutable per benchmark definition)
  - The decision-gate metrics: blocker rate (4 v4-blockers), CI95 of mean_ach
  - Diagnostics for fairness checks (max_steps cap fixed at 500, seeds 42..)

The agent (which edits fmc_mutable.py) calls evaluate(impl_module) here to get
its score. This is the SOLE source of truth for ranking experiments.

The wall budget creates a Pareto pressure: faster FMC configs get more seeds
(tighter CI), slower configs get fewer (looser CI). The agent must trade off
depth vs throughput, exactly as in Karpathy's autoresearch.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Force CPU backend; JAX Metal blocks Craftax import
os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# IMMUTABLE CONSTANTS — analogous to prepare.py constants in Karpathy's repo
# ---------------------------------------------------------------------------

ENV_NAME = "Craftax-Classic-Symbolic-v1"
MAX_STEPS = 500                  # episode cap (Hafner standard for Crafter)
WALL_BUDGET_S = 20 * 60          # 20 minutes wall clock per experiment
SEED_START = 42                  # start of seed sequence for fairness
MAX_SEEDS = 60                   # hard cap to prevent infinite seed loops

# 22 official Craftax-Classic achievements (in canonical order, do not reorder)
CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]

# The 4 v4-blockers (achievements that vanilla FMC v4_p02_delta never unlocks
# in 115 episodes from run 007). Tracking these is the decision-gate signal.
BLOCKER_ACHIEVEMENTS = [
    "collect_diamond", "make_iron_pickaxe", "make_iron_sword", "eat_plant",
]

# Historical baseline (run 007 30-seed validation, locked artifact)
BASELINE_CRAFTER = 29.27   # (N=512, M=40) v4_p02_delta config
BASELINE_ACH_MEAN = 12.77
BASELINE_ACH_CI95 = 1.04
BASELINE_LABEL = "v4_p02_delta @ N=512 M=40 (30 seeds, run_007)"


# ---------------------------------------------------------------------------
# Crafter score formula (Hafner 2021, locked)
# ---------------------------------------------------------------------------

def crafter_score(success_rates: dict) -> float:
    """S = exp(mean(log(1 + 100*s_i))) - 1, s_i in [0,1].

    Returns score in 0..100. T7.1 unit test verifies corner cases:
      - all 100% -> 100; all 50% -> 50; all 0% -> 0; 1/22 at 100% -> ~0.23.
    """
    log_terms = []
    for ach in CRAFTAX_CLASSIC_ACHIEVEMENTS:
        s = success_rates.get(ach, 0.0)
        log_terms.append(math.log(1.0 + 100.0 * s))
    mean_log = sum(log_terms) / len(log_terms)
    return math.exp(mean_log) - 1.0


# ---------------------------------------------------------------------------
# Evaluation result struct
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    crafter_score: float
    mean_ach: float
    ach_ci95: float
    n_seeds_completed: int
    n_episodes_attempted: int
    achievement_freq: dict
    blocker_freq: dict
    wall_s: float
    decisions_per_sec: float
    raw_runs: list = field(default_factory=list)

    def vs_baseline_pp(self) -> float:
        """Delta in percentage points vs the run_007 baseline of 29.27."""
        return self.crafter_score - BASELINE_CRAFTER

    def is_significant_improvement(self, min_delta_pp: float = 1.0) -> bool:
        """True if the score is at least min_delta_pp above baseline.

        With 5 seeds the CI95 on Crafter score is roughly +/-3pp; with 30 seeds
        it's about +/-1pp. min_delta_pp=1.0 is a conservative gate for the
        first-stage smoke test under Karpathy's autonomous loop.
        """
        return self.crafter_score >= BASELINE_CRAFTER + min_delta_pp

    def any_blocker_fired(self) -> bool:
        return any(rate > 0 for rate in self.blocker_freq.values())

    def to_dict(self) -> dict:
        return {
            "crafter_score": round(self.crafter_score, 2),
            "mean_ach": round(self.mean_ach, 2),
            "ach_ci95": round(self.ach_ci95, 2),
            "n_seeds_completed": self.n_seeds_completed,
            "n_episodes_attempted": self.n_episodes_attempted,
            "wall_s": round(self.wall_s, 1),
            "decisions_per_sec": round(self.decisions_per_sec, 2),
            "blocker_freq": {k: round(v, 4) for k, v in self.blocker_freq.items()},
            "achievement_freq": {k: round(v, 4) for k, v in self.achievement_freq.items()},
        }


# ---------------------------------------------------------------------------
# The single evaluate() function — entry point for agent's experiments
# ---------------------------------------------------------------------------

def evaluate(impl_module, wall_budget_s: float = WALL_BUDGET_S,
             seed_start: int = SEED_START, max_seeds: int = MAX_SEEDS,
             verbose: bool = True) -> EvalResult:
    """Run impl_module.run_episode(seed, ...) repeatedly until wall_budget_s.

    The agent's `fmc_mutable.py` MUST expose:
      run_episode(seed: int, max_steps: int = 500, env_name: str = ENV_NAME) -> dict
      where dict has keys: 'reward', 'achievements_list', 'achievements_unlocked',
                           'n_steps_decisions', 'wall_time_s', 'decisions_per_sec'

    Returns EvalResult with:
      - crafter_score: Crafter % over completed seeds
      - mean_ach +/- ach_ci95: mean achievements unlocked per episode
      - blocker_freq: rate of unlock for the 4 v4-blockers (decision gate signal)
    """
    if not hasattr(impl_module, "run_episode"):
        raise AttributeError(
            "impl_module must expose run_episode(seed, max_steps, env_name) -> dict"
        )

    raw_runs = []
    t0 = time.time()
    seed = seed_start
    deadline = t0 + wall_budget_s

    while time.time() < deadline and (seed - seed_start) < max_seeds:
        per_run_remaining = deadline - time.time()
        if per_run_remaining < 5.0:
            break  # not enough time left for even a tiny run

        if verbose:
            elapsed = time.time() - t0
            print(f"  seed={seed} (t={elapsed:.0f}s/{wall_budget_s:.0f}s, "
                  f"runs={len(raw_runs)})...", file=sys.stderr, flush=True)

        try:
            r = impl_module.run_episode(
                seed=seed, max_steps=MAX_STEPS, env_name=ENV_NAME,
            )
        except Exception as e:
            if verbose:
                print(f"    seed={seed} CRASH: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
            break  # crash = abort the experiment; agent will see partial results

        if not isinstance(r, dict) or "achievements_list" not in r:
            raise ValueError(
                f"run_episode returned malformed dict: {type(r).__name__} {r!r}"
            )
        r["seed"] = seed
        raw_runs.append(r)
        seed += 1

    wall = time.time() - t0
    n = len(raw_runs)

    if n == 0:
        # Could not complete a single seed within budget
        return EvalResult(
            crafter_score=0.0, mean_ach=0.0, ach_ci95=0.0,
            n_seeds_completed=0, n_episodes_attempted=seed - seed_start,
            achievement_freq={a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS},
            blocker_freq={a: 0.0 for a in BLOCKER_ACHIEVEMENTS},
            wall_s=wall, decisions_per_sec=0.0, raw_runs=raw_runs,
        )

    # Aggregate
    achs = [r["achievements_unlocked"] for r in raw_runs]
    mu = sum(achs) / n
    var = sum((x - mu) ** 2 for x in achs) / (n - 1) if n > 1 else 0.0
    ci95 = 1.96 * math.sqrt(var / n) if n > 1 else 0.0

    freq = {a: 0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for r in raw_runs:
        for a in r["achievements_list"]:
            if a in freq:
                freq[a] += 1
    rates = {a: freq[a] / n for a in freq}

    score = crafter_score(rates)
    blocker_rates = {a: rates[a] for a in BLOCKER_ACHIEVEMENTS}

    # Throughput — useful diagnostic for agent
    total_decisions = sum(r.get("n_steps_decisions", 0) for r in raw_runs)
    dec_per_s = total_decisions / wall if wall > 0 else 0.0

    return EvalResult(
        crafter_score=score, mean_ach=mu, ach_ci95=ci95,
        n_seeds_completed=n, n_episodes_attempted=seed - seed_start,
        achievement_freq=rates, blocker_freq=blocker_rates,
        wall_s=wall, decisions_per_sec=dec_per_s, raw_runs=raw_runs,
    )


# ---------------------------------------------------------------------------
# Sanity check entry point (called from evaluate.py once at session start)
# ---------------------------------------------------------------------------

def sanity_check_environment() -> dict:
    """Verify Craftax + JAX install + that the env can be reset/stepped.

    Returns a dict with diagnostics. Called once before each agent session.
    """
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    params = env.default_params
    n_actions = int(env.action_space(params).n)

    # Single reset+step to verify the pipe works
    rng = jax.random.PRNGKey(0)
    rng, k_reset = jax.random.split(rng)
    obs, state = env.reset(k_reset, params)
    rng, k_step = jax.random.split(rng)
    obs2, state2, reward, done, info = env.step(k_step, state, 0, params)

    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "env_name": ENV_NAME,
        "n_actions": n_actions,
        "obs_shape": tuple(obs.shape) if hasattr(obs, "shape") else None,
        "wall_budget_s": WALL_BUDGET_S,
        "max_steps": MAX_STEPS,
        "n_achievements": len(CRAFTAX_CLASSIC_ACHIEVEMENTS),
        "baseline_crafter": BASELINE_CRAFTER,
        "baseline_label": BASELINE_LABEL,
    }


if __name__ == "__main__":
    info = sanity_check_environment()
    import json
    print(json.dumps(info, indent=2))
