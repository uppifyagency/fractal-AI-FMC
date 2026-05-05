"""prepare_plasma.py — Karpathy autoresearch eval harness for plasma FMC.

Wraps the M15 published targets + M16 real TCV shot into a single
benchmark suite usable in a tight experiment loop:

    score = -mean truth_err across all (scenario × seed × tick)
    higher score = better tracking

Uses SIM-truth (linear M2 sim with shape_err formula) as fast proxy.
Final validation against freegs oracle is a separate, slower milestone.

Usage (from autoresearch driver):

    import prepare_plasma as P
    result = P.evaluate(impl_module=fmc_mutable_plasma,
                       wall_budget_s=600,
                       verbose=True)
    print(result.score, result.physicality_pct, result.per_scenario)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / "scripts"))

import jax
import jax.numpy as jnp

from plasma_simulator_jax import make_jit_step
from calibrated_sim import build_calibrated_jax_params as build_jax_params
from tcv_published_targets import PUBLISHED_TARGETS

# Wall budget per experiment — generous default for first runs.
WALL_BUDGET_S = 600  # 10 min

# Baseline: established by the first run of vanilla FMC, persisted here.
# Updated when the agent runs the baseline experiment.
BASELINE_SCORE = None  # populated by results.tsv first row

# Eval config
N_SEEDS_PER_SCENARIO = 3
DT = 1e-3
HORIZON = 10
N_WALKERS = 64

# Real TCV-X21 shot 65402 target as a static-target scenario
REAL_TCV_TARGET = np.array([0.889, -0.056, 1.71, 0.12], dtype=np.float32)
REAL_TCV_DURATION_S = 0.5  # 500 ticks


# ---- Score & summary ----

@dataclass
class EvalResult:
    score: float
    mean_err: float
    physicality_pct: float
    n_scenarios_completed: int
    n_seeds_completed: int
    wall_s: float
    decisions_per_sec: float
    per_scenario: dict[str, dict[str, float]] = field(default_factory=dict)
    raw_runs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": float(self.score),
            "mean_err": float(self.mean_err),
            "physicality_pct": float(self.physicality_pct),
            "n_scenarios_completed": int(self.n_scenarios_completed),
            "n_seeds_completed": int(self.n_seeds_completed),
            "wall_s": float(self.wall_s),
            "decisions_per_sec": float(self.decisions_per_sec),
            "per_scenario": self.per_scenario,
        }

    def vs_baseline(self) -> float:
        if BASELINE_SCORE is None:
            return 0.0
        return self.score - BASELINE_SCORE

    def is_significant_improvement(self, min_delta: float = 1.0) -> bool:
        return self.vs_baseline() >= min_delta


# ---- Episode runner ----

def shape_err_np(state: np.ndarray, target: np.ndarray) -> float:
    N = 20
    return float(
        100.0 * (state[N + 3] - target[0]) ** 2
        + 100.0 * (state[N + 4] - target[1]) ** 2
        + 10.0 * (state[N + 5] - target[2]) ** 2
        + 10.0 * (state[N + 6] - target[3]) ** 2
    )


def is_physical(state: np.ndarray, err: float) -> bool:
    """Proxy for physicality: I_p > 50 kA, finite err, err < 1e6."""
    N = 20
    I_p_MA = abs(float(state[N])) / 1e6
    return I_p_MA > 0.05 and np.isfinite(err) and err < 1e6


def run_scenario(
    controller,
    sim_p,
    x0: np.ndarray,
    target_fn,           # callable t_seconds -> np.array(4,)
    duration_s: float,
    dt: float,
) -> dict:
    """Run a single closed-loop episode against a (possibly time-varying)
    target. Returns per-tick metrics."""
    step_jit = make_jit_step(sim_p)
    state = np.asarray(x0).copy()
    P_aux = jnp.asarray(5e5, dtype=jnp.float32)
    gas = jnp.asarray(1e21, dtype=jnp.float32)

    n_ticks = int(np.ceil(duration_s / dt))
    errs = np.zeros(n_ticks, dtype=np.float32)
    phys = np.zeros(n_ticks, dtype=bool)
    decision_us = np.zeros(n_ticks, dtype=np.float32)

    for t in range(n_ticks):
        t_sec = t * dt
        target = np.asarray(target_fn(t_sec), dtype=np.float32)
        t0 = time.perf_counter()
        decision = controller.decide(state, target)
        decision_us[t] = (time.perf_counter() - t0) * 1e6

        V = jnp.asarray(decision["V_coils"], dtype=jnp.float32)
        new_state = step_jit(jnp.asarray(state), V, P_aux, gas, dt)
        state = np.asarray(new_state)

        errs[t] = shape_err_np(state, target)
        phys[t] = is_physical(state, errs[t])

    return {
        "errs": errs,
        "physical": phys,
        "decision_us": decision_us,
        "n_ticks": n_ticks,
        "mean_err": float(errs.mean()),
        "max_err": float(errs.max()),
        "min_err": float(errs.min()),
        "physicality_pct": float(phys.mean() * 100.0),
        "decision_us_p50": float(np.median(decision_us)),
    }


# ---- Main eval ----

def _build_real_target_fn(target_4: np.ndarray):
    def fn(t_sec):
        return target_4
    return fn


def evaluate(
    impl_module: Any,
    wall_budget_s: int = WALL_BUDGET_S,
    n_seeds: int = N_SEEDS_PER_SCENARIO,
    dt: float = DT,
    verbose: bool = True,
) -> EvalResult:
    """Run the full benchmark on the controller from impl_module.

    Args:
      impl_module: must expose `make_controller(sim_p, seed=0)` returning
                   a controller with `.decide(state, target_4) -> dict`
                   containing at least `"V_coils"`.
      wall_budget_s: hard cap on wall time.
      n_seeds: number of seeds per scenario.
      dt: integration step.
    """
    t0 = time.time()
    sim_p, x0 = build_jax_params()

    # Build the scenario suite (M15 published × 6 + M16 real × 1 = 7 scenarios)
    scenarios: list[tuple[str, callable, float]] = []
    for s in PUBLISHED_TARGETS:
        scenarios.append((s.name, s.target_at, s.duration_s))
    scenarios.append((
        "M16_real_TCV_65402",
        _build_real_target_fn(REAL_TCV_TARGET),
        REAL_TCV_DURATION_S,
    ))

    per_scenario: dict[str, dict[str, float]] = {}
    raw_runs: list[dict] = []
    n_completed_seeds_total = 0

    for sname, target_fn, dur_s in scenarios:
        if time.time() - t0 > wall_budget_s:
            if verbose:
                print(f"  [budget exceeded — skip scenario {sname}]",
                      file=sys.stderr)
            break

        if verbose:
            print(f"  scenario={sname}  duration={dur_s:.2f}s",
                  file=sys.stderr, flush=True)

        seed_runs = []
        for seed in range(n_seeds):
            if time.time() - t0 > wall_budget_s:
                break
            ctrl = impl_module.make_controller(sim_p, seed=seed)
            tr = run_scenario(ctrl, sim_p, x0, target_fn, dur_s, dt)
            tr["scenario"] = sname
            tr["seed"] = seed
            seed_runs.append(tr)
            raw_runs.append({
                "scenario": sname, "seed": seed,
                "mean_err": tr["mean_err"],
                "physicality_pct": tr["physicality_pct"],
                "n_ticks": tr["n_ticks"],
                "decision_us_p50": tr["decision_us_p50"],
            })
            n_completed_seeds_total += 1
            if verbose:
                print(
                    f"    seed={seed}: mean_err={tr['mean_err']:7.2f}  "
                    f"phys={tr['physicality_pct']:5.1f}%  "
                    f"dec={tr['decision_us_p50']:.0f}µs",
                    file=sys.stderr, flush=True,
                )

        if seed_runs:
            mean_err_arr = np.array([r["mean_err"] for r in seed_runs])
            phys_arr = np.array([r["physicality_pct"] for r in seed_runs])
            per_scenario[sname] = {
                "mean_err_mean": float(mean_err_arr.mean()),
                "mean_err_std": float(mean_err_arr.std(ddof=1))
                                if len(mean_err_arr) > 1 else 0.0,
                "physicality_pct_mean": float(phys_arr.mean()),
                "n_seeds": len(seed_runs),
                "duration_s": dur_s,
            }

    # Aggregate score: average of (per-scenario mean_err) — equally
    # weighted across scenarios. Inverted so improvements are positive.
    if not per_scenario:
        return EvalResult(
            score=-1e6, mean_err=1e6, physicality_pct=0.0,
            n_scenarios_completed=0, n_seeds_completed=0,
            wall_s=time.time() - t0, decisions_per_sec=0.0,
            per_scenario={}, raw_runs=raw_runs,
        )

    mean_err_overall = float(np.mean([
        v["mean_err_mean"] for v in per_scenario.values()
    ]))
    phys_overall = float(np.mean([
        v["physicality_pct_mean"] for v in per_scenario.values()
    ]))

    # Score = -mean_err_overall + bonus for full physicality
    # bonus puts ~1pp physicality drop on par with ~0.1 err change.
    score = -mean_err_overall + 0.1 * phys_overall

    total_decisions = sum(r["n_ticks"] for r in raw_runs)
    wall_total = time.time() - t0

    return EvalResult(
        score=score,
        mean_err=mean_err_overall,
        physicality_pct=phys_overall,
        n_scenarios_completed=len(per_scenario),
        n_seeds_completed=n_completed_seeds_total,
        wall_s=wall_total,
        decisions_per_sec=total_decisions / max(wall_total, 1e-3),
        per_scenario=per_scenario,
        raw_runs=raw_runs,
    )


def sanity_check_environment() -> dict:
    return {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "n_published_targets": len(PUBLISHED_TARGETS),
        "n_real_targets": 1,
        "total_targets": len(PUBLISHED_TARGETS) + 1,
        "wall_budget_s": WALL_BUDGET_S,
        "n_seeds_default": N_SEEDS_PER_SCENARIO,
    }


if __name__ == "__main__":
    info = sanity_check_environment()
    print("plasma autoresearch eval harness")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
