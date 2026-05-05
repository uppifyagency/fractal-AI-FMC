"""M18 closed-loop comparison: baseline FMC vs hierarchical FMC.

Two scenarios:
  S1 (near-target, M16-like): target = real TCV-X21 shot 65402
       (0.889, -0.056, 1.71, 0.12). Default x0 starts at err≈0.65.
       Achievement bonus is mostly latched at root → expect parity
       (no harm, no help — verifies the augmentation doesn't break
        existing-good performance).

  S2 (far-target, large-displacement): target shifted to push err > 20
       at root, exercising tier 1-3 of achievement chain. Tests
       whether ach-fire bonus discovers the tracking trajectory faster.

Three policies:
  P_vanilla:      original FMCPlasmaJaxController (no ach-fire)
  P_hier_default: hierarchical FMC, weights [10, 30, 60, 100, 200, 300]
  P_hier_amplified: hierarchical FMC, weights ×1.4 (Conjecture D sweet spot)

Metric: mean shape err over 30 ticks × 50 ms = 1.5 s episode, n_seeds=5.
Reports per-policy mean, std, and physicality (% ticks with valid I_p).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

# Make scripts/ importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

from plasma_simulator_jax import build_jax_params, make_jit_step
from fmc_plasma_jax import FMCPlasmaJaxController
from fmc_plasma_hierarchical import (
    FMCPlasmaHierarchicalController,
    DEFAULT_TIER_WEIGHTS,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DT = 1e-3
HORIZON = 10
N_WALKERS = 64
N_TICKS = 30
N_SEEDS = 5

# Two scenarios
TARGET_S1 = np.array([0.889, -0.056, 1.71, 0.12], dtype=np.float32)
TARGET_S2 = np.array([1.00, 0.10, 2.20, -0.45], dtype=np.float32)


def shape_err_np(state, target):
    """Match shape_err_jax convention with weights [100, 100, 10, 10]."""
    N = 20
    return float(
        100.0 * (state[N + 3] - target[0]) ** 2
        + 100.0 * (state[N + 4] - target[1]) ** 2
        + 10.0 * (state[N + 5] - target[2]) ** 2
        + 10.0 * (state[N + 6] - target[3]) ** 2
    )


def run_episode(controller, sim_p, x0, target, n_ticks, dt):
    """Run a closed-loop episode; return per-tick metrics."""
    step_jit = make_jit_step(sim_p)
    state = np.asarray(x0).copy()
    P_aux = jnp.asarray(5e5, dtype=jnp.float32)
    gas = jnp.asarray(1e21, dtype=jnp.float32)
    errs = []
    I_p_history = []
    physical = []
    walkers_alive_history = []
    ach_mean_history = []
    decision_times = []

    for t in range(n_ticks):
        t0 = time.perf_counter()
        decision = controller.decide(state, target)
        decision_times.append(time.perf_counter() - t0)

        V = jnp.asarray(decision["V_coils"], dtype=jnp.float32)
        new_state = step_jit(jnp.asarray(state), V, P_aux, gas, dt)
        state = np.asarray(new_state)

        err = shape_err_np(state, target)
        errs.append(err)
        I_p_history.append(float(state[20]) / 1e6)  # MA
        # physicality proxy: I_p > 50 kA AND not NaN AND err finite
        is_phys = (
            abs(I_p_history[-1]) > 0.05
            and np.isfinite(err)
            and err < 1e6
        )
        physical.append(bool(is_phys))
        walkers_alive_history.append(int(decision["walkers_alive"]))
        ach_mean_history.append(float(decision.get("ach_mean", 0.0)))

    return {
        "errs": errs,
        "I_p_MA": I_p_history,
        "physical": physical,
        "walkers_alive": walkers_alive_history,
        "ach_mean": ach_mean_history,
        "decision_times_s": decision_times,
    }


def summarize(traj):
    """Compute summary metrics from a single episode trajectory."""
    errs = np.array(traj["errs"])
    phys = np.array(traj["physical"], dtype=bool)
    return {
        "mean_err": float(errs.mean()),
        "steady_err_last10": float(errs[-10:].mean()),
        "min_err": float(errs.min()),
        "max_err": float(errs.max()),
        "physicality_pct": float(phys.mean() * 100.0),
        "ach_mean_last": float(traj["ach_mean"][-1]),
        "decision_us_p50": float(np.median(traj["decision_times_s"]) * 1e6),
    }


def aggregate_seeds(per_seed_results):
    """Aggregate metrics across seeds."""
    keys = ["mean_err", "steady_err_last10", "min_err", "max_err",
            "physicality_pct", "ach_mean_last", "decision_us_p50"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in per_seed_results])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{k}_per_seed"] = vals.tolist()
    return out


def run_policy_on_scenario(name, controller_factory, sim_p, x0, target,
                           n_ticks, dt, n_seeds):
    print(f"  policy={name}", flush=True)
    per_seed = []
    for seed in range(n_seeds):
        ctrl = controller_factory(seed)
        traj = run_episode(ctrl, sim_p, x0, target, n_ticks, dt)
        s = summarize(traj)
        per_seed.append(s)
        print(
            f"    seed={seed}: mean_err={s['mean_err']:7.2f} "
            f"steady={s['steady_err_last10']:7.2f} "
            f"phys={s['physicality_pct']:5.1f}%  "
            f"ach={s['ach_mean_last']:.2f}",
            flush=True,
        )
    return aggregate_seeds(per_seed)


def main():
    print("M18 closed-loop comparison")
    print("=" * 70)
    print(f"  N_walkers={N_WALKERS}, horizon={HORIZON}, n_ticks={N_TICKS}, "
          f"n_seeds={N_SEEDS}, dt={DT}")
    print()

    sim_p, x0 = build_jax_params()

    # Tier weights: default + amplified ×1.4 (Conjecture D sweet spot)
    tw_default = np.asarray(DEFAULT_TIER_WEIGHTS)
    tw_amp = tw_default * 1.4

    def make_vanilla(seed):
        return FMCPlasmaJaxController(
            sim_p, n_walkers=N_WALKERS, horizon=HORIZON,
            dt=DT, seed=seed,
        )

    def make_hier_default(seed):
        return FMCPlasmaHierarchicalController(
            sim_p, n_walkers=N_WALKERS, horizon=HORIZON,
            dt=DT, seed=seed,
            tier_weights=tw_default, ach_alpha=1.0,
        )

    def make_hier_amp(seed):
        return FMCPlasmaHierarchicalController(
            sim_p, n_walkers=N_WALKERS, horizon=HORIZON,
            dt=DT, seed=seed,
            tier_weights=tw_amp, ach_alpha=1.0,
        )

    results = {
        "config": {
            "N_walkers": N_WALKERS, "horizon": HORIZON,
            "n_ticks": N_TICKS, "n_seeds": N_SEEDS, "dt": DT,
            "tier_weights_default": tw_default.tolist(),
            "tier_weights_amplified": tw_amp.tolist(),
        },
        "scenarios": {},
    }

    for scenario_name, target in [("S1_near_M16", TARGET_S1),
                                   ("S2_far_displacement", TARGET_S2)]:
        print(f"--- Scenario {scenario_name} ---")
        print(f"  target = {target.tolist()}")
        err0 = shape_err_np(np.asarray(x0), target)
        print(f"  err at x0 (root) = {err0:.3f}")
        print()

        scen = {"target": target.tolist(), "err_at_root": err0, "policies": {}}

        for pname, factory in [
            ("vanilla", make_vanilla),
            ("hier_default", make_hier_default),
            ("hier_amplified_1.4x", make_hier_amp),
        ]:
            t_start = time.perf_counter()
            agg = run_policy_on_scenario(
                pname, factory, sim_p, x0, target,
                N_TICKS, DT, N_SEEDS,
            )
            agg["wall_s"] = time.perf_counter() - t_start
            scen["policies"][pname] = agg
            print(
                f"    AGGREGATE: mean={agg['mean_err_mean']:.2f}±"
                f"{agg['mean_err_std']:.2f}  "
                f"steady={agg['steady_err_last10_mean']:.2f}±"
                f"{agg['steady_err_last10_std']:.2f}  "
                f"phys={agg['physicality_pct_mean']:.1f}%  "
                f"wall={agg['wall_s']:.1f}s",
                flush=True,
            )
            print()

        results["scenarios"][scenario_name] = scen

    out_path = RESULTS_DIR / "m18_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")

    # Print final summary table
    print()
    print("=" * 70)
    print("FINAL SUMMARY (mean ± std over seeds)")
    print("=" * 70)
    for scen_name, scen in results["scenarios"].items():
        print(f"\n  {scen_name}  (err at root = {scen['err_at_root']:.2f})")
        print(f"  {'policy':<25s}  {'mean_err':>15s}  "
              f"{'steady_last10':>15s}  {'phys%':>8s}")
        for pname, agg in scen["policies"].items():
            print(
                f"  {pname:<25s}  "
                f"{agg['mean_err_mean']:7.2f} ± {agg['mean_err_std']:5.2f}  "
                f"{agg['steady_err_last10_mean']:7.2f} ± "
                f"{agg['steady_err_last10_std']:5.2f}  "
                f"{agg['physicality_pct_mean']:6.1f}%"
            )


if __name__ == "__main__":
    main()
