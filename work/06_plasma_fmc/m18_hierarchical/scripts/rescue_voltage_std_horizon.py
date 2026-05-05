"""M18 rescue test: increase walker divergence by voltage_std and horizon.

Iteration 1 produced IDENTICAL results between vanilla and hierarchical
because all walkers fire achievements simultaneously → uniform fire_bonus
→ relativize standardizes to zero → no effect on cloning argmax.

Hypothesis: increasing voltage_std (50 → 200 V) and horizon (10 → 20)
produces enough walker divergence that some walkers fire higher-tier
achievements before others. If Conjecture D applies to plasma, the
hierarchical FMC should now outperform vanilla.

Three scenarios:
  S2  far-displacement (target = (1.00, 0.10, 2.20, -0.45))   err≈10.6
  S3  very-far    (target = (1.05, 0.15, 2.40, -0.60))         err≈26.5
  S4  extreme     (target = (1.10, 0.20, 2.60, -0.75))         err≈53.0
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
HORIZON = 20         # rescue: 10 → 20
N_WALKERS = 64
N_TICKS = 30
N_SEEDS = 3
VOLTAGE_STD = 200.0  # rescue: 50 → 200

TARGETS = {
    "S2_far":     np.array([1.00, 0.10, 2.20, -0.45], dtype=np.float32),
    "S3_very_far": np.array([1.05, 0.15, 2.40, -0.60], dtype=np.float32),
    "S4_extreme": np.array([1.10, 0.20, 2.60, -0.75], dtype=np.float32),
}


def shape_err_np(state, target):
    N = 20
    return float(
        100.0 * (state[N + 3] - target[0]) ** 2
        + 100.0 * (state[N + 4] - target[1]) ** 2
        + 10.0 * (state[N + 5] - target[2]) ** 2
        + 10.0 * (state[N + 6] - target[3]) ** 2
    )


def run_episode(controller, sim_p, x0, target, n_ticks, dt):
    step_jit = make_jit_step(sim_p)
    state = np.asarray(x0).copy()
    P_aux = jnp.asarray(5e5, dtype=jnp.float32)
    gas = jnp.asarray(1e21, dtype=jnp.float32)
    errs = []
    I_p_history = []
    physical = []
    ach_mean_history = []

    for t in range(n_ticks):
        decision = controller.decide(state, target)
        V = jnp.asarray(decision["V_coils"], dtype=jnp.float32)
        new_state = step_jit(jnp.asarray(state), V, P_aux, gas, dt)
        state = np.asarray(new_state)

        err = shape_err_np(state, target)
        errs.append(err)
        I_p_history.append(float(state[20]) / 1e6)
        is_phys = (
            abs(I_p_history[-1]) > 0.05
            and np.isfinite(err) and err < 1e6
        )
        physical.append(bool(is_phys))
        ach_mean_history.append(float(decision.get("ach_mean", 0.0)))

    return {
        "errs": errs,
        "physical": physical,
        "ach_mean": ach_mean_history,
    }


def summarize(traj):
    errs = np.array(traj["errs"])
    phys = np.array(traj["physical"], dtype=bool)
    return {
        "mean_err": float(errs.mean()),
        "steady_err_last10": float(errs[-10:].mean()),
        "min_err": float(errs.min()),
        "physicality_pct": float(phys.mean() * 100.0),
        "ach_mean_last": float(traj["ach_mean"][-1]),
        "ach_mean_avg": float(np.mean(traj["ach_mean"])),
    }


def main():
    print("M18 RESCUE: voltage_std=200, horizon=20")
    print("=" * 70)

    sim_p, x0 = build_jax_params()
    tw_default = np.asarray(DEFAULT_TIER_WEIGHTS)

    def make_vanilla(seed):
        return FMCPlasmaJaxController(
            sim_p, n_walkers=N_WALKERS, horizon=HORIZON, dt=DT,
            voltage_std=VOLTAGE_STD, seed=seed,
        )

    def make_hier(seed):
        return FMCPlasmaHierarchicalController(
            sim_p, n_walkers=N_WALKERS, horizon=HORIZON, dt=DT,
            voltage_std=VOLTAGE_STD, seed=seed,
            tier_weights=tw_default, ach_alpha=1.0,
        )

    results = {}

    for sname, target in TARGETS.items():
        print(f"\n--- {sname}  target={target.tolist()} ---")
        err0 = shape_err_np(np.asarray(x0), target)
        print(f"  err at x0 = {err0:.2f}")

        scen = {"target": target.tolist(), "err_at_root": err0,
                "policies": {}}

        for pname, factory in [("vanilla", make_vanilla),
                               ("hierarchical", make_hier)]:
            print(f"  policy={pname}")
            seeds_data = []
            for seed in range(N_SEEDS):
                ctrl = factory(seed)
                traj = run_episode(ctrl, sim_p, x0, target, N_TICKS, DT)
                s = summarize(traj)
                seeds_data.append(s)
                print(
                    f"    seed={seed}: mean={s['mean_err']:7.2f}  "
                    f"steady={s['steady_err_last10']:7.2f}  "
                    f"phys={s['physicality_pct']:5.1f}%  "
                    f"ach_avg={s['ach_mean_avg']:.2f}"
                )
            mean_err = np.mean([s["mean_err"] for s in seeds_data])
            std_err = np.std([s["mean_err"] for s in seeds_data], ddof=1) \
                if N_SEEDS > 1 else 0
            steady_mean = np.mean([s["steady_err_last10"] for s in seeds_data])
            steady_std = np.std([s["steady_err_last10"] for s in seeds_data], ddof=1) \
                if N_SEEDS > 1 else 0
            phys_mean = np.mean([s["physicality_pct"] for s in seeds_data])
            ach_mean = np.mean([s["ach_mean_avg"] for s in seeds_data])
            print(f"    AGGREGATE: mean_err={mean_err:.2f}±{std_err:.2f}  "
                  f"steady={steady_mean:.2f}±{steady_std:.2f}  "
                  f"phys={phys_mean:.1f}%  ach={ach_mean:.2f}")
            scen["policies"][pname] = {
                "mean_err_mean": float(mean_err), "mean_err_std": float(std_err),
                "steady_mean": float(steady_mean), "steady_std": float(steady_std),
                "phys_pct_mean": float(phys_mean),
                "ach_mean": float(ach_mean),
                "per_seed": seeds_data,
            }
        results[sname] = scen

    out_path = RESULTS_DIR / "m18_rescue.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Final summary: delta hierarchical vs vanilla
    print()
    print("=" * 70)
    print("DELTA hierarchical vs vanilla (negative = hierarchical wins)")
    print("=" * 70)
    print(f"  {'scenario':<15s}  {'err_root':>9s}  "
          f"{'Δ_mean_err':>12s}  {'Δ_steady':>10s}  {'verdict':>15s}")
    for sname, scen in results.items():
        v = scen["policies"]["vanilla"]
        h = scen["policies"]["hierarchical"]
        d_mean = h["mean_err_mean"] - v["mean_err_mean"]
        d_steady = h["steady_mean"] - v["steady_mean"]
        # noise-vs-signal: significant if |delta| > std_combined / sqrt(N)
        std_combined = np.sqrt(v["mean_err_std"] ** 2 + h["mean_err_std"] ** 2)
        if abs(d_mean) < 0.01 and abs(d_steady) < 0.01:
            verdict = "IDENTICAL"
        elif d_mean < -std_combined / np.sqrt(N_SEEDS):
            verdict = "hier WINS"
        elif d_mean > std_combined / np.sqrt(N_SEEDS):
            verdict = "vanilla WINS"
        else:
            verdict = "noise-tie"
        print(f"  {sname:<15s}  {scen['err_at_root']:9.2f}  "
              f"{d_mean:+12.3f}  {d_steady:+10.3f}  {verdict:>15s}")


if __name__ == "__main__":
    main()
