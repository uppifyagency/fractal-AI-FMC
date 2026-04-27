"""Milestone 15 — Evaluate policies on TCV published target scenarios.

Closes the loop: sim → GS oracle → published-experimental-targets.

For each of 6 scenarios from Degrave 2022 / Reimerdes 2022:
1. Time-discretize the target trajectory (10 ms ticks).
2. Run each policy in closed-loop on its native simulator.
3. Per tick: compute truth-err using the M14 FreeGS oracle.
4. Record physicality rate (% LCFS-valid steps).
5. Aggregate per-policy metrics.

The result is a benchmark table comparable across policies that uses
literature-published targets and physics-grounded ground truth.
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

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from calibrated_sim import build_calibrated_jax_params
from fmc_plasma_jax import FMCPlasmaJaxController
from freegs_oracle_robust import FreeGSOracle
from plasma_simulator_jax import DTYPE, make_jit_step
from plasma_simulator_nn_shape import (
    SimParamsNN, build_nn_sim_params, make_jit_step_nn, predict_shape,
)
from policy import TrainedPolicy
from tcv_published_targets import PUBLISHED_TARGETS, discretize

RESULTS_DIR = Path(__file__).parent.parent / "results"


def truth_shape_freegs(I_coils, oracle, nn_fallback):
    def fb(I):
        s = np.array(predict_shape(nn_fallback, jnp.asarray(I, dtype=DTYPE)))
        s[0] = np.clip(s[0], 0.624, 1.136)
        s[1] = np.clip(s[1], -0.75, 0.75)
        s[2] = np.clip(s[2], 1.0, 2.8)
        s[3] = np.clip(s[3], -0.7, 1.0)
        return s
    res = oracle.shape_from_coils(np.asarray(I_coils), fallback_fn=fb)
    return np.array([res.R_p, res.Z_p, res.kappa, res.delta]), res.source


def make_initial_state(sim_p):
    a = sim_p.a_eff
    V_plasma = 2 * np.pi**2 * sim_p.R_ref * a**2 * sim_p.kappa_ref
    n_bar = 5e19
    T_e_keV = 1.0
    W = 3 * n_bar * V_plasma * (T_e_keV * 1e3 * 1.602176634e-19)
    return jnp.concatenate([
        jnp.asarray(sim_p.I_ref, dtype=DTYPE),
        jnp.array([200_000.0, W, n_bar,
                   sim_p.R_ref, sim_p.Z_ref,
                   sim_p.kappa_ref, sim_p.delta_ref], dtype=DTYPE),
    ])


def run_scenario(policy_fn, sim_step, sim_p, oracle, nn_fallback,
                 targets: np.ndarray, weights):
    """Run one scenario, per-tick freegs truth.

    targets: shape (n_ticks, 4)
    Returns dict with truth_errs, self_errs, freegs_count, fallback_count.
    """
    N = sim_p.N
    x = np.asarray(make_initial_state(sim_p)).copy()
    truth_errs, self_errs = [], []
    n_freegs = n_fb = 0
    n_nan = 0
    for t_idx, target in enumerate(targets):
        target = target.astype(np.float32)
        V = policy_fn(x, target)
        x_new = sim_step(
            jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(5e5, dtype=DTYPE),
            jnp.asarray(1e21, dtype=DTYPE),
            jnp.asarray(1e-3, dtype=DTYPE),
        )
        x = np.array(x_new)
        if np.isnan(x).any() or np.isinf(x).any():
            n_nan += 1
            break
        true_shape, source = truth_shape_freegs(x[:N], oracle, nn_fallback)
        self_R, self_Z, self_K, self_D = x[N+3], x[N+4], x[N+5], x[N+6]
        err_t = float(
            weights[0] * (true_shape[0] - target[0]) ** 2
            + weights[1] * (true_shape[1] - target[1]) ** 2
            + weights[2] * (true_shape[2] - target[2]) ** 2
            + weights[3] * (true_shape[3] - target[3]) ** 2
        )
        err_s = float(
            weights[0] * (self_R - target[0]) ** 2
            + weights[1] * (self_Z - target[1]) ** 2
            + weights[2] * (self_K - target[2]) ** 2
            + weights[3] * (self_D - target[3]) ** 2
        )
        truth_errs.append(err_t)
        self_errs.append(err_s)
        if source == "freegs":
            n_freegs += 1
        elif source == "nn_fallback":
            n_fb += 1
    return {
        "truth_errs": truth_errs,
        "self_errs": self_errs,
        "n_freegs": n_freegs,
        "n_fallback": n_fb,
        "n_nan": n_nan,
        "n_total": len(targets),
    }


def run_fmc_scenario(jx_fmc, sim_step, sim_p, oracle, nn_fallback,
                     targets, weights):
    N = sim_p.N
    x = np.asarray(make_initial_state(sim_p)).copy()
    truth_errs, self_errs = [], []
    n_freegs = n_fb = 0
    n_nan = 0
    for t_idx, target in enumerate(targets):
        target = target.astype(np.float32)
        V = jx_fmc.decide(x, target)["V_coils"]
        x_new = sim_step(
            jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(5e5, dtype=DTYPE),
            jnp.asarray(1e21, dtype=DTYPE),
            jnp.asarray(1e-3, dtype=DTYPE),
        )
        x = np.array(x_new)
        if np.isnan(x).any():
            n_nan += 1
            break
        true_shape, source = truth_shape_freegs(x[:N], oracle, nn_fallback)
        err_t = sum(weights[i] * (true_shape[i] - target[i])**2 for i in range(4))
        err_s = sum(weights[i] * (x[N+3+i] - target[i])**2 for i in range(4))
        truth_errs.append(float(err_t))
        self_errs.append(float(err_s))
        if source == "freegs":
            n_freegs += 1
        elif source == "nn_fallback":
            n_fb += 1
    return {"truth_errs": truth_errs, "self_errs": self_errs,
            "n_freegs": n_freegs, "n_fallback": n_fb, "n_nan": n_nan,
            "n_total": len(targets)}


def main():
    print("=" * 72)
    print("Milestone 15 — Eval on published TCV experimental targets")
    print("=" * 72)

    weights = np.array([100.0, 100.0, 10.0, 10.0])
    dt = 0.05  # 50 ms per tick = 20 Hz (close to TCV PCS 10 kHz subsampled)

    print("\n[1] Initializing FreeGS oracle (M14)...")
    oracle = FreeGSOracle(verbose=True)

    print("\n[2] Loading sims and policies...")
    sim_p_lin, _ = build_calibrated_jax_params()
    sim_step_lin = make_jit_step(sim_p_lin)
    policy_m10 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger_jax.npz")
    policy_m6 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")
    policy_m5 = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    sim_p_nn = build_nn_sim_params()
    sim_step_nn = make_jit_step_nn(sim_p_nn)
    policy_m12 = TrainedPolicy.load(RESULTS_DIR / "policy_nn_shape.npz")

    jx_fmc = FMCPlasmaJaxController(sim_p_lin, n_walkers=64, horizon=10, seed=15)
    target_warm = np.array([sim_p_lin.R_ref, sim_p_lin.Z_ref,
                             sim_p_lin.kappa_ref, sim_p_lin.delta_ref],
                            dtype=np.float32)
    jx_fmc.decide(np.zeros(27, dtype=np.float32), target_warm)

    setups = {
        "M5_BC":       (sim_p_lin, sim_step_lin, policy_m5),
        "M6_DAgger3":  (sim_p_lin, sim_step_lin, policy_m6),
        "M10_DAggerN": (sim_p_lin, sim_step_lin, policy_m10),
        "M12_NNshape": (sim_p_nn, sim_step_nn, policy_m12),
    }

    all_results = {}
    t_start = time.perf_counter()

    for scenario in PUBLISHED_TARGETS:
        t_arr, targets = discretize(scenario, dt=dt)
        n_ticks = len(targets)
        print(f"\n[3] Scenario: {scenario.name} "
              f"({n_ticks} ticks @ {dt*1000:.0f} ms = {scenario.duration_s} s)")

        scn_results = {"scenario": scenario.name,
                        "citation": scenario.citation,
                        "n_ticks": n_ticks,
                        "duration_s": scenario.duration_s,
                        "policies": {}}

        for label, (sim_p, sim_step, pol) in setups.items():
            res = run_scenario(
                lambda x, t: pol(x, t),
                sim_step, sim_p, oracle, sim_p_nn,
                targets, weights,
            )
            mean_truth = float(np.mean(res["truth_errs"])) if res["truth_errs"] else float("inf")
            mean_self = float(np.mean(res["self_errs"])) if res["self_errs"] else float("inf")
            phys = res["n_freegs"] / max(1, res["n_freegs"] + res["n_fallback"])
            print(f"    {label:14s} | truth {mean_truth:7.2f} | "
                  f"self {mean_self:7.2f} | phys {100*phys:.0f}% "
                  f"(n_steps={len(res['truth_errs'])}/{res['n_total']})")
            scn_results["policies"][label] = {
                "mean_truth_err": mean_truth,
                "mean_self_err": mean_self,
                "physicality": phys,
                "n_steps": len(res["truth_errs"]),
                "n_nan": res["n_nan"],
            }

        # FMC online
        res_fmc = run_fmc_scenario(jx_fmc, sim_step_lin, sim_p_lin,
                                     oracle, sim_p_nn, targets, weights)
        mt = float(np.mean(res_fmc["truth_errs"])) if res_fmc["truth_errs"] else float("inf")
        ms = float(np.mean(res_fmc["self_errs"])) if res_fmc["self_errs"] else float("inf")
        ph = res_fmc["n_freegs"] / max(1, res_fmc["n_freegs"] + res_fmc["n_fallback"])
        print(f"    {'FMC_online':14s} | truth {mt:7.2f} | "
              f"self {ms:7.2f} | phys {100*ph:.0f}% "
              f"(n_steps={len(res_fmc['truth_errs'])}/{res_fmc['n_total']})")
        scn_results["policies"]["FMC_online"] = {
            "mean_truth_err": mt, "mean_self_err": ms,
            "physicality": ph,
            "n_steps": len(res_fmc["truth_errs"]),
            "n_nan": res_fmc["n_nan"],
        }
        all_results[scenario.name] = scn_results

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # Aggregate: per-policy mean across all scenarios
    print(f"\n[4] Cross-scenario aggregate ranking")
    print("-" * 72)
    aggregate = {}
    for label in ["M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"]:
        truths = [all_results[s]["policies"][label]["mean_truth_err"]
                  for s in all_results]
        selfs = [all_results[s]["policies"][label]["mean_self_err"]
                 for s in all_results]
        physs = [all_results[s]["policies"][label]["physicality"]
                 for s in all_results]
        aggregate[label] = {
            "mean_truth_across_scenarios": float(np.mean(truths)),
            "mean_self_across_scenarios": float(np.mean(selfs)),
            "mean_physicality": float(np.mean(physs)),
            "max_truth": float(np.max(truths)),
            "min_truth": float(np.min(truths)),
        }
    ranking = sorted(aggregate.items(),
                      key=lambda kv: kv[1]["mean_truth_across_scenarios"])
    for label, agg in ranking:
        print(f"  {label:14s} | mean truth {agg['mean_truth_across_scenarios']:7.2f}"
              f" (range {agg['min_truth']:6.2f}-{agg['max_truth']:6.2f})"
              f" | self {agg['mean_self_across_scenarios']:7.2f}"
              f" | phys {100*agg['mean_physicality']:.0f}%")

    out = {
        "method": "tcv_published_targets_via_m14_freegs_oracle",
        "weights": weights.tolist(),
        "tick_dt_s": dt,
        "wall_clock_s": elapsed,
        "scenarios": all_results,
        "aggregate": aggregate,
    }
    out_path = RESULTS_DIR / "milestone_15_published_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
