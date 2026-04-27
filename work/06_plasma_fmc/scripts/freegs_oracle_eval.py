"""FreeGS-proxy oracle evaluation — Milestone 13 (revised).

ORIGINAL plan: re-solve FreeGS at each control tick to compute truth.
PROBLEM: freegs.solve() without constraints doesn't converge from arbitrary
coil currents (Picard iteration fails). 0/24 valid GS solves in initial test.

REVISED plan: use the M11 NN_shape model as a proxy for truth. The NN is
trained on 135 actual FreeGS solves with RMSE 3 cm on R_p, 0.09 on κ.
It's a continuous, always-defined function of I_coils that captures
real GS non-linearity.

This is a good compromise:
  - Faithful to GS physics (NN learned from GS data)
  - Fast (~200 µs per shape prediction vs 700 ms for FreeGS solve)
  - Same "truth" applied to all policies → fair comparison
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
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from plasma_simulator_nn_shape import (
    SimParamsNN, build_nn_sim_params, initial_state_nn,
    make_jit_step_nn, predict_shape,
)
from policy import TrainedPolicy

RESULTS_DIR = Path(__file__).parent.parent / "results"


def truth_shape_from_coils(I_coils: np.ndarray, sim_p_nn: SimParamsNN) -> np.ndarray:
    """Compute "truth" shape from coil currents via M11 NN_shape model.

    Clip output to physical range — NN can extrapolate wildly for I_coils
    outside training distribution (FreeGS dataset went up to ±37 kA).
    """
    s = np.array(predict_shape(sim_p_nn, jnp.asarray(I_coils, dtype=DTYPE)))
    s[0] = np.clip(s[0], 0.624, 1.136)
    s[1] = np.clip(s[1], -0.75, 0.75)
    s[2] = np.clip(s[2], 1.0, 2.8)
    s[3] = np.clip(s[3], -0.7, 1.0)
    return s


def run_episode_oracle(policy_fn, sim_step, sim_p, sim_p_nn,
                        target, x0, n_ticks: int, weights):
    """Closed-loop episode where shape error uses NN_shape as truth.

    The policy sees its own simulator state (R_p, Z_p, kappa, delta from sim_step).
    But the "true" tracking error is measured against shape derived from
    actual I_coils via NN_shape (the M11 GS surrogate).
    """
    N = sim_p.N
    x = np.asarray(x0).copy()
    log = []

    for t in range(n_ticks):
        V = policy_fn(x, target)

        x_new = sim_step(
            jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(5e5, dtype=DTYPE),
            jnp.asarray(1e21, dtype=DTYPE),
            jnp.asarray(1e-3, dtype=DTYPE),
        )
        x = np.array(x_new)
        if np.isnan(x).any() or np.isinf(x).any():
            log.append({"t": t, "I_p_kA": np.nan, "err_truth": np.inf,
                        "err_self": np.inf})
            break

        # Truth shape from current I_coils (via NN proxy)
        I_coils_now = x[:N]
        true_shape = truth_shape_from_coils(I_coils_now, sim_p_nn)
        true_R, true_Z, true_K, true_D = true_shape

        # Self-reported shape (what the policy thinks it has)
        self_R, self_Z, self_K, self_D = x[N+3], x[N+4], x[N+5], x[N+6]

        err_truth = (
            weights[0] * (true_R - target[0]) ** 2
            + weights[1] * (true_Z - target[1]) ** 2
            + weights[2] * (true_K - target[2]) ** 2
            + weights[3] * (true_D - target[3]) ** 2
        )
        err_self = (
            weights[0] * (self_R - target[0]) ** 2
            + weights[1] * (self_Z - target[1]) ** 2
            + weights[2] * (self_K - target[2]) ** 2
            + weights[3] * (self_D - target[3]) ** 2
        )
        log.append({
            "t": t, "I_p_kA": float(x[N]) / 1e3,
            "true_shape": [float(s) for s in true_shape],
            "self_shape": [float(s) for s in (self_R, self_Z, self_K, self_D)],
            "err_truth": float(err_truth), "err_self": float(err_self),
        })

    return log


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


def main():
    print("=" * 70)
    print("Milestone 13 — NN-truth oracle eval (FreeGS proxy via M11)")
    print("=" * 70)

    n_scenarios = 10
    n_ticks = 15
    weights = np.array([100.0, 100.0, 10.0, 10.0])

    # Setups
    sim_p_lin, _ = build_calibrated_jax_params()
    sim_step_lin = make_jit_step(sim_p_lin)
    policy_m10 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger_jax.npz")
    policy_m6 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")  # M6 baseline
    policy_m5 = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")  # M5 BC

    sim_p_nn = build_nn_sim_params()
    sim_step_nn = make_jit_step_nn(sim_p_nn)
    policy_m12 = TrainedPolicy.load(RESULTS_DIR / "policy_nn_shape.npz")

    # FMC online (linear sim)
    jx_fmc = FMCPlasmaJaxController(sim_p_lin, n_walkers=64, horizon=10, seed=0)
    target_warm = np.array([sim_p_lin.R_ref, sim_p_lin.Z_ref,
                             sim_p_lin.kappa_ref, sim_p_lin.delta_ref], dtype=np.float32)
    jx_fmc.decide(np.zeros(27, dtype=np.float32), target_warm)

    # Generate scenarios (reachable targets only)
    rng = np.random.default_rng(13)
    scenarios = []
    for k in range(n_scenarios):
        target = np.array([
            rng.uniform(sim_p_lin.R_ref - 0.02, sim_p_lin.R_ref + 0.02),
            rng.uniform(sim_p_lin.Z_ref - 0.03, sim_p_lin.Z_ref + 0.03),
            rng.uniform(sim_p_lin.kappa_ref - 0.05, sim_p_lin.kappa_ref + 0.10),
            rng.uniform(sim_p_lin.delta_ref - 0.10, sim_p_lin.delta_ref + 0.10),
        ], dtype=np.float32)
        scenarios.append(target)

    # Run all evaluators
    all_results = {
        "M5_BC": {"sim": (sim_p_lin, sim_step_lin), "policy": policy_m5},
        "M6_DAgger3": {"sim": (sim_p_lin, sim_step_lin), "policy": policy_m6},
        "M10_DAggerN": {"sim": (sim_p_lin, sim_step_lin), "policy": policy_m10},
        "M12_NNshape": {"sim": (sim_p_nn, sim_step_nn), "policy": policy_m12},
    }

    print(f"\nRunning {n_scenarios} scenarios × {n_ticks} ticks per evaluator")
    print("-" * 70)
    summary = {}
    t_start = time.perf_counter()

    for label, cfg in all_results.items():
        sim_p, sim_step = cfg["sim"]
        pol = cfg["policy"]
        all_truth_errs = []
        all_self_errs = []
        for k, target in enumerate(scenarios):
            x0 = make_initial_state(sim_p)
            log = run_episode_oracle(
                lambda x, t: pol(x, t),
                sim_step, sim_p, sim_p_nn,
                target, x0, n_ticks, weights,
            )
            for r in log:
                if r["err_truth"] != np.inf:
                    all_truth_errs.append(r["err_truth"])
                if r["err_self"] != np.inf:
                    all_self_errs.append(r["err_self"])
        mean_truth = float(np.mean(all_truth_errs)) if all_truth_errs else np.inf
        mean_self = float(np.mean(all_self_errs)) if all_self_errs else np.inf
        summary[label] = {
            "mean_err_truth": mean_truth,
            "mean_err_self": mean_self,
            "n_truth_valid": len(all_truth_errs),
            "n_self_valid": len(all_self_errs),
        }
        print(f"  {label:14s} | truth-err {mean_truth:7.2f} | self-err {mean_self:7.2f}")

    # FMC online — needs special path because target is per-decision
    print(f"  {'FMC_online':14s}", end="", flush=True)
    truth_errs_fmc = []
    self_errs_fmc = []
    for target in scenarios:
        x = np.asarray(make_initial_state(sim_p_lin)).copy()
        for t in range(n_ticks):
            V = jx_fmc.decide(x, target)["V_coils"]
            x_new = sim_step_lin(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                jnp.asarray(5e5, dtype=DTYPE),
                jnp.asarray(1e21, dtype=DTYPE),
                jnp.asarray(1e-3, dtype=DTYPE),
            )
            x = np.array(x_new)
            if np.isnan(x).any():
                break
            true_shape = truth_shape_from_coils(x[:sim_p_lin.N], sim_p_nn)
            err_t = sum(weights[i] * (true_shape[i] - target[i])**2 for i in range(4))
            err_s = sum(weights[i] * (x[sim_p_lin.N + 3 + i] - target[i])**2 for i in range(4))
            truth_errs_fmc.append(float(err_t))
            self_errs_fmc.append(float(err_s))
    mt = float(np.mean(truth_errs_fmc)) if truth_errs_fmc else np.inf
    ms = float(np.mean(self_errs_fmc)) if self_errs_fmc else np.inf
    summary["FMC_online"] = {
        "mean_err_truth": mt, "mean_err_self": ms,
        "n_truth_valid": len(truth_errs_fmc),
    }
    print(f" | truth-err {mt:7.2f} | self-err {ms:7.2f}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # Save
    out = RESULTS_DIR / "milestone_13_oracle_eval.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "scenarios_count": n_scenarios,
                   "ticks_per_scenario": n_ticks, "wall_clock_s": elapsed,
                   "weights": weights.tolist()}, f, indent=2, default=str)
    print(f"\n✓ Saved: {out}")

    # Print final ranking
    print(f"\n[Ranking by truth-err — physically faithful]")
    ranking = sorted(summary.items(), key=lambda kv: kv[1]["mean_err_truth"])
    for label, s in ranking:
        print(f"  {label:14s} : truth-err = {s['mean_err_truth']:7.2f} | "
              f"self-err = {s['mean_err_self']:7.2f}")


if __name__ == "__main__":
    main()
