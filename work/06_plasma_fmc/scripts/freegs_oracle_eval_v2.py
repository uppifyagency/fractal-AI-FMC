"""Oracle eval with REAL FreeGS truth — Milestone 14.

This is the M13 oracle eval, but using the M14 robust FreeGS oracle as
ground truth instead of the M11 NN_shape proxy. This answers: do the M13
"all policies ≈ same truth-err" finding hold up against real GS physics,
or was it an artifact of the NN proxy?

Setup
-----
- Same 5 policies: M5 BC, M6 DAgger×3, M10 DAggerN, M12 NN-shape, FMC online
- Same 10 scenarios × 15 ticks
- Per-tick truth shape: oracle.shape_from_coils(I_coils) using
  vacuum + plasma-residual decomposition (24 ms/shape, 90% conv)
- NN_shape used as fallback only when freegs extraction fails

Output
------
- results/milestone_14_oracle_eval.json: per-policy truth-err and self-err
- comparison vs M13 (NN proxy) included
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
from freegs_oracle_robust import COIL_ORDER, FreeGSOracle, OracleResult
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from plasma_simulator_nn_shape import (
    SimParamsNN, build_nn_sim_params, make_jit_step_nn, predict_shape,
)
from policy import TrainedPolicy

RESULTS_DIR = Path(__file__).parent.parent / "results"


def truth_shape_freegs(I_coils: np.ndarray, oracle: FreeGSOracle,
                        nn_fallback: SimParamsNN) -> tuple[np.ndarray, str]:
    """Get truth shape from FreeGS oracle, with NN fallback.

    Returns (shape_array_4d, source_label).
    """
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


def run_episode(policy_fn, sim_step, sim_p, oracle, nn_fallback,
                target, x0, n_ticks: int, weights):
    """Closed-loop episode; per-tick truth via FreeGS oracle."""
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
            log.append({"t": t, "err_truth": np.inf, "err_self": np.inf,
                        "source": "nan"})
            break
        I_coils_now = x[:N]
        true_shape, source = truth_shape_freegs(I_coils_now, oracle, nn_fallback)
        # Self-reported shape (what policy thinks it has)
        self_R, self_Z, self_K, self_D = x[N+3], x[N+4], x[N+5], x[N+6]
        err_truth = float(
            weights[0] * (true_shape[0] - target[0]) ** 2
            + weights[1] * (true_shape[1] - target[1]) ** 2
            + weights[2] * (true_shape[2] - target[2]) ** 2
            + weights[3] * (true_shape[3] - target[3]) ** 2
        )
        err_self = float(
            weights[0] * (self_R - target[0]) ** 2
            + weights[1] * (self_Z - target[1]) ** 2
            + weights[2] * (self_K - target[2]) ** 2
            + weights[3] * (self_D - target[3]) ** 2
        )
        log.append({"t": t, "err_truth": err_truth, "err_self": err_self,
                    "true_shape": true_shape.tolist(), "source": source,
                    "I_p_kA": float(x[N]) / 1e3})
    return log


def main():
    print("=" * 72)
    print("Milestone 14 — Oracle eval v2 (REAL FreeGS truth)")
    print("=" * 72)

    n_scenarios = 10
    n_ticks = 15
    weights = np.array([100.0, 100.0, 10.0, 10.0])

    print("\n[1] Initializing FreeGS oracle (baseline solve ~1 s)...")
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

    jx_fmc = FMCPlasmaJaxController(sim_p_lin, n_walkers=64, horizon=10, seed=0)
    target_warm = np.array([sim_p_lin.R_ref, sim_p_lin.Z_ref,
                             sim_p_lin.kappa_ref, sim_p_lin.delta_ref],
                            dtype=np.float32)
    jx_fmc.decide(np.zeros(27, dtype=np.float32), target_warm)

    rng = np.random.default_rng(14)
    scenarios = []
    for k in range(n_scenarios):
        target = np.array([
            rng.uniform(sim_p_lin.R_ref - 0.02, sim_p_lin.R_ref + 0.02),
            rng.uniform(sim_p_lin.Z_ref - 0.03, sim_p_lin.Z_ref + 0.03),
            rng.uniform(sim_p_lin.kappa_ref - 0.05, sim_p_lin.kappa_ref + 0.10),
            rng.uniform(sim_p_lin.delta_ref - 0.10, sim_p_lin.delta_ref + 0.10),
        ], dtype=np.float32)
        scenarios.append(target)

    setups = {
        "M5_BC":       (sim_p_lin, sim_step_lin, policy_m5),
        "M6_DAgger3":  (sim_p_lin, sim_step_lin, policy_m6),
        "M10_DAggerN": (sim_p_lin, sim_step_lin, policy_m10),
        "M12_NNshape": (sim_p_nn, sim_step_nn, policy_m12),
    }

    print(f"\n[3] Running {n_scenarios} scenarios × {n_ticks} ticks per evaluator")
    print("    (~150 freegs solves per policy × 4 policies = ~600, "
          "expected ~15 sec)")
    print("-" * 72)
    summary = {}
    t_start = time.perf_counter()

    for label, (sim_p, sim_step, pol) in setups.items():
        all_truth, all_self = [], []
        n_freegs = n_fb = 0
        for target in scenarios:
            x0 = make_initial_state(sim_p)
            log = run_episode(
                lambda x, t: pol(x, t),
                sim_step, sim_p, oracle, sim_p_nn,
                target, x0, n_ticks, weights,
            )
            for r in log:
                if r["err_truth"] != np.inf:
                    all_truth.append(r["err_truth"])
                if r["err_self"] != np.inf:
                    all_self.append(r["err_self"])
                if r.get("source") == "freegs":
                    n_freegs += 1
                elif r.get("source") == "nn_fallback":
                    n_fb += 1
        mt = float(np.mean(all_truth)) if all_truth else np.inf
        ms = float(np.mean(all_self)) if all_self else np.inf
        summary[label] = {
            "mean_err_truth": mt, "mean_err_self": ms,
            "n_truth_valid": len(all_truth), "n_self_valid": len(all_self),
            "n_freegs": n_freegs, "n_nn_fallback": n_fb,
        }
        print(f"  {label:14s} | truth-err {mt:7.2f} | self-err {ms:7.2f} | "
              f"freegs {n_freegs}/{n_freegs+n_fb}")

    # FMC online
    print(f"  {'FMC_online':14s}", end="", flush=True)
    truth_errs_fmc, self_errs_fmc = [], []
    n_freegs_fmc = n_fb_fmc = 0
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
            true_shape, source = truth_shape_freegs(
                x[:sim_p_lin.N], oracle, sim_p_nn,
            )
            err_t = sum(weights[i] * (true_shape[i] - target[i])**2 for i in range(4))
            err_s = sum(weights[i] * (x[sim_p_lin.N + 3 + i] - target[i])**2 for i in range(4))
            truth_errs_fmc.append(float(err_t))
            self_errs_fmc.append(float(err_s))
            if source == "freegs":
                n_freegs_fmc += 1
            elif source == "nn_fallback":
                n_fb_fmc += 1
    mt = float(np.mean(truth_errs_fmc)) if truth_errs_fmc else np.inf
    ms = float(np.mean(self_errs_fmc)) if self_errs_fmc else np.inf
    summary["FMC_online"] = {
        "mean_err_truth": mt, "mean_err_self": ms,
        "n_truth_valid": len(truth_errs_fmc),
        "n_freegs": n_freegs_fmc, "n_nn_fallback": n_fb_fmc,
    }
    print(f" | truth-err {mt:7.2f} | self-err {ms:7.2f} | "
          f"freegs {n_freegs_fmc}/{n_freegs_fmc+n_fb_fmc}")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # Compare against M13 (NN proxy) — load if available
    m13_path = RESULTS_DIR / "milestone_13_oracle_eval.json"
    m13_compare = None
    if m13_path.exists():
        with open(m13_path) as f:
            m13 = json.load(f)
        m13_compare = m13.get("summary", {})

    out = {
        "method": "real_freegs_via_vacuum_plus_plasma_residual",
        "summary": summary,
        "scenarios_count": n_scenarios,
        "ticks_per_scenario": n_ticks,
        "wall_clock_s": elapsed,
        "weights": weights.tolist(),
        "m13_comparison_nn_proxy": m13_compare,
    }
    out_path = RESULTS_DIR / "milestone_14_oracle_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")

    # Ranking
    print(f"\n[Ranking by truth-err — REAL FreeGS truth]")
    ranking = sorted(summary.items(), key=lambda kv: kv[1]["mean_err_truth"])
    for label, s in ranking:
        m13_ref = ""
        if m13_compare and label in m13_compare:
            m13_truth = m13_compare[label].get("mean_err_truth", float("nan"))
            m13_ref = f"  (M13 NN-proxy: {m13_truth:.2f})"
        print(f"  {label:14s} : truth-err = {s['mean_err_truth']:7.2f}"
              f"  self-err = {s['mean_err_self']:7.2f}{m13_ref}")


if __name__ == "__main__":
    main()
