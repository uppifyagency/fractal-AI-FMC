"""Benchmark NN policy vs FMC: runtime + closed-loop tracking quality.

Runs:
  A) Latency: NN policy single decision vs FMC single decision
  B) Closed-loop tracking: same (init state, target) tracked by both
  C) Generalization probe: target distribution disjoint from training
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import json
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import TrainedPolicy

RESULTS_DIR = Path(__file__).parent.parent / "results"


def time_call(fn, n_warmup=3, n_runs=50):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.percentile(times, 95))


def main():
    print("=" * 70)
    print("Milestone 5 — NN policy benchmark vs FMC")
    print("=" * 70)

    sim_p, x0 = build_jax_params()
    policy_path = RESULTS_DIR / "policy_params.npz"
    policy = TrainedPolicy.load(policy_path)
    print(f"Policy loaded: {policy_path.name}, "
          f"{sum(p.size for p in jax.tree.leaves(policy.params))} params")

    sim_step = make_jit_step(sim_p)
    target = ShapeTarget(R_p=0.90, Z_p=0.0, kappa=1.85, delta=0.3)
    target_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                          dtype=np.float32)

    # ---- A) Latency benchmark ----
    print("\n[A] Single-decision latency")
    print("-" * 70)

    state = np.asarray(x0)
    # NN policy
    nn_med, nn_p95 = time_call(
        lambda: policy(state, target_arr),
        n_warmup=5, n_runs=200,
    )
    print(f"  NN policy (1 forward pass)   : "
          f"median = {nn_med*1e6:8.1f} µs   p95 = {nn_p95*1e6:8.1f} µs")

    # FMC (same config as training)
    cfg_fmc = FMCConfig(n_walkers=32, horizon=8, voltage_std=50.0)
    ctrl = FMCPlasmaController(sim_p, target, cfg_fmc, seed=0)
    fmc_med, fmc_p95 = time_call(
        lambda: ctrl.decide(state),
        n_warmup=2, n_runs=20,
    )
    print(f"  FMC (M={cfg_fmc.n_walkers}, H={cfg_fmc.horizon})           : "
          f"median = {fmc_med*1e6:8.1f} µs   p95 = {fmc_p95*1e6:8.1f} µs")

    # FMC at higher quality (M=200, H=20, like real-time setting)
    cfg_fmc_full = FMCConfig(n_walkers=200, horizon=20, voltage_std=50.0)
    ctrl_full = FMCPlasmaController(sim_p, target, cfg_fmc_full, seed=0)
    fmc_full_med, fmc_full_p95 = time_call(
        lambda: ctrl_full.decide(state),
        n_warmup=1, n_runs=5,
    )
    print(f"  FMC (M={cfg_fmc_full.n_walkers}, H={cfg_fmc_full.horizon})         : "
          f"median = {fmc_full_med*1e6:8.1f} µs   p95 = {fmc_full_p95*1e6:8.1f} µs")

    speedup_small = fmc_med / nn_med
    speedup_full = fmc_full_med / nn_med
    print(f"\n  Speedup NN vs FMC(small) : {speedup_small:.0f}×")
    print(f"  Speedup NN vs FMC(full)  : {speedup_full:.0f}×")

    # ---- B) Closed-loop tracking ----
    print("\n[B] Closed-loop tracking comparison (50 control ticks)")
    print("-" * 70)
    dt = 1e-3
    horizon_s = 50e-3

    def run_closed_loop(controller_fn, label):
        x = np.asarray(x0).copy()
        log = []
        t0 = time.perf_counter()
        for k in range(50):
            V = controller_fn(x)
            x_new = sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                jnp.asarray(5e5, dtype=DTYPE),
                jnp.asarray(1e21, dtype=DTYPE),
                jnp.asarray(dt, dtype=DTYPE),
            )
            x = np.array(x_new)
            N = sim_p.N
            log.append({
                "t_ms": (k + 1) * 1.0,
                "R_p": float(x[N + 3]), "Z_p": float(x[N + 4]),
                "kappa": float(x[N + 5]), "delta": float(x[N + 6]),
                "I_p_kA": float(x[N]) / 1e3,
            })
        wall = time.perf_counter() - t0
        # Mean shape error over the run
        err = np.array([
            target.w_R * (r["R_p"] - target.R_p)**2
            + target.w_Z * (r["Z_p"] - target.Z_p)**2
            + target.w_kappa * (r["kappa"] - target.kappa)**2
            + target.w_delta * (r["delta"] - target.delta)**2
            for r in log
        ])
        print(f"  {label:18s} | wall {wall*1e3:6.1f} ms | "
              f"mean shape err {err.mean():6.3f} | "
              f"final R_p={log[-1]['R_p']:.3f} κ={log[-1]['kappa']:.3f} "
              f"I_p={log[-1]['I_p_kA']:.0f} kA")
        return log, wall

    nn_log, nn_wall = run_closed_loop(
        lambda x: policy(x, target_arr), "NN policy",
    )
    # Re-create controller for FMC closed-loop (full quality)
    ctrl_full2 = FMCPlasmaController(sim_p, target, cfg_fmc_full, seed=0)
    fmc_log, fmc_wall = run_closed_loop(
        lambda x: ctrl_full2.decide(x)["V_coils"], "FMC (M=200, H=20)",
    )

    print(f"\n  Wall-clock 50-tick episode:")
    print(f"    NN policy : {nn_wall*1e3:6.1f} ms")
    print(f"    FMC       : {fmc_wall*1e3:6.1f} ms ({fmc_wall/nn_wall:.0f}× slower)")

    # Save logs
    out = RESULTS_DIR / "milestone_5_benchmark.json"
    with open(out, "w") as f:
        json.dump({
            "latency": {
                "nn_policy_us": {"median": nn_med * 1e6, "p95": nn_p95 * 1e6},
                "fmc_small_us": {"median": fmc_med * 1e6, "p95": fmc_p95 * 1e6},
                "fmc_full_us": {"median": fmc_full_med * 1e6, "p95": fmc_full_p95 * 1e6},
                "speedup_vs_small": speedup_small,
                "speedup_vs_full": speedup_full,
            },
            "tracking": {
                "target": {"R_p": target.R_p, "Z_p": target.Z_p,
                           "kappa": target.kappa, "delta": target.delta},
                "nn_log": nn_log, "fmc_log": fmc_log,
                "nn_wall_ms": nn_wall * 1e3, "fmc_wall_ms": fmc_wall * 1e3,
            },
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
