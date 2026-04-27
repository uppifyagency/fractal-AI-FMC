"""Benchmark M8 (extended DAgger) vs M6 (3-iter DAgger) vs M5 BC vs FMC."""
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

import jax.numpy as jnp
from fmc_plasma_jax import FMCPlasmaJaxController
from generate_expert_dataset import sample_initial_state, sample_target
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import TrainedPolicy

RESULTS_DIR = Path(__file__).parent.parent / "results"


def time_call(fn, n_warmup=3, n_runs=200):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.percentile(times, 95))


def closed_loop_err(decide_fn, sim_p, sim_step, n_eval=10,
                    episode_length=30, seed=99):
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
        tgt_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                           dtype=np.float32)
        ep_err = []
        for t in range(episode_length):
            V = decide_fn(x, tgt_arr, target)
            x_new = sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                jnp.asarray(5e5, dtype=DTYPE),
                jnp.asarray(1e21, dtype=DTYPE),
                jnp.asarray(1e-3, dtype=DTYPE),
            )
            x = np.array(x_new)
            N = sim_p.N
            err = (target.w_R * (x[N+3] - target.R_p)**2
                   + target.w_Z * (x[N+4] - target.Z_p)**2
                   + target.w_kappa * (x[N+5] - target.kappa)**2
                   + target.w_delta * (x[N+6] - target.delta)**2)
            ep_err.append(float(err))
            if abs(x[N]) / 1e6 < 0.05:
                quenches += 1
                break
        errors.append(np.mean(ep_err))
    return {"mean_err": float(np.mean(errors)),
            "median_err": float(np.median(errors)),
            "quench": quenches, "n_eval": n_eval}


def main():
    print("=" * 70)
    print("Milestone 8 — extended DAgger benchmark")
    print("=" * 70)

    sim_p, x0 = build_jax_params()
    sim_step = make_jit_step(sim_p)

    bc = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    dagger3 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")
    dagger_jax = TrainedPolicy.load(RESULTS_DIR / "policy_dagger_jax.npz")

    target_demo = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)

    # ---- Latency ----
    print("\n[A] Latency")
    print("-" * 70)
    bc_lat, _ = time_call(lambda: bc(np.asarray(x0), target_demo))
    d3_lat, _ = time_call(lambda: dagger3(np.asarray(x0), target_demo))
    djax_lat, _ = time_call(lambda: dagger_jax(np.asarray(x0), target_demo))

    jx = FMCPlasmaJaxController(sim_p, n_walkers=200, horizon=20, seed=0)
    jx.decide(np.asarray(x0), target_demo)  # warmup
    fmc_lat, _ = time_call(lambda: jx.decide(np.asarray(x0), target_demo),
                            n_warmup=2, n_runs=20)

    print(f"  M5 BC          : {bc_lat*1e6:8.1f} µs")
    print(f"  M6 DAgger×3    : {d3_lat*1e6:8.1f} µs")
    print(f"  M8 DAgger×N    : {djax_lat*1e6:8.1f} µs")
    print(f"  FMC online (full) : {fmc_lat*1e6:8.1f} µs ({fmc_lat/djax_lat:.0f}× slower)")

    # ---- Quality ----
    print("\n[B] Closed-loop quality (10 random scenarios, 30 tick)")
    print("-" * 70)
    e_bc = closed_loop_err(lambda x, t, _: bc(x, t), sim_p, sim_step,
                            n_eval=10, seed=99)
    e_d3 = closed_loop_err(lambda x, t, _: dagger3(x, t), sim_p, sim_step,
                            n_eval=10, seed=99)
    e_djax = closed_loop_err(lambda x, t, _: dagger_jax(x, t), sim_p, sim_step,
                              n_eval=10, seed=99)

    # FMC online with same JIT controller (warm)
    def fmc_call(x, tgt_arr, _):
        return jx.decide(x, tgt_arr)["V_coils"]
    e_fmc = closed_loop_err(fmc_call, sim_p, sim_step, n_eval=10, seed=99)

    print(f"  M5 BC          : err {e_bc['mean_err']:6.2f} | quench {e_bc['quench']}/10")
    print(f"  M6 DAgger×3    : err {e_d3['mean_err']:6.2f} | quench {e_d3['quench']}/10")
    print(f"  M8 DAgger×N    : err {e_djax['mean_err']:6.2f} | quench {e_djax['quench']}/10")
    print(f"  FMC online     : err {e_fmc['mean_err']:6.2f} | quench {e_fmc['quench']}/10  (ground truth)")

    # Compute key metrics
    if e_djax["mean_err"] > 0:
        gap = e_djax["mean_err"] / e_fmc["mean_err"]
        print(f"\n  M8 DAgger gap to FMC online: {gap:.2f}× worse")

    speedup_quality_combined = (fmc_lat / djax_lat) * (e_fmc["mean_err"] / e_djax["mean_err"])
    print(f"  Speed × quality factor (vs FMC): {speedup_quality_combined:.1f}× "
          f"(higher = NN policy better tradeoff)")

    out = RESULTS_DIR / "milestone_8_benchmark.json"
    with open(out, "w") as f:
        json.dump({
            "latency_us": {
                "bc": bc_lat * 1e6, "dagger3": d3_lat * 1e6,
                "dagger_jax": djax_lat * 1e6, "fmc": fmc_lat * 1e6,
            },
            "tracking": {"bc": e_bc, "dagger3": e_d3,
                         "dagger_jax": e_djax, "fmc": e_fmc},
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
