"""Compare DAgger policy vs M5 BC policy vs FMC on identical scenarios."""
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
from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget
from generate_expert_dataset import sample_initial_state, sample_target
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import TrainedPolicy

RESULTS_DIR = Path(__file__).parent.parent / "results"


def time_call(fn, n_warmup=3, n_runs=100):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.percentile(times, 95))


def closed_loop_err(policy_callable, sim_p, sim_step, n_eval=10,
                    episode_length=30, seed=99):
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
        target_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                              dtype=np.float32)
        ep_err = []
        for t in range(episode_length):
            V = policy_callable(x, target_arr, target)
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
            "quench": quenches, "n_eval": n_eval}


def main():
    print("=" * 70)
    print("Milestone 6 — DAgger vs M5 BC vs FMC")
    print("=" * 70)

    sim_p, x0 = build_jax_params()
    sim_step = make_jit_step(sim_p)

    # Load both policies
    bc = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    dagger = TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")

    target_demo = ShapeTarget(R_p=0.90, Z_p=0.0, kappa=1.85, delta=0.3)
    tgt_arr = np.array([target_demo.R_p, target_demo.Z_p,
                        target_demo.kappa, target_demo.delta], dtype=np.float32)

    # ---- Latency ----
    print("\n[A] Latency (single decision)")
    print("-" * 70)
    bc_med, _ = time_call(lambda: bc(np.asarray(x0), tgt_arr), n_runs=200)
    dg_med, _ = time_call(lambda: dagger(np.asarray(x0), tgt_arr), n_runs=200)
    cfg_full = FMCConfig(n_walkers=200, horizon=20, voltage_std=50.0)
    ctrl = FMCPlasmaController(sim_p, target_demo, cfg_full, seed=0)
    fmc_med, _ = time_call(lambda: ctrl.decide(np.asarray(x0)), n_warmup=1, n_runs=5)
    print(f"  BC policy (M5)        : {bc_med*1e6:8.1f} µs")
    print(f"  DAgger policy (M6)    : {dg_med*1e6:8.1f} µs")
    print(f"  FMC (M=200, H=20)     : {fmc_med*1e6:8.1f} µs ({fmc_med/dg_med:.0f}× slower than DAgger)")

    # ---- Closed-loop quality on N=10 randomized scenarios ----
    print("\n[B] Closed-loop tracking — 10 random scenarios, 30 tick each")
    print("-" * 70)

    e_bc = closed_loop_err(lambda x, t, _: bc(x, t),
                            sim_p, sim_step, n_eval=10, seed=99)
    e_dg = closed_loop_err(lambda x, t, _: dagger(x, t),
                            sim_p, sim_step, n_eval=10, seed=99)

    # FMC closed-loop (slow, n_eval reduced)
    def fmc_call(x, target_arr, target_obj):
        c = FMCPlasmaController(sim_p, target_obj,
                                FMCConfig(n_walkers=64, horizon=10), seed=99)
        return c.decide(x)["V_coils"]
    e_fmc = closed_loop_err(fmc_call, sim_p, sim_step,
                            n_eval=5, episode_length=30, seed=99)

    print(f"  BC policy (M5)     : mean err {e_bc['mean_err']:7.2f} | "
          f"quench {e_bc['quench']}/{e_bc['n_eval']}")
    print(f"  DAgger policy (M6) : mean err {e_dg['mean_err']:7.2f} | "
          f"quench {e_dg['quench']}/{e_dg['n_eval']}")
    print(f"  FMC (online)       : mean err {e_fmc['mean_err']:7.2f} | "
          f"quench {e_fmc['quench']}/{e_fmc['n_eval']} (slow ground truth)")

    if e_bc['mean_err'] > 0:
        print(f"\n  Quality improvement (BC → DAgger): {e_bc['mean_err'] / e_dg['mean_err']:.1f}× better")
    if e_fmc['mean_err'] > 0:
        print(f"  Remaining gap (DAgger vs FMC)    : {e_dg['mean_err'] / e_fmc['mean_err']:.1f}× worse")

    out = RESULTS_DIR / "milestone_6_benchmark.json"
    with open(out, "w") as f:
        json.dump({
            "latency_us": {"bc": bc_med * 1e6, "dagger": dg_med * 1e6,
                            "fmc": fmc_med * 1e6},
            "tracking": {"bc": e_bc, "dagger": e_dg, "fmc": e_fmc},
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
