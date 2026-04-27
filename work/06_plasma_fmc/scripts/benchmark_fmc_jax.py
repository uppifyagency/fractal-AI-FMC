"""Benchmark JIT FMC vs Python FMC on:
  1. Single decision latency
  2. Dataset generation rate (samples/sec) — what matters for DAgger
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

from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget
from fmc_plasma_jax import FMCPlasmaJaxController
from generate_expert_dataset import sample_initial_state, sample_target
from plasma_simulator_jax import build_jax_params

RESULTS_DIR = Path(__file__).parent.parent / "results"


def bench_latency(name, decide_fn, n_warmup=2, n_runs=20):
    for _ in range(n_warmup):
        decide_fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        decide_fn()
    elapsed = (time.perf_counter() - t0) / n_runs
    print(f"  {name:20s}: median {elapsed*1e6:7.1f} µs")
    return elapsed


def bench_dataset(name, sample_fn, n_samples=50):
    """Time end-to-end dataset sample generation (= what DAgger does)."""
    t0 = time.perf_counter()
    for i in range(n_samples):
        sample_fn(i)
    elapsed = time.perf_counter() - t0
    rate = n_samples / elapsed
    per_sample = elapsed / n_samples * 1e3  # ms
    print(f"  {name:30s}: {n_samples} samples in {elapsed:.2f}s "
          f"({rate:.1f} samples/sec, {per_sample:.1f} ms/sample)")
    return rate, per_sample


def main():
    sim_p, x0 = build_jax_params()
    target_arr = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
    target_obj = ShapeTarget(0.90, 0.0, 1.85, 0.3)

    print("=" * 70)
    print("Milestone 7 — JIT FMC benchmark")
    print("=" * 70)

    # ---- A) Single-decision latency ----
    print("\n[A] Single-decision latency (M=32, H=8, dataset config)")
    print("-" * 70)
    py_ctrl = FMCPlasmaController(sim_p, target_obj,
                                  FMCConfig(n_walkers=32, horizon=8), seed=0)
    jx_ctrl = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=8, seed=0)
    py_lat = bench_latency("Python FMC (M3 impl)",
                           lambda: py_ctrl.decide(np.asarray(x0)))
    jx_lat = bench_latency("JIT FMC (M7 impl)",
                           lambda: jx_ctrl.decide(np.asarray(x0), target_arr))
    print(f"  → JIT speedup: {py_lat/jx_lat:.1f}×")

    print("\n[A2] Single-decision latency (M=200, H=20, real-time config)")
    print("-" * 70)
    py_ctrl_full = FMCPlasmaController(sim_p, target_obj,
                                       FMCConfig(n_walkers=200, horizon=20), seed=0)
    jx_ctrl_full = FMCPlasmaJaxController(sim_p, n_walkers=200, horizon=20, seed=0)
    py_lat_full = bench_latency("Python FMC", lambda: py_ctrl_full.decide(np.asarray(x0)))
    jx_lat_full = bench_latency("JIT FMC",
                                lambda: jx_ctrl_full.decide(np.asarray(x0), target_arr))
    print(f"  → JIT speedup: {py_lat_full/jx_lat_full:.1f}×")

    # ---- B) Dataset generation rate ----
    print("\n[B] Dataset generation rate (full pipeline: random sample → FMC → save)")
    print("-" * 70)
    rng = np.random.default_rng(42)

    # Reuse sample_initial_state + sample_target from generate_expert_dataset
    def py_sample(i):
        s = sample_initial_state(rng, sim_p)
        tgt = sample_target(rng)
        ctrl = FMCPlasmaController(sim_p, tgt, FMCConfig(n_walkers=32, horizon=8),
                                    seed=i)
        ctrl.decide(s)

    rng2 = np.random.default_rng(42)  # same seed for fair compare
    # JIT controller: reuse across samples — JIT cache stays warm
    jx_persistent = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=8, seed=0)
    # Warm up jit compilation
    s_warm = sample_initial_state(np.random.default_rng(99), sim_p)
    jx_persistent.decide(s_warm, target_arr)

    def jx_sample(i):
        s = sample_initial_state(rng2, sim_p)
        tgt = sample_target(rng2)
        tgt_arr = np.array([tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta], dtype=np.float32)
        jx_persistent.decide(s, tgt_arr)

    py_rate, py_ms = bench_dataset("Python FMC dataset gen", py_sample, n_samples=20)
    jx_rate, jx_ms = bench_dataset("JIT FMC dataset gen", jx_sample, n_samples=20)
    speedup = jx_rate / py_rate
    print(f"  → Dataset gen speedup: {speedup:.1f}×")

    # Estimate: 1000-sample DAgger iter
    print(f"\n  Time to generate 1000 samples:")
    print(f"    Python FMC : {1000/py_rate:6.1f} sec")
    print(f"    JIT FMC    : {1000/jx_rate:6.1f} sec")

    out = RESULTS_DIR / "milestone_7_benchmark.json"
    with open(out, "w") as f:
        json.dump({
            "latency_us": {
                "python_small": py_lat * 1e6, "jit_small": jx_lat * 1e6,
                "python_full": py_lat_full * 1e6, "jit_full": jx_lat_full * 1e6,
                "speedup_small": py_lat / jx_lat,
                "speedup_full": py_lat_full / jx_lat_full,
            },
            "dataset_gen": {
                "python_samples_per_sec": py_rate,
                "jit_samples_per_sec": jx_rate,
                "speedup": speedup,
            },
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
