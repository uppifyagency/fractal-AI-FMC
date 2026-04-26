"""Benchmark plasma simulator step latency on M1 Pro.

Three implementations measured:
1. NumPy reference          (one walker, one step)
2. JAX jit (CPU)            (one walker, one step)
3. JAX jit + vmap (CPU)     (B walkers, one step)
4. JAX jit + vmap + scan    (B walkers, H-step rollout)

Targets (FMC budget @ 1 kHz control):
- single step          : < 100 µs
- 200 walkers × 20 steps = 4000 evals : < 1 ms

Note: jax-metal 0.1.1 is incompatible with current jax 0.10. CPU benchmarks
only. Metal results to be added when plugin catches up.
"""
from __future__ import annotations

import os
import time

import numpy as np

# Force CPU explicitly (Metal plugin is broken; see Milestone 2 doc)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp

from plasma_simulator import Control, build_default_simulator
from plasma_simulator_jax import (
    DTYPE,
    build_jax_params,
    make_batched_rollout,
    make_batched_step,
    make_jit_step,
)


def time_it(fn, n_warmup: int = 5, n_runs: int = 1000) -> tuple[float, float]:
    """Returns (median_seconds, p95_seconds)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return float(np.median(times)), float(np.percentile(times, 95))


def bench_numpy(n_runs: int = 1000):
    sim, state = build_default_simulator()
    rng = np.random.default_rng(0)
    V = rng.normal(0, 50.0, size=sim.N)
    ctrl = Control(V_coils=V, P_aux=5e5, gas_puff=2e21)

    def call():
        sim.step(state, ctrl, 1e-3)

    med, p95 = time_it(call, n_runs=n_runs)
    return med, p95


def bench_jax_single(n_runs: int = 1000):
    params, x0 = build_jax_params()
    step = make_jit_step(params)
    rng = np.random.default_rng(0)
    V = jnp.asarray(rng.normal(0, 50.0, size=params.N), dtype=DTYPE)
    P = jnp.asarray(5e5, dtype=DTYPE)
    g = jnp.asarray(2e21, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)

    # Warm jit
    step(x0, V, P, g, dt).block_until_ready()

    def call():
        step(x0, V, P, g, dt).block_until_ready()

    med, p95 = time_it(call, n_runs=n_runs)
    return med, p95


def bench_jax_batched(B: int, n_runs: int = 200):
    params, x0 = build_jax_params()
    bstep = make_batched_step(params)

    rng = np.random.default_rng(0)
    state_b = jnp.broadcast_to(x0, (B, x0.shape[0]))
    V_b = jnp.asarray(rng.normal(0, 50.0, size=(B, params.N)), dtype=DTYPE)
    P_b = jnp.full((B,), 5e5, dtype=DTYPE)
    g_b = jnp.full((B,), 2e21, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)

    bstep(state_b, V_b, P_b, g_b, dt).block_until_ready()

    def call():
        bstep(state_b, V_b, P_b, g_b, dt).block_until_ready()

    med, p95 = time_it(call, n_runs=n_runs)
    return med, p95


def bench_jax_rollout(B: int, H: int, n_runs: int = 100):
    params, x0 = build_jax_params()
    rollout = make_batched_rollout(params, horizon=H)

    rng = np.random.default_rng(0)
    state_b = jnp.broadcast_to(x0, (B, x0.shape[0]))
    V_seq = jnp.asarray(rng.normal(0, 50.0, size=(H, B, params.N)), dtype=DTYPE)
    P_seq = jnp.full((H, B), 5e5, dtype=DTYPE)
    g_seq = jnp.full((H, B), 2e21, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)

    rollout(state_b, V_seq, P_seq, g_seq, dt).block_until_ready()

    def call():
        rollout(state_b, V_seq, P_seq, g_seq, dt).block_until_ready()

    med, p95 = time_it(call, n_runs=n_runs)
    return med, p95


def main():
    print("=" * 70)
    print(f"Plasma simulator benchmark — Apple M1 Pro, JAX backend: {jax.devices()}")
    print("=" * 70)

    # 1. NumPy single-step
    med, p95 = bench_numpy(n_runs=2000)
    print(f"\n[1] NumPy single-step (1 walker, 1 step)")
    print(f"    median = {med*1e6:7.2f} µs   p95 = {p95*1e6:7.2f} µs")
    np_med = med

    # 2. JAX single-step
    med, p95 = bench_jax_single(n_runs=2000)
    print(f"\n[2] JAX jit single-step (1 walker, 1 step)")
    print(f"    median = {med*1e6:7.2f} µs   p95 = {p95*1e6:7.2f} µs")
    print(f"    speedup vs NumPy: {np_med/med:.2f}×")

    # 3. JAX batched single-step
    for B in (32, 128, 512):
        med, p95 = bench_jax_batched(B=B, n_runs=500)
        per = med / B
        print(f"\n[3] JAX vmap+jit (B={B} walkers, 1 step)")
        print(f"    median = {med*1e6:7.2f} µs total = {per*1e6:6.3f} µs/walker"
              f"   p95 = {p95*1e6:7.2f} µs")

    # 4. Rollout (batched horizon)
    for B, H in [(32, 20), (128, 20), (256, 20), (200, 30)]:
        med, p95 = bench_jax_rollout(B=B, H=H, n_runs=200)
        per_eval = med / (B * H)
        print(f"\n[4] JAX scan rollout (B={B} walkers × H={H} steps = {B*H} evals)")
        print(f"    median = {med*1e6:7.2f} µs total = {per_eval*1e6:6.3f} µs/eval"
              f"   p95 = {p95*1e6:7.2f} µs")

    print()
    print("=" * 70)
    print("FMC budget reminder: 1 ms per control decision @ 1 kHz")
    print("Target: M=200 walkers × N=20 steps = 4000 evals < 1000 µs")
    print("=" * 70)


if __name__ == "__main__":
    main()
