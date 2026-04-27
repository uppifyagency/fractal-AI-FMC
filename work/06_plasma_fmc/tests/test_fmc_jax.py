"""Tests for Milestone 7 — JIT-compiled FMC.

Verifies:
- relativize_jax matches relativize_np
- Single decision returns valid V_coils (correct shape, finite, near V_ref)
- Same target+state across many calls produces consistent decisions
- Latency is improved vs Python FMC (regression check)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import jax
import jax.numpy as jnp
from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget, relativize_np
from fmc_plasma_jax import (
    FMCPlasmaJaxController,
    FMCStaticCfg,
    relativize_jax,
)
from plasma_simulator_jax import build_jax_params


class TestRelativizeJax:
    def test_matches_numpy(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            x = rng.normal(0, 5, 50).astype(np.float32)
            np_out = relativize_np(x)
            jx_out = np.asarray(relativize_jax(jnp.asarray(x)))
            assert np.allclose(np_out, jx_out, atol=1e-5), (
                f"max diff = {np.max(np.abs(np_out - jx_out))}"
            )

    def test_constant_input(self):
        x = jnp.full(10, 3.14, dtype=jnp.float32)
        out = np.asarray(relativize_jax(x))
        assert np.allclose(out, 1.0)


class TestFMCJaxBasic:
    def test_decision_valid(self):
        sim_p, x0 = build_jax_params()
        ctrl = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=5, seed=0)
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        d = ctrl.decide(np.asarray(x0), target)
        assert d["V_coils"].shape == (sim_p.N,)
        assert np.all(np.isfinite(d["V_coils"]))
        assert d["walkers_alive"] >= 0

    def test_returns_v_near_ref_when_feasible(self):
        """V command shouldn't be wildly far from V_ref."""
        sim_p, x0 = build_jax_params()
        V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
        ctrl = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=5, seed=0)
        target = np.array([0.88, 0.0, 1.7, 0.3], dtype=np.float32)  # = ref
        d = ctrl.decide(np.asarray(x0), target)
        # ΔV should be modest (< 1000 V on average)
        dV = d["V_coils"] - V_ref
        assert np.linalg.norm(dV) < 5000.0, (
            f"|V - V_ref| = {np.linalg.norm(dV)} too large"
        )

    def test_seed_determinism(self):
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        c1 = FMCPlasmaJaxController(sim_p, n_walkers=16, horizon=4, seed=42)
        c2 = FMCPlasmaJaxController(sim_p, n_walkers=16, horizon=4, seed=42)
        d1 = c1.decide(np.asarray(x0), target)
        d2 = c2.decide(np.asarray(x0), target)
        assert np.array_equal(d1["V_coils"], d2["V_coils"])


class TestFMCJaxLatencyRegression:
    def test_faster_than_python_for_small_config(self):
        """JIT version must be at least 2× faster than Python for M=32, H=8."""
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        target_obj = ShapeTarget(0.90, 0.0, 1.85, 0.3)

        # Python FMC
        py = FMCPlasmaController(
            sim_p, target_obj,
            FMCConfig(n_walkers=32, horizon=8, voltage_std=50.0), seed=0,
        )

        # JAX FMC
        jx = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=8, seed=0)

        # Warmup
        for _ in range(2):
            py.decide(np.asarray(x0))
            jx.decide(np.asarray(x0), target)

        # Time
        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            py.decide(np.asarray(x0))
        py_time = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for _ in range(n):
            jx.decide(np.asarray(x0), target)
        jx_time = (time.perf_counter() - t0) / n

        print(f"\n  Python FMC : {py_time*1e6:.0f} µs")
        print(f"  JAX FMC    : {jx_time*1e6:.0f} µs ({py_time/jx_time:.1f}× speedup)")
        assert jx_time < py_time / 2.0, (
            f"JIT FMC ({jx_time*1e6:.0f}µs) not 2× faster than "
            f"Python FMC ({py_time*1e6:.0f}µs)"
        )


if __name__ == "__main__":
    test_classes = [TestRelativizeJax, TestFMCJaxBasic, TestFMCJaxLatencyRegression]
    n_pass = n_fail = 0
    for cls in test_classes:
        instance = cls()
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(instance, attr)()
                print(f"  ✓ {cls.__name__}.{attr}")
                n_pass += 1
            except Exception as e:
                print(f"  ✗ {cls.__name__}.{attr}: {e}")
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
