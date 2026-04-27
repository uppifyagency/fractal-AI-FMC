"""Tests for Milestone 5 — NN policy distillation.

Verifies:
- Feature builder shapes
- Normalizer round-trip
- State rescaling correctness
- Trained policy loads + outputs valid voltages
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from policy import (
    Normalizer,
    PolicyMLP,
    TrainedPolicy,
    build_features,
    rescale_state,
)
from plasma_simulator_jax import build_jax_params

POLICY_PATH = Path(__file__).parent.parent / "results" / "policy_params.npz"


class TestNormalizer:
    def test_roundtrip(self):
        x = np.random.default_rng(0).normal(5.0, 3.0, (100, 8))
        n = Normalizer.fit(x)
        z = n.transform(x)
        x_back = n.inverse(z)
        assert np.allclose(x, x_back, atol=1e-5)

    def test_mean_zero_std_one(self):
        x = np.random.default_rng(0).normal(7.0, 2.0, (1000, 4))
        n = Normalizer.fit(x)
        z = n.transform(x)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(z.std(axis=0), 1, atol=1e-2)

    def test_dict_save_load(self):
        x = np.random.default_rng(0).normal(0, 1, (50, 5))
        n = Normalizer.fit(x)
        d = n.to_dict()
        n2 = Normalizer.from_dict(d)
        assert np.array_equal(n.mean, n2.mean)
        assert np.array_equal(n.std, n2.std)


class TestRescaleState:
    def test_units(self):
        # Build a representative state
        s = np.zeros(27, dtype=np.float32)
        s[:20] = 5000.0   # I_coils 5 kA
        s[20] = 200_000   # I_p 200 kA
        s[21] = 40_000    # W 40 kJ
        s[22] = 5e19      # n_bar
        s[23] = 0.88
        s[24] = 0.0
        s[25] = 1.7
        s[26] = 0.3

        sr = rescale_state(s)
        assert np.allclose(sr[:20], 5.0)      # 5 kA
        assert np.allclose(sr[20], 0.2)       # 0.2 MA
        assert np.allclose(sr[21], 40.0)      # 40 kJ
        assert np.allclose(sr[22], 5.0)       # 5 in 10^19
        # Geometric values unchanged
        assert sr[23] == 0.88
        assert sr[25] == 1.7

    def test_no_overflow(self):
        s = np.zeros(27, dtype=np.float32)
        s[22] = 5e19
        sr = rescale_state(s)
        # Without rescale, sr[22]² = 2.5e39 → overflow in f32.
        # After rescale, sr[22] = 5, sr[22]² = 25 → safe.
        sq = sr ** 2
        assert np.all(np.isfinite(sq))


class TestBuildFeatures:
    def test_shape_single(self):
        s = np.arange(27.0)
        t = np.array([0.9, 0.0, 1.85, 0.3])
        i = np.arange(20.0) / 100
        f = build_features(s, t, i)
        assert f.shape == (51,)

    def test_shape_batch(self):
        s = np.arange(8 * 27.0).reshape(8, 27)
        t = np.array([[0.9, 0.0, 1.85, 0.3]] * 8)
        i = np.arange(20.0) / 100
        f = build_features(s, t, i)
        assert f.shape == (8, 51)

    def test_concatenation_order(self):
        s = np.full(27, 1.0)
        t = np.array([2.0, 3.0, 4.0, 5.0])
        i = np.full(20, 6.0)
        f = build_features(s, t, i)
        assert np.allclose(f[:27], 1.0)
        assert np.allclose(f[27:31], [2, 3, 4, 5])
        assert np.allclose(f[31:], 6.0)


class TestTrainedPolicy:
    def test_load_and_call(self):
        if not POLICY_PATH.exists():
            import pytest
            pytest.skip(f"No trained policy at {POLICY_PATH}")
        p = TrainedPolicy.load(POLICY_PATH)
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        V = p(np.asarray(x0), target)
        assert V.shape == (sim_p.N,)
        assert np.all(np.isfinite(V))

    def test_clipping_active(self):
        """Output ΔV must be within ±max_dV."""
        if not POLICY_PATH.exists():
            return
        p = TrainedPolicy.load(POLICY_PATH)
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        V = p(np.asarray(x0), target, max_dV=10.0)
        dV = V - p.V_ref
        assert np.all(np.abs(dV) <= 10.0 + 1e-3)

    def test_determinism(self):
        if not POLICY_PATH.exists():
            return
        p = TrainedPolicy.load(POLICY_PATH)
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        V1 = p(np.asarray(x0), target)
        V2 = p(np.asarray(x0), target)
        assert np.array_equal(V1, V2)


if __name__ == "__main__":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")

    test_classes = [TestNormalizer, TestRescaleState, TestBuildFeatures, TestTrainedPolicy]
    n_pass = n_fail = n_skip = 0
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
                if "skip" in str(e).lower():
                    print(f"  - {cls.__name__}.{attr}: skipped")
                    n_skip += 1
                else:
                    print(f"  ✗ {cls.__name__}.{attr}: {e}")
                    n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(0 if n_fail == 0 else 1)
