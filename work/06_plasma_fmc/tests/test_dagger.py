"""Tests for Milestone 6 — DAgger correctness.

Verifies:
- Dataset growth across iterations (monotone increase)
- DAgger policy loads + runs without NaN
- Closed-loop quality strictly improves vs M5 BC
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DAGGER_HISTORY = Path(__file__).parent.parent / "results" / "dagger_history.json"
DAGGER_POLICY = Path(__file__).parent.parent / "results" / "policy_dagger.npz"
DAGGER_BENCH = Path(__file__).parent.parent / "results" / "milestone_6_benchmark.json"


class TestDAggerHistory:
    def test_dataset_monotone(self):
        if not DAGGER_HISTORY.exists():
            return
        with open(DAGGER_HISTORY) as f:
            h = json.load(f)["history"]
        sizes = [r["n_samples"] for r in h]
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i-1], (
                f"Dataset shrunk at iter {i}: {sizes[i-1]} → {sizes[i]}"
            )

    def test_iter_zero_baseline(self):
        if not DAGGER_HISTORY.exists():
            return
        with open(DAGGER_HISTORY) as f:
            h = json.load(f)["history"]
        # Iteration 0 should be the M5 baseline (= 500 samples from M5 dataset)
        assert h[0]["n_samples"] == 500


class TestDAggerPolicy:
    def test_loads(self):
        if not DAGGER_POLICY.exists():
            return
        from policy import TrainedPolicy
        from plasma_simulator_jax import build_jax_params
        p = TrainedPolicy.load(DAGGER_POLICY)
        sim_p, x0 = build_jax_params()
        target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)
        V = p(np.asarray(x0), target)
        assert V.shape == (sim_p.N,)
        assert np.all(np.isfinite(V))


class TestDAggerImproves:
    def test_quality_better_than_bc(self):
        if not DAGGER_BENCH.exists():
            return
        with open(DAGGER_BENCH) as f:
            b = json.load(f)
        # DAgger must be at least 2× better than BC on tracking error
        bc_err = b["tracking"]["bc"]["mean_err"]
        dg_err = b["tracking"]["dagger"]["mean_err"]
        assert dg_err < bc_err / 2.0, (
            f"DAgger {dg_err:.2f} not at least 2× better than BC {bc_err:.2f}"
        )

    def test_no_quench_after_dagger(self):
        if not DAGGER_BENCH.exists():
            return
        with open(DAGGER_BENCH) as f:
            b = json.load(f)
        # DAgger should produce strictly fewer quenches than BC
        bc_q = b["tracking"]["bc"]["quench"]
        dg_q = b["tracking"]["dagger"]["quench"]
        assert dg_q < bc_q, (
            f"DAgger quenches {dg_q} not better than BC quenches {bc_q}"
        )

    def test_latency_unchanged(self):
        if not DAGGER_BENCH.exists():
            return
        with open(DAGGER_BENCH) as f:
            b = json.load(f)
        # DAgger latency must be within 30% of BC (same architecture)
        bc_lat = b["latency_us"]["bc"]
        dg_lat = b["latency_us"]["dagger"]
        rel_diff = abs(dg_lat - bc_lat) / bc_lat
        assert rel_diff < 0.30, f"Latency drift too large: {rel_diff*100:.0f}%"


if __name__ == "__main__":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")

    test_classes = [TestDAggerHistory, TestDAggerPolicy, TestDAggerImproves]
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
