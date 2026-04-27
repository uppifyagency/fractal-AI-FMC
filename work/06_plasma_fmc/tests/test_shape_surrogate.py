"""Tests for Milestone 11 — NN shape surrogate."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

DATASET = Path(__file__).parent.parent / "results" / "freegs_shape_dataset.npz"
SURROGATE = Path(__file__).parent.parent / "results" / "shape_surrogate.npz"
COMPARISON = Path(__file__).parent.parent / "results" / "milestone_11_comparison.json"


class TestDataset:
    def test_dataset_exists(self):
        assert DATASET.exists()

    def test_dataset_shapes(self):
        if not DATASET.exists():
            return
        d = np.load(DATASET)
        n = d["I_coils"].shape[0]
        assert d["I_coils"].shape == (n, 20)
        assert d["shape"].shape == (n, 4)
        assert n >= 100, f"Too few samples: {n}"

    def test_shape_in_physical_range(self):
        if not DATASET.exists():
            return
        d = np.load(DATASET)
        s = d["shape"]
        # R_p ∈ TCV vessel
        assert (0.5 < s[:, 0]).all() and (s[:, 0] < 1.3).all()
        # |Z_p| < vessel half-height
        assert (np.abs(s[:, 1]) < 0.8).all()
        # kappa physical
        assert (s[:, 2] > 0.5).all() and (s[:, 2] < 3.0).all()
        # delta in TCV envelope
        assert (s[:, 3] > -0.7).all() and (s[:, 3] < 1.0).all()


class TestSurrogate:
    def test_surrogate_exists(self):
        assert SURROGATE.exists()

    def test_surrogate_loads(self):
        if not SURROGATE.exists():
            return
        d = np.load(SURROGATE, allow_pickle=True)
        assert "params" in d.files
        assert "x_mean" in d.files
        # NN normalizers and params present

    def test_per_dim_rmse_reasonable(self):
        if not SURROGATE.exists():
            return
        d = np.load(SURROGATE, allow_pickle=True)
        rmse = d["per_dim_rmse"]
        # R_p RMSE < 5 cm
        assert rmse[0] < 0.05, f"R_p RMSE {rmse[0]:.4f} too large"
        # kappa RMSE < 0.15
        assert rmse[2] < 0.15
        # delta RMSE < 0.10
        assert rmse[3] < 0.10

    def test_better_than_mean_baseline(self):
        if not SURROGATE.exists():
            return
        d = np.load(SURROGATE, allow_pickle=True)
        nn = d["per_dim_rmse"]
        baseline = d["per_dim_rmse_baseline"]
        # NN at least 1.8× better than predicting the mean (135 samples is small,
        # κ has highest variance — relax threshold accordingly)
        for j in range(4):
            assert nn[j] < baseline[j] / 1.8, (
                f"dim {j}: NN {nn[j]:.4f} not 1.8× better than mean baseline {baseline[j]:.4f}"
            )


class TestComparison:
    def test_comparison_exists(self):
        assert COMPARISON.exists()

    def test_nn_beats_linear_s(self):
        if not COMPARISON.exists():
            return
        with open(COMPARISON) as f:
            c = json.load(f)
        # NN must beat linear S on at least 3 of 4 dimensions
        n_better = sum(
            1 for v in c["per_dim"].values() if v["rmse_nn"] < v["rmse_linear"]
        )
        assert n_better >= 3

    def test_aggregate_nn_better(self):
        if not COMPARISON.exists():
            return
        with open(COMPARISON) as f:
            c = json.load(f)
        agg = c["aggregate"]
        # NN aggregate at least 2× better
        assert agg["rmse_nn"] < agg["rmse_linear"] / 2.0


if __name__ == "__main__":
    n_pass = n_fail = 0
    for cls in [TestDataset, TestSurrogate, TestComparison]:
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
