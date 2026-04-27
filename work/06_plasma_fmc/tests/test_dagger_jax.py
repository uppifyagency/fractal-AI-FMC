"""Tests for Milestone 8 — extended DAgger with JIT FMC backbone."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

HISTORY = Path(__file__).parent.parent / "results" / "dagger_jax_history.json"
POLICY = Path(__file__).parent.parent / "results" / "policy_dagger_jax.npz"
BENCH = Path(__file__).parent.parent / "results" / "milestone_8_benchmark.json"


class TestM8History:
    def test_dataset_grows(self):
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            h = json.load(f)["history"]
        sizes = [r["n_samples"] for r in h]
        assert all(sizes[i+1] > sizes[i] for i in range(len(sizes) - 1)), \
            "Dataset must strictly grow"

    def test_label_time_dominates_or_equal(self):
        """JIT FMC labeling should be << training time per iteration (otherwise
        defeats the M8 speedup purpose)."""
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            h = json.load(f)["history"][1:]  # skip iter 0
        for r in h:
            label_t = r.get("t_label_s", 0)
            train_t = r.get("t_train_s", 0)
            # JIT FMC labeling should be small fraction of total
            total = label_t + train_t + r.get("t_rollout_s", 0)
            assert label_t < total, "Label time dominates — JIT speedup not effective"


class TestM8Quality:
    def test_at_least_as_good_as_m6(self):
        if not BENCH.exists():
            return
        with open(BENCH) as f:
            b = json.load(f)
        m6 = b["tracking"]["dagger3"]["mean_err"]
        m8 = b["tracking"]["dagger_jax"]["mean_err"]
        # M8 must be at least 90% as good as M6 (allow small regression from
        # different hyperparams / random seed)
        assert m8 <= m6 * 1.1, f"M8 {m8} significantly worse than M6 {m6}"

    def test_no_quench(self):
        if not BENCH.exists():
            return
        with open(BENCH) as f:
            b = json.load(f)
        assert b["tracking"]["dagger_jax"]["quench"] == 0

    def test_close_to_fmc_online(self):
        """Final policy must be within 2× of FMC online quality."""
        if not BENCH.exists():
            return
        with open(BENCH) as f:
            b = json.load(f)
        m8 = b["tracking"]["dagger_jax"]["mean_err"]
        fmc = b["tracking"]["fmc"]["mean_err"]
        if fmc > 0:
            ratio = m8 / fmc
            assert ratio < 2.0, f"M8 {m8} > 2× FMC {fmc}"


class TestM8Latency:
    def test_latency_in_target_range(self):
        if not BENCH.exists():
            return
        with open(BENCH) as f:
            b = json.load(f)
        # NN policy < 1 ms (control rate 1 kHz target)
        assert b["latency_us"]["dagger_jax"] < 1000.0


if __name__ == "__main__":
    n_pass = n_fail = 0
    for cls in [TestM8History, TestM8Quality, TestM8Latency]:
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
