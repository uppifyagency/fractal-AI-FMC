"""Tests for Milestone 10 — calibrated simulator + pipeline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from calibrated_sim import (
    M9_BASELINE, build_calibrated_jax_params, calibrated_target_ranges,
)

HISTORY = Path(__file__).parent.parent / "results" / "calibrated_history.json"
POLICY = Path(__file__).parent.parent / "results" / "policy_calibrated.npz"


class TestCalibratedSim:
    def test_ref_matches_m9(self):
        sim_p, _ = build_calibrated_jax_params()
        assert abs(sim_p.kappa_ref - M9_BASELINE["kappa"]) < 1e-3
        assert abs(sim_p.delta_ref - M9_BASELINE["delta"]) < 1e-3
        assert abs(sim_p.R_ref - M9_BASELINE["R_p"]) < 1e-3

    def test_S_scaled(self):
        sim_p, _ = build_calibrated_jax_params(s_scale=10.0)
        # Max S coeff with s_scale=10 should be 10× larger than baseline 2e-6
        max_S = float(np.max(np.abs(np.asarray(sim_p.S))))
        assert max_S > 1e-5, f"S not scaled up: max={max_S:.2e}"

    def test_initial_state_consistent(self):
        sim_p, x0 = build_calibrated_jax_params()
        N = sim_p.N
        # Initial R_p, kappa, delta in state must match ref
        assert abs(float(x0[N + 3]) - sim_p.R_ref) < 1e-3
        assert abs(float(x0[N + 5]) - sim_p.kappa_ref) < 1e-3

    def test_target_ranges_around_ref(self):
        sim_p, _ = build_calibrated_jax_params()
        ranges = calibrated_target_ranges()
        # Each range must straddle the reference
        for k_short, k_full in [("kappa", "kappa"), ("delta", "delta")]:
            lo, hi = ranges[k_short]
            ref = getattr(sim_p, f"{k_full}_ref")
            assert lo <= ref <= hi, f"{k_short} ref {ref} not in [{lo}, {hi}]"


class TestCalibratedPipeline:
    def test_history_exists(self):
        assert HISTORY.exists(), "Run scripts/calibrated_pipeline.py first"

    def test_dagger_improves_bc(self):
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            d = json.load(f)
        bc_err = d["history"][0]["mean_err"]
        last_err = d["history"][-1]["mean_err"]
        # DAgger should reduce error vs BC alone
        assert last_err < bc_err / 2.0, (
            f"DAgger ({last_err:.2f}) not at least 2× better than BC ({bc_err:.2f})"
        )

    def test_no_quench_after_dagger(self):
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            d = json.load(f)
        # Final iter should have 0 quenches
        assert d["history"][-1]["quench"] == 0


if __name__ == "__main__":
    n_pass = n_fail = 0
    for cls in [TestCalibratedSim, TestCalibratedPipeline]:
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
