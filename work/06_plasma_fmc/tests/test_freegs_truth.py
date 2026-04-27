"""Tests for Milestone 9 — FreeGS truth coupling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

TRUTH_PATH = Path(__file__).parent.parent / "results" / "freegs_truth.json"


class TestBaseline:
    def test_results_exist(self):
        assert TRUTH_PATH.exists(), "Run scripts/freegs_truth.py first"

    def test_baseline_solve_fast(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        # FreeGS baseline solve should complete in < 5 sec on M1 Pro
        t = float(d["baseline"]["info"]["solve_time_s"])
        assert t < 5.0, f"FreeGS solve took {t:.1f}s, too slow"

    def test_baseline_Ip_matches(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        Ip = float(d["baseline"]["info"]["I_p_actual"])
        # Solver target was 200 kA — must match within 1%
        assert abs(Ip - 2e5) / 2e5 < 0.01

    def test_baseline_axis_near_R0(self):
        """Magnetic axis must be near TCV's geometric center (R≈0.88)."""
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        R_axis = float(d["baseline"]["shape"]["R_axis"])
        # Expect within 5 cm of TCV major radius
        assert 0.83 < R_axis < 0.93


class TestShapeExtraction:
    def test_shape_keys_present(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        for key in ("R_p", "Z_p", "kappa", "delta", "a"):
            assert key in d["baseline"]["shape"]

    def test_kappa_in_physical_range(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        kappa = float(d["baseline"]["shape"]["kappa"])
        # TCV envelope: 1 < κ < 2.8
        assert 1.0 < kappa < 2.8

    def test_delta_in_physical_range(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        delta = float(d["baseline"]["shape"]["delta"])
        # TCV envelope: -0.7 < δ < +1.0
        assert -0.7 < delta < 1.0


class TestConstraintSensitivity:
    def test_perturbation_runs(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        # At least one perturbation must have succeeded
        sens = d["constraint_perturbation"]
        n_ok = sum(1 for v in sens.values() if "error" not in v)
        assert n_ok >= 1, f"All perturbations failed: {sens}"

    def test_perturbation_changes_shape(self):
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        for k, v in d["constraint_perturbation"].items():
            if "error" in v:
                continue
            ds = v["d_shape"]
            # At least one shape parameter must change non-trivially
            mag = max(abs(ds[k]) for k in ("R_p", "Z_p", "kappa", "delta"))
            assert mag > 1e-4, f"{k}: shape barely changed (max |Δ| = {mag})"

    def test_perturbation_finite_currents(self):
        """Coil current changes must be finite + within engineering limits."""
        if not TRUTH_PATH.exists():
            return
        with open(TRUTH_PATH) as f:
            d = json.load(f)
        for k, v in d["constraint_perturbation"].items():
            if "error" in v:
                continue
            for clabel, dI in v["d_currents"].items():
                assert np.isfinite(dI), f"{k} {clabel}: ΔI = {dI}"
                # Engineering limit: TCV E/F coils 7.7 kA absolute,
                # perturbation can plausibly approach but not wildly exceed
                assert abs(dI) < 50e3, f"{k} {clabel}: ΔI = {dI} A unrealistic"


if __name__ == "__main__":
    n_pass = n_fail = 0
    for cls in [TestBaseline, TestShapeExtraction, TestConstraintSensitivity]:
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
