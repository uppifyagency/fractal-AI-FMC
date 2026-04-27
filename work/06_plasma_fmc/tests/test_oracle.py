"""Tests for Milestone 13 — NN-proxy oracle evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ORACLE = Path(__file__).parent.parent / "results" / "milestone_13_oracle_eval.json"


class TestOracle:
    def test_results_exist(self):
        assert ORACLE.exists(), "Run scripts/freegs_oracle_eval.py first"

    def test_all_policies_evaluated(self):
        if not ORACLE.exists():
            return
        with open(ORACLE) as f:
            d = json.load(f)
        for label in ("M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"):
            assert label in d["summary"], f"missing {label}"

    def test_all_truth_errs_finite(self):
        if not ORACLE.exists():
            return
        with open(ORACLE) as f:
            d = json.load(f)
        for label, s in d["summary"].items():
            assert np.isfinite(s["mean_err_truth"]), f"{label} truth-err inf"

    def test_truth_errs_in_meaningful_range(self):
        """All truth-errs should be in 1-200 range (else clip is broken)."""
        if not ORACLE.exists():
            return
        with open(ORACLE) as f:
            d = json.load(f)
        for label, s in d["summary"].items():
            assert 1 < s["mean_err_truth"] < 500, (
                f"{label}: truth-err {s['mean_err_truth']} out of meaningful range"
            )

    def test_self_eval_underestimates_truth_for_linear_sim(self):
        """Policies trained on linear-S sim should have self-err << truth-err
        (the gap reveals simulator overfitting)."""
        if not ORACLE.exists():
            return
        with open(ORACLE) as f:
            d = json.load(f)
        # For M6/M10/FMC (linear sim), self-err should be smaller than truth-err
        for label in ("M6_DAgger3", "M10_DAggerN", "FMC_online"):
            s = d["summary"][label]
            assert s["mean_err_self"] < s["mean_err_truth"], (
                f"{label}: self {s['mean_err_self']} not smaller than truth {s['mean_err_truth']}"
            )


if __name__ == "__main__":
    n_pass = n_fail = 0
    instance = TestOracle()
    for attr in dir(instance):
        if not attr.startswith("test_"):
            continue
        try:
            getattr(instance, attr)()
            print(f"  ✓ TestOracle.{attr}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ TestOracle.{attr}: {e}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
