"""Tests for M15 published-target benchmark.

Validates:
1. All 6 scenarios are loadable and have valid waypoints
2. discretize() produces correctly-sized trajectories
3. Targets stay within M14 oracle physical envelope
4. Eval results JSON has all 5 policies × 6 scenarios populated
5. Best policy aggregate truth-err is below 'comparable to GS RMSE' bar (~10)
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tcv_published_targets import (PUBLISHED_TARGETS, discretize, get_scenario,
                                    list_scenarios)


RESULTS_DIR = Path(__file__).parent.parent / "results"


class TestM15Published(unittest.TestCase):

    def test_all_scenarios_loadable(self):
        """All 6 published targets are loaded with metadata."""
        names = list_scenarios()
        self.assertEqual(len(names), 6)
        for n in names:
            s = get_scenario(n)
            self.assertGreater(len(s.waypoints), 1)
            self.assertGreater(s.duration_s, 0)
            self.assertNotEqual(s.citation, "")

    def test_targets_within_oracle_envelope(self):
        """All target waypoints fall within M14 oracle's tested envelope."""
        # Envelope verified in M14 grid:
        #   R_p ∈ [0.7, 1.0], Z_p ∈ [-0.2, 0.2],
        #   κ ∈ [1.2, 2.5], δ ∈ [-0.7, 0.8]
        for s in PUBLISHED_TARGETS:
            for t, target in s.waypoints:
                R, Z, K, D = target
                self.assertTrue(0.7 <= R <= 1.0,
                                f"{s.name} R={R} out of envelope")
                self.assertTrue(-0.2 <= Z <= 0.2,
                                f"{s.name} Z={Z} out of envelope")
                self.assertTrue(1.2 <= K <= 2.5,
                                f"{s.name} κ={K} out of envelope")
                self.assertTrue(-0.7 <= D <= 0.8,
                                f"{s.name} δ={D} out of envelope")

    def test_discretize_correctness(self):
        """discretize() produces trajectories of expected length and bounds."""
        for s in PUBLISHED_TARGETS:
            t_arr, targets = discretize(s, dt=0.05)
            n_expected = int(np.ceil(s.duration_s / 0.05)) + 1
            self.assertEqual(len(t_arr), n_expected, s.name)
            self.assertEqual(targets.shape, (n_expected, 4))
            # First and last waypoint match
            np.testing.assert_array_almost_equal(targets[0],
                                                  s.waypoints[0][1])
            np.testing.assert_array_almost_equal(targets[-1],
                                                  s.waypoints[-1][1])

    def test_eval_results_complete(self):
        """milestone_15_published_eval.json has all policies × scenarios."""
        path = RESULTS_DIR / "milestone_15_published_eval.json"
        if not path.exists():
            self.skipTest("Run scripts/m15_eval_published.py first")
        with open(path) as f:
            d = json.load(f)
        self.assertIn("aggregate", d)
        self.assertIn("scenarios", d)
        self.assertEqual(len(d["scenarios"]), 6)
        for s_name, s in d["scenarios"].items():
            self.assertEqual(len(s["policies"]), 5,
                             f"scenario {s_name} missing policies")

    def test_best_policy_meets_quality_bar(self):
        """At least one policy aggregate truth-err <= 10 (deployable threshold)."""
        path = RESULTS_DIR / "milestone_15_published_eval.json"
        if not path.exists():
            self.skipTest("Run scripts/m15_eval_published.py first")
        with open(path) as f:
            d = json.load(f)
        best_truth = min(d["aggregate"][p]["mean_truth_across_scenarios"]
                          for p in d["aggregate"])
        self.assertLessEqual(best_truth, 10.0,
            f"Best policy truth-err = {best_truth:.2f} > 10 threshold")

    def test_physicality_separates_policies(self):
        """Top-3 policies have phys >= 80%, bottom-2 < 50%."""
        path = RESULTS_DIR / "milestone_15_published_eval.json"
        if not path.exists():
            self.skipTest("Run scripts/m15_eval_published.py first")
        with open(path) as f:
            d = json.load(f)
        policies_sorted = sorted(d["aggregate"].items(),
                                  key=lambda kv: -kv[1]["mean_physicality"])
        top3 = policies_sorted[:3]
        bottom2 = policies_sorted[-2:]
        for label, agg in top3:
            self.assertGreaterEqual(agg["mean_physicality"], 0.80,
                f"top-3 {label} phys = {100*agg['mean_physicality']:.0f}%")
        for label, agg in bottom2:
            self.assertLess(agg["mean_physicality"], 0.50,
                f"bottom-2 {label} phys = {100*agg['mean_physicality']:.0f}%")


def main():
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestM15Published)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
