"""Tests for M16 real TCV experimental validation."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from m16_tcv_x21 import EQDSK_PATH, load_real_tcv_lcfs

RESULTS_DIR = Path(__file__).parent.parent / "results"


class TestM16RealTCV(unittest.TestCase):

    def test_eqdsk_file_present(self):
        """TCV-X21 eqdsk file is present in data/tcv_x21/."""
        self.assertTrue(EQDSK_PATH.exists(),
                        f"{EQDSK_PATH} missing — download from TCV-X21")
        self.assertGreater(EQDSK_PATH.stat().st_size, 1_000_000,
                           "eqdsk should be > 1 MB")

    def test_load_real_tcv(self):
        """Real TCV shape extracted with sensible values."""
        shape = load_real_tcv_lcfs()
        # Physical envelope
        self.assertTrue(0.7 < shape["R_p"] < 1.0, f"R_p={shape['R_p']}")
        self.assertTrue(-0.3 < shape["Z_p"] < 0.3, f"Z_p={shape['Z_p']}")
        self.assertTrue(0.15 < shape["a"] < 0.30, f"a={shape['a']}")
        self.assertTrue(1.3 < shape["kappa"] < 2.0, f"κ={shape['kappa']}")
        self.assertTrue(-0.5 < shape["delta"] < 0.5, f"δ={shape['delta']}")
        # Real TCV-X21 was an L-mode shot
        self.assertGreater(abs(shape["I_p"]), 100_000,
                           "I_p should be > 100 kA")

    def test_real_target_in_oracle_envelope(self):
        """Real TCV target is within M14 oracle's tested envelope."""
        shape = load_real_tcv_lcfs()
        self.assertTrue(0.7 < shape["R_p"] < 1.0)
        self.assertTrue(-0.2 < shape["Z_p"] < 0.2)
        self.assertTrue(1.2 < shape["kappa"] < 2.5)
        self.assertTrue(-0.7 < shape["delta"] < 0.8)

    def test_results_json_present(self):
        """milestone_16_real_tcv.json exists after eval run."""
        path = RESULTS_DIR / "milestone_16_real_tcv.json"
        if not path.exists():
            self.skipTest("Run scripts/m16_tcv_x21.py first")
        with open(path) as f:
            d = json.load(f)
        self.assertIn("real_shape", d)
        self.assertIn("policy_results", d)
        self.assertEqual(len(d["policy_results"]), 5)

    def test_best_policy_achieves_target(self):
        """At least one policy achieves steady-state truth-err <= 10."""
        path = RESULTS_DIR / "milestone_16_real_tcv.json"
        if not path.exists():
            self.skipTest("Run scripts/m16_tcv_x21.py first")
        with open(path) as f:
            d = json.load(f)
        best_steady = min(
            d["policy_results"][p]["steady_state_truth_err_last10"]
            for p in d["policy_results"]
        )
        self.assertLessEqual(best_steady, 10.0,
            f"No policy met steady-state quality bar: "
            f"best = {best_steady:.2f}")

    def test_top_policy_achieves_high_physicality(self):
        """Best policy by truth-err also has physicality >= 90%."""
        path = RESULTS_DIR / "milestone_16_real_tcv.json"
        if not path.exists():
            self.skipTest("Run scripts/m16_tcv_x21.py first")
        with open(path) as f:
            d = json.load(f)
        best_label = min(
            d["policy_results"],
            key=lambda p: d["policy_results"][p]["steady_state_truth_err_last10"],
        )
        best_phys = d["policy_results"][best_label]["physicality"]
        self.assertGreaterEqual(best_phys, 0.90,
            f"Best policy {best_label} has phys = {100*best_phys:.0f}% < 90%")


def main():
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestM16RealTCV)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
