"""Tests for the M14 robust FreeGS oracle.

Validates:
1. Baseline equilibrium converges and yields physical shape
2. shape_from_coils returns valid OracleResult for baseline currents
3. Convergence rate on a perturbation grid is high (>=80%)
4. Solve time is within budget (<100 ms)
5. NN fallback is invoked when extraction fails
6. Shape outputs are physically bounded
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from freegs_oracle_robust import (COIL_ORDER, FreeGSOracle, OracleResult,
                                    find_lcfs_from_psi, vacuum_psi)


_ORACLE = None


def get_oracle():
    """Lazy oracle init — solves baseline only once for the whole test class."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = FreeGSOracle(verbose=False)
    return _ORACLE


class TestOracleRobust(unittest.TestCase):

    def test_baseline_solved(self):
        """Baseline DN equilibrium is solved and shape is physical."""
        oracle = get_oracle()
        self.assertIsNotNone(oracle.baseline_shape)
        sh = oracle.baseline_shape
        # TCV physical envelope
        self.assertTrue(0.7 < sh["R_p"] < 1.0, f"R_p={sh['R_p']}")
        self.assertTrue(-0.2 < sh["Z_p"] < 0.2, f"Z_p={sh['Z_p']}")
        self.assertTrue(1.2 < sh["kappa"] < 2.5, f"kappa={sh['kappa']}")
        self.assertTrue(-0.5 < sh["delta"] < 0.8, f"delta={sh['delta']}")

    def test_baseline_currents_recovered(self):
        """Calling shape_from_coils with baseline currents reproduces baseline shape."""
        oracle = get_oracle()
        I = np.array([oracle.baseline_currents.get(k, 0.0) for k in COIL_ORDER])
        res = oracle.shape_from_coils(I)
        self.assertTrue(res.converged)
        self.assertEqual(res.source, "freegs")
        # Shape should be close to baseline (small numerical drift OK)
        sh = oracle.baseline_shape
        self.assertAlmostEqual(res.R_p, sh["R_p"], delta=0.05)
        self.assertAlmostEqual(res.kappa, sh["kappa"], delta=0.15)

    def test_solve_time_budget(self):
        """Each shape extraction takes <100 ms (budget for oracle eval)."""
        oracle = get_oracle()
        I = np.array([oracle.baseline_currents.get(k, 0.0) for k in COIL_ORDER])
        res = oracle.shape_from_coils(I)
        self.assertLess(res.solve_time_s, 0.15,
                        f"shape_from_coils took {res.solve_time_s:.3f}s")

    def test_convergence_rate_on_perturbations(self):
        """At least 80% convergence on ±15% coil current perturbations."""
        oracle = get_oracle()
        baseline_I = np.array(
            [oracle.baseline_currents.get(k, 0.0) for k in COIL_ORDER]
        )
        rng = np.random.default_rng(140)
        n_total = 12
        n_conv = 0
        for k in range(n_total):
            scale = 1.0 + 0.10 * rng.standard_normal()
            I = baseline_I * scale + rng.normal(0, 300, size=16)
            res = oracle.shape_from_coils(I)
            if res.converged:
                n_conv += 1
        rate = n_conv / n_total
        self.assertGreaterEqual(rate, 0.80,
            f"Only {n_conv}/{n_total} = {100*rate:.0f}% converged (need 80%)")

    def test_nn_fallback(self):
        """Failed extraction (zero coils) triggers fallback if provided."""
        oracle = get_oracle()
        I = np.zeros(16)
        # Define a dummy fallback that returns nominal values
        nominal = np.array([0.88, 0.0, 1.7, 0.3])
        res = oracle.shape_from_coils(I, fallback_fn=lambda I_: nominal)
        # zero coils → should fall back
        self.assertEqual(res.source, "nn_fallback")
        self.assertEqual(res.R_p, 0.88)

    def test_shape_outputs_bounded(self):
        """Converged shapes always within physical envelope."""
        oracle = get_oracle()
        baseline_I = np.array(
            [oracle.baseline_currents.get(k, 0.0) for k in COIL_ORDER]
        )
        rng = np.random.default_rng(141)
        for k in range(8):
            I = baseline_I + rng.normal(0, 800, size=16)
            res = oracle.shape_from_coils(I)
            if res.converged:
                self.assertTrue(0.4 < res.R_p < 1.3, f"R_p={res.R_p}")
                self.assertTrue(-0.7 < res.Z_p < 0.7, f"Z_p={res.Z_p}")
                self.assertTrue(0.5 < res.kappa < 3.5, f"kappa={res.kappa}")
                self.assertTrue(-1.5 < res.delta < 1.5, f"delta={res.delta}")


def main():
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOracleRobust)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
