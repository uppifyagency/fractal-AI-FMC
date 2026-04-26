"""Mathematical correctness tests for the plasma simulator.

Run with: python -m pytest tests/test_simulator.py -v
or:       python tests/test_simulator.py

These tests verify *mathematical* properties (symmetry, conservation,
analytical limits). Tests of physical accuracy at TCV operating points
require the full GS coupling (Milestone 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable regardless of where pytest is run from
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from mutual_inductance import (
    MU_0,
    mutual_inductance_pair,
    mutual_matrix,
    mutual_to_plasma,
    self_inductance,
)
from plasma_simulator import (
    Control,
    PlasmaSimulator,
    PlasmaState,
    build_default_simulator,
    spitzer_resistance,
    tau_E_IPB98,
    temperature_from_W,
)
from tcv_geometry import load_tcv


# ============================================================
# Mutual inductance properties
# ============================================================

class TestMutualInductance:
    def test_symmetry(self):
        """M(R1, R2, dZ) == M(R2, R1, dZ)."""
        a = mutual_inductance_pair(0.5, 1.5, 0.3)
        b = mutual_inductance_pair(1.5, 0.5, 0.3)
        assert abs(a - b) < 1e-15

    def test_distance_decay(self):
        """M decreases as separation grows."""
        m_close = mutual_inductance_pair(1.0, 1.0, 0.1)
        m_mid = mutual_inductance_pair(1.0, 1.0, 1.0)
        m_far = mutual_inductance_pair(1.0, 1.0, 100.0)
        assert m_close > m_mid > m_far > 0
        assert m_far < 1e-9  # essentially zero at 100m

    def test_reference_value(self):
        """Two identical 1m loops at 1m sep:
           k² = 4/5 = 0.8 → M = μ₀ · 0.394 ≈ 4.95e-7 H."""
        m = mutual_inductance_pair(1.0, 1.0, 1.0)
        assert abs(m - 4.95e-7) / 4.95e-7 < 0.01

    def test_self_inductance_positive(self):
        L = self_inductance(1.0, 0.01)
        assert L > 0
        assert 5e-6 < L < 1e-5  # ~6 µH expected

    def test_matrix_symmetric(self):
        R = np.array([0.5, 1.0, 1.5, 0.8])
        Z = np.array([-0.3, 0.0, 0.5, 0.2])
        M = mutual_matrix(R, Z)
        assert M.shape == (4, 4)
        assert np.max(np.abs(M - M.T)) < 1e-15

    def test_matrix_positive_definite(self):
        """Inductance matrix must be PD (all eigenvalues > 0).
        Required for the implicit-Euler solve to be stable."""
        tcv = load_tcv()
        R = np.array([c.R for c in tcv.shaping_coils])
        Z = np.array([c.Z for c in tcv.shaping_coils])
        M = mutual_matrix(R, Z)
        eigs = np.linalg.eigvalsh(M)
        assert eigs.min() > 0, f"Min eigenvalue: {eigs.min()}"

    def test_diagonal_dominant_self_term(self):
        """Self-inductance >> any pairwise mutual for typical TCV coil sep."""
        tcv = load_tcv()
        R = np.array([c.R for c in tcv.shaping_coils])
        Z = np.array([c.Z for c in tcv.shaping_coils])
        M = mutual_matrix(R, Z)
        for i in range(M.shape[0]):
            row = M[i].copy()
            row[i] = 0
            assert M[i, i] > np.max(np.abs(row)), (
                f"row {i}: diag={M[i,i]:.3e}, max off-diag={np.max(np.abs(row)):.3e}"
            )


# ============================================================
# Energy / temperature relations
# ============================================================

class TestEnergyTemperature:
    def test_W_T_inversion(self):
        """T = W/(3 n V) round-trip."""
        n = 5e19
        V = 1.7
        T_keV_target = 1.5
        # W = 3 n V T  with T in J
        W = 3 * n * V * (T_keV_target * 1e3 * 1.602176634e-19)
        T = temperature_from_W(W, n, V)
        assert abs(T - T_keV_target) < 1e-9

    def test_T_zero_for_empty_plasma(self):
        assert temperature_from_W(1000.0, 0.0, 1.0) == 0.0
        assert temperature_from_W(1000.0, 1e19, 0.0) == 0.0

    def test_spitzer_scaling(self):
        """η ∝ T^(-3/2): doubling T should drop R by factor 2^(3/2)."""
        R1 = spitzer_resistance(1.0, 0.88, 0.24, 1.7)
        R2 = spitzer_resistance(2.0, 0.88, 0.24, 1.7)
        ratio = R1 / R2
        expected = 2.0**1.5
        assert abs(ratio - expected) / expected < 1e-9


# ============================================================
# IPB98 confinement scaling
# ============================================================

class TestIPB98:
    def test_basic_scaling(self):
        """Check known scaling exponents."""
        base = tau_E_IPB98(0.2, 1.43, 1.0, 5.0, 0.88, 0.284, 1.7)
        # Double I_p: τ_E should grow by 2^0.93
        scaled_I = tau_E_IPB98(0.4, 1.43, 1.0, 5.0, 0.88, 0.284, 1.7)
        assert abs(scaled_I / base - 2**0.93) / (2**0.93) < 1e-9

        # Double power: τ_E should drop by 2^-0.69
        scaled_P = tau_E_IPB98(0.2, 1.43, 2.0, 5.0, 0.88, 0.284, 1.7)
        assert abs(scaled_P / base - 2**(-0.69)) / (2**(-0.69)) < 1e-9

    def test_zero_inputs_safe(self):
        """Edge cases must not crash."""
        assert tau_E_IPB98(0.2, 1.43, 0.0, 5.0, 0.88, 0.284, 1.7) == 0.0
        assert tau_E_IPB98(0.2, 1.43, 1.0, 0.0, 0.88, 0.284, 1.7) == 0.0

    def test_tcv_typical_value(self):
        """For TCV typical (200 kA, 1 MW heating, n=5e19), τ_E ~ tens of ms.
        IPB98 returns the ELMy H-mode confinement time."""
        tau = tau_E_IPB98(
            I_p_MA=0.2, B_T=1.43, P_loss_MW=1.0, n_e_e19=5.0,
            R_p=0.88, eps=0.284, kappa=1.7,
        )
        # H98=1 baseline; for low-power TCV ohmic-only, tau ~ 5-50 ms
        assert 1e-3 < tau < 1.0, f"tau_E = {tau} s out of expected range"


# ============================================================
# Simulator one-step properties
# ============================================================

class TestSimulator:
    @pytest.fixture
    def sim_state(self):
        return build_default_simulator()

    def test_initial_state_consistent(self, sim_state):
        sim, state = sim_state
        # Initial T_e ~ 1 keV
        d = sim.diagnostics(state)
        assert 0.5 < d["T_e_keV"] < 2.0

    def test_circuit_zero_voltage_decays(self, sim_state):
        """With V=0, |I_coils| must monotonically decrease (resistive decay)."""
        sim, state = sim_state
        ctrl = Control(V_coils=np.zeros(sim.N))
        # Use small dt to isolate circuit decay (avoid plasma feedback artifact
        # from the very first step's large dI/dt)
        I_norm_prev = np.linalg.norm(state.I_coils)
        s = state.copy()
        for _ in range(5):
            s = sim.step(s, ctrl, dt=1e-4)
            I_norm = np.linalg.norm(s.I_coils)
            assert I_norm < I_norm_prev, "Currents not decaying with V=0"
            I_norm_prev = I_norm

    def test_circuit_steady_state(self, sim_state):
        """With V = R·I_ref, coil currents must stay at I_ref (steady state)."""
        sim, state = sim_state
        V_ss = sim.R_diag * sim.I_ref
        ctrl = Control(V_coils=V_ss)
        # Start exactly at I_ref
        s = state.copy()
        s.I_coils = sim.I_ref.copy()
        for _ in range(10):
            s = sim.step(s, ctrl, dt=1e-3)
        max_drift = np.max(np.abs(s.I_coils - sim.I_ref))
        assert max_drift < 1e-6, (
            f"Coils drifted from steady state by {max_drift} A "
            f"(should be ~0 for V=R·I)"
        )

    def test_state_immutability(self, sim_state):
        """step() must NOT mutate the input state."""
        sim, state = sim_state
        ctrl = Control(V_coils=np.ones(sim.N))
        I_before = state.I_coils.copy()
        W_before = state.W
        _ = sim.step(state, ctrl, dt=1e-3)
        assert np.array_equal(state.I_coils, I_before)
        assert state.W == W_before

    def test_shape_response_signs(self, sim_state):
        """Linearized shape response must respect physical signs:
        - Increasing F-up - F-down should move plasma UP (Z_p > 0)
        - Increasing F symmetric should ELONGATE (kappa > ref)
        """
        sim, state = sim_state
        # Z-shift test: F1 (Z=-0.77) decreases by 1000A, F8 (Z=+0.77) increases
        dI = np.zeros(sim.N)
        # F1 is index 8, F8 is index 15
        dI[15] = +1000.0  # F8 up
        dI[8] = -1000.0   # F1 down
        delta_shape = sim.S @ dI
        assert delta_shape[1] > 0, f"Z shift should be positive, got {delta_shape[1]}"

        # Elongation test: all F coils +1000A symmetrically
        dI = np.zeros(sim.N)
        for i in range(8, 16):  # F1-F8
            dI[i] = +1000.0
        delta_shape = sim.S @ dI
        assert delta_shape[2] > 0, f"κ change should be positive, got {delta_shape[2]}"

    def test_purity(self, sim_state):
        """Same input state + same control → same output (deterministic)."""
        sim, state = sim_state
        ctrl = Control(V_coils=np.linspace(-100, 100, sim.N), P_aux=5e5)
        s1 = sim.step(state, ctrl, 1e-3)
        s2 = sim.step(state, ctrl, 1e-3)
        assert np.array_equal(s1.I_coils, s2.I_coils)
        assert s1.W == s2.W
        assert s1.I_p == s2.I_p

    def test_plasma_volume_consistency(self, sim_state):
        """V_plasma = 2π² R_p a² κ — diagnostic must match formula."""
        sim, state = sim_state
        d = sim.diagnostics(state)
        a_eff = sim.tcv.a_minor * 0.96
        V_expected = 2.0 * np.pi**2 * state.R_p * a_eff**2 * state.kappa
        assert abs(d["V_plasma_m3"] - V_expected) < 1e-9


# ============================================================
# Aggregate sanity
# ============================================================

def test_mutual_to_plasma_decreases_with_distance():
    R_coils = np.array([0.5, 0.5])
    Z_coils = np.array([0.1, 1.0])
    M_pc = mutual_to_plasma(R_coils, Z_coils, R_p=0.88, Z_p=0.0)
    assert M_pc[0] > M_pc[1] > 0


if __name__ == "__main__":
    # Run as script: provides clear pass/fail per test
    import unittest

    # Convert pytest classes to unittest by inspection
    test_classes = [
        TestMutualInductance, TestEnergyTemperature, TestIPB98, TestSimulator,
    ]
    standalone_tests = [test_mutual_to_plasma_decreases_with_distance]

    n_pass = 0
    n_fail = 0
    sim_state = build_default_simulator()

    for cls in test_classes:
        instance = cls()
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            try:
                fn = getattr(instance, attr)
                # Inject sim_state if signature requires it
                import inspect
                sig = inspect.signature(fn)
                if "sim_state" in sig.parameters:
                    fn(sim_state)
                else:
                    fn()
                print(f"  ✓ {cls.__name__}.{attr}")
                n_pass += 1
            except AssertionError as e:
                print(f"  ✗ {cls.__name__}.{attr}: {e}")
                n_fail += 1
            except Exception as e:
                print(f"  ✗ {cls.__name__}.{attr}: {type(e).__name__}: {e}")
                n_fail += 1

    for fn in standalone_tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
