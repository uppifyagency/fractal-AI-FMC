"""mpc_baseline_controller.py — Linear MPC / DLQR baseline for plasma shape control.

Falsifiable claim under test (M21 milestone closing question):
  Linear MPC computed via DLQR on the calibrated S matrix beats the M21 BEST
  FMC controller (steady_truth_err 0.58 on M16 TCV-X21 65402) at lower latency
  with zero seed variance.

Math:
  Coil sub-system (linearized around M9 baseline):
    M_cc · dI/dt + R_diag · I = V
    Forward Euler discretization, Δt = dt:
      I[t+1] = (I - dt · M_cc⁻¹ · R_diag) I[t] + (dt · M_cc⁻¹) V[t]
            = A · I[t] + B · V[t]

  Shape model (linearized):
    shape = ref + S · (I - I_ref)   ⇒   I_target = I_ref + S⁺ · (target - ref)
    Steady-state at I_target requires V_target = R_diag · I_target

  LQR cost (penalize shape error squared, weighted by W):
    J = Σ ‖shape_err‖²_W + ρ ‖ΔV‖²
      = Σ ΔIᵀ Sᵀ W S ΔI + ρ ΔVᵀ ΔV
    Q = SᵀWS + ε·I  (ε for conditioning)
    R = ρ · I_N

  Solve DARE → P, gain K = (R + BᵀPB)⁻¹ BᵀPA
  Control law: V[t] = V_target - K · (I[t] - I_target)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import sys
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
from scipy.linalg import solve_discrete_are


# H1 hypothesis: voltage rails ±1500 V on E/F coils (REFERENCES.md)
V_LIMIT = 1500.0


class LinearMPCController:
    """Deterministic DLQR controller. .decide() returns voltages computed
    in O(N²) (single matrix-vector product). On a 20-coil TCV that's
    ~50µs Python overhead, ~µs in compiled BLAS. Zero seed variance.
    """

    def __init__(self, sim_p, *, dt: float = 1e-3,
                 R_weight: float = 1e-6, eps_Q: float = 1e-6,
                 P_aux: float = 5e5, gas_puff: float = 1e21,
                 voltage_limit: float = V_LIMIT, seed: int = 0):
        # Required attributes for the eval harness
        self.dt = dt
        self.P_aux = P_aux
        self.gas_puff = gas_puff
        self.voltage_limit = voltage_limit

        # Extract sim matrices (numpy from JAX)
        N = int(sim_p.N)
        M_cc = np.asarray(sim_p.M_cc, dtype=np.float64)
        R_diag = np.asarray(sim_p.R_diag, dtype=np.float64)  # shape (N,)
        S = np.asarray(sim_p.S, dtype=np.float64)            # shape (4, N)
        I_ref = np.asarray(sim_p.I_ref, dtype=np.float64)

        # State-space coil dynamics (forward Euler discretization)
        M_cc_inv = np.linalg.inv(M_cc)
        A = np.eye(N) - dt * (M_cc_inv @ np.diag(R_diag))
        B = dt * M_cc_inv

        # Shape weights (must match harness shape_err_np weights)
        W = np.diag([100.0, 100.0, 10.0, 10.0])

        # State cost: minimise (S ΔI)ᵀ W (S ΔI) = ΔIᵀ Sᵀ W S ΔI
        # Add small ε I for conditioning (Q is rank≤4 in 20-D state)
        Q = S.T @ W @ S + eps_Q * np.eye(N)
        R = R_weight * np.eye(N)

        # Solve DARE for P, then K
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

        # Reference frame
        ref_shape = np.array([
            float(sim_p.R_ref), float(sim_p.Z_ref),
            float(sim_p.kappa_ref), float(sim_p.delta_ref),
        ])
        S_pinv = np.linalg.pinv(S)  # 20×4 (Moore-Penrose)

        self.A = A
        self.B = B
        self.K = K
        self.S = S
        self.S_pinv = S_pinv
        self.I_ref = I_ref
        self.R_diag = R_diag
        self.ref_shape = ref_shape
        self.N = N
        # diagnostics
        self.K_norm = float(np.linalg.norm(K, ord=2))
        self.A_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        self.A_BK_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(A - B @ K)))
        )

    def decide(self, state, target_4):
        """state: (27,) packed plasma state. target_4: (4,) shape target.
        Returns dict with 'V_coils': (N,) float32 voltages.
        """
        I = np.asarray(state[:self.N], dtype=np.float64)
        target = np.asarray(target_4, dtype=np.float64)

        # Map shape target → coil currents that achieve it (linear inversion)
        I_target = self.I_ref + self.S_pinv @ (target - self.ref_shape)

        # Steady-state voltage to hold I_target
        V_target = self.R_diag * I_target

        # LQR feedback law
        V = V_target - self.K @ (I - I_target)

        # Hardware safety clip (H1: ±1500 V coil rails)
        V = np.clip(V, -self.voltage_limit, self.voltage_limit)

        return {"V_coils": np.asarray(V, dtype=np.float32)}


def diagnostics(sim_p, **kwargs):
    """Print MPC controller stability diagnostics."""
    ctrl = LinearMPCController(sim_p, **kwargs)
    print(f"MPC diagnostics:")
    print(f"  N coils                = {ctrl.N}")
    print(f"  dt                     = {ctrl.dt}")
    print(f"  open-loop ρ(A)         = {ctrl.A_spectral_radius:.6f}")
    print(f"  closed-loop ρ(A-BK)    = {ctrl.A_BK_spectral_radius:.6f}  (must be < 1)")
    print(f"  ‖K‖₂                   = {ctrl.K_norm:.3e}")
    print(f"  voltage limit          = ±{ctrl.voltage_limit:.0f} V (H1 hypothesis)")
    return ctrl


if __name__ == "__main__":
    from calibrated_sim import build_calibrated_jax_params
    sim_p, x0 = build_calibrated_jax_params()
    print("=" * 60)
    print("Linear MPC baseline controller — calibrated TCV sim")
    print("=" * 60)
    ctrl = diagnostics(sim_p)

    # Latency test
    import time
    target = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)
    state = np.asarray(x0)
    # warmup
    for _ in range(5):
        ctrl.decide(state, target)
    t0 = time.perf_counter()
    n_iter = 1000
    for _ in range(n_iter):
        out = ctrl.decide(state, target)
    elapsed = (time.perf_counter() - t0) * 1e6 / n_iter
    print(f"\n  decision latency       = {elapsed:.1f} µs  (1000-iter avg)")
    print(f"  V_coils norm           = {np.linalg.norm(out['V_coils']):.2f}")
    print(f"  V_coils max abs        = {np.max(np.abs(out['V_coils'])):.2f}  (rail limit {V_LIMIT})")
