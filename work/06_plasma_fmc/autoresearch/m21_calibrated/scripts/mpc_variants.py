"""mpc_variants.py — Autoresearch sweep of MPC formulations.

After m23_mpc_validation showed the baseline MPC (R=1e-6, full S⁺
inversion) catastrophically extrapolates on M15 aggressive targets
(0% physicality on κ=1.85 / δ=0.45), this module sweeps MPC variants
to find a tuning that matches FMC's robustness.

Variants tested:
  v00 baseline     — DLQR R=1e-6, full S⁺, V±1500
  v01 conservative — DLQR R=1e-3 (1000× higher control penalty)
  v02 medium       — DLQR R=1e-4
  v03 I_target_clip — DLQR R=1e-6 + I_target clipped to ±7.7 kA (TCV
                     current limit, REFERENCES.md G2)
  v04 V_limit_low  — DLQR R=1e-6 + V clip ±100V (FMC-like envelope)
  v05 prop_shape   — proportional shape feedback (no model inversion)
  v06 prop_clip    — v05 + V±50
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
from scipy.linalg import solve_discrete_are

V_LIMIT_DEFAULT = 1500.0
I_LIMIT_TCV = 7700.0  # E/F coil current limit (REFERENCES TCV-current-limits)


class MPCVariant:
    """Configurable linear MPC. Three formulations selected by `mode`:
      'dlqr' — classical DLQR with cost ΔIᵀ(SᵀWS)ΔI + ρ‖V‖²
      'prop' — proportional shape feedback V = -α S⁺ shape_err
    """

    def __init__(self, sim_p, *, mode: str = "dlqr",
                 dt: float = 1e-3, R_weight: float = 1e-6,
                 eps_Q: float = 1e-6, prop_alpha: float = 1.0,
                 P_aux: float = 5e5, gas_puff: float = 1e21,
                 voltage_limit: float = V_LIMIT_DEFAULT,
                 i_target_clip: float | None = None, seed: int = 0):
        self.dt = dt
        self.P_aux = P_aux
        self.gas_puff = gas_puff
        self.voltage_limit = voltage_limit
        self.i_target_clip = i_target_clip
        self.mode = mode
        self.prop_alpha = prop_alpha

        N = int(sim_p.N)
        M_cc = np.asarray(sim_p.M_cc, dtype=np.float64)
        R_diag = np.asarray(sim_p.R_diag, dtype=np.float64)
        S = np.asarray(sim_p.S, dtype=np.float64)
        I_ref = np.asarray(sim_p.I_ref, dtype=np.float64)
        ref_shape = np.array([
            float(sim_p.R_ref), float(sim_p.Z_ref),
            float(sim_p.kappa_ref), float(sim_p.delta_ref),
        ])
        S_pinv = np.linalg.pinv(S)

        if mode == "dlqr":
            M_cc_inv = np.linalg.inv(M_cc)
            A = np.eye(N) - dt * (M_cc_inv @ np.diag(R_diag))
            B = dt * M_cc_inv
            W_shape = np.diag([100.0, 100.0, 10.0, 10.0])
            Q = S.T @ W_shape @ S + eps_Q * np.eye(N)
            R = R_weight * np.eye(N)
            P = solve_discrete_are(A, B, Q, R)
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            self.A = A; self.B = B; self.K = K
            self.A_BK_spectral = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
        elif mode == "prop":
            # Proportional shape feedback: V = -α · S⁺ · (shape - target)·W_diag
            W_shape = np.array([100.0, 100.0, 10.0, 10.0])
            self.K_shape = prop_alpha * S_pinv * W_shape
        else:
            raise ValueError(f"unknown mode {mode}")

        self.S_pinv = S_pinv
        self.I_ref = I_ref
        self.R_diag = R_diag
        self.ref_shape = ref_shape
        self.N = N

    def decide(self, state, target_4):
        I = np.asarray(state[:self.N], dtype=np.float64)
        target = np.asarray(target_4, dtype=np.float64)

        if self.mode == "dlqr":
            I_target = self.I_ref + self.S_pinv @ (target - self.ref_shape)
            if self.i_target_clip is not None:
                I_target = np.clip(I_target, -self.i_target_clip, self.i_target_clip)
            V_target = self.R_diag * I_target
            V = V_target - self.K @ (I - I_target)
        elif self.mode == "prop":
            shape = self.ref_shape + self.S_pinv.T @ (I - self.I_ref)  # 4-vec via SI=Sshape⁻ map
            # Actually shape = ref + S(I - I_ref) — direct
            shape = self.ref_shape + np.zeros(4)  # placeholder
            # Use S directly (the linear shape model)
            S = self.K_shape  # K_shape already is α S⁺ W
            # Shape error from current state
            S_full = np.linalg.pinv(self.S_pinv)  # recover S
            shape = self.ref_shape + S_full @ (I - self.I_ref)
            shape_err = shape - target
            V = -S @ shape_err

        V = np.clip(V, -self.voltage_limit, self.voltage_limit)
        return {"V_coils": np.asarray(V, dtype=np.float32)}


def variant_specs():
    """Return list of (name, config_kwargs)."""
    return [
        ("v00_dlqr_baseline", dict(mode="dlqr", R_weight=1e-6)),
        ("v01_dlqr_R1e-3",    dict(mode="dlqr", R_weight=1e-3)),
        ("v02_dlqr_R1e-4",    dict(mode="dlqr", R_weight=1e-4)),
        ("v03_dlqr_Iclip",    dict(mode="dlqr", R_weight=1e-6,
                                    i_target_clip=I_LIMIT_TCV)),
        ("v04_dlqr_Vclip100", dict(mode="dlqr", R_weight=1e-6, voltage_limit=100.0)),
        ("v05_dlqr_Iclip_R1e-3", dict(mode="dlqr", R_weight=1e-3,
                                       i_target_clip=I_LIMIT_TCV)),
        ("v06_prop_alpha1",   dict(mode="prop", prop_alpha=1.0,
                                    voltage_limit=V_LIMIT_DEFAULT)),
        ("v07_prop_alpha10",  dict(mode="prop", prop_alpha=10.0,
                                    voltage_limit=V_LIMIT_DEFAULT)),
    ]


if __name__ == "__main__":
    from calibrated_sim import build_calibrated_jax_params
    sim_p, x0 = build_calibrated_jax_params()
    print("MPC variants smoke test on calibrated TCV sim")
    print("=" * 60)
    target = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)
    state = np.asarray(x0)
    for name, kw in variant_specs():
        ctrl = MPCVariant(sim_p, **kw)
        out = ctrl.decide(state, target)
        v_norm = float(np.linalg.norm(out["V_coils"]))
        v_max = float(np.max(np.abs(out["V_coils"])))
        print(f"  {name:25s}  ‖V‖={v_norm:>8.2f}  max|V|={v_max:>8.2f}")
