"""Plasma simulator with NN shape model integrated end-to-end.

Replaces the linearized `delta_shape = S @ dI` of plasma_simulator_jax
with `shape = NN_shape(I_coils)` (M11 surrogate). The NN is trained on
135 FreeGS solves and predicts shape with ~3 cm RMSE on R_p.

Architecture:
- `SimParamsNN` extends SimParams with NN params + normalizers
- `step_jax_nn` is a new step function that calls the NN inside JIT
- `FMCPlasmaJaxControllerNN` mirrors the M7 controller but uses
  `step_jax_nn` instead of `step_jax`

Mathematical equivalence: identical to M7/M10 simulator EXCEPT for the
shape update block (5):
    M10 (linear)  : delta_shape = p.S @ dI;  shape = ref + delta_shape
    M12 (NN)      : shape       = NN_shape(I_coils, params, normalizers)

Tests in test_nn_sim.py verify same behavior on physical conservation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from plasma_simulator_jax import (
    DTYPE, _T_keV, _spitzer_R, _tau_E,
    SimParams, build_jax_params, pack_state,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---- Shape NN module (matches train_shape_surrogate.py) ----

class ShapeNN(nn.Module):
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(4, kernel_init=nn.initializers.he_normal())(x)


@dataclass(frozen=True)
class SimParamsNN:
    """SimParams + NN shape model bundled for jit-friendly use."""
    # Inherit all fields from SimParams (manually expanded)
    N: int
    M_cc: jnp.ndarray
    M_pc: jnp.ndarray
    R_diag: jnp.ndarray
    L_p: float
    I_ref: jnp.ndarray
    R_ref: float
    Z_ref: float
    kappa_ref: float
    delta_ref: float
    a_eff: float
    B_T: float
    eps: float
    H98: float
    R_plasma_calib: float
    # NN shape model
    nn_params: object  # Flax PyTree of params (nested dict)
    x_mean: jnp.ndarray  # (20,) coil current means in kA
    x_std: jnp.ndarray   # (20,) coil current stds in kA
    y_mean: jnp.ndarray  # (4,)  shape means
    y_std: jnp.ndarray   # (4,)  shape stds
    # Reuse the same Flax module (stateless, class-level)
    nn_module_hidden: tuple = (64, 64)


def predict_shape(params: SimParamsNN, I_coils):
    """NN forward: I_coils (A) → shape (R_p, Z_p, κ, δ).

    Args:
        params: SimParamsNN (carries NN weights + normalizers)
        I_coils: jnp.ndarray of shape (20,) in Amps
    Returns:
        shape: jnp.ndarray of shape (4,)
    """
    # Convert A → kA, normalize
    x_kA = I_coils / 1000.0
    x_norm = (x_kA - params.x_mean) / params.x_std
    model = ShapeNN(hidden_sizes=params.nn_module_hidden)
    y_norm = model.apply(params.nn_params, x_norm[None])[0]
    y = y_norm * params.y_std + params.y_mean
    return y


def step_jax_nn(state, V_coils, P_aux, gas_puff, dt, p: SimParamsNN):
    """One simulator step with NN shape model.

    Same physics as plasma_simulator_jax.step_jax, except shape is now
    `NN(I_coils)` rather than `ref + S @ (I - I_ref)`.
    """
    N = p.N

    # Unpack
    I_coils = state[:N]
    I_p = state[N]
    W = state[N + 1]
    n_bar = state[N + 2]
    R_p = state[N + 3]
    Z_p = state[N + 4]
    kappa = state[N + 5]
    delta = state[N + 6]

    V_plasma = 2.0 * jnp.pi**2 * R_p * p.a_eff**2 * kappa

    # (1) Coil circuit (implicit Euler)
    A = p.M_cc + dt * jnp.diag(p.R_diag)
    b = p.M_cc @ I_coils + dt * V_coils
    I_new = jnp.linalg.solve(A, b)

    # (2) Loop voltage + plasma current
    dI_dt = (I_new - I_coils) / dt
    V_loop = -(p.M_pc @ dI_dt)
    T = _T_keV(W, n_bar, V_plasma)
    R_plasma = _spitzer_R(T, R_p, p.a_eff, kappa) * p.R_plasma_calib
    I_p_new = (I_p + dt * V_loop / p.L_p) / (1.0 + dt * R_plasma / p.L_p)

    # (3) Energy balance
    I_p_MA = jnp.abs(I_p_new) / 1e6
    P_ohm = R_plasma * I_p_new**2
    P_loss = jnp.maximum(P_ohm + P_aux, 1e3)
    n_e19 = jnp.maximum(n_bar / 1e19, 1e-3)
    tau_E = jnp.maximum(_tau_E(
        I_p_MA, p.B_T, P_loss / 1e6, n_e19, R_p, p.eps, kappa, p.H98,
    ), 1e-4)
    dW_dt = P_aux + P_ohm - W / tau_E
    W_new = jnp.maximum(W + dt * dW_dt, 0.0)

    # (4) Particle balance
    tau_p = 3.0 * tau_E
    dn_dt = gas_puff / V_plasma - n_bar / tau_p
    n_new = jnp.maximum(n_bar + dt * dn_dt, 1e15)

    # (5) Shape via NN (this is the M12 change)
    shape_pred = predict_shape(p, I_new)
    R_p_new = jnp.clip(shape_pred[0], 0.624, 1.136)
    Z_p_new = jnp.clip(shape_pred[1], -0.75, 0.75)
    kappa_new = jnp.clip(shape_pred[2], 1.0, 2.8)
    delta_new = jnp.clip(shape_pred[3], -0.7, 1.0)

    return jnp.concatenate([
        I_new,
        jnp.array([I_p_new, W_new, n_new, R_p_new, Z_p_new, kappa_new, delta_new]),
    ])


def build_nn_sim_params(s_scale_unused: float = 0.0,
                        sim_p_base: SimParams | None = None) -> SimParamsNN:
    """Build SimParamsNN by combining a calibrated SimParams + M11 NN weights."""
    if sim_p_base is None:
        from calibrated_sim import build_calibrated_jax_params
        sim_p_base, _ = build_calibrated_jax_params()

    # Load M11 NN
    nn_data = np.load(RESULTS_DIR / "shape_surrogate.npz", allow_pickle=True)
    nn_params = nn_data["params"].item()
    x_mean = jnp.asarray(nn_data["x_mean"], dtype=DTYPE)
    x_std = jnp.asarray(nn_data["x_std"], dtype=DTYPE)
    y_mean = jnp.asarray(nn_data["y_mean"], dtype=DTYPE)
    y_std = jnp.asarray(nn_data["y_std"], dtype=DTYPE)

    return SimParamsNN(
        N=sim_p_base.N,
        M_cc=sim_p_base.M_cc,
        M_pc=sim_p_base.M_pc,
        R_diag=sim_p_base.R_diag,
        L_p=sim_p_base.L_p,
        I_ref=sim_p_base.I_ref,
        R_ref=sim_p_base.R_ref,
        Z_ref=sim_p_base.Z_ref,
        kappa_ref=sim_p_base.kappa_ref,
        delta_ref=sim_p_base.delta_ref,
        a_eff=sim_p_base.a_eff,
        B_T=sim_p_base.B_T,
        eps=sim_p_base.eps,
        H98=sim_p_base.H98,
        R_plasma_calib=sim_p_base.R_plasma_calib,
        nn_params=nn_params,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    )


def make_jit_step_nn(p: SimParamsNN):
    @jax.jit
    def f(state, V_coils, P_aux, gas_puff, dt):
        return step_jax_nn(state, V_coils, P_aux, gas_puff, dt, p)
    return f


def initial_state_nn(p: SimParamsNN) -> jnp.ndarray:
    """Initial state at the calibrated reference."""
    a = p.a_eff
    V_plasma = 2 * jnp.pi**2 * p.R_ref * a**2 * p.kappa_ref
    n_bar = 5e19
    T_e_keV = 1.0
    W = 3 * n_bar * V_plasma * (T_e_keV * 1e3 * 1.602176634e-19)
    return pack_state(
        I_coils=p.I_ref,
        I_p=200_000.0, W=float(W), n_bar=n_bar,
        R_p=p.R_ref, Z_p=p.Z_ref,
        kappa=p.kappa_ref, delta=p.delta_ref,
    )


if __name__ == "__main__":
    print("Plasma simulator with NN shape model (M12)")
    print("=" * 60)
    p = build_nn_sim_params()
    print(f"Channels        : {p.N}")
    print(f"Reference       : R_p={p.R_ref:.4f}, κ={p.kappa_ref:.3f}, "
          f"δ={p.delta_ref:+.3f}")

    # Test shape prediction at I_ref
    shape_at_ref = np.asarray(predict_shape(p, jnp.asarray(p.I_ref)))
    print(f"\nShape at I_ref via NN:")
    print(f"  R_p   = {shape_at_ref[0]:.4f}  (calibrated ref: {p.R_ref:.4f})")
    print(f"  Z_p   = {shape_at_ref[1]:+.4f}  (calibrated ref: {p.Z_ref:+.4f})")
    print(f"  kappa = {shape_at_ref[2]:.4f}  (calibrated ref: {p.kappa_ref:.4f})")
    print(f"  delta = {shape_at_ref[3]:+.4f}  (calibrated ref: {p.delta_ref:+.4f})")

    # Sanity: a step
    step = make_jit_step_nn(p)
    x0 = initial_state_nn(p)
    print(f"\nInitial state shape: {x0.shape}")

    V_ref = jnp.asarray(p.R_diag) * jnp.asarray(p.I_ref)
    V = V_ref
    P = jnp.asarray(0.0, dtype=DTYPE)
    g = jnp.asarray(0.0, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)

    print(f"\nStep test (V=V_ref, no aux, dt=1ms × 5 steps):")
    x = x0
    N = p.N
    for k in range(5):
        x = step(x, V, P, g, dt)
        print(f"  t={k+1} ms: I_p={float(x[N])/1e3:6.1f} kA, "
              f"R_p={float(x[N+3]):.4f}, "
              f"κ={float(x[N+5]):.3f}, "
              f"δ={float(x[N+6]):+.3f}")
