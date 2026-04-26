"""TCV plasma simulator — JAX version with vmap/jit, runnable on Metal GPU.

Same physics as `plasma_simulator.py` (which is the readable NumPy reference),
but rewritten as pure functions so JAX can:
- jit-compile the step function
- vmap the step over a *batch* of states (one per FMC walker)

This is the version FMC will actually use. With M=200 walkers and
N=20 lookahead steps, we need 4000 step evaluations per FMC decision.
At 1 kHz control rate, the budget is 1 ms = 250 ns per evaluation
in batched form.

Backend selection happens automatically: if Metal is available JAX
will use it; otherwise CPU. Set `JAX_PLATFORMS=cpu` to force CPU.

NB: Metal backend is experimental (JAX 0.4+). Some ops fall back to
CPU via XLA — measure with the benchmark script before relying on
GPU speedup for tiny matrices.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# Metal backend currently requires f32 (no f64 device support).
# Set BEFORE importing jax for the config to take effect.
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np

from mutual_inductance import mutual_matrix, mutual_to_plasma, self_inductance
from tcv_geometry import TCVMachine, load_tcv

E_CHARGE = 1.602176634e-19
DTYPE = jnp.float32


# ---------- JAX-friendly state (flat array) ----------
#
# State packing convention (length = N + 7):
#   0 .. N-1     : I_coils (A)
#   N            : I_p (A)
#   N+1          : W (J)
#   N+2          : n_bar (m⁻³)
#   N+3          : R_p (m)
#   N+4          : Z_p (m)
#   N+5          : kappa
#   N+6          : delta
#
# Time is tracked externally (we only need it for diagnostics).

@dataclass(frozen=True)
class SimParams:
    """Static (compile-time) simulator parameters — no JAX arrays here."""
    N: int                  # number of control channels
    M_cc: jnp.ndarray       # (N, N) coil mutual inductance [H]
    M_pc: jnp.ndarray       # (N,) plasma-coil mutual inductance [H]
    R_diag: jnp.ndarray     # (N,) coil resistances [Ω]
    L_p: float              # plasma self-inductance [H]
    S: jnp.ndarray          # (4, N) shape response matrix
    I_ref: jnp.ndarray      # (N,) reference coil currents [A]
    R_ref: float            # reference R_p [m]
    Z_ref: float            # reference Z_p [m]
    kappa_ref: float
    delta_ref: float
    a_eff: float            # effective minor radius [m]
    B_T: float              # toroidal field at R0 [T]
    eps: float              # inverse aspect ratio
    H98: float              # confinement enhancement
    # Calibration: multiplies Spitzer η to match TCV measured τ_res ~ 30-100 ms.
    # Reasons: (a) profile averaging — current is peaked, T_e profile-averaged;
    # (b) neoclassical bootstrap reduces effective resistance.
    # Empirical default 0.05 calibrates τ_res ≈ 30 ms at 1 keV.
    R_plasma_calib: float = 0.05


def pack_state(I_coils, I_p, W, n_bar, R_p, Z_p, kappa, delta) -> jnp.ndarray:
    return jnp.concatenate([
        jnp.asarray(I_coils, dtype=DTYPE),
        jnp.array([I_p, W, n_bar, R_p, Z_p, kappa, delta], dtype=DTYPE),
    ])


def unpack_state(x: jnp.ndarray, N: int) -> dict:
    return {
        "I_coils": x[:N],
        "I_p": x[N],
        "W": x[N + 1],
        "n_bar": x[N + 2],
        "R_p": x[N + 3],
        "Z_p": x[N + 4],
        "kappa": x[N + 5],
        "delta": x[N + 6],
    }


# ---------- Pure-function physics (jittable) ----------

def _T_keV(W, n_bar, V_plasma):
    """T_e [keV] from stored energy, density, plasma volume."""
    n_safe = jnp.maximum(n_bar, 1.0)
    V_safe = jnp.maximum(V_plasma, 1e-3)
    T_J = W / (3.0 * n_safe * V_safe)
    return T_J / (1e3 * E_CHARGE)


def _spitzer_R(T_keV, R_p, a_eff, kappa):
    """Spitzer R [Ω], floored at 0.01 keV to avoid divergence."""
    T_safe = jnp.maximum(T_keV, 0.01)
    eta = 5.2e-5 * 17.0 * T_safe ** (-1.5)
    L = 2.0 * jnp.pi * R_p
    A = jnp.pi * a_eff**2 * kappa
    return eta * L / A


def _tau_E(I_p_MA, B_T, P_MW, n_e19, R_p, eps, kappa, H98):
    P_safe = jnp.maximum(P_MW, 1e-3)
    n_safe = jnp.maximum(n_e19, 1e-3)
    Ip_safe = jnp.maximum(I_p_MA, 1e-6)
    return (
        0.0562 * H98
        * Ip_safe**0.93 * B_T**0.15 * P_safe**(-0.69)
        * n_safe**0.41 * 2.0**0.19
        * R_p**1.97 * eps**0.58 * kappa**0.78
    )


def step_jax(
    state: jnp.ndarray,
    V_coils: jnp.ndarray,
    P_aux: float,
    gas_puff: float,
    dt: float,
    p: SimParams,
) -> jnp.ndarray:
    """One simulator step — pure function, jit-compatible.

    Args:
        state: packed state, shape (N+7,)
        V_coils: applied voltage, shape (N,)
        P_aux: aux heating [W] (scalar)
        gas_puff: particle source [s⁻¹] (scalar)
        dt: integration step [s]
        p: SimParams (static)

    Returns:
        new packed state, shape (N+7,)
    """
    N = p.N
    s = unpack_state(state, N)

    V_plasma = 2.0 * jnp.pi**2 * s["R_p"] * p.a_eff**2 * s["kappa"]

    # (1) Coil circuit — implicit Euler:
    #     (M + dt·R) I_new = M I_old + dt V
    A = p.M_cc + dt * jnp.diag(p.R_diag)
    b = p.M_cc @ s["I_coils"] + dt * V_coils
    I_new = jnp.linalg.solve(A, b)

    # (2) V_loop induced + plasma current (implicit Euler):
    dI_dt = (I_new - s["I_coils"]) / dt
    V_loop = -(p.M_pc @ dI_dt)

    T = _T_keV(s["W"], s["n_bar"], V_plasma)
    R_plasma = _spitzer_R(T, s["R_p"], p.a_eff, s["kappa"]) * p.R_plasma_calib

    I_p_new = (s["I_p"] + dt * V_loop / p.L_p) / (1.0 + dt * R_plasma / p.L_p)

    # (3) Energy balance (explicit; τ_E >> dt for stable cases)
    I_p_MA = jnp.abs(I_p_new) / 1e6
    P_ohm = R_plasma * I_p_new**2
    P_loss = jnp.maximum(P_ohm + P_aux, 1e3)
    n_e19 = jnp.maximum(s["n_bar"] / 1e19, 1e-3)

    tau_E = jnp.maximum(_tau_E(
        I_p_MA, p.B_T, P_loss / 1e6, n_e19,
        s["R_p"], p.eps, s["kappa"], p.H98,
    ), 1e-4)

    dW_dt = P_aux + P_ohm - s["W"] / tau_E
    W_new = jnp.maximum(s["W"] + dt * dW_dt, 0.0)

    # (4) Particle balance
    tau_p = 3.0 * tau_E
    dn_dt = gas_puff / V_plasma - s["n_bar"] / tau_p
    n_new = jnp.maximum(s["n_bar"] + dt * dn_dt, 1e15)

    # (5) Shape response
    dI = I_new - p.I_ref
    delta_shape = p.S @ dI
    R_p_new = p.R_ref + delta_shape[0]
    Z_p_new = p.Z_ref + delta_shape[1]
    kappa_new = jnp.maximum(p.kappa_ref + delta_shape[2], 1.0)
    delta_new = jnp.clip(p.delta_ref + delta_shape[3], -0.7, 1.0)

    return jnp.concatenate([
        I_new,
        jnp.array([I_p_new, W_new, n_new, R_p_new, Z_p_new, kappa_new, delta_new]),
    ])


# ---------- Convenience: build params from TCV machine ----------

def build_jax_params(tcv: TCVMachine | None = None) -> tuple[SimParams, jnp.ndarray]:
    """Construct JAX SimParams + initial state at typical TCV operating point.

    Returns:
        (params, initial_state)
    """
    if tcv is None:
        tcv = load_tcv()

    # Same coil layout as NumPy version
    coil_R = np.concatenate([
        np.array([c.R for c in tcv.shaping_coils]),
        np.array([c.R for c in tcv.t_coils]),
        np.array([tcv.solenoid["R"]]),
    ])
    coil_Z = np.concatenate([
        np.array([c.Z for c in tcv.shaping_coils]),
        np.array([c.Z for c in tcv.t_coils]),
        np.array([(tcv.solenoid["Z_min"] + tcv.solenoid["Z_max"]) / 2.0]),
    ])
    N = coil_R.shape[0]

    M_cc = mutual_matrix(coil_R, coil_Z, a_wire=0.01)
    M_pc = mutual_to_plasma(coil_R, coil_Z, R_p=tcv.R_major, Z_p=0.0)
    L_p = self_inductance(tcv.R_major, a_wire=tcv.R_major / 10.0) * (1 + 1.7 * 0.3)

    # Multi-turn solenoid calibration: TCV OH has 100 turns.
    # For a multi-turn coil, all coupling integrals scale with turns count:
    #   L_OH = N² · L_single_loop
    #   M(OH, other) = N · M_single_loop
    # Implementation: scale row 19 AND column 19 of M_cc by N (diag becomes N²
    # automatically), then scale M_pc[19] by N.
    N_turns_OH = float(tcv.solenoid["N_turns"])
    M_cc[19, :] *= N_turns_OH
    M_cc[:, 19] *= N_turns_OH
    # Sanity: M_cc[19,19] now = N² · M_self_single ✓
    M_pc[19] *= N_turns_OH

    # Neoclassical/profile resistivity correction: real plasma R ~ 0.1× Spitzer
    # because of profile averaging (current peaked, T_e profile averaged) and
    # neoclassical bootstrap. Standard tokamak modeling uses Z_eff×profile_factor.
    # Empirical TCV: τ_res ~ 30-100 ms at 1 keV, vs Spitzer-naive ~300 µs.
    R_plasma_calib = 0.05  # multiplicative factor on Spitzer to match measured τ_res

    R_diag = np.full(N, tcv.R_coil_uniform)
    # OH circuit has higher resistance (multi-turn copper)
    R_diag[19] = tcv.R_coil_uniform * N_turns_OH

    # Reference currents (must match NumPy version)
    I_ref = np.zeros(N)
    for i, c in enumerate(tcv.shaping_coils):
        I_ref[i] = -1500.0 if c.name.startswith("E") else +2200.0
    I_ref[19] = +5000.0

    # Linearized shape response — same construction as NumPy
    S = np.zeros((4, N))
    for i, c in enumerate(tcv.shaping_coils):
        sign_Z = np.sign(c.Z) if c.Z != 0 else 1.0
        is_F = c.name.startswith("F")
        S[1, i] = (1.5e-6 * sign_Z) if is_F else (0.5e-6 * sign_Z)
        S[0, i] = (-2.0e-6) if is_F else (+1.5e-6)
        S[2, i] = (4.0e-8 * abs(c.Z)) if is_F else (-1.0e-8 * abs(c.Z))
        S[3, i] = (2.0e-8 * abs(c.Z)) * (1.0 if is_F else -0.5)

    params = SimParams(
        N=N,
        M_cc=jnp.asarray(M_cc, dtype=DTYPE),
        M_pc=jnp.asarray(M_pc, dtype=DTYPE),
        R_diag=jnp.asarray(R_diag, dtype=DTYPE),
        L_p=float(L_p),
        S=jnp.asarray(S, dtype=DTYPE),
        I_ref=jnp.asarray(I_ref, dtype=DTYPE),
        R_ref=float(tcv.R_major),
        Z_ref=0.0,
        kappa_ref=1.7,
        delta_ref=0.3,
        a_eff=float(tcv.a_minor * 0.96),
        B_T=1.43,
        eps=float(tcv.epsilon),
        H98=1.0,
    )

    initial_state = pack_state(
        I_coils=I_ref,
        I_p=200_000.0,
        W=40_800.0,
        n_bar=5.0e19,
        R_p=tcv.R_major,
        Z_p=0.0,
        kappa=1.7,
        delta=0.3,
    )

    return params, initial_state


# ---------- High-level wrappers (jit + vmap) ----------

def make_jit_step(params: SimParams):
    """Return a jit-compiled single-step function bound to fixed params.

    Usage:
        params, x0 = build_jax_params()
        step = make_jit_step(params)
        x1 = step(x0, V, P_aux, gas_puff, dt)
    """
    @jax.jit
    def f(state, V_coils, P_aux, gas_puff, dt):
        return step_jax(state, V_coils, P_aux, gas_puff, dt, params)
    return f


def make_batched_step(params: SimParams):
    """Vectorize step over a batch dimension (FMC walkers).

    Input shapes (B = batch):
        state    : (B, N+7)
        V_coils  : (B, N)
        P_aux    : (B,) or scalar
        gas_puff : (B,) or scalar
        dt       : scalar

    Returns: (B, N+7)
    """
    base = make_jit_step(params)
    return jax.jit(jax.vmap(base, in_axes=(0, 0, 0, 0, None)))


def make_batched_rollout(params: SimParams, horizon: int):
    """Roll out `horizon` steps for a batch of walkers.

    Returns a function (state_batch, V_seq, P_seq, gas_seq, dt) → final state batch.
    V_seq has shape (horizon, B, N).

    Uses lax.scan for efficient unrolling on accelerator.
    """
    @jax.jit
    def rollout(state_batch, V_seq, P_seq, gas_seq, dt):
        # vmap one-step
        def one_step(carry, inputs):
            V, P, gas = inputs
            new = jax.vmap(step_jax, in_axes=(0, 0, 0, 0, None, None))(
                carry, V, P, gas, dt, params,
            )
            return new, None

        final_state, _ = jax.lax.scan(one_step, state_batch, (V_seq, P_seq, gas_seq))
        return final_state
    return rollout


# ---------- Self-test (cross-check vs NumPy version) ----------

def _cross_check_vs_numpy():
    """Verify JAX step matches NumPy step within float precision."""
    from plasma_simulator import build_default_simulator, Control

    np_sim, np_state = build_default_simulator()
    jx_params, jx_state = build_jax_params()

    # Apply identical control
    rng = np.random.default_rng(42)
    V = rng.normal(0, 50.0, size=np_sim.N)
    P_aux = 5.0e5
    gas = 2e21
    dt = 1e-3

    # NumPy step
    np_ctrl = Control(V_coils=V, P_aux=P_aux, gas_puff=gas)
    np_next = np_sim.step(np_state, np_ctrl, dt)

    # JAX step
    step = make_jit_step(jx_params)
    jx_next = step(jx_state,
                   jnp.asarray(V, dtype=DTYPE),
                   jnp.asarray(P_aux, dtype=DTYPE),
                   jnp.asarray(gas, dtype=DTYPE),
                   jnp.asarray(dt, dtype=DTYPE))
    jx_next_np = np.asarray(jx_next)

    # Compare
    np_packed = np.concatenate([
        np_next.I_coils,
        [np_next.I_p, np_next.W, np_next.n_bar, np_next.R_p,
         np_next.Z_p, np_next.kappa, np_next.delta],
    ])

    abs_err = np.abs(np_packed - jx_next_np)
    rel_err = abs_err / (np.abs(np_packed) + 1e-12)
    max_rel = np.max(rel_err)
    print(f"  Max abs err : {abs_err.max():.3e}")
    print(f"  Max rel err : {max_rel:.3e}")
    print(f"  Component-wise diff (first 5): {abs_err[:5]}")
    return max_rel


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    params, x0 = build_jax_params()
    print(f"Initial state shape: {x0.shape}, dtype: {x0.dtype}")
    print(f"Channels: {params.N}")
    print(f"M_cc condition number: {np.linalg.cond(np.asarray(params.M_cc)):.3e}")

    print("\n--- Cross-check NumPy vs JAX (one step) ---")
    max_rel = _cross_check_vs_numpy()
    if max_rel < 1e-5:
        print("  ✓ JAX implementation matches NumPy")
    else:
        print(f"  ⚠ Mismatch — max relative error {max_rel:.3e}")

    # Quick batched demo
    print("\n--- Batched rollout (B=64 walkers, H=20 steps) ---")
    B, H = 64, 20
    rng = np.random.default_rng(0)
    state_batch = jnp.broadcast_to(x0, (B, x0.shape[0]))
    V_seq = jnp.asarray(rng.normal(0, 30.0, size=(H, B, params.N)), dtype=DTYPE)
    P_seq = jnp.full((H, B), 5e5, dtype=DTYPE)
    gas_seq = jnp.full((H, B), 2e21, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)

    rollout = make_batched_rollout(params, horizon=H)
    final = rollout(state_batch, V_seq, P_seq, gas_seq, dt)
    final.block_until_ready()
    print(f"  Final state shape: {final.shape}")
    print(f"  Sample I_p[batch=0]: {float(final[0, params.N]):.1f} A")
    print(f"  Sample W[batch=0]:   {float(final[0, params.N + 1]):.1f} J")
    print(f"  Sample R_p[batch=0]: {float(final[0, params.N + 3]):.4f} m")
