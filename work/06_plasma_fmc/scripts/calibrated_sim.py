"""Calibrated SimParams using FreeGS truth findings (Milestone 10).

M9 baseline DN equilibrium has:
  R_p=0.901, Z_p=-0.109, a=0.390, κ=1.616, δ=+0.003

So the proper reference state is NOT (κ=1.7, δ=+0.3) — that was a Miller-
parametric "wishlist" target, not an actually achievable shape with the
TCV coil set + standard DN constraints.

S sensitivity: M9 constraint perturbation gave Δκ/Δ|coil| ≈ 1e-5 /A,
vs synthetic S coefficient of 4e-7·|Z| ≈ 3e-7 /A.
→ Synthetic S was ~30× under-scaled. We bump it ~10× (conservative; the
M9 number is an aggregate over multiple coils, so per-coil scaling could
be 3-10× rather than 30×).

Also: the larger plasma (a=0.39 vs 0.24 in M2) requires reproducing a
realistic energy balance — which the same Miller volume formula gives
us automatically (V = 2π² R a² κ ≈ 2.7 m³ vs 1.7).
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import numpy as np

from mutual_inductance import mutual_matrix, mutual_to_plasma, self_inductance
from plasma_simulator_jax import DTYPE, SimParams, pack_state
from tcv_geometry import TCVMachine, load_tcv

# ---- M9-derived baseline ----
M9_BASELINE = {
    "R_p": 0.901,
    "Z_p": -0.109,
    "a": 0.390,
    "kappa": 1.616,
    "delta": 0.003,
}

# ---- Scaling factors derived from M9 sensitivity vs M3 synthetic ----
S_SCALE_FACTOR = 10.0  # multiply per-coil S coefficients by this


def build_calibrated_jax_params(
    s_scale: float = S_SCALE_FACTOR,
    tcv: TCVMachine | None = None,
) -> tuple[SimParams, jnp.ndarray]:
    """Build a SimParams using M9 baseline + scaled S matrix.

    Same machinery as plasma_simulator_jax.build_jax_params() but with:
    - reference state from M9 DN equilibrium (not Miller wishlist)
    - shape response S × s_scale (default 10×)
    """
    if tcv is None:
        tcv = load_tcv()

    # Same coil layout
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
    M_pc = mutual_to_plasma(coil_R, coil_Z,
                            R_p=M9_BASELINE["R_p"], Z_p=M9_BASELINE["Z_p"])
    L_p = self_inductance(M9_BASELINE["R_p"],
                           a_wire=M9_BASELINE["R_p"] / 10.0) \
          * (1 + M9_BASELINE["kappa"] * 0.3)

    # Multi-turn OH calibration (same as M3)
    N_turns_OH = float(tcv.solenoid["N_turns"])
    M_cc[19, :] *= N_turns_OH
    M_cc[:, 19] *= N_turns_OH
    M_pc[19] *= N_turns_OH

    R_diag = np.full(N, tcv.R_coil_uniform)
    R_diag[19] = tcv.R_coil_uniform * N_turns_OH

    # Reference currents — use the SAME currents that M9 GS solver found
    # for the baseline DN equilibrium (read from results/freegs_truth.json)
    import json
    truth_path = Path(__file__).parent.parent / "results" / "freegs_truth.json"
    if truth_path.exists():
        with open(truth_path) as f:
            truth = json.load(f)
        empirical_currents = truth["baseline"]["currents"]
        I_ref = np.zeros(N)
        # Map empirical currents to our coil ordering
        for i, c in enumerate(tcv.shaping_coils):
            I_ref[i] = empirical_currents.get(c.name, 0.0)
        for i, c in enumerate(tcv.t_coils, start=16):
            I_ref[i] = empirical_currents.get(c.name, 0.0)
        # OH circuit elements lumped — average C1/C2/D1/D2
        oh_currents = [empirical_currents.get(k, 0)
                        for k in ("C1", "C2", "D1", "D2")]
        I_ref[19] = float(np.mean(oh_currents))
    else:
        # Fall back to M3 hand-tuned currents
        I_ref = np.zeros(N)
        for i, c in enumerate(tcv.shaping_coils):
            I_ref[i] = -1500.0 if c.name.startswith("E") else +2200.0
        I_ref[19] = +5000.0

    # Calibrated S matrix — base structure same as M3 but × s_scale
    S = np.zeros((4, N))
    for i, c in enumerate(tcv.shaping_coils):
        sign_Z = np.sign(c.Z) if c.Z != 0 else 1.0
        is_F = c.name.startswith("F")
        S[1, i] = ((1.5e-6 * sign_Z) if is_F else (0.5e-6 * sign_Z)) * s_scale
        S[0, i] = ((-2.0e-6) if is_F else (+1.5e-6)) * s_scale
        S[2, i] = ((4.0e-7 * abs(c.Z)) if is_F else (-1.0e-7 * abs(c.Z))) * s_scale
        S[3, i] = ((2.0e-7 * abs(c.Z)) * (1.0 if is_F else -0.5)) * s_scale

    params = SimParams(
        N=N,
        M_cc=jnp.asarray(M_cc, dtype=DTYPE),
        M_pc=jnp.asarray(M_pc, dtype=DTYPE),
        R_diag=jnp.asarray(R_diag, dtype=DTYPE),
        L_p=float(L_p),
        S=jnp.asarray(S, dtype=DTYPE),
        I_ref=jnp.asarray(I_ref, dtype=DTYPE),
        R_ref=float(M9_BASELINE["R_p"]),
        Z_ref=float(M9_BASELINE["Z_p"]),
        kappa_ref=float(M9_BASELINE["kappa"]),
        delta_ref=float(M9_BASELINE["delta"]),
        a_eff=float(M9_BASELINE["a"]),
        B_T=1.43,
        eps=float(M9_BASELINE["a"] / M9_BASELINE["R_p"]),
        H98=1.0,
    )

    # Initial state at the new ref
    a = M9_BASELINE["a"]
    V_plasma = 2 * np.pi**2 * M9_BASELINE["R_p"] * a**2 * M9_BASELINE["kappa"]
    n_bar = 5e19
    T_e_keV = 1.0
    W = 3 * n_bar * V_plasma * (T_e_keV * 1e3 * 1.602176634e-19)
    initial = pack_state(
        I_coils=I_ref,
        I_p=200_000.0,
        W=W,
        n_bar=n_bar,
        R_p=M9_BASELINE["R_p"],
        Z_p=M9_BASELINE["Z_p"],
        kappa=M9_BASELINE["kappa"],
        delta=M9_BASELINE["delta"],
    )
    return params, initial


def calibrated_target_ranges() -> dict:
    """Realistic target ranges for the calibrated reference.

    Tighter than M5: avoids targets that are physically unreachable
    given the linear-S approximation around the M9 baseline.
    """
    base = M9_BASELINE
    return {
        "R_p": (base["R_p"] - 0.03, base["R_p"] + 0.03),       # ±3 cm
        "Z_p": (base["Z_p"] - 0.05, base["Z_p"] + 0.05),       # ±5 cm
        "kappa": (base["kappa"] - 0.15, base["kappa"] + 0.30),  # 1.47 .. 1.92
        "delta": (base["delta"] - 0.30, base["delta"] + 0.50),  # -0.30 .. +0.50
    }


if __name__ == "__main__":
    print("Calibrated SimParams (Milestone 10)")
    print("=" * 60)
    params, x0 = build_calibrated_jax_params()
    print(f"  Channels       : {params.N}")
    print(f"  Reference state: R_p={params.R_ref:.4f}, Z_p={params.Z_ref:+.4f}")
    print(f"                  κ={params.kappa_ref:.3f}, δ={params.delta_ref:+.3f}")
    print(f"  a_eff          : {params.a_eff:.4f} m")
    print(f"  ε = a/R        : {params.eps:.4f}")
    print(f"  S max coeff    : {float(jnp.max(jnp.abs(params.S))):.2e}")

    # Initial state diagnostics
    N = params.N
    print(f"\n  I_p initial    : {float(x0[N]):.0f} A")
    print(f"  W initial      : {float(x0[N+1]):.0f} J")
    print(f"  n_bar          : {float(x0[N+2]):.2e} m⁻³")

    # Target ranges
    print(f"\n  Calibrated target ranges:")
    for k, (lo, hi) in calibrated_target_ranges().items():
        print(f"    {k:6s} : [{lo:+.3f}, {hi:+.3f}]")

    # Quick free-decay test
    from plasma_simulator_jax import make_jit_step
    step = make_jit_step(params)
    V = jnp.zeros(params.N, dtype=DTYPE)
    P = jnp.asarray(0.0, dtype=DTYPE)
    g = jnp.asarray(0.0, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)
    x = x0
    print(f"\n  Free decay (V=0, no aux):")
    for k in [0, 5, 10, 20]:
        if k > 0:
            for _ in range(k - (k - 5 if k > 5 else 0)):
                x = step(x, V, P, g, dt)
        I_p_kA = float(x[N]) / 1e3
        kappa_now = float(x[N + 5])
        R_p_now = float(x[N + 3])
        print(f"    t={k} ms: I_p={I_p_kA:6.1f} kA, R_p={R_p_now:.4f} m, κ={kappa_now:.3f}")
