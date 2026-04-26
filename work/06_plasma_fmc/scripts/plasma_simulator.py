"""TCV plasma simulator — 0D + linear PF circuit (NumPy reference impl).

This is the **fast inner simulator** suitable for FMC rollouts.
It is *not* a Grad-Shafranov solver — that role is Milestone 3 (FreeGS as
slow ground-truth oracle).

State vector (continuous):
    I_coils[20]      coil currents [A]
    I_p              plasma current [A]
    W                plasma stored thermal energy [J]
    n_bar            volume-averaged electron density [m⁻³]
    R_p, Z_p         plasma centroid [m]
    kappa, delta     shape parameters [-]

Control input (per-step):
    V_coils[20]      voltages applied to each control channel [V]
    P_aux            auxiliary heating power (ECRH+NBI) [W]
    gas_puff         particle source [s⁻¹]

Physics implemented:

(1) Coil circuit equation (REFERENCES §D.6, Walker-Humphreys 2006):
        M · dI/dt + R · I = V
    where M includes plasma-coil mutual coupling via the (frozen-shape)
    plasma filament approximation. Linearized about a reference state.

(2) Plasma current evolution (lumped 0D):
        dI_p/dt = (1/L_p) [V_loop - R_p_resistive · I_p]
    V_loop = -d(M_pc · I_coils)/dt = -M_pc · dI/dt   (induced loop voltage)
    R_p_resistive = Spitzer-like, ∝ T_e^(-3/2)

(3) 0D energy balance (REFERENCES §D.4, IPB98(y,2)):
        dW/dt = P_aux + P_ohm - W/τ_E
    P_ohm = R_p_resistive · I_p²
    τ_E = IPB98(y,2) scaling (with H98 = 1.0)

(4) Particle balance (simple):
        dn_bar/dt = (gas_puff)/V_plasma - n_bar/τ_p
    τ_p ≈ 3·τ_E (typical scaling)

(5) Linearized shape response (Wesson §11.4 + Walker-Humphreys):
        [δR_p, δZ_p, δκ, δδ]ᵀ = S · (I_coils - I_coils_ref)
    S is a 4×N response matrix, identified once around a reference
    equilibrium. For Milestone 2 we use a synthetic but physical S
    (vertical instability sign correct, etc.). Real S comes from the
    GS solver in Milestone 3.

Numerical integration: implicit-Euler for the circuit equation
(stable for any Δt vs the L/R timescales), explicit-Euler for the
slower energy/density (timescales ~10 ms vs Δt = 1 ms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mutual_inductance import mutual_matrix, mutual_to_plasma
from tcv_geometry import TCVMachine, load_tcv

# ---------- Physical constants ----------
MU_0 = 4 * np.pi * 1e-7
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
M_E = 9.1093837015e-31


# ---------- Plasma state (continuous) ----------

@dataclass
class PlasmaState:
    """Mutable simulator state. Keep dimensions small — copied for FMC walkers."""
    I_coils: np.ndarray   # (N_channels,) [A]
    I_p: float            # [A]
    W: float              # [J]
    n_bar: float          # [m⁻³]
    R_p: float            # [m]
    Z_p: float            # [m]
    kappa: float
    delta: float
    t: float              # [s]

    def copy(self) -> "PlasmaState":
        return PlasmaState(
            I_coils=self.I_coils.copy(),
            I_p=self.I_p, W=self.W, n_bar=self.n_bar,
            R_p=self.R_p, Z_p=self.Z_p,
            kappa=self.kappa, delta=self.delta, t=self.t,
        )

    def as_vector(self) -> np.ndarray:
        return np.concatenate([
            self.I_coils,
            np.array([self.I_p, self.W, self.n_bar,
                      self.R_p, self.Z_p, self.kappa, self.delta, self.t]),
        ])


@dataclass
class Control:
    """One-step actuation."""
    V_coils: np.ndarray   # (N_channels,) [V]
    P_aux: float = 0.0    # [W]
    gas_puff: float = 0.0 # [s⁻¹]


# ---------- Helpers ----------

def temperature_from_W(W: float, n_bar: float, V_plasma: float) -> float:
    """T_e = T_i [keV] from W = 3 n V T (assuming Z_eff=1, T_e=T_i)."""
    if n_bar <= 0 or V_plasma <= 0:
        return 0.0
    # W [J] = 3 · n [m⁻³] · V [m³] · T [J]   (factor 3 = 3/2·2 species)
    T_J = W / (3.0 * n_bar * V_plasma)
    return T_J / (1e3 * E_CHARGE)  # → keV


def spitzer_resistance(T_keV: float, R_p: float, a_eff: float, kappa: float) -> float:
    """Spitzer plasma resistance [Ω] (simplified, Z_eff=1, ln Λ=17).

    R_plasma = η_Spitzer · L / A
       η_Sp = 5.2e-5 · ln Λ · T_e[keV]^(-3/2)   [Ω·m]
       L = 2π R_p,  A = π a² κ   (cross-section)
    """
    if T_keV <= 0:
        return float("inf")
    eta = 5.2e-5 * 17.0 * T_keV ** (-1.5)  # Ω·m
    L = 2.0 * np.pi * R_p
    A = np.pi * a_eff**2 * kappa
    return eta * L / A


def tau_E_IPB98(
    I_p_MA: float, B_T: float, P_loss_MW: float, n_e_e19: float,
    R_p: float, eps: float, kappa: float, M_amu: float = 2.0, H98: float = 1.0,
) -> float:
    """IPB98(y,2) energy confinement time [s] (REFERENCES §D.4).

    τ_E = 0.0562 · H98 · I_p^0.93 · B_T^0.15 · P^-0.69 · n_e^0.41
                 · M^0.19 · R^1.97 · ε^0.58 · κ^0.78
    """
    if P_loss_MW <= 0 or n_e_e19 <= 0:
        return 0.0
    return (
        0.0562 * H98
        * I_p_MA**0.93 * B_T**0.15 * P_loss_MW**(-0.69)
        * n_e_e19**0.41 * M_amu**0.19
        * R_p**1.97 * eps**0.58 * kappa**0.78
    )


# ---------- The simulator ----------

class PlasmaSimulator:
    """One-step Markov-style simulator — pure function (state, control) → state'.

    Designed to be cheap so an FMC walker pool of M=200, looking N=20 steps
    ahead, fits in a few ms wall-clock.

    Parameters fixed at construction:
    - Mutual inductance matrix M (frozen-shape approximation, computed once)
    - Resistance matrix R (diagonal)
    - Linearized shape response matrix S
    - Reference operating point (used as expansion point for S)
    """

    def __init__(
        self,
        tcv: TCVMachine,
        I_ref_coils: np.ndarray,
        ref_state: PlasmaState,
        a_wire: float = 0.01,
        H98: float = 1.0,
        R_plasma_calib: float = 0.05,
    ):
        self.tcv = tcv
        self.H98 = H98
        self.a_wire = a_wire
        # See SimParams.R_plasma_calib — neoclassical/profile correction
        self.R_plasma_calib = R_plasma_calib

        # Active coils that interact with the plasma:
        # 16 shaping + OH circuit lumped (5 elements: solenoid handled below + C1/C2/D1/D2)
        # We lump T coils into the plasma response only weakly — for now treat
        # them as 3 independent control channels with own L/R.
        self.coil_R = np.concatenate([
            np.array([c.R for c in tcv.shaping_coils]),
            np.array([c.R for c in tcv.t_coils]),
            np.array([tcv.solenoid["R"]]),  # OH = single equivalent loop
        ])
        self.coil_Z = np.concatenate([
            np.array([c.Z for c in tcv.shaping_coils]),
            np.array([c.Z for c in tcv.t_coils]),
            np.array([(tcv.solenoid["Z_min"] + tcv.solenoid["Z_max"]) / 2.0]),
        ])
        self.N = self.coil_R.shape[0]
        assert self.N == tcv.n_control_channels, (
            f"channel count mismatch: {self.N} vs {tcv.n_control_channels}"
        )

        # Mutual inductance matrix between coils (Neumann, frozen plasma)
        self.M_cc = mutual_matrix(self.coil_R, self.coil_Z, a_wire=a_wire)

        # Plasma-coil coupling (treats plasma as filamentary loop)
        self.M_pc = mutual_to_plasma(
            self.coil_R, self.coil_Z, ref_state.R_p, ref_state.Z_p,
        )

        # Multi-turn OH solenoid: scale row/col 19 by N_turns, M_pc[19] by N_turns
        N_turns_OH = float(tcv.solenoid["N_turns"])
        self.M_cc[19, :] *= N_turns_OH
        self.M_cc[:, 19] *= N_turns_OH
        self.M_pc[19] *= N_turns_OH

        # Plasma self-inductance (Wesson §3.7, plasma loop)
        from mutual_inductance import self_inductance
        self.L_p = self_inductance(ref_state.R_p, a_wire=ref_state.R_p / 10.0) \
                   * (1 + ref_state.kappa * 0.3)  # rough κ correction

        # Diagonal coil resistances (uniform — H2 in REFERENCES §G)
        # OH gets N_turns × resistance (multi-turn copper)
        self.R_diag = np.full(self.N, tcv.R_coil_uniform)
        self.R_diag[19] = tcv.R_coil_uniform * N_turns_OH

        # Reference operating point
        self.I_ref = I_ref_coils.copy()
        self.ref_state = ref_state.copy()

        # Linearized shape response S (4 × N)
        # Built synthetically but physically: F coils control κ/δ, E vs F balance
        # controls R_p, vertical asymmetry of F drives Z_p (and is unstable).
        self.S = self._build_shape_response()

        # Vessel (passive) approximated as zero — all flux goes through coils
        # for simplicity. Real model has 192 toroidal vessel filaments.

    def _build_shape_response(self) -> np.ndarray:
        """4 × N response matrix [δR, δZ, δκ, δδ] vs δI_coils.

        Constructed from physical first-principles arguments:
        - Vertical position Z_p: sum of F-up minus F-down currents.
          Coefficient unstable (positive feedback); we pick small magnitude
          so explicit Euler is stable for typical Δt.
        - Radial R_p: balance between E (inboard, pushes plasma out) and F.
        - Elongation κ: enhanced by symmetric F-up + F-down current.
        - Triangularity δ: by ratio of inner-F vs outer-F currents.

        Magnitudes are scaled so 1 kA on a typical coil produces ~1 cm shift
        and ~0.05 in shape parameters — order-of-magnitude correct for TCV.
        """
        S = np.zeros((4, self.N))
        coils = self.tcv.shaping_coils  # 16 entries

        # Per-coil sensitivities scaled so that *full ensemble* response
        # (16 coils acting together) gives ~ few cm shift / ~0.05 shape
        # change for typical control current swing of 1 kA per coil.
        # Divide by 16 vs naive "1 kA → 1 cm per coil" to avoid additive blowup.
        for i, c in enumerate(coils):
            sign_Z = np.sign(c.Z) if c.Z != 0 else 1.0
            is_F = c.name.startswith("F")

            # δZ_p / δI: F-coils antisymmetric in Z, marginally unstable
            S[1, i] = (1.5e-6 * sign_Z) if is_F else (0.5e-6 * sign_Z)
            # δR_p / δI: E pushes outward, F pulls inward (symmetric in Z)
            S[0, i] = (-2.0e-6) if is_F else (+1.5e-6)
            # δκ / δI: outer-most F coils enhance elongation
            S[2, i] = (4.0e-8 * abs(c.Z)) if is_F else (-1.0e-8 * abs(c.Z))
            # δδ / δI: F coils with large |Z| → triangularity
            S[3, i] = (2.0e-8 * abs(c.Z)) * (1.0 if is_F else -0.5)

        # T coils + OH: small effect on shape, mainly heat plasma & drive I_p
        return S

    def _circuit_step(
        self, I: np.ndarray, V: np.ndarray, dt: float,
    ) -> np.ndarray:
        """Implicit-Euler step for M·dI/dt + R·I = V.

        I_{k+1} = (M + dt·R)⁻¹ · (M·I_k + dt·V)
        L/R timescale ~ M_diag / R_diag ~ 6 µH / 1 mΩ = 6 ms — implicit-Euler
        is unconditionally stable.
        """
        A = self.M_cc + dt * np.diag(self.R_diag)
        b = self.M_cc @ I + dt * V
        return np.linalg.solve(A, b)

    def step(self, state: PlasmaState, control: Control, dt: float) -> PlasmaState:
        """Advance the simulator by Δt seconds. Pure function, returns NEW state.

        Args:
            state: current PlasmaState
            control: applied voltages + heating
            dt: integration step [s]; recommended 1e-3 (1 ms = control rate)

        Returns:
            next PlasmaState
        """
        s = state

        # Plasma volume from current shape (Wesson §1.4)
        a_eff = self.tcv.a_minor * 0.96  # roughly the ref shape a
        V_plasma = 2.0 * np.pi**2 * s.R_p * a_eff**2 * s.kappa

        # (1) Coil circuit
        I_new = self._circuit_step(s.I_coils, control.V_coils, dt)

        # (2) Loop voltage induced on plasma + plasma current dynamics
        dI_dt = (I_new - s.I_coils) / dt
        V_loop = -float(self.M_pc @ dI_dt)  # induced + sign convention

        # Plasma resistance (Spitzer × calibration for profile/neoclassical)
        T_keV = temperature_from_W(s.W, s.n_bar, V_plasma)
        T_keV = max(T_keV, 0.01)  # floor to avoid /0
        R_plasma = spitzer_resistance(T_keV, s.R_p, a_eff, s.kappa) * self.R_plasma_calib

        # dI_p/dt = (V_loop - R_plasma · I_p) / L_p
        # IMPLICIT Euler — at low T_e, R_plasma can be large (Ω scale) and
        # explicit-Euler with dt=1ms would blow up. Solve:
        #   I_p_new (1 + dt·R/L_p) = I_p + dt·V_loop/L_p
        I_p_new = (s.I_p + dt * V_loop / self.L_p) \
                  / (1.0 + dt * R_plasma / self.L_p)

        # (3) 0D energy balance
        I_p_MA = abs(I_p_new) / 1e6
        B_T_typical = 1.43  # T (held fixed — TF coil is independent)
        eps = self.tcv.epsilon
        P_ohm = R_plasma * I_p_new**2
        P_aux = control.P_aux
        P_loss = max(P_ohm + P_aux, 1e3)  # floor 1 kW to avoid τ→∞
        n_e19 = s.n_bar / 1e19
        n_e19 = max(n_e19, 1e-3)

        tau_E = tau_E_IPB98(
            I_p_MA, B_T_typical, P_loss / 1e6, n_e19,
            s.R_p, eps, s.kappa, M_amu=2.0, H98=self.H98,
        )
        tau_E = max(tau_E, 1e-4)

        dW_dt = P_aux + P_ohm - s.W / tau_E
        W_new = max(s.W + dt * dW_dt, 0.0)

        # (4) Particle balance
        tau_p = 3.0 * tau_E
        dn_dt = control.gas_puff / V_plasma - s.n_bar / tau_p
        n_new = max(s.n_bar + dt * dn_dt, 1e15)

        # (5) Shape response (linearized about reference)
        dI = I_new - self.I_ref
        delta_shape = self.S @ dI  # [δR, δZ, δκ, δδ]
        R_p_new = self.ref_state.R_p + delta_shape[0]
        Z_p_new = self.ref_state.Z_p + delta_shape[1]
        kappa_new = max(self.ref_state.kappa + delta_shape[2], 1.0)  # κ ≥ 1
        delta_new = np.clip(
            self.ref_state.delta + delta_shape[3],
            self.tcv.delta_min, self.tcv.delta_max,
        )

        return PlasmaState(
            I_coils=I_new, I_p=I_p_new, W=W_new, n_bar=n_new,
            R_p=R_p_new, Z_p=Z_p_new,
            kappa=kappa_new, delta=delta_new,
            t=s.t + dt,
        )

    # --- Diagnostic getters (cheap; for reward / safety checks) ---

    def diagnostics(self, state: PlasmaState) -> dict:
        a_eff = self.tcv.a_minor * 0.96
        V_p = 2.0 * np.pi**2 * state.R_p * a_eff**2 * state.kappa
        T_keV = temperature_from_W(state.W, state.n_bar, V_p)
        I_p_MA = abs(state.I_p) / 1e6
        B_T = 1.43
        # Greenwald
        n_GW = (I_p_MA / (np.pi * a_eff**2)) * 1e20
        # q95 cylindrical
        q95 = (5.0 * a_eff**2 * B_T * (1 + state.kappa**2) / 2.0) \
              / (state.R_p * max(I_p_MA, 1e-3))
        # β_max via Troyon
        beta_max = 2.8 * I_p_MA / (a_eff * B_T) if I_p_MA > 0 else 0.0

        return {
            "T_e_keV": T_keV,
            "V_plasma_m3": V_p,
            "I_p_MA": I_p_MA,
            "n_GW_m3": n_GW,
            "q95_cyl": q95,
            "beta_max_pct": beta_max,
            "n_bar": state.n_bar,
            "n_over_nGW": state.n_bar / n_GW if n_GW > 0 else 0.0,
        }


# ---------- Convenience builder ----------

def build_default_simulator() -> tuple[PlasmaSimulator, PlasmaState]:
    """Construct a simulator at a typical TCV operating point (200 kA, T_e~1 keV)."""
    tcv = load_tcv()

    # Nominal coil currents — order-of-magnitude TCV flat-top
    # E coils balanced symmetric, F coils symmetric for shape, OH winds down
    N = tcv.n_control_channels
    I_ref = np.zeros(N)
    # Index map: 0..7 = E1..E8, 8..15 = F1..F8, 16..18 = T1..T3, 19 = OH
    for i, c in enumerate(tcv.shaping_coils):
        if c.name.startswith("E"):
            I_ref[i] = -1500.0  # inboard, polarized
        else:
            I_ref[i] = +2200.0  # outboard, polarized
    I_ref[19] = +5000.0  # OH

    # W = 3 n V T_e (assumes T_e = T_i, Z_eff=1)
    # For T_e=1 keV, n=5e19, V=1.7 m³ → W = 3·5e19·1.7·(1e3·1.602e-19) ≈ 40.8 kJ
    ref_state = PlasmaState(
        I_coils=I_ref.copy(),
        I_p=200_000.0,         # 200 kA
        W=40_800.0,            # 40.8 kJ → T_e ≈ 1 keV
        n_bar=5.0e19,
        R_p=tcv.R_major,
        Z_p=0.0,
        kappa=1.7,
        delta=0.3,
        t=0.0,
    )

    sim = PlasmaSimulator(tcv, I_ref, ref_state)
    return sim, ref_state


if __name__ == "__main__":
    sim, state = build_default_simulator()
    print("Plasma simulator initialized")
    print(f"  Channels      : {sim.N}")
    print(f"  M_cc shape    : {sim.M_cc.shape}, det={np.linalg.det(sim.M_cc):.3e}")
    print(f"  M_cc symmetric: max|M-M.T|={np.max(np.abs(sim.M_cc - sim.M_cc.T)):.2e}")
    eigs = np.linalg.eigvalsh(sim.M_cc)
    print(f"  M_cc eigenvalues range: {eigs.min()*1e6:.3f} .. {eigs.max()*1e6:.3f} µH")
    print(f"  L_plasma      : {sim.L_p*1e6:.3f} µH")
    print(f"  R_coil (diag) : {sim.R_diag[0]*1e3:.3f} mΩ")

    print(f"\nInitial state diagnostics:")
    diag = sim.diagnostics(state)
    for k, v in diag.items():
        print(f"  {k:18s} = {v:.4e}")

    # Test 1: Free decay — no voltage, no aux. Plasma quenches realistically.
    print(f"\n--- Test 1: Free decay (V=0, P_aux=0), 20 ms ---")
    dt = 1e-3
    ctrl_zero = Control(V_coils=np.zeros(sim.N), P_aux=0.0, gas_puff=0.0)
    s = state.copy()
    for k in range(20):
        s = sim.step(s, ctrl_zero, dt)
        if k % 4 == 0:
            d = sim.diagnostics(s)
            print(f"  t={s.t*1e3:5.1f} ms | "
                  f"I_p={d['I_p_MA']*1e3:6.1f} kA | "
                  f"T_e={d['T_e_keV']:6.3f} keV | "
                  f"|I|max={np.max(np.abs(s.I_coils)):7.0f} A | "
                  f"R_p={s.R_p:.4f} m, Z_p={s.Z_p*1e3:+5.1f} mm")

    # Test 2: Sustained — hold voltages at -R·I_ref (steady-state Ohm's law),
    # apply 1 MW aux heating. Plasma should remain near reference.
    print(f"\n--- Test 2: Steady drive (V = R·I_ref + OH ramp, P_aux=1 MW), 30 ms ---")
    V_hold = sim.R_diag * sim.I_ref  # steady-state: V = R·I → no current change
    V_hold[19] += 5.0  # add small OH loop voltage to sustain I_p
    ctrl_drive = Control(V_coils=V_hold, P_aux=1.0e6, gas_puff=1e21)
    s = state.copy()
    for k in range(30):
        s = sim.step(s, ctrl_drive, dt)
        if k % 5 == 0:
            d = sim.diagnostics(s)
            print(f"  t={s.t*1e3:5.1f} ms | "
                  f"I_p={d['I_p_MA']*1e3:6.1f} kA | "
                  f"T_e={d['T_e_keV']:6.3f} keV | "
                  f"W={s.W/1e3:5.1f} kJ | "
                  f"R_p={s.R_p:.4f} m, Z_p={s.Z_p*1e3:+5.1f} mm")
