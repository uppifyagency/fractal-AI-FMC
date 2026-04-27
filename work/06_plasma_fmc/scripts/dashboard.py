"""Streamlit dashboard — interactive plasma shape + FMC tracking viewer.

Run with:
    cd work/06_plasma_fmc
    streamlit run scripts/dashboard.py

Three tabs:
  1. Geometry — TCV cross-section + Miller LCFS playground (sliders for κ, δ)
  2. Simulator — single-step inspection of the plasma simulator
  3. FMC tracking — interactive controller demo with target sliders
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent))

from reference_shapes import ReferenceShape, all_reference_shapes
from tcv_geometry import load_tcv

# JAX setup before import
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
from plasma_simulator_jax import (
    DTYPE,
    build_jax_params,
    make_jit_step,
)


@st.cache_resource
def get_tcv():
    return load_tcv()


@st.cache_resource
def get_simulator():
    sim_p, x0 = build_jax_params()
    step = make_jit_step(sim_p)
    # Warm jit
    V = jnp.zeros(sim_p.N, dtype=DTYPE)
    P = jnp.asarray(0.0, dtype=DTYPE)
    g = jnp.asarray(0.0, dtype=DTYPE)
    dt = jnp.asarray(1e-3, dtype=DTYPE)
    step(x0, V, P, g, dt).block_until_ready()
    return sim_p, x0, step


def page_geometry():
    st.header("TCV geometry + Miller plasma shapes")
    st.markdown(
        "Adjust **κ** (elongation) and **δ** (triangularity) to see the Miller LCFS update. "
        "Coil positions are loaded from the validated `tcv_geometry.yaml` "
        "(cross-checked against `freegs.machine.TCV()`)."
    )

    tcv = get_tcv()

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Plasma shape")
        kappa = st.slider("κ (elongation)", min_value=1.0, max_value=2.8,
                          value=1.7, step=0.05)
        delta = st.slider("δ (triangularity)", min_value=-0.7, max_value=1.0,
                          value=0.3, step=0.05)
        Z0 = st.slider("Z₀ (vertical offset) [cm]", min_value=-30.0, max_value=30.0,
                       value=0.0, step=1.0)
        a_frac = st.slider("a fraction of TCV minor radius",
                           min_value=0.5, max_value=1.0, value=0.96, step=0.02)

        # Derived quantities
        a_eff = tcv.a_minor * a_frac
        V = 2 * np.pi**2 * tcv.R_major * a_eff**2 * kappa
        S = 4 * np.pi**2 * tcv.R_major * a_eff * np.sqrt((1 + kappa**2) / 2)
        eps = a_eff / tcv.R_major
        st.markdown(
            f"**Derived (Wesson §1.4):**\n"
            f"- a_eff = {a_eff*100:.1f} cm\n"
            f"- ε = {eps:.3f}\n"
            f"- V = {V:.3f} m³\n"
            f"- S = {S:.3f} m²"
        )

        I_p_kA = st.number_input("I_p [kA]", min_value=10.0, max_value=1000.0,
                                  value=200.0, step=10.0)
        B_T = st.number_input("B_T [T]", min_value=0.5, max_value=1.5,
                               value=1.43, step=0.1)
        I_p_MA = I_p_kA / 1000.0
        n_GW = (I_p_MA / (np.pi * a_eff**2)) * 1e20
        q95 = (5.0 * a_eff**2 * B_T * (1 + kappa**2) / 2.0) \
              / (tcv.R_major * I_p_MA)
        beta_max = 2.8 * I_p_MA / (a_eff * B_T)
        st.markdown(
            f"**Operating point (REFERENCES §D):**\n"
            f"- n_GW = {n_GW:.2e} m⁻³ (Greenwald)\n"
            f"- q₉₅ ≈ {q95:.2f} (Wesson §3.6)\n"
            f"- β_max ≈ {beta_max:.2f}% (Troyon 2.8)"
        )

    with col2:
        # Plot
        fig, ax = plt.subplots(figsize=(8, 9))
        # Vessel rectangle
        vrect = plt.Rectangle(
            (tcv.vessel["inner_R"], -tcv.vessel["height"] / 2),
            tcv.vessel["outer_R"] - tcv.vessel["inner_R"],
            tcv.vessel["height"],
            linewidth=1, edgecolor="gray", facecolor="lightyellow", alpha=0.3,
        )
        ax.add_patch(vrect)

        # Coils
        e_coils = [c for c in tcv.shaping_coils if c.name.startswith("E")]
        f_coils = [c for c in tcv.shaping_coils if c.name.startswith("F")]
        ax.scatter([c.R for c in e_coils], [c.Z for c in e_coils],
                   s=120, c="royalblue", marker="s", edgecolor="black", label="E coils")
        ax.scatter([c.R for c in f_coils], [c.Z for c in f_coils],
                   s=120, c="crimson", marker="s", edgecolor="black", label="F coils")
        ax.scatter([c.R for c in tcv.t_coils], [c.Z for c in tcv.t_coils],
                   s=80, c="orange", marker="^", edgecolor="black", label="T coils")
        ax.scatter([c.R for c in tcv.ohmic_coils], [c.Z for c in tcv.ohmic_coils],
                   s=100, c="forestgreen", marker="D", edgecolor="black", label="OH C/D")
        # Solenoid
        sol = tcv.solenoid
        srect = plt.Rectangle(
            (sol["R"] - 0.02, sol["Z_min"]), 0.04,
            sol["Z_max"] - sol["Z_min"],
            linewidth=1, edgecolor="darkgreen", facecolor="forestgreen", alpha=0.5,
        )
        ax.add_patch(srect)

        # Live LCFS from sliders
        shape = ReferenceShape(
            name="interactive", R0=tcv.R_major, Z0=Z0 / 100.0, a=a_eff,
            kappa=kappa, delta=delta, description="",
        )
        R_lcfs, Z_lcfs = shape.lcfs(300)
        R_lcfs = np.append(R_lcfs, R_lcfs[0])
        Z_lcfs = np.append(Z_lcfs, Z_lcfs[0])
        ax.fill(R_lcfs, Z_lcfs, color="lightcoral", alpha=0.4)
        ax.plot(R_lcfs, Z_lcfs, "-", color="darkred", linewidth=2.5,
                label=f"Plasma (κ={kappa}, δ={delta:+.2f})")
        ax.plot(tcv.R_major, Z0 / 100.0, "k+", markersize=12, mew=2)

        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(0.2, 1.9)
        ax.set_ylim(-1.4, 1.4)
        ax.set_title("TCV poloidal cross-section + interactive plasma shape")

        st.pyplot(fig)
        plt.close(fig)


def page_simulator():
    st.header("Plasma simulator — interactive single-step")
    st.markdown(
        "Apply a control input and see how the simulator state evolves "
        "over `n_steps` × `dt = 1 ms` integration."
    )

    sim_p, x0, sim_step = get_simulator()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control input")
        V_scale = st.slider("V_coils scale (multiplier on V_ref)",
                            min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        V_jitter = st.slider("V_coils jitter [V]",
                             min_value=0.0, max_value=200.0, value=0.0, step=10.0)
        P_aux_MW = st.slider("P_aux [MW]",
                             min_value=0.0, max_value=4.0, value=0.5, step=0.1)
        gas_puff_e21 = st.slider("Gas puff [10²¹ s⁻¹]",
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        n_steps = st.slider("Number of 1ms steps",
                            min_value=1, max_value=200, value=50, step=10)

    # V_ref = R · I_ref
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
    V = V_scale * V_ref + np.random.default_rng(0).normal(0, V_jitter, sim_p.N)

    # Run rollout
    x = np.asarray(x0).copy()
    log = []
    t0 = time.perf_counter()
    for k in range(n_steps):
        x_jax = sim_step(
            jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(P_aux_MW * 1e6, dtype=DTYPE),
            jnp.asarray(gas_puff_e21 * 1e21, dtype=DTYPE),
            jnp.asarray(1e-3, dtype=DTYPE),
        )
        x = np.asarray(x_jax)
        N = sim_p.N
        log.append({
            "t_ms": (k + 1) * 1.0,
            "I_p_kA": float(x[N]) / 1e3,
            "T_e_keV": float(x[N + 1] / (3 * x[N + 2] * 2 * np.pi**2 * x[N + 3]
                                          * (sim_p.a_eff)**2 * x[N + 5]) / 1.602e-16),
            "n_bar_e19": float(x[N + 2]) / 1e19,
            "R_p": float(x[N + 3]),
            "Z_p_mm": float(x[N + 4]) * 1e3,
            "kappa": float(x[N + 5]),
            "delta": float(x[N + 6]),
        })
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    with col2:
        st.subheader(f"Result ({elapsed_ms:.1f} ms wall-clock for {n_steps} sim steps)")
        t = [r["t_ms"] for r in log]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(t, [r["I_p_kA"] for r in log], "b-")
        axes[0, 0].set_xlabel("t [ms]"); axes[0, 0].set_ylabel("I_p [kA]")
        axes[0, 0].grid(alpha=0.3); axes[0, 0].set_title("Plasma current")
        axes[0, 1].plot(t, [r["T_e_keV"] for r in log], "r-")
        axes[0, 1].set_xlabel("t [ms]"); axes[0, 1].set_ylabel("T_e [keV]")
        axes[0, 1].grid(alpha=0.3); axes[0, 1].set_title("Temperature")
        axes[1, 0].plot(t, [r["R_p"] for r in log], "g-")
        axes[1, 0].set_xlabel("t [ms]"); axes[1, 0].set_ylabel("R_p [m]")
        axes[1, 0].grid(alpha=0.3); axes[1, 0].set_title("Centroid R")
        axes[1, 1].plot(t, [r["Z_p_mm"] for r in log], "m-")
        axes[1, 1].set_xlabel("t [ms]"); axes[1, 1].set_ylabel("Z_p [mm]")
        axes[1, 1].grid(alpha=0.3); axes[1, 1].set_title("Centroid Z")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"**Final state:** I_p = {log[-1]['I_p_kA']:.1f} kA, "
                    f"T_e = {log[-1]['T_e_keV']:.2f} keV, "
                    f"R_p = {log[-1]['R_p']:.4f} m, "
                    f"κ = {log[-1]['kappa']:.3f}")


def page_tracking():
    st.header("FMC tracking log viewer")
    st.markdown(
        "Visualize the latest FMC tracking experiment from "
        "`results/milestone_3_tracking.json`. Re-run "
        "`python scripts/fmc_plasma.py` to refresh."
    )

    log_path = Path(__file__).parent.parent / "results" / "milestone_3_tracking.json"
    if not log_path.exists():
        st.error(f"No tracking log found at {log_path}. "
                 "Run `python scripts/fmc_plasma.py` first.")
        return

    with open(log_path) as f:
        data = json.load(f)

    target = data["target"]
    log = data["log"]
    t = [r["t_ms"] for r in log]

    cols = st.columns(4)
    cols[0].metric("Target R_p", f"{target['R_p']} m")
    cols[1].metric("Target Z_p", f"{target['Z_p']} m")
    cols[2].metric("Target κ", f"{target['kappa']}")
    cols[3].metric("Target δ", f"{target['delta']:+.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    plots = [
        (axes[0, 0], "R_p", target["R_p"], "R_p [m]", "Centroid R"),
        (axes[0, 1], "Z_p", target["Z_p"], "Z_p [m]", "Centroid Z"),
        (axes[0, 2], "kappa", target["kappa"], "κ", "Elongation"),
        (axes[1, 0], "delta", target["delta"], "δ", "Triangularity"),
    ]
    for ax, key, tgt, ylabel, title in plots:
        ax.plot(t, [r[key] for r in log], "-", linewidth=2)
        ax.axhline(tgt, color="r", ls="--", label=f"target {tgt}")
        ax.set_xlabel("t [ms]"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    axes[1, 1].plot(t, [r["I_p_kA"] for r in log], "k-")
    axes[1, 1].set_xlabel("t [ms]"); axes[1, 1].set_ylabel("I_p [kA]")
    axes[1, 1].set_title("Plasma current"); axes[1, 1].grid(alpha=0.3)
    axes[1, 2].plot(t, [r["alive"] for r in log], "g-")
    axes[1, 2].set_xlabel("t [ms]"); axes[1, 2].set_ylabel("count")
    axes[1, 2].set_title(f"Walkers alive (M={data['config']['n_walkers']})")
    axes[1, 2].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Raw config"):
        st.json(data["config"])


def main():
    st.set_page_config(page_title="TCV FMC Plasma Control",
                       layout="wide", page_icon="🔥")
    st.title("TCV FMC Plasma Control — research dashboard")
    st.caption(
        "Milestones 1-4 of `work/06_plasma_fmc/`. "
        "Geometry validated against `freegs.machine.TCV()`. "
        "Physics: Wesson 4th ed. + IPB98(y,2) + Spitzer × 0.005 calibrated."
    )

    page = st.sidebar.radio(
        "Page",
        ["1. Geometry", "2. Simulator", "3. FMC tracking"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Sources** (all in [REFERENCES.md](../REFERENCES.md)):\n\n"
        "- TCV geometry: EPFL LRP-755-13 + freegs\n"
        "- Miller LCFS: Phys. Plasmas 5:973 (1998)\n"
        "- IPB98(y,2): Nucl. Fusion 39:2175 (1999)\n"
        "- FMC: arXiv:1803.05049v5 (Hernández-Cerezo & Duran-Ballester)"
    )

    if page.startswith("1"):
        page_geometry()
    elif page.startswith("2"):
        page_simulator()
    else:
        page_tracking()


if __name__ == "__main__":
    main()
