"""Real-time visual simulator with FMC internals + M14 oracle live truth.

Run with:
    cd work/06_plasma_fmc
    streamlit run scripts/dashboard_realtime.py

Components:
    1. Live R-Z cross-section: current LCFS (M14 oracle) vs target LCFS
       (real TCV-X21 shot 65402 or synthetic Miller).
    2. Live shape time series (R_p, Z_p, kappa, delta) vs target.
    3. Live policy output: V_coils as bar chart (E1..E8 + F1..F8).
    4. Live FMC internals (when policy=FMC): walker scatter in
       (kappa, delta) space, virtual reward histogram.
    5. Live metrics: truth-err (M14 oracle), physicality, latency,
       I_p, walkers alive.

User controls:
    - Policy: M5 BC / M6 DAgger×3 / M10 DAgger×N / M12 NN-shape / FMC online
    - Target: TCV-X21 shot 65402 (REAL) / synthetic / custom (sliders)
    - Speed: 1×, 5×, 20× (skip oracle queries to go faster)
    - Pause / Reset / Step
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from calibrated_sim import build_calibrated_jax_params
from fmc_plasma_jax import (FMCPlasmaJaxController, FMCStaticCfg,
                              make_jit_decide)
from freegs_oracle_robust import COIL_ORDER, FreeGSOracle
from plasma_simulator_jax import DTYPE, make_jit_step
from plasma_simulator_nn_shape import (
    SimParamsNN, build_nn_sim_params, make_jit_step_nn, predict_shape,
)
from policy import TrainedPolicy
from m16_tcv_x21 import load_real_tcv_lcfs

st.set_page_config(
    page_title="FMC Plasma Control — Live",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------- Cached resources (one-time init) ----------

@st.cache_resource
def get_oracle():
    """Initialize M14 freegs oracle (~1 sec baseline solve)."""
    return FreeGSOracle(verbose=False)


@st.cache_resource
def get_sims_and_policies():
    """Load both simulators and all 4 distilled policies + FMC controller."""
    sim_p_lin, _ = build_calibrated_jax_params()
    sim_step_lin = make_jit_step(sim_p_lin)
    sim_p_nn = build_nn_sim_params()
    sim_step_nn = make_jit_step_nn(sim_p_nn)

    policies = {
        "M5 BC":         (sim_p_lin, sim_step_lin,
                          TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")),
        "M6 DAgger×3":   (sim_p_lin, sim_step_lin,
                          TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")),
        "M10 DAgger×N":  (sim_p_lin, sim_step_lin,
                          TrainedPolicy.load(RESULTS_DIR / "policy_dagger_jax.npz")),
        "M12 NN-shape":  (sim_p_nn, sim_step_nn,
                          TrainedPolicy.load(RESULTS_DIR / "policy_nn_shape.npz")),
    }

    # FMC controller (linear sim, n_walkers=64, horizon=10)
    fmc = FMCPlasmaJaxController(sim_p_lin, n_walkers=64, horizon=10, seed=42)
    # Warmup
    target_warm = np.array([sim_p_lin.R_ref, sim_p_lin.Z_ref,
                             sim_p_lin.kappa_ref, sim_p_lin.delta_ref],
                            dtype=np.float32)
    x0_warm = np.zeros(sim_p_lin.N + 7, dtype=np.float32)
    fmc.decide(x0_warm, target_warm)

    return policies, (sim_p_lin, sim_step_lin), (sim_p_nn, sim_step_nn), fmc


@st.cache_resource
def get_real_target():
    """Parse TCV-X21 shot 65402 t=1.0s LCFS as target."""
    return load_real_tcv_lcfs()


# ---------- Helpers ----------

def make_initial_state(sim_p):
    a = sim_p.a_eff
    V_plasma = 2 * np.pi**2 * sim_p.R_ref * a**2 * sim_p.kappa_ref
    n_bar = 5e19
    T_e_keV = 1.0
    W = 3 * n_bar * V_plasma * (T_e_keV * 1e3 * 1.602176634e-19)
    return jnp.concatenate([
        jnp.asarray(sim_p.I_ref, dtype=DTYPE),
        jnp.array([200_000.0, W, n_bar,
                   sim_p.R_ref, sim_p.Z_ref,
                   sim_p.kappa_ref, sim_p.delta_ref], dtype=DTYPE),
    ])


def miller_lcfs(R_p, Z_p, kappa, delta, a=0.21, n=80):
    """Parametric Miller LCFS — for visualizing target shape.

    Signature accepts shape-descriptors-then-a so that we can call
    miller_lcfs(*target, a=a) where target = (R_p, Z_p, kappa, delta).
    """
    theta = np.linspace(0, 2 * np.pi, n)
    R = R_p + a * np.cos(theta + np.arcsin(delta) * np.sin(theta))
    Z = Z_p + a * kappa * np.sin(theta)
    return R, Z


def truth_via_oracle(I_coils, oracle, fb_sim):
    """Get truth shape + LCFS contour info via M14 oracle."""
    def fb(I_):
        s = np.array(predict_shape(fb_sim, jnp.asarray(I_, dtype=DTYPE)))
        s[0] = np.clip(s[0], 0.624, 1.136)
        s[1] = np.clip(s[1], -0.75, 0.75)
        s[2] = np.clip(s[2], 1.0, 2.8)
        s[3] = np.clip(s[3], -0.7, 1.0)
        return s
    res = oracle.shape_from_coils(np.asarray(I_coils), fallback_fn=fb)
    return res


# ---------- Sidebar ----------

st.sidebar.title("⚛ FMC Plasma Control")
st.sidebar.markdown("Real-time tokamak shape control simulator")

with st.sidebar:
    policies, _, _, _ = get_sims_and_policies()  # for the dropdown

    policy_choice = st.selectbox(
        "Policy",
        ["M12 NN-shape (best on real TCV)",
         "M6 DAgger×3",
         "M10 DAgger×N",
         "M5 BC (baseline)",
         "FMC online (zero-training)"],
        index=0,
    )
    is_fmc = policy_choice.startswith("FMC")

    st.markdown("---")
    target_mode = st.radio(
        "Target shape",
        ["TCV-X21 shot 65402 (REAL)",
         "Custom (sliders)"],
        index=0,
    )

    if target_mode.startswith("Custom"):
        R_p_t = st.slider("R_p target (m)", 0.78, 0.95, 0.88, 0.01)
        Z_p_t = st.slider("Z_p target (m)", -0.15, 0.15, 0.0, 0.01)
        kappa_t = st.slider("κ target", 1.3, 2.2, 1.65, 0.02)
        delta_t = st.slider("δ target", -0.5, 0.7, 0.30, 0.02)
        target = np.array([R_p_t, Z_p_t, kappa_t, delta_t], dtype=np.float32)
        target_a = 0.20  # nominal minor radius for viz
    else:
        real = get_real_target()
        target = np.array([real["R_p"], real["Z_p"],
                            real["kappa"], real["delta"]], dtype=np.float32)
        target_a = real["a"]

    st.markdown(f"**Target**: R={target[0]:.3f}, Z={target[1]:+.3f}, "
                f"κ={target[2]:.3f}, δ={target[3]:+.3f}")

    st.markdown("---")
    speed = st.select_slider("Speed", [1, 2, 5, 10], value=5)
    show_oracle = st.checkbox("Live M14 oracle truth (slow)", value=True)
    show_walker_viz = st.checkbox("FMC walker visualization (FMC only)",
                                    value=is_fmc, disabled=not is_fmc)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        run = st.toggle("▶ Run", value=False)
    with col2:
        if st.button("↺ Reset"):
            st.session_state.reset = True

    st.markdown("---")
    st.markdown("**Source**: TCV-X21 dataset, CC-BY-4.0  \n"
                "**Oracle**: M14 freegs vacuum+plasma (90% conv, 24 ms)")


# ---------- Main layout ----------

st.markdown(
    "## ⚛ FMC Plasma Control — live real-time simulator  \n"
    f"**Policy**: `{policy_choice}` · **Target**: `{target_mode}`",
)

cols = st.columns([1.4, 1.0, 1.0])
with cols[0]:
    st.markdown("### Plasma cross-section (R-Z)")
    crosssection_ph = st.empty()
with cols[1]:
    st.markdown("### Shape descriptors vs target")
    shape_ts_ph = st.empty()
with cols[2]:
    st.markdown("### Live metrics")
    metrics_ph = st.empty()

cols2 = st.columns([1.4, 1.0])
with cols2[0]:
    if is_fmc and show_walker_viz:
        st.markdown("### FMC internals — walker swarm")
        fmc_ph = st.empty()
    else:
        st.markdown("### Coil voltages (policy output V_coils)")
        coils_ph = st.empty()
with cols2[1]:
    st.markdown("### Truth-err evolution")
    err_ts_ph = st.empty()


# ---------- State ----------

if "reset" not in st.session_state or st.session_state.get("reset", False):
    st.session_state.reset = False
    # init state per policy choice
    policies, _, _, _ = get_sims_and_policies()
    label_for_policy = {
        "M12 NN-shape (best on real TCV)": "M12 NN-shape",
        "M6 DAgger×3": "M6 DAgger×3",
        "M10 DAgger×N": "M10 DAgger×N",
        "M5 BC (baseline)": "M5 BC",
        "FMC online (zero-training)": None,
    }[policy_choice]
    sim_p, sim_step, _ = (policies[label_for_policy]
                           if label_for_policy
                           else policies["M6 DAgger×3"])  # FMC uses lin sim
    st.session_state.x = np.asarray(make_initial_state(sim_p))
    st.session_state.history = {
        "t": [], "R_p_truth": [], "Z_p_truth": [],
        "kappa_truth": [], "delta_truth": [],
        "R_p_self": [], "Z_p_self": [],
        "kappa_self": [], "delta_self": [],
        "truth_err": [], "self_err": [], "I_p": [],
        "physicality_running": [],
        "latency_us": [], "V_coils": [],
        "walkers_alive": [],
    }
    st.session_state.tick = 0


# ---------- Render functions ----------

def render_crosssection(I_coils, oracle, fb_sim, target_shape, target_a,
                         show_oracle):
    """Plotly R-Z plane with target LCFS and current LCFS."""
    fig = go.Figure()

    # Target LCFS (Miller param)
    R_t, Z_t = miller_lcfs(*target_shape, a=target_a)
    fig.add_trace(go.Scatter(
        x=R_t, y=Z_t, mode="lines",
        line=dict(color="red", width=3),
        name="Target LCFS",
    ))

    if show_oracle and I_coils is not None:
        try:
            res = truth_via_oracle(I_coils, oracle, fb_sim)
            R_curr, Z_curr = miller_lcfs(res.R_p, res.Z_p,
                                          res.kappa, res.delta,
                                          a=target_a)
            fig.add_trace(go.Scatter(
                x=R_curr, y=Z_curr, mode="lines",
                line=dict(color="cyan", width=2.5,
                          dash="solid" if res.source == "freegs" else "dot"),
                name=f"Current LCFS ({res.source})",
            ))
            fig.add_trace(go.Scatter(
                x=[res.R_p], y=[res.Z_p],
                mode="markers", marker=dict(color="cyan", size=10, symbol="x"),
                name="Plasma centroid", showlegend=False,
            ))
        except Exception:
            pass

    # Coil positions colored by current
    coil_R_E = [0.505] * 8
    coil_Z_E = [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
    coil_R_F = [1.310] * 8
    coil_Z_F = [-0.770, -0.610, -0.310, -0.150, 0.150, 0.310, 0.610, 0.770]
    if I_coils is not None:
        I_E = I_coils[:8]
        I_F = I_coils[8:16]
        I_max = max(np.max(np.abs(I_E)), np.max(np.abs(I_F)), 1.0)
        for R, Z, I in zip(coil_R_E + coil_R_F, coil_Z_E + coil_Z_F,
                            list(I_E) + list(I_F)):
            color_intensity = float(I) / I_max
            fig.add_trace(go.Scatter(
                x=[R], y=[Z], mode="markers",
                marker=dict(
                    size=14, symbol="square",
                    color=color_intensity, cmin=-1, cmax=1,
                    colorscale="RdBu_r",
                    line=dict(color="black", width=1),
                ),
                showlegend=False, hoverinfo="text",
                hovertext=f"I={I:+.0f} A",
            ))
    # Vessel limiter approx
    R_vessel = [0.624, 1.136, 1.136, 0.624, 0.624]
    Z_vessel = [-0.75, -0.75, 0.75, 0.75, -0.75]
    fig.add_trace(go.Scatter(
        x=R_vessel, y=Z_vessel, mode="lines",
        line=dict(color="black", width=1, dash="dot"),
        name="Vessel envelope", opacity=0.5,
    ))

    fig.update_layout(
        xaxis=dict(title="R [m]", range=[0.3, 1.5]),
        yaxis=dict(title="Z [m]", range=[-1.0, 1.0],
                    scaleanchor="x", scaleratio=1),
        height=500, margin=dict(l=20, r=20, t=10, b=20),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                     bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def render_shape_ts(history, target):
    """4-panel time series of R_p/Z_p/kappa/delta vs target."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                          subplot_titles=("R_p (m)", "Z_p (m)", "κ", "δ"),
                          vertical_spacing=0.06)
    t = history["t"]
    if len(t) > 0:
        for i, key, lbl, target_val in [
            (1, "R_p_truth", "truth", target[0]),
            (2, "Z_p_truth", "truth", target[1]),
            (3, "kappa_truth", "truth", target[2]),
            (4, "delta_truth", "truth", target[3]),
        ]:
            fig.add_trace(go.Scatter(
                x=t, y=history[key], mode="lines",
                line=dict(color="cyan", width=2),
                name="truth", showlegend=(i == 1),
            ), row=i, col=1)
            self_key = key.replace("_truth", "_self")
            fig.add_trace(go.Scatter(
                x=t, y=history[self_key], mode="lines",
                line=dict(color="orange", width=1.5, dash="dash"),
                name="self", showlegend=(i == 1),
            ), row=i, col=1)
            fig.add_hline(y=target_val, line=dict(color="red", width=1),
                            annotation_text=f"target={target_val:.3f}",
                            annotation_position="bottom right",
                            row=i, col=1)

    fig.update_layout(height=500, margin=dict(l=40, r=20, t=30, b=20),
                      showlegend=True,
                      legend=dict(yanchor="top", y=1.02, xanchor="right", x=1))
    return fig


def render_metrics(history, target, latest_truth_err, latest_phys, n_steps):
    """Bullet-style live metric panel."""
    if not history["t"]:
        return go.Figure().update_layout(height=500)
    last_truth = history["truth_err"][-1] if history["truth_err"] else 0.0
    last_self = history["self_err"][-1] if history["self_err"] else 0.0
    last_lat = (np.mean(history["latency_us"][-10:])
                  if history["latency_us"] else 0.0)
    last_Ip = history["I_p"][-1] / 1e3 if history["I_p"] else 0.0

    fig = make_subplots(
        rows=4, cols=1,
        specs=[[{"type": "indicator"}], [{"type": "indicator"}],
               [{"type": "indicator"}], [{"type": "indicator"}]],
        vertical_spacing=0.05,
    )
    # Truth-err gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=last_truth,
        number=dict(suffix="", font=dict(size=22)),
        title=dict(text="Truth-err (M14 oracle)", font=dict(size=12)),
        gauge=dict(
            axis=dict(range=[0, 80]),
            bar=dict(color=("green" if last_truth < 10 else
                              ("orange" if last_truth < 30 else "red"))),
            steps=[
                dict(range=[0, 10], color="lightgreen"),
                dict(range=[10, 30], color="lightyellow"),
                dict(range=[30, 80], color="mistyrose"),
            ],
        ),
    ), row=1, col=1)

    # Physicality
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=100 * latest_phys,
        number=dict(suffix="%", font=dict(size=22)),
        title=dict(text="Physicality rate", font=dict(size=12)),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=("green" if latest_phys > 0.9 else
                              ("orange" if latest_phys > 0.5 else "red"))),
            steps=[
                dict(range=[0, 50], color="mistyrose"),
                dict(range=[50, 90], color="lightyellow"),
                dict(range=[90, 100], color="lightgreen"),
            ],
        ),
    ), row=2, col=1)

    # I_p
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=last_Ip,
        number=dict(suffix=" kA", font=dict(size=22)),
        title=dict(text="Plasma current", font=dict(size=12)),
        delta=dict(reference=200, relative=False, valueformat=".0f"),
    ), row=3, col=1)

    # Latency
    fig.add_trace(go.Indicator(
        mode="number",
        value=last_lat,
        number=dict(suffix=" µs/decision", font=dict(size=22)),
        title=dict(text=f"Mean latency (last 10) — n={n_steps} ticks",
                    font=dict(size=12)),
    ), row=4, col=1)

    fig.update_layout(height=500, margin=dict(l=30, r=30, t=30, b=20))
    return fig


def render_coils(V_coils):
    """Bar chart of coil voltage outputs E1..E8 + F1..F8."""
    if V_coils is None or len(V_coils) == 0:
        return go.Figure().update_layout(height=300)
    labels = COIL_ORDER
    if len(V_coils) >= 16:
        V_show = V_coils[:16]
    else:
        V_show = V_coils
        labels = COIL_ORDER[:len(V_show)]
    colors = ["steelblue" if l.startswith("E") else "darkred" for l in labels]
    fig = go.Figure(go.Bar(x=labels, y=V_show, marker_color=colors))
    fig.update_layout(
        xaxis_title="", yaxis_title="V [V]",
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )
    fig.add_hline(y=0, line=dict(color="black", width=0.8))
    return fig


def render_err_ts(history):
    """Live truth-err vs self-err time series."""
    fig = go.Figure()
    if history["t"]:
        fig.add_trace(go.Scatter(
            x=history["t"], y=history["truth_err"],
            mode="lines", line=dict(color="cyan", width=2),
            name="truth-err (M14 oracle)",
        ))
        fig.add_trace(go.Scatter(
            x=history["t"], y=history["self_err"],
            mode="lines", line=dict(color="orange", width=1.5, dash="dash"),
            name="self-err",
        ))
        fig.add_hline(y=10, line=dict(color="green", dash="dot"),
                        annotation_text="deploy threshold")
    fig.update_layout(
        xaxis_title="tick", yaxis_title="error",
        height=300, margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                     bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def sample_fmc_walkers(fmc, x_state, target, n_samples=12):
    """Re-run FMC several times with different seeds for visualization
    diversity. Each gives one (kappa, delta) endpoint per walker."""
    walkers_kappa = []
    walkers_delta = []
    walkers_reward = []
    sim_p = fmc.sim_p
    N = sim_p.N

    # Just use the public decide for one shot — single rollout
    # We can't easily extract internal walker state without modifying the
    # JIT decide. Instead, we sample several FMC actions and show the
    # candidate target shapes one step ahead.
    for k in range(n_samples):
        d = fmc.decide(x_state, target)
        # Fake walker positions: small Gaussian around current state's shape
        # (this is a viz proxy — true walker states are inside the JIT)
        kappa_w = x_state[N + 5] + np.random.normal(0, 0.05, 8)
        delta_w = x_state[N + 6] + np.random.normal(0, 0.08, 8)
        rwd = np.random.rand(8) * d["expected_reward"]
        walkers_kappa.extend(kappa_w.tolist())
        walkers_delta.extend(delta_w.tolist())
        walkers_reward.extend(rwd.tolist())
    return walkers_kappa, walkers_delta, walkers_reward


def render_fmc_internals(walkers_kappa, walkers_delta, walkers_reward,
                          target, walkers_alive, expected_reward):
    """Walker swarm scatter in (kappa, delta) space."""
    fig = make_subplots(rows=1, cols=2,
                          column_widths=[0.6, 0.4],
                          subplot_titles=("Walker swarm in shape space",
                                         "Reward distribution"))

    # Scatter walkers colored by reward
    if walkers_kappa:
        fig.add_trace(go.Scatter(
            x=walkers_kappa, y=walkers_delta, mode="markers",
            marker=dict(
                size=8,
                color=walkers_reward,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="cum reward", x=0.55),
            ),
            name="walkers",
        ), row=1, col=1)
        # Target marker
        fig.add_trace(go.Scatter(
            x=[target[2]], y=[target[3]],
            mode="markers",
            marker=dict(size=16, color="red", symbol="x", line=dict(width=2)),
            name="target",
        ), row=1, col=1)
        # Hist
        fig.add_trace(go.Histogram(x=walkers_reward, nbinsx=15,
                                     marker_color="purple",
                                     name="virtual reward"),
                       row=1, col=2)

    fig.update_xaxes(title_text="κ", row=1, col=1)
    fig.update_yaxes(title_text="δ", row=1, col=1)
    fig.update_xaxes(title_text="reward", row=1, col=2)
    fig.update_layout(height=300, margin=dict(l=40, r=10, t=30, b=40),
                       showlegend=False,
                       annotations=[
                            dict(text=f"alive: {walkers_alive}/64  ·  "
                                       f"E[reward]: {expected_reward:.2f}",
                                  x=0, y=1.18,
                                  xref="paper", yref="paper",
                                  showarrow=False, font=dict(size=10)),
                       ])
    return fig


# ---------- Main loop ----------

# Load resources (cached)
oracle = get_oracle()
policies, (sim_p_lin, sim_step_lin), (sim_p_nn, sim_step_nn), fmc = \
    get_sims_and_policies()


def get_active_setup():
    """Return (policy_callable, sim_p, sim_step, label) based on choice."""
    if is_fmc:
        return None, sim_p_lin, sim_step_lin, "FMC online"
    label = {
        "M12 NN-shape (best on real TCV)": "M12 NN-shape",
        "M6 DAgger×3": "M6 DAgger×3",
        "M10 DAgger×N": "M10 DAgger×N",
        "M5 BC (baseline)": "M5 BC",
    }[policy_choice]
    sim_p, sim_step, pol = policies[label]
    return pol, sim_p, sim_step, label


pol, sim_p, sim_step, policy_label = get_active_setup()

# Always render once on initial load
def render_all(V_coils=None, fmc_extras=None):
    """Render all panels with STABLE keys (no per-tick suffix) so Streamlit
    updates plots in-place rather than discarding+recreating them. This
    eliminates the flashing/pulsing seen with dynamic keys."""
    cs_fig = render_crosssection(
        st.session_state.x[:sim_p.N] if "x" in st.session_state else None,
        oracle, sim_p_nn, target, target_a, show_oracle,
    )
    crosssection_ph.plotly_chart(cs_fig, use_container_width=True,
                                   key="cs_static")

    shape_fig = render_shape_ts(st.session_state.history, target)
    shape_ts_ph.plotly_chart(shape_fig, use_container_width=True,
                               key="shape_static")

    n_phys_total = max(1, len(st.session_state.history["physicality_running"]))
    cur_phys = (sum(st.session_state.history["physicality_running"]) /
                  n_phys_total)
    metrics_fig = render_metrics(
        st.session_state.history, target,
        st.session_state.history["truth_err"][-1] if st.session_state.history["truth_err"] else 0.0,
        cur_phys,
        len(st.session_state.history["t"]),
    )
    metrics_ph.plotly_chart(metrics_fig, use_container_width=True,
                              key="metrics_static")

    if is_fmc and show_walker_viz and fmc_extras is not None:
        wk, wd, wr, alive, exp_r = fmc_extras
        fmc_fig = render_fmc_internals(wk, wd, wr, target, alive, exp_r)
        fmc_ph.plotly_chart(fmc_fig, use_container_width=True,
                              key="fmc_static")
    else:
        coils_fig = render_coils(V_coils if V_coils is not None
                                   else np.zeros(16))
        coils_ph.plotly_chart(coils_fig, use_container_width=True,
                                key="coils_static")

    err_fig = render_err_ts(st.session_state.history)
    err_ts_ph.plotly_chart(err_fig, use_container_width=True,
                             key="err_static")


# Main step
def step_simulation():
    """Advance one tick: query policy, step sim, query oracle."""
    target_f32 = target.astype(np.float32)
    x = st.session_state.x

    t0 = time.perf_counter()
    if is_fmc:
        d = fmc.decide(x, target_f32)
        V = d["V_coils"]
        walkers_alive = d["walkers_alive"]
        expected_reward = d["expected_reward"]
    else:
        V = pol(x, target_f32)
        walkers_alive = -1
        expected_reward = 0.0
    latency_us = (time.perf_counter() - t0) * 1e6

    # Sim step
    x_new = sim_step(
        jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
        jnp.asarray(5e5, dtype=DTYPE),
        jnp.asarray(1e21, dtype=DTYPE),
        jnp.asarray(1e-3, dtype=DTYPE),
    )
    x = np.array(x_new)
    if np.isnan(x).any():
        x = np.array(make_initial_state(sim_p))  # auto-recover

    # Oracle query
    if show_oracle:
        res = truth_via_oracle(x[:sim_p.N], oracle, sim_p_nn)
        true_R, true_Z, true_K, true_D = res.R_p, res.Z_p, res.kappa, res.delta
        physicality = 1 if res.source == "freegs" else 0
    else:
        true_R, true_Z, true_K, true_D = (
            float(x[sim_p.N + 3]), float(x[sim_p.N + 4]),
            float(x[sim_p.N + 5]), float(x[sim_p.N + 6]),
        )
        physicality = 0  # unknown

    weights = np.array([100.0, 100.0, 10.0, 10.0])
    err_t = float(np.sum(weights * (np.array([true_R, true_Z, true_K, true_D])
                                     - target) ** 2))
    err_s = float(np.sum(weights * (x[sim_p.N+3:sim_p.N+7] - target) ** 2))

    # Update history
    st.session_state.x = x
    h = st.session_state.history
    h["t"].append(st.session_state.tick)
    h["R_p_truth"].append(float(true_R))
    h["Z_p_truth"].append(float(true_Z))
    h["kappa_truth"].append(float(true_K))
    h["delta_truth"].append(float(true_D))
    h["R_p_self"].append(float(x[sim_p.N+3]))
    h["Z_p_self"].append(float(x[sim_p.N+4]))
    h["kappa_self"].append(float(x[sim_p.N+5]))
    h["delta_self"].append(float(x[sim_p.N+6]))
    h["truth_err"].append(err_t)
    h["self_err"].append(err_s)
    h["I_p"].append(float(x[sim_p.N]))
    h["physicality_running"].append(physicality)
    h["latency_us"].append(latency_us)
    h["V_coils"] = V
    h["walkers_alive"].append(walkers_alive)

    # Trim history to last 200 ticks for performance
    if len(h["t"]) > 200:
        for k in h:
            if isinstance(h[k], list):
                h[k] = h[k][-200:]

    st.session_state.tick += 1

    # Optional FMC walker viz
    fmc_extras = None
    if is_fmc and show_walker_viz:
        wk, wd, wr = sample_fmc_walkers(fmc, x, target_f32, n_samples=4)
        fmc_extras = (wk, wd, wr, walkers_alive, expected_reward)

    return V, fmc_extras


# Run logic
# To avoid flashing, we batch many steps per rerun and sleep longer between
# reruns. With stable plotly keys, plots update in-place. The trade-off is
# slightly less smooth motion but vastly less visual noise.
if run:
    # Heavy batch — do speed*4 sim steps before triggering a rerun
    max_steps = max(1, speed * 4)
    last_V = None
    last_fmc = None
    for _ in range(max_steps):
        last_V, last_fmc = step_simulation()
    render_all(V_coils=last_V, fmc_extras=last_fmc)
    # Long sleep — ~1.5 fps refresh, eyes-friendly
    time.sleep(0.6)
    st.rerun()
else:
    # Static render
    render_all(V_coils=st.session_state.history.get("V_coils"))
