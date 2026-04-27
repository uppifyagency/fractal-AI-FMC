"""Milestone 16 — Validation against REAL TCV experimental equilibrium.

Source: TCV-X21 FAIR dataset (Oliveira & Body et al. 2021), CC-BY-4.0
https://github.com/SPCData/TCV-X21

The dataset includes shot 65402 at t=1s in eqdsk format. We use it as
our FIRST real TCV experimental shape target. This closes the chain:
    in-sim → freegs proxy → published-derived → REAL EXPERIMENTAL DATA.

Pipeline
--------
1. Parse `data/tcv_x21/65402_t1.eqdsk` via freeqdsk (low-level)
2. Extract real LCFS contour (rbbbs, zbbbs) from the experimental fit
3. Compute (R_p, Z_p, kappa, delta) directly from the experimental contour
4. Run all 5 policies tracking this REAL shape as target
5. Use M14 freegs oracle for truth-err — same metric as M14, M15

Note on coil currents
---------------------
The eqdsk format does not include explicit coil currents (the
experimental equilibrium reconstruction LIUQE produces psi but the
coil-current fit is in a separate file not provided in TCV-X21). So we
cannot directly validate M14 oracle's coil→shape mapping against the
real shot. We can only validate **policy tracking** of the real target.
A future M17 could obtain LIUQE current fits from EPFL.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import freegs
import freegs.critical
from freegs.machine import TCV
from freeqdsk import geqdsk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibrated_sim import build_calibrated_jax_params
from fmc_plasma_jax import FMCPlasmaJaxController
from freegs_oracle_robust import COIL_ORDER, FreeGSOracle
from plasma_simulator_jax import DTYPE, make_jit_step
from plasma_simulator_nn_shape import (
    SimParamsNN, build_nn_sim_params, make_jit_step_nn, predict_shape,
)
from policy import TrainedPolicy

DATA_DIR = Path(__file__).parent.parent / "data" / "tcv_x21"
RESULTS_DIR = Path(__file__).parent.parent / "results"
EQDSK_PATH = DATA_DIR / "65402_t1.eqdsk"


def load_real_tcv_lcfs():
    """Parse TCV-X21 shot 65402 eqdsk file. Returns shape descriptors."""
    if not EQDSK_PATH.exists():
        raise FileNotFoundError(
            f"{EQDSK_PATH} not found — download from TCV-X21 first")
    with open(EQDSK_PATH) as fh:
        data = geqdsk.read(fh)

    # LCFS from experimental boundary
    R_lcfs = data.rbbbs
    Z_lcfs = data.zbbbs
    R_p = 0.5 * (R_lcfs.max() + R_lcfs.min())
    a = 0.5 * (R_lcfs.max() - R_lcfs.min())
    Z_p = 0.5 * (Z_lcfs.max() + Z_lcfs.min())
    b = 0.5 * (Z_lcfs.max() - Z_lcfs.min())
    kappa = b / a
    idx_top = int(np.argmax(Z_lcfs))
    delta = (R_p - R_lcfs[idx_top]) / a

    return {
        "R_p": float(R_p),
        "Z_p": float(Z_p),
        "a": float(a),
        "kappa": float(kappa),
        "delta": float(delta),
        "I_p": float(data.current),
        "R_axis": float(data.rmaxis),
        "Z_axis": float(data.zmaxis),
        "psi_axis": float(data.simagx),
        "psi_bndry": float(data.sibdry),
        "n_lcfs_pts": int(data.nbbbs),
        "lcfs_R": R_lcfs.tolist(),
        "lcfs_Z": Z_lcfs.tolist(),
        "psi_grid_shape": list(data.psirz.shape),
        "B0_T_at_R0": float(data.bcentr),
        "R0_m": float(data.rcentr),
    }


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


def truth(I, oracle, fb_sim, weights, target):
    def fb(I_):
        s = np.array(predict_shape(fb_sim, jnp.asarray(I_, dtype=DTYPE)))
        s[0] = np.clip(s[0], 0.624, 1.136)
        s[1] = np.clip(s[1], -0.75, 0.75)
        s[2] = np.clip(s[2], 1.0, 2.8)
        s[3] = np.clip(s[3], -0.7, 1.0)
        return s
    r = oracle.shape_from_coils(np.asarray(I), fallback_fn=fb)
    return np.array([r.R_p, r.Z_p, r.kappa, r.delta]), r.source


def main():
    print("=" * 72)
    print("Milestone 16 — Validation against REAL TCV experimental equilibrium")
    print("=" * 72)

    print(f"\n[1] Parsing TCV-X21 shot 65402 t=1 from {EQDSK_PATH.name}...")
    t0 = time.perf_counter()
    real = load_real_tcv_lcfs()
    print(f"    Parsed in {1000*(time.perf_counter()-t0):.0f} ms")

    print(f"\n[2] Real TCV experimental shape (LIUQE reconstruction):")
    print(f"    R_p   = {real['R_p']:.4f} m")
    print(f"    Z_p   = {real['Z_p']:+.4f} m")
    print(f"    a     = {real['a']:.4f} m")
    print(f"    kappa = {real['kappa']:.4f}")
    print(f"    delta = {real['delta']:+.4f}")
    print(f"    I_p   = {real['I_p']/1e3:+.1f} kA")
    print(f"    LCFS  = {real['n_lcfs_pts']} contour points")
    print(f"    B_T   = {real['B0_T_at_R0']:.4f} T at R={real['R0_m']:.3f}")

    target_real = np.array([real["R_p"], real["Z_p"],
                              real["kappa"], real["delta"]])

    # Check target is within M14 envelope
    in_envelope = (
        0.7 < real["R_p"] < 1.0
        and -0.2 < real["Z_p"] < 0.2
        and 1.2 < real["kappa"] < 2.5
        and -0.7 < real["delta"] < 0.8
    )
    print(f"\n[3] M14 oracle envelope check: "
          f"{'PASS — inside tested range' if in_envelope else 'WARN — outside'}")

    # ---- Setup oracle, sims, policies ----
    print(f"\n[4] Initializing M14 oracle + sims + policies...")
    oracle = FreeGSOracle(verbose=True)
    sim_p_lin, _ = build_calibrated_jax_params()
    sim_step_lin = make_jit_step(sim_p_lin)
    policy_m10 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger_jax.npz")
    policy_m6 = TrainedPolicy.load(RESULTS_DIR / "policy_dagger.npz")
    policy_m5 = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    sim_p_nn = build_nn_sim_params()
    sim_step_nn = make_jit_step_nn(sim_p_nn)
    policy_m12 = TrainedPolicy.load(RESULTS_DIR / "policy_nn_shape.npz")

    weights = np.array([100.0, 100.0, 10.0, 10.0])
    target_f32 = target_real.astype(np.float32)

    jx_fmc = FMCPlasmaJaxController(sim_p_lin, n_walkers=64,
                                     horizon=10, seed=16)
    jx_fmc.decide(np.zeros(27, dtype=np.float32), target_f32)

    setups = {
        "M5_BC":       (sim_p_lin, sim_step_lin, policy_m5),
        "M6_DAgger3":  (sim_p_lin, sim_step_lin, policy_m6),
        "M10_DAggerN": (sim_p_lin, sim_step_lin, policy_m10),
        "M12_NNshape": (sim_p_nn, sim_step_nn, policy_m12),
    }

    n_ticks = 30
    print(f"\n[5] Closed-loop tracking {n_ticks} ticks @ 50ms = 1.5s")
    print(f"    Target: REAL TCV shot 65402 shape "
          f"(R={target_real[0]:.4f}, Z={target_real[1]:+.4f}, "
          f"κ={target_real[2]:.3f}, δ={target_real[3]:+.3f})")
    print("-" * 72)

    policy_results = {}

    for label, (sim_p, sim_step, pol) in setups.items():
        x = np.asarray(make_initial_state(sim_p)).copy()
        truth_errs, self_errs = [], []
        n_freegs = n_fb = 0
        for _ in range(n_ticks):
            V = pol(x, target_f32)
            x_new = sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                jnp.asarray(5e5, dtype=DTYPE),
                jnp.asarray(1e21, dtype=DTYPE),
                jnp.asarray(1e-3, dtype=DTYPE),
            )
            x = np.array(x_new)
            if np.isnan(x).any():
                break
            true_shape, source = truth(x[:sim_p.N], oracle, sim_p_nn,
                                         weights, target_real)
            self_R, self_Z, self_K, self_D = (
                x[sim_p.N+3], x[sim_p.N+4],
                x[sim_p.N+5], x[sim_p.N+6],
            )
            err_t = float(np.sum(weights * (true_shape - target_real) ** 2))
            err_s = float(np.sum(weights *
                                  (np.array([self_R, self_Z, self_K, self_D])
                                    - target_real) ** 2))
            truth_errs.append(err_t)
            self_errs.append(err_s)
            if source == "freegs":
                n_freegs += 1
            elif source == "nn_fallback":
                n_fb += 1
        mt = float(np.mean(truth_errs)) if truth_errs else float("inf")
        ms = float(np.mean(self_errs)) if self_errs else float("inf")
        phys = n_freegs / max(1, n_freegs + n_fb)
        steady_truth = float(np.mean(truth_errs[-10:])) if len(truth_errs) >= 10 else mt
        policy_results[label] = {
            "mean_truth_err": mt,
            "mean_self_err": ms,
            "physicality": phys,
            "n_steps": len(truth_errs),
            "steady_state_truth_err_last10": steady_truth,
        }
        print(f"    {label:14s} | mean truth {mt:7.2f} | "
              f"steady (last10) {steady_truth:7.2f} | "
              f"phys {100*phys:.0f}%")

    # FMC online
    x = np.asarray(make_initial_state(sim_p_lin)).copy()
    truth_errs_fmc, self_errs_fmc = [], []
    n_freegs_fmc = n_fb_fmc = 0
    for _ in range(n_ticks):
        V = jx_fmc.decide(x, target_f32)["V_coils"]
        x_new = sim_step_lin(
            jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
            jnp.asarray(5e5, dtype=DTYPE),
            jnp.asarray(1e21, dtype=DTYPE),
            jnp.asarray(1e-3, dtype=DTYPE),
        )
        x = np.array(x_new)
        if np.isnan(x).any():
            break
        true_shape, source = truth(x[:sim_p_lin.N], oracle, sim_p_nn,
                                     weights, target_real)
        err_t = float(np.sum(weights * (true_shape - target_real) ** 2))
        err_s = float(np.sum(weights *
                              (x[sim_p_lin.N+3:sim_p_lin.N+7]
                                - target_real) ** 2))
        truth_errs_fmc.append(err_t)
        self_errs_fmc.append(err_s)
        if source == "freegs":
            n_freegs_fmc += 1
        elif source == "nn_fallback":
            n_fb_fmc += 1
    mt = float(np.mean(truth_errs_fmc)) if truth_errs_fmc else float("inf")
    ms = float(np.mean(self_errs_fmc)) if self_errs_fmc else float("inf")
    phys = n_freegs_fmc / max(1, n_freegs_fmc + n_fb_fmc)
    steady_truth = (float(np.mean(truth_errs_fmc[-10:]))
                     if len(truth_errs_fmc) >= 10 else mt)
    policy_results["FMC_online"] = {
        "mean_truth_err": mt,
        "mean_self_err": ms,
        "physicality": phys,
        "n_steps": len(truth_errs_fmc),
        "steady_state_truth_err_last10": steady_truth,
    }
    print(f"    {'FMC_online':14s} | mean truth {mt:7.2f} | "
          f"steady (last10) {steady_truth:7.2f} | "
          f"phys {100*phys:.0f}%")

    # ---- Save ----
    out = {
        "shot": "TCV 65402 t=1.0s",
        "source": "TCV-X21 (Oliveira & Body et al. 2021), CC-BY-4.0",
        "physical_parameters": {
            "B0_T": 0.929, "Te0_eV": 41.3, "n0_per_m3": 1e19,
            "R0_m": 0.906, "Mi_amu": 2,
        },
        "real_shape": {k: v for k, v in real.items()
                        if not isinstance(v, list)},
        "in_oracle_envelope": in_envelope,
        "weights": weights.tolist(),
        "n_ticks": n_ticks,
        "policy_results": policy_results,
    }
    out_path = RESULTS_DIR / "milestone_16_real_tcv.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: real LCFS contour vs M14 baseline
    ax = axes[0]
    R_lcfs = np.array(real["lcfs_R"])
    Z_lcfs = np.array(real["lcfs_Z"])
    ax.plot(R_lcfs, Z_lcfs, "r-", linewidth=2.5, label="REAL TCV 65402 LCFS")
    ax.plot(real["R_axis"], real["Z_axis"], "k+", markersize=14, mew=2,
            label="real magnetic axis")
    # Vessel envelope
    from freegs_oracle_robust import VESSEL_R_RANGE, VESSEL_Z_RANGE
    rect_R = [VESSEL_R_RANGE[0], VESSEL_R_RANGE[1], VESSEL_R_RANGE[1],
              VESSEL_R_RANGE[0], VESSEL_R_RANGE[0]]
    rect_Z = [VESSEL_Z_RANGE[0], VESSEL_Z_RANGE[0], VESSEL_Z_RANGE[1],
              VESSEL_Z_RANGE[1], VESSEL_Z_RANGE[0]]
    ax.plot(rect_R, rect_Z, "k:", linewidth=1, alpha=0.5,
            label="M14 vessel envelope")
    # Limiter from eqdsk
    from freeqdsk import geqdsk
    with open(EQDSK_PATH) as fh:
        data = geqdsk.read(fh)
    ax.plot(data.rlim, data.zlim, "k-", linewidth=0.5, alpha=0.4,
            label="TCV limiter")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.set_title(
        f"REAL TCV shot 65402 LCFS\n"
        f"R_p={real['R_p']:.3f} m, Z_p={real['Z_p']:+.3f} m, "
        f"κ={real['kappa']:.3f}, δ={real['delta']:+.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(-1.0, 1.0)

    # Panel 2: policy ranking
    ax = axes[1]
    policies = ["M5_BC", "M6_DAgger3", "M10_DAggerN", "M12_NNshape", "FMC_online"]
    truths = [policy_results[p]["mean_truth_err"] for p in policies]
    steady = [policy_results[p]["steady_state_truth_err_last10"]
              for p in policies]
    physs = [policy_results[p]["physicality"] for p in policies]
    colors = ["lightcoral" if p < 0.5 else
              ("yellow" if t > 30 else "mediumseagreen")
              for t, p in zip(steady, physs)]
    x_pos = np.arange(len(policies))
    w = 0.4
    ax.bar(x_pos - w/2, truths, w, color=colors, edgecolor="black",
           label="mean (all 30 ticks)", alpha=0.7)
    ax.bar(x_pos + w/2, steady, w, color=colors, edgecolor="black",
           hatch="//", label="steady-state (last 10)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("Truth-err vs REAL TCV target")
    ax.set_title("Policy ranking on REAL TCV 65402 shape\n(M14 oracle)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    for i, (t, s, p) in enumerate(zip(truths, steady, physs)):
        ax.text(i, max(t, s) + 1, f"phys={100*p:.0f}%",
                ha="center", fontsize=8)

    fig.suptitle(
        "Milestone 16 — Validation on REAL TCV experimental data "
        "(TCV-X21 shot 65402, CC-BY-4.0)", fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out_png = RESULTS_DIR / "milestone_16_real_tcv.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")


if __name__ == "__main__":
    main()
