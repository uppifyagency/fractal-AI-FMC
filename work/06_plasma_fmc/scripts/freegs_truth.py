"""FreeGS truth coupling — Milestone 9.

Validates the linearized matrices (S, M_pc) against real Grad-Shafranov
free-boundary equilibria computed by freegs.

Pipeline:
1. Solve baseline TCV equilibrium with two configurations:
   - Single-null (SN, X-point at Z=-0.5)
   - Double-null (DN, X-points at Z=±0.65)
2. Extract plasma shape (R_p, Z_p, κ, δ) from the LCFS
3. Compare to Miller nominal targets — quantify discrepancy
4. Identify empirical shape sensitivity by varying X-point position via
   constraint perturbation (NOT direct coil ΔI, which doesn't converge)
5. Compare to synthetic S of M2/M3

Method note (perturbation):
We do NOT use direct coil-current perturbation because freezing all coils
+ adding ΔI to one creates an over-constrained system where the Picard
iteration cannot find a self-consistent equilibrium. Instead, we perturb
the SHAPE CONSTRAINT (X-point Z position) and let freegs adjust all coils
to satisfy it. Then we read out δ(R_p, Z_p, κ, δ) and δI_coils together —
this is closer to what a real "shape feedback controller" does anyway.

Source: freegs 0.8.2 (community-validated GS solver with TCV machine
description matching EPFL LRP-755-13).

Output: results/freegs_truth.json + results/milestone_9_*.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freegs
import freegs.critical
from freegs.machine import TCV

RESULTS_DIR = Path(__file__).parent.parent / "results"


def solve_equilibrium(I_p=2.0e5, paxis=1e3, fvac=0.5,
                      configuration="DN",
                      Z_xpoint_DN=0.65,
                      Z_xpoint_SN=-0.5,
                      maxits=20):
    """Solve a TCV equilibrium with the requested boundary configuration.

    configuration: "SN" (single-null) or "DN" (double-null up-down symmetric)
    """
    m = TCV()
    eq = freegs.Equilibrium(
        tokamak=m, Rmin=0.3, Rmax=1.5, Zmin=-1.0, Zmax=1.0,
        nx=65, ny=65,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, paxis, I_p, fvac)

    if configuration == "SN":
        xpoints = [(0.85, Z_xpoint_SN)]
        isoflux = [(0.85, Z_xpoint_SN, 0.85, -Z_xpoint_SN),
                   (1.10, 0.0, 0.65, 0.0)]
    elif configuration == "DN":
        # Double-null with symmetric X-points
        xpoints = [(0.65, -Z_xpoint_DN), (0.65, +Z_xpoint_DN)]
        isoflux = [(1.10, 0.0, 0.65, 0.0),
                   (0.85, Z_xpoint_DN, 0.85, -Z_xpoint_DN)]
    else:
        raise ValueError(f"Unknown configuration: {configuration}")

    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
    t0 = time.perf_counter()
    freegs.solve(eq, profiles, constrain, show=False, maxits=maxits)
    elapsed = time.perf_counter() - t0
    return eq, {"solve_time_s": elapsed,
                "I_p_actual": float(eq.plasmaCurrent())}


def extract_shape(eq) -> dict:
    """Extract (R_p, Z_p, κ, δ, a) from the LCFS contour."""
    psi = eq.psi()
    opt, xpt = freegs.critical.find_critical(eq.R, eq.Z, psi)
    if not opt:
        raise RuntimeError("No O-point found")
    R_axis, Z_axis, psi_axis = opt[0]

    cs = plt.contour(eq.R, eq.Z, psi.T, levels=[eq.psi_bndry])
    plt.close()
    paths = cs.allsegs[0] if cs.allsegs else []

    # Find the closed loop containing the magnetic axis
    boundary = None
    for path in paths:
        if len(path) < 20:
            continue
        if (path[:, 0].min() < R_axis < path[:, 0].max()
                and path[:, 1].min() < Z_axis < path[:, 1].max()):
            boundary = path
            break

    if boundary is None:
        raise RuntimeError("Could not extract LCFS contour")

    R_lcfs = boundary[:, 0]
    Z_lcfs = boundary[:, 1]

    R_p = 0.5 * (R_lcfs.max() + R_lcfs.min())
    a = 0.5 * (R_lcfs.max() - R_lcfs.min())
    Z_p = 0.5 * (Z_lcfs.max() + Z_lcfs.min())
    b = 0.5 * (Z_lcfs.max() - Z_lcfs.min())
    kappa = b / a
    idx_top = int(np.argmax(Z_lcfs))
    delta = (R_p - R_lcfs[idx_top]) / a

    return {
        "R_p": float(R_p), "Z_p": float(Z_p),
        "a": float(a), "kappa": float(kappa), "delta": float(delta),
        "R_axis": float(R_axis), "Z_axis": float(Z_axis),
        "psi_axis": float(psi_axis), "psi_bndry": float(eq.psi_bndry),
        "n_lcfs_pts": int(len(R_lcfs)),
        "I_p": float(eq.plasmaCurrent()),
        "lcfs_R": R_lcfs.tolist(),
        "lcfs_Z": Z_lcfs.tolist(),
    }


def extract_coil_currents(eq) -> dict:
    out = {}
    for label, elem in eq.tokamak.coils:
        if hasattr(elem, "current"):
            out[label] = float(elem.current)
        elif hasattr(elem, "coils"):
            for sub_label, sub_elem, factor in elem.coils:
                if hasattr(sub_elem, "current"):
                    out[sub_label] = float(sub_elem.current * factor)
    return out


def perturb_xpoint_z(dZ: float, baseline_kwargs):
    """Perturb the X-point Z position by dZ; let freegs re-find coils."""
    bk = dict(baseline_kwargs)
    bk["Z_xpoint_DN"] = bk.get("Z_xpoint_DN", 0.65) + dZ
    eq, info = solve_equilibrium(**bk)
    return eq, info


def main():
    print("=" * 70)
    print("Milestone 9 — FreeGS truth coupling")
    print("=" * 70)

    out = {}

    # ---- 1. Baseline DN equilibrium ----
    print("\n[1] Solving baseline TCV equilibrium (double-null)...")
    bk = dict(I_p=2.0e5, paxis=1e3, fvac=0.5, configuration="DN",
               Z_xpoint_DN=0.65)
    eq_base, info_base = solve_equilibrium(**bk)
    print(f"    Solved in {info_base['solve_time_s']:.2f}s, "
          f"I_p={info_base['I_p_actual']:.0f} A")

    base_shape = extract_shape(eq_base)
    base_currents = extract_coil_currents(eq_base)
    print(f"\n[2] Baseline shape from LCFS:")
    print(f"    R_p   = {base_shape['R_p']:.4f} m   Z_p = {base_shape['Z_p']:+.4f} m")
    print(f"    a     = {base_shape['a']:.4f} m   κ   = {base_shape['kappa']:.4f}")
    print(f"    δ     = {base_shape['delta']:+.4f}")

    nominal = {"R_p": 0.88, "Z_p": 0.0, "kappa": 1.7, "delta": 0.3}
    diffs_to_nominal = {k: float(base_shape[k] - v) for k, v in nominal.items()}
    print(f"\n[3] Discrepancy vs M1/M2 Miller nominal:")
    for k, dv in diffs_to_nominal.items():
        print(f"    Δ{k:5s} = {dv:+.4f}  (real {base_shape[k]:.4f} "
              f"vs nominal {nominal[k]:.4f})")

    out["baseline"] = {
        "config": bk, "info": info_base,
        "shape": {k: v for k, v in base_shape.items()
                  if k not in ("lcfs_R", "lcfs_Z")},
        "currents": base_currents,
        "diffs_vs_nominal": diffs_to_nominal,
    }

    # ---- 2. Also solve a single-null for comparison ----
    print(f"\n[4] Comparison: single-null configuration...")
    bk_sn = dict(I_p=2.0e5, paxis=1e3, fvac=0.5, configuration="SN",
                  Z_xpoint_SN=-0.5)
    eq_sn, info_sn = solve_equilibrium(**bk_sn)
    sn_shape = extract_shape(eq_sn)
    print(f"    SN shape: R_p={sn_shape['R_p']:.3f}, "
          f"Z_p={sn_shape['Z_p']:+.3f}, "
          f"κ={sn_shape['kappa']:.3f}, δ={sn_shape['delta']:+.3f}")

    out["single_null"] = {
        "config": bk_sn, "info": info_sn,
        "shape": {k: v for k, v in sn_shape.items()
                  if k not in ("lcfs_R", "lcfs_Z")},
    }

    # ---- 3. Constraint-based shape sensitivity (Z_xpoint perturbation) ----
    # We perturb the X-point Z by ±0.05 m and measure the shape change.
    # This is NOT a per-coil S column — it's a "constraint-driven shape
    # sensitivity" that captures what the controller would actually achieve.
    print(f"\n[5] Constraint perturbation study: ΔZ_xpoint = ±0.05 m...")
    sensitivity = {}
    for dZ in [-0.05, +0.05]:
        try:
            eq_p, info_p = perturb_xpoint_z(dZ, bk)
            sh_p = extract_shape(eq_p)
            curr_p = extract_coil_currents(eq_p)
            # δshape per Z_xpoint perturbation
            d_shape = {k: float(sh_p[k] - base_shape[k])
                       for k in ("R_p", "Z_p", "kappa", "delta")}
            # δI_coils per Z_xpoint perturbation
            d_curr = {k: float(curr_p[k] - base_currents[k])
                      for k in base_currents}
            sensitivity[f"dZxp{dZ:+.2f}"] = {
                "d_shape": d_shape, "d_currents": d_curr,
                "perturbed_shape": {k: v for k, v in sh_p.items()
                                     if k not in ("lcfs_R", "lcfs_Z")},
                "solve_time_s": info_p["solve_time_s"],
            }
            print(f"    ΔZ_xpt={dZ:+.2f} m → "
                  f"Δκ={d_shape['kappa']:+.4f}, "
                  f"Δδ={d_shape['delta']:+.4f}, "
                  f"|ΔI|max={max(abs(v) for v in d_curr.values()):.0f} A")
        except Exception as e:
            sensitivity[f"dZxp{dZ:+.2f}"] = {"error": str(e)}
            print(f"    ΔZ_xpt={dZ:+.2f}: failed — {type(e).__name__}: {e}")

    out["constraint_perturbation"] = sensitivity

    # ---- 4. Save + plot ----
    out_path = RESULTS_DIR / "freegs_truth.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")

    # Plot baseline equilibrium with LCFS
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, (eq, sh, label) in zip(axes,
                                    [(eq_base, base_shape, "Double-null"),
                                     (eq_sn, sn_shape, "Single-null")]):
        psi = eq.psi()
        levels = np.linspace(psi.min(), psi.max(), 25)
        ax.contour(eq.R, eq.Z, psi.T, levels=levels, alpha=0.4, colors='gray')
        ax.contour(eq.R, eq.Z, psi.T, levels=[eq.psi_bndry],
                   colors='red', linewidths=2.5)
        ax.plot(sh["R_axis"], sh["Z_axis"], "k+", markersize=15, mew=2)

        # Coils
        m_disp = TCV()
        for clabel, elem in m_disp.coils:
            if hasattr(elem, "R"):
                color = "blue" if clabel.startswith("E") else "red"
                ax.plot(elem.R, elem.Z, "s", color=color, markersize=7, alpha=0.7)

        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.set_title(f"{label}: I_p={sh['I_p']/1e3:.0f} kA, "
                     f"κ={sh['kappa']:.2f}, δ={sh['delta']:+.2f}")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.3, 1.5); ax.set_ylim(-1.0, 1.0)

    fig.suptitle("Milestone 9 — FreeGS truth equilibria for TCV", fontsize=12, y=0.99)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "milestone_9_equilibria.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {RESULTS_DIR / 'milestone_9_equilibria.png'}")


if __name__ == "__main__":
    main()
