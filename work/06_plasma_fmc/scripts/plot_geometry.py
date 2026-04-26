"""Generate Milestone 1 visualization: TCV cross-section + reference shapes.

Output: results/milestone_1_geometry.png and results/milestone_1_shapes.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from reference_shapes import all_reference_shapes
from tcv_geometry import load_tcv

RESULTS_DIR = Path(__file__).parent.parent / "results"


def plot_machine_cross_section(tcv, ax):
    """Draw TCV poloidal cross-section: vessel + all coils."""
    # Vessel rectangle (D-shape simplified to rectangle for plotting)
    vessel_rect = patches.Rectangle(
        (tcv.vessel["inner_R"], -tcv.vessel["height"] / 2),
        tcv.vessel["outer_R"] - tcv.vessel["inner_R"],
        tcv.vessel["height"],
        linewidth=1.5,
        edgecolor="gray",
        facecolor="lightyellow",
        alpha=0.3,
        label="Vacuum vessel (approx)",
    )
    ax.add_patch(vessel_rect)

    # Magnetic axis crosshair
    ax.plot(tcv.R_major, 0, "k+", markersize=12, mew=2, label=f"R₀={tcv.R_major} m")

    # Shaping coils E (inboard) - blue
    e_coils = [c for c in tcv.shaping_coils if c.name.startswith("E")]
    e_R = [c.R for c in e_coils]
    e_Z = [c.Z for c in e_coils]
    ax.scatter(
        e_R, e_Z, s=180, c="royalblue", marker="s", edgecolor="black",
        linewidth=1.2, label="E coils (8 inboard)", zorder=5,
    )
    for c in e_coils:
        ax.annotate(
            c.name, (c.R, c.Z), xytext=(-12, 0),
            textcoords="offset points", fontsize=7, ha="right", va="center",
        )

    # Shaping coils F (outboard) - red
    f_coils = [c for c in tcv.shaping_coils if c.name.startswith("F")]
    f_R = [c.R for c in f_coils]
    f_Z = [c.Z for c in f_coils]
    ax.scatter(
        f_R, f_Z, s=180, c="crimson", marker="s", edgecolor="black",
        linewidth=1.2, label="F coils (8 outboard)", zorder=5,
    )
    for c in f_coils:
        ax.annotate(
            c.name, (c.R, c.Z), xytext=(12, 0),
            textcoords="offset points", fontsize=7, ha="left", va="center",
        )

    # T coils - orange
    t_R = [c.R for c in tcv.t_coils]
    t_Z = [c.Z for c in tcv.t_coils]
    ax.scatter(
        t_R, t_Z, s=120, c="orange", marker="^", edgecolor="black",
        linewidth=1.2, label="T coils (3)", zorder=5,
    )

    # OH coils - green
    oh_R = [c.R for c in tcv.ohmic_coils]
    oh_Z = [c.Z for c in tcv.ohmic_coils]
    ax.scatter(
        oh_R, oh_Z, s=160, c="forestgreen", marker="D", edgecolor="black",
        linewidth=1.2, label="OH circuit coils", zorder=5,
    )

    # Solenoid (vertical bar)
    sol = tcv.solenoid
    sol_rect = patches.Rectangle(
        (sol["R"] - 0.02, sol["Z_min"]),
        0.04,
        sol["Z_max"] - sol["Z_min"],
        linewidth=1, edgecolor="darkgreen", facecolor="forestgreen", alpha=0.5,
        label=f"Solenoid ({sol['N_turns']} turns)",
    )
    ax.add_patch(sol_rect)

    ax.set_xlabel("R [m]", fontsize=11)
    ax.set_ylabel("Z [m]", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.set_title(
        f"TCV poloidal cross-section\n"
        f"R₀={tcv.R_major} m, a={tcv.a_minor} m, ε={tcv.epsilon:.3f}, "
        f"{tcv.n_control_channels} control channels",
        fontsize=11,
    )
    ax.set_xlim(0.2, 1.9)
    ax.set_ylim(-1.4, 1.4)


def plot_reference_shapes(tcv, ax):
    """Overlay the 3 reference plasma shapes on the machine cross-section."""
    # Light machine background
    vessel_rect = patches.Rectangle(
        (tcv.vessel["inner_R"], -tcv.vessel["height"] / 2),
        tcv.vessel["outer_R"] - tcv.vessel["inner_R"],
        tcv.vessel["height"],
        linewidth=1, edgecolor="gray", facecolor="lightyellow", alpha=0.2,
    )
    ax.add_patch(vessel_rect)

    # All coils as small markers
    for coils, color in [
        (tcv.shaping_coils, "gray"),
        (tcv.ohmic_coils, "gray"),
    ]:
        for c in coils:
            ax.plot(c.R, c.Z, "s", color=color, markersize=5, alpha=0.4)

    # Plot each reference LCFS
    colors = {"standard_single_null": "navy", "snowflake": "darkorange",
              "negative_triangularity": "darkviolet"}
    for shape in all_reference_shapes(tcv):
        R, Z = shape.lcfs(300)
        # Close the curve
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
        c = colors[shape.name]
        ax.plot(R, Z, "-", color=c, linewidth=2.2,
                label=f"{shape.name}: κ={shape.kappa}, δ={shape.delta:+.1f}")
        # Mark the geometric center
        ax.plot(shape.R0, shape.Z0, "+", color=c, markersize=10, mew=1.5)

    ax.set_xlabel("R [m]", fontsize=11)
    ax.set_ylabel("Z [m]", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        "Reference plasma shapes (Miller parametric LCFS)\n"
        "Source: Miller et al. Phys. Plasmas 5:973 (1998)",
        fontsize=11,
    )
    ax.set_xlim(0.2, 1.9)
    ax.set_ylim(-1.4, 1.4)


def plot_combined():
    tcv = load_tcv()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    plot_machine_cross_section(tcv, ax1)
    plot_reference_shapes(tcv, ax2)
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_1_geometry.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")

    return fig


def plot_individual_shapes():
    """Per-shape detail plots with derived physics quantities."""
    tcv = load_tcv()
    shapes = all_reference_shapes(tcv)

    B_T_typical = 1.43
    I_p_typical = 200_000.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, shape in zip(axes, shapes):
        # Vessel
        vessel_rect = patches.Rectangle(
            (tcv.vessel["inner_R"], -tcv.vessel["height"] / 2),
            tcv.vessel["outer_R"] - tcv.vessel["inner_R"],
            tcv.vessel["height"],
            linewidth=1, edgecolor="gray", facecolor="lightyellow", alpha=0.2,
        )
        ax.add_patch(vessel_rect)

        # Coils
        for c in tcv.shaping_coils:
            color = "royalblue" if c.name.startswith("E") else "crimson"
            ax.plot(c.R, c.Z, "s", color=color, markersize=6, alpha=0.7)

        # LCFS
        R, Z = shape.lcfs(300)
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
        ax.fill(R, Z, color="lightcoral", alpha=0.35, label="Plasma")
        ax.plot(R, Z, "-", color="darkred", linewidth=2.0, label="LCFS")
        ax.plot(shape.R0, shape.Z0, "k+", markersize=12, mew=2)

        # Derived quantities annotation
        d = shape.derived(B_T_typical, I_p_typical)
        info = (
            f"κ = {shape.kappa}\n"
            f"δ = {shape.delta:+.1f}\n"
            f"V = {d['volume_m3']:.2f} m³\n"
            f"q₉₅ ≈ {d['q95_cylindrical']:.2f}\n"
            f"n_GW = {d['n_GW_m3']:.2e} m⁻³\n"
            f"β_max = {d['beta_max_pct']:.2f} %"
        )
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85, "pad": 5},
        )

        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(shape.name.replace("_", " "))
        ax.set_xlim(0.2, 1.9)
        ax.set_ylim(-1.4, 1.4)

    fig.suptitle(
        f"TCV reference shapes @ B_T={B_T_typical} T, I_p={I_p_typical/1e3:.0f} kA",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_1_shapes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")
    return fig


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Generating Milestone 1 plots...")
    plot_combined()
    plot_individual_shapes()
    print("Done.")
