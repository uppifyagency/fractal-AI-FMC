"""gap9_fig4.py — Figure 4: two-component reward + relativize regime separation.

Programmatic schematic via matplotlib. Two panels:
    LEFT: stacked R_inv (dense) + R_ach (sparse) bars over rollout time.
    RIGHT: relativize(z) curve with firing-walker / non-firing-walker markers.

Saves figures/fig4_schematic.{pdf,png} at 300 DPI.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Wong colorblind-safe palette
COLOR_BLUE  = "#0072B2"
COLOR_RED   = "#D55E00"
COLOR_GREY  = "#888888"
COLOR_LIGHT_BLUE = "#56B4E9"


def relativize(z):
    return np.where(z > 0, 1.0 + np.log1p(np.maximum(z, 0)),
                    np.exp(np.minimum(z, 0)) / np.e)


def main():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 3.6),
                                     gridspec_kw={"width_ratios": [1, 1.2]})

    # ---------- LEFT panel: two-component reward ----------
    T = 40  # rollout horizon
    t = np.arange(T)

    # R_inv: small uniform increments (dense)
    r_inv = np.cumsum(np.full(T, 1.5)) * 0.4
    # R_ach: zero except one big spike at t=27
    r_ach = np.zeros(T)
    r_ach[27] = 200.0  # cum-bonus at firing point
    # cumulative ach (step function after fire)
    r_ach_cum = np.cumsum(r_ach)

    # Stacked area
    ax_l.fill_between(t, 0, r_inv, color=COLOR_BLUE, alpha=0.7,
                      label=r"$R_{\mathrm{inv}}$ (dense)")
    ax_l.fill_between(t, r_inv, r_inv + r_ach_cum, color=COLOR_RED, alpha=0.7,
                      label=r"$R_{\mathrm{ach}}$ (sparse)")
    # Mark the spike
    ax_l.annotate(r"$w_j = 200$" + "\nach. fires",
                  xy=(27, r_inv[27] + r_ach_cum[27]),
                  xytext=(15, 235), fontsize=8, color=COLOR_RED,
                  arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.7))
    ax_l.set_xlabel("rollout time step $t$")
    ax_l.set_ylabel("walker cumulative reward $r$")
    ax_l.set_title("(a) Two-component walker reward")
    ax_l.legend(loc="upper left", framealpha=0.95)
    ax_l.set_xlim(0, T - 1)
    ax_l.set_ylim(0, 300)
    ax_l.grid(True, alpha=0.2, linewidth=0.5)

    # ---------- RIGHT panel: relativize regime separation ----------
    z_neg = np.linspace(-3, 0, 200)
    z_pos = np.linspace(0, 6, 400)
    rh_neg = relativize(z_neg)
    rh_pos = relativize(z_pos)

    ax_r.fill_between(z_neg, 0, rh_neg, color=COLOR_GREY, alpha=0.25,
                      label="exp regime: $\\hat r = e^{z}/e$")
    ax_r.fill_between(z_pos, 0, rh_pos, color=COLOR_LIGHT_BLUE, alpha=0.30,
                      label="log regime: $\\hat r = 1 + \\log(1+z)$")
    ax_r.plot(z_neg, rh_neg, color=COLOR_GREY, linewidth=1.5)
    ax_r.plot(z_pos, rh_pos, color=COLOR_BLUE, linewidth=1.5)

    ax_r.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    # Non-firing walkers cluster around z = 0
    rng = np.random.default_rng(0)
    z_non = rng.normal(0.0, 0.3, 14)
    rh_non = relativize(z_non)
    ax_r.scatter(z_non, rh_non, color=COLOR_BLUE, s=22, alpha=0.85,
                 edgecolor="white", linewidth=0.5, zorder=10,
                 label="non-firing walkers")
    # Firing walker at z ≈ 4.5
    z_fire = 4.5
    rh_fire = relativize(np.array([z_fire]))[0]
    ax_r.scatter([z_fire], [rh_fire], color=COLOR_RED, s=110, marker="*",
                 edgecolor="black", linewidth=0.5, zorder=20,
                 label="firing walker")

    # Annotations
    ax_r.annotate(f"$\\hat r \\approx {rh_fire:.2f}$",
                  xy=(z_fire, rh_fire), xytext=(z_fire - 0.6, rh_fire + 0.4),
                  fontsize=8, color=COLOR_RED,
                  arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.6))
    ax_r.annotate(r"$\hat r \approx 1/e$",
                  xy=(0, 1 / np.e), xytext=(-2.5, 0.55), fontsize=8,
                  color=COLOR_GREY,
                  arrowprops=dict(arrowstyle="->", color=COLOR_GREY, lw=0.6))

    ax_r.set_xlabel(r"$z = (r - \bar r) / \sigma_r$")
    ax_r.set_ylabel(r"$\hat r = \mathrm{relativize}(r)$")
    ax_r.set_title("(b) Relativize regime separation")
    ax_r.legend(loc="upper left", fontsize=7, framealpha=0.95)
    ax_r.set_xlim(-3, 6)
    ax_r.set_ylim(0, 3.0)
    ax_r.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_schematic.pdf")
    fig.savefig(FIG_DIR / "fig4_schematic.png")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig4_schematic.pdf'}")


if __name__ == "__main__":
    main()
