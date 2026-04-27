"""Plot Milestone 3 tracking results from results/milestone_3_tracking.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    log_file = RESULTS_DIR / "milestone_3_tracking.json"
    with open(log_file) as f:
        data = json.load(f)

    target = data["target"]
    log = data["log"]
    t = [r["t_ms"] for r in log]
    R_p = [r["R_p"] for r in log]
    Z_p = [r["Z_p"] * 1e3 for r in log]  # mm
    kappa = [r["kappa"] for r in log]
    delta = [r["delta"] for r in log]
    I_p_kA = [r["I_p_kA"] for r in log]
    alive = [r["alive"] for r in log]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # R_p
    ax = axes[0, 0]
    ax.plot(t, R_p, "b-", label="actual")
    ax.axhline(target["R_p"], color="r", ls="--", label="target")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("R_p [m]"); ax.set_title("Plasma centroid R")
    ax.legend(); ax.grid(alpha=0.3)

    # Z_p
    ax = axes[0, 1]
    ax.plot(t, Z_p, "g-", label="actual")
    ax.axhline(target["Z_p"] * 1e3, color="r", ls="--", label="target")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("Z_p [mm]"); ax.set_title("Plasma centroid Z")
    ax.legend(); ax.grid(alpha=0.3)

    # kappa
    ax = axes[0, 2]
    ax.plot(t, kappa, "m-", label="actual")
    ax.axhline(target["kappa"], color="r", ls="--", label="target")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("κ"); ax.set_title("Elongation")
    ax.legend(); ax.grid(alpha=0.3)

    # delta
    ax = axes[1, 0]
    ax.plot(t, delta, "c-", label="actual")
    ax.axhline(target["delta"], color="r", ls="--", label="target")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("δ"); ax.set_title("Triangularity")
    ax.legend(); ax.grid(alpha=0.3)

    # I_p
    ax = axes[1, 1]
    ax.plot(t, I_p_kA, "k-")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("I_p [kA]"); ax.set_title("Plasma current")
    ax.grid(alpha=0.3)

    # walkers alive
    ax = axes[1, 2]
    ax.plot(t, alive, "r-")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("count")
    ax.set_title(f"FMC walkers alive (M={data['config']['n_walkers']})")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, data["config"]["n_walkers"] * 1.1)

    fig.suptitle(
        f"FMC tracking demo (Milestone 3) — "
        f"target: R_p={target['R_p']:.3f} m, κ={target['kappa']:.2f}, "
        f"M={data['config']['n_walkers']}, H={data['config']['horizon']}",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_3_tracking.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
