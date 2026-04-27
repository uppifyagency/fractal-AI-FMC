"""Plot Milestone 5 distillation results: training curves + NN-vs-FMC tracking."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    # Load
    with open(RESULTS_DIR / "training_log.json") as f:
        train = json.load(f)
    with open(RESULTS_DIR / "milestone_5_benchmark.json") as f:
        bench = json.load(f)

    fig = plt.figure(figsize=(16, 9))

    # 1) Training curves
    ax = fig.add_subplot(2, 3, 1)
    epochs = [r["epoch"] for r in train["log"]]
    ax.semilogy(epochs, [r["train_loss"] for r in train["log"]], "b-", label="train MSE")
    ax.semilogy(epochs, [r["val_loss"] for r in train["log"]], "r-", label="val MSE")
    ax.axhline(train["final_val_loss"], color="gray", ls=":", label=f"final val={train['final_val_loss']:.3f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalized)")
    ax.set_title("Training curves (behavioral cloning)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Latency comparison (log scale bar chart)
    ax = fig.add_subplot(2, 3, 2)
    lat = bench["latency"]
    methods = ["NN policy", "FMC small\n(M=32, H=8)", "FMC full\n(M=200, H=20)"]
    medians = [lat["nn_policy_us"]["median"], lat["fmc_small_us"]["median"],
               lat["fmc_full_us"]["median"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(methods, medians, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("median latency [µs]")
    ax.set_title(f"Decision latency — NN {lat['speedup_vs_full']:.0f}× vs FMC(full)")
    for b, v in zip(bars, medians):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f} µs",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 3) Tracking R_p
    ax = fig.add_subplot(2, 3, 3)
    nn_log = bench["tracking"]["nn_log"]
    fmc_log = bench["tracking"]["fmc_log"]
    target = bench["tracking"]["target"]
    t = [r["t_ms"] for r in fmc_log]
    ax.plot([r["t_ms"] for r in nn_log], [r["R_p"] for r in nn_log],
            "g-", label="NN policy", linewidth=2)
    ax.plot(t, [r["R_p"] for r in fmc_log], "r-", label="FMC", linewidth=2)
    ax.axhline(target["R_p"], color="k", ls="--", label=f"target {target['R_p']}")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("R_p [m]"); ax.set_title("Tracking R_p")
    ax.legend(); ax.grid(alpha=0.3)

    # 4) Tracking κ
    ax = fig.add_subplot(2, 3, 4)
    ax.plot([r["t_ms"] for r in nn_log], [r["kappa"] for r in nn_log],
            "g-", label="NN policy", linewidth=2)
    ax.plot(t, [r["kappa"] for r in fmc_log], "r-", label="FMC", linewidth=2)
    ax.axhline(target["kappa"], color="k", ls="--", label=f"target {target['kappa']}")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("κ"); ax.set_title("Tracking κ (elongation)")
    ax.legend(); ax.grid(alpha=0.3)

    # 5) I_p
    ax = fig.add_subplot(2, 3, 5)
    ax.plot([r["t_ms"] for r in nn_log], [r["I_p_kA"] for r in nn_log],
            "g-", label="NN policy", linewidth=2)
    ax.plot(t, [r["I_p_kA"] for r in fmc_log], "r-", label="FMC", linewidth=2)
    ax.set_xlabel("t [ms]"); ax.set_ylabel("I_p [kA]")
    ax.set_title("Plasma current")
    ax.legend(); ax.grid(alpha=0.3)

    # 6) Episode wall-clock
    ax = fig.add_subplot(2, 3, 6)
    ax.bar(["NN policy", "FMC"],
           [bench["tracking"]["nn_wall_ms"], bench["tracking"]["fmc_wall_ms"]],
           color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("wall-clock for 50-tick episode [ms]")
    ratio = bench["tracking"]["fmc_wall_ms"] / bench["tracking"]["nn_wall_ms"]
    ax.set_title(f"Episode runtime — FMC is {ratio:.1f}× slower")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate([bench["tracking"]["nn_wall_ms"], bench["tracking"]["fmc_wall_ms"]]):
        ax.text(i, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        f"Milestone 5 — FMC-to-policy distillation (500 expert samples, "
        f"3380-param MLP, behavioral cloning)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()

    out = RESULTS_DIR / "milestone_5_distillation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
