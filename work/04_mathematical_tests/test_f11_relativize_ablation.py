"""F11 — Reward-negativity ablation: with vs without `relativize`.

Sergio's claim (video, righe 280-283):
    "si usáramos recompensas negativas como aquí chocarse [...] cuando hacen la
    suma digamos ven que seguir andando le da recompensa negativa con lo cual
    lo que hacen es frenar y entonces se vuelven miedosos".

Concretely Sergio asserts that on a mixed-sign reward landscape, removing
`relativize` produces a qualitatively different, "fearful" agent: it stalls,
freezes near the start, and avoids the negative regions even when crossing
them is the only path to a positive goal.

Test setup:

  Mixed-sign reward landscape:
    - one positive Gaussian peak at (8, 5)         R_max ~ +1.2
    - one negative Gaussian well between start and peak (5, 5)  R_min ~ -1.5
    - bounded domain [0, 10]^2
    - all walkers initialised at (1, 5) — the goal is to reach (8, 5),
      crossing or skirting the negative well.

  Three conditions:
    (A) FMC with `relativize`         — canonical
    (B) FMC without `relativize`      — raw rewards (potentially negative)
    (C) Random walk baseline          — diffusion only

  Metrics over time:
    - Mean position (x, y)              ← does the swarm advance?
    - Coverage area (convex hull)       ← does it explore?
    - Mean instantaneous reward         ← are walkers in productive regions?
    - Probability of reaching x > 6     ← did it cross the well?
    - Speed (mean step magnitude)       ← is it "freezing"?
    - Variance of walker positions      ← collapse vs spread

We expect:
    A.  walkers advance, cross the well, accumulate at the peak
    B.  walkers freeze near the start (Sergio's "fearful" prediction)
    C.  walkers diffuse uniformly with no advance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from fmc_core import FMCConfig, FMCSwarm
from toy_environment import (BoundedDomain, mixed_sign_reward,
                             smooth_gradient_negative_reward, grid_evaluate)


HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def fixed_init(n_walkers: int, x0: tuple[float, float],
               jitter: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array(x0, dtype=np.float64)
    return base + rng.normal(0, jitter, size=(n_walkers, 2))


def safe_hull_area(points: np.ndarray) -> float:
    """Convex hull area; 0 if degenerate."""
    if len(points) < 3:
        return 0.0
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)  # 2D hull -> volume == area
    except Exception:
        return 0.0


def run_one(label: str, domain: BoundedDomain, R_fn, n_walkers: int,
            n_steps: int, seed: int, mode: str) -> dict:
    """mode: 'fmc_relativize' | 'fmc_raw_clip' | 'fmc_raw_signed' | 'random' """
    init = fixed_init(n_walkers, x0=(1.0, 5.0), seed=seed)
    if mode == "random":
        rng = np.random.default_rng(seed)
        states = init.copy()
        history = [states.copy()]
        for _ in range(n_steps):
            states = domain.step_fn(states, rng)
            history.append(states.copy())
        rewards_history = [R_fn(s) for s in history]
    else:
        if mode == "fmc_relativize":
            use_rel, clip = True, True  # canonical (clip flag is irrelevant when relativize on)
        elif mode == "fmc_raw_clip":
            use_rel, clip = False, True
        elif mode == "fmc_raw_signed":
            use_rel, clip = False, False
        else:
            raise ValueError(f"unknown mode: {mode}")
        cfg = FMCConfig(n_walkers=n_walkers, balance=1.0,
                        use_relativize_reward=use_rel,
                        use_relativize_distance=use_rel,
                        clip_negative_reward_at_zero=clip,
                        rng_seed=seed)
        swarm = FMCSwarm(step_fn=domain.step_fn, reward_fn=R_fn,
                         init_states=init, config=cfg)
        history = [swarm.states.copy()]
        rewards_history = [swarm.rewards.copy()]
        for _ in range(n_steps):
            swarm.step()
            history.append(swarm.states.copy())
            rewards_history.append(swarm.rewards.copy())

    # ---- metrics over time ----
    mean_x_t = np.array([h[:, 0].mean() for h in history])
    mean_y_t = np.array([h[:, 1].mean() for h in history])
    var_t = np.array([h.var(axis=0).sum() for h in history])
    hull_t = np.array([safe_hull_area(h) for h in history])
    speed_t = np.array([np.linalg.norm(history[t] - history[t - 1], axis=1).mean()
                        for t in range(1, len(history))])
    speed_t = np.concatenate([[0.0], speed_t])
    mean_r_t = np.array([r.mean() for r in rewards_history])
    crossed_t = np.array([(h[:, 0] > 6.0).mean() for h in history])
    reached_peak_t = np.array([
        (np.linalg.norm(h - np.array([8.0, 5.0]), axis=1) < 1.0).mean()
        for h in history
    ])

    return {
        "label": label,
        "mode": mode,
        "history": history,
        "mean_x": mean_x_t.tolist(),
        "mean_y": mean_y_t.tolist(),
        "var": var_t.tolist(),
        "hull": hull_t.tolist(),
        "speed": speed_t.tolist(),
        "mean_reward": mean_r_t.tolist(),
        "frac_crossed_x6": crossed_t.tolist(),
        "frac_reached_peak": reached_peak_t.tolist(),
        "final_state": history[-1],
        "n_steps": n_steps,
    }


def run_scenario(scenario_name: str, R, domain: BoundedDomain,
                 n_walkers: int, n_steps: int, n_seeds: int, n_grid: int,
                 modes: dict, suffix: str) -> dict:
    """Run all conditions on a given reward landscape and emit plots+summary."""
    X, Y, Z = grid_evaluate(R, domain.L, n_grid=n_grid)
    by_cond: dict[str, list[dict]] = {name: [] for name in modes}
    for seed in range(n_seeds):
        for name, m in modes.items():
            r = run_one(name, domain, R, n_walkers, n_steps, seed, m)
            by_cond[name].append(r)
            print(f"  [{scenario_name}] seed={seed} {name:24s} "
                  f"final mean x={r['mean_x'][-1]:.3f}  "
                  f"frac_crossed={r['frac_crossed_x6'][-1]:.3f}  "
                  f"frac_at_peak={r['frac_reached_peak'][-1]:.3f}  "
                  f"hull={r['hull'][-1]:.2f}")
    summary = {"scenario": scenario_name, "n_walkers": n_walkers,
               "n_steps": n_steps, "n_seeds": n_seeds,
               "domain_L": domain.L, "step_sigma": domain.step_sigma,
               "landscape": {"R_min": float(Z.min()), "R_max": float(Z.max())},
               "per_condition": {}}
    for name, runs in by_cond.items():
        arrs = {k: np.array([r[k] for r in runs]) for k in
                ["mean_x", "mean_y", "var", "hull", "speed",
                 "mean_reward", "frac_crossed_x6", "frac_reached_peak"]}
        summary["per_condition"][name] = {
            f"{k}_mean": arrs[k].mean(axis=0).tolist() for k in arrs
        }
        for k in arrs:
            summary["per_condition"][name][f"{k}_std"] = arrs[k].std(axis=0).tolist()
        summary["per_condition"][name]["final"] = {
            k: float(arrs[k][:, -1].mean()) for k in arrs
        }
        summary["per_condition"][name]["final_std"] = {
            k: float(arrs[k][:, -1].std()) for k in arrs
        }
    _plot_landscape_trajectories(by_cond, X, Y, Z, domain.L,
                                 RESULTS / f"f11_{suffix}_trajectories.png")
    _plot_metrics_over_time(summary, modes,
                            RESULTS / f"f11_{suffix}_metrics_over_time.png")
    _plot_final_distributions(by_cond, X, Y, Z, domain.L,
                              RESULTS / f"f11_{suffix}_final_distributions.png")
    return summary


def main(n_walkers: int = 400, n_steps: int = 300, n_seeds: int = 4,
         n_grid: int = 80) -> dict:
    domain = BoundedDomain(L=10.0, step_sigma=0.30)

    # ----- Scenario 1: Wells & peak, global negative offset -----
    R_a = mixed_sign_reward(
        pos_centers=[(8.0, 5.0)], pos_sigmas=[1.0], pos_weights=[2.5],
        neg_centers=[(5.0, 5.0)], neg_sigmas=[1.0], neg_weights=[1.5],
        global_offset=-0.5,
    )

    # ----- Scenario 2: Smooth monotone gradient, all values mostly negative -----
    R_b = smooth_gradient_negative_reward(slope=0.10, offset=-0.7, L=10.0)

    print("=" * 60)
    print("F11 — relativize ablation, two scenarios")
    print("=" * 60)
    print(f"  Scenario A (wells+peak, offset=-0.5)")
    print(f"  Scenario B (smooth gradient, R = -0.7 + 0.10 x  → R ∈ [-0.7, +0.3])")
    print(f"  Start: all walkers at (1, 5);  Goal proxy: x > 6")
    print(f"  N={n_walkers} walkers, T={n_steps} steps, {n_seeds} seeds")
    print(f"  Start: all walkers at (1, 5);  Goal: x > 6 (cross well to reach peak)")
    print(f"  N={n_walkers} walkers, T={n_steps} steps, {n_seeds} seeds")
    modes = {
        "FMC_with_relativize": "fmc_relativize",
        "FMC_raw_clip_at_0": "fmc_raw_clip",
        "FMC_raw_signed_negatives": "fmc_raw_signed",
        "Random_walk": "random",
    }

    print("\n--- Scenario A: wells + peak, offset=-0.5 ---")
    sum_a = run_scenario("A_wells_offset", R_a, domain, n_walkers, n_steps,
                          n_seeds, n_grid, modes, suffix="A_wells_offset")
    print("\n--- Scenario B: smooth monotone gradient, all-negative ---")
    sum_b = run_scenario("B_smooth_gradient", R_b, domain, n_walkers, n_steps,
                          n_seeds, n_grid, modes, suffix="B_smooth_gradient")

    summary = {"scenario_A": sum_a, "scenario_B": sum_b}
    out_json = RESULTS / "f11_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[F11] Wrote summary -> {out_json}")

    # ----- automated assertions -----
    for tag, sm in [("Scenario A", sum_a), ("Scenario B", sum_b)]:
        print("\n" + "=" * 60)
        print(f"Automated verification of F11 claim — {tag}")
        print("=" * 60)
        finals = {name: sm["per_condition"][name]["final"] for name in modes}
        header = "  " + "metric".ljust(28) + "".join(f"{n:>22s}" for n in modes)
        print(header)
        for k in ("frac_crossed_x6", "frac_reached_peak", "mean_x", "hull", "mean_reward", "speed"):
            line = "  " + k.ljust(28) + "".join(f"{finals[n][k]:>22.4f}" for n in modes)
            print(line)
        rel = finals["FMC_with_relativize"]
        raw_signed = finals["FMC_raw_signed_negatives"]
        raw_clip = finals["FMC_raw_clip_at_0"]
        rnd = finals["Random_walk"]
        print()
        print("Hypothesis tests (Sergio's claim):")
        print(f"  H1: relativize beats raw_signed on advance        -> "
              f"{rel['mean_x'] > raw_signed['mean_x']}")
        print(f"  H2: raw_signed exhibits 'fearful' (mean_x < 3)    -> "
              f"{raw_signed['mean_x'] < 3.0}")
        print(f"  H3: relativize beats raw_signed on mean reward    -> "
              f"{rel['mean_reward'] > raw_signed['mean_reward']}")
        print(f"  H4: relativize beats random_walk on advance       -> "
              f"{rel['mean_x'] > rnd['mean_x']}")

    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_landscape_trajectories(by_cond, X, Y, Z, L, out: Path) -> None:
    """Heat-map of R(x,y) overlaid with a few walker trajectories per condition."""
    n_cond = len(by_cond)
    fig, axes = plt.subplots(1, n_cond, figsize=(4.2 * n_cond, 4.2))
    if n_cond == 1:
        axes = [axes]
    for ax, (name, runs) in zip(axes, by_cond.items()):
        im = ax.imshow(Z, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r",
                       vmin=-Z.max(), vmax=Z.max())
        # overlay first 12 walker trajectories from seed 0
        history = runs[0]["history"]
        T = len(history)
        # subsample timesteps
        ts = np.linspace(0, T - 1, 30, dtype=int)
        for w in range(12):
            xs = [history[t][w, 0] for t in ts]
            ys = [history[t][w, 1] for t in ts]
            ax.plot(xs, ys, "-", color="black", alpha=0.35, linewidth=0.8)
            ax.plot(xs[0], ys[0], "o", color="green", markersize=4)
            ax.plot(xs[-1], ys[-1], "x", color="yellow", markersize=6)
        ax.set_title(name.replace("_", " "), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        # mark goal
        ax.plot(8, 5, "*", color="lime", markersize=14, markeredgecolor="black")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="R(x,y)")
    fig.suptitle("F11 — trajectories on mixed-sign landscape (start: green; end: yellow x; goal: green star)",
                 y=1.02)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics_over_time(summary, modes, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [("mean_x", "mean walker x  (start=1, goal=8)"),
               ("frac_crossed_x6", "fraction of walkers with x > 6"),
               ("frac_reached_peak", "fraction within 1.0 of peak (8,5)"),
               ("hull", "convex hull area"),
               ("speed", "mean step length"),
               ("mean_reward", "mean instantaneous reward")]
    colors = {"FMC_with_relativize": "#1f77b4",
              "FMC_raw_clip_at_0": "#ff7f0e",
              "FMC_raw_signed_negatives": "#d62728",
              "Random_walk": "#2ca02c"}
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for name in modes:
            mean_curve = np.array(summary["per_condition"][name][f"{key}_mean"])
            std_curve = np.array(summary["per_condition"][name][f"{key}_std"])
            ax.plot(mean_curve, label=name, color=colors[name], linewidth=1.6)
            ax.fill_between(np.arange(len(mean_curve)),
                            mean_curve - std_curve, mean_curve + std_curve,
                            color=colors[name], alpha=0.18)
        ax.set_xlabel("step"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("F11 — relativize ablation: dynamics on mixed-sign landscape", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _plot_final_distributions(by_cond, X, Y, Z, L, out: Path) -> None:
    n_grid = Z.shape[0]
    fig, axes = plt.subplots(1, len(by_cond) + 1, figsize=(3.4 * (len(by_cond) + 1), 3.6))
    axes[0].imshow(Z, origin="lower", extent=[0, L, 0, L], cmap="RdBu_r",
                   vmin=-Z.max(), vmax=Z.max())
    axes[0].set_title("Reward landscape R")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].plot(1, 5, "o", color="green", markersize=8)
    axes[0].plot(8, 5, "*", color="lime", markersize=14, markeredgecolor="black")
    for j, (name, runs) in enumerate(by_cond.items()):
        # pool final states across seeds
        finals = np.concatenate([r["history"][-1] for r in runs], axis=0)
        H, _, _ = np.histogram2d(finals[:, 0], finals[:, 1],
                                 bins=n_grid, range=[[0, L], [0, L]])
        H = H.T
        axes[j + 1].imshow(H, origin="lower", extent=[0, L, 0, L], cmap="viridis")
        axes[j + 1].set_title(f"final P_walker | {name.replace('_', ' ')}")
        axes[j + 1].set_xticks([]); axes[j + 1].set_yticks([])
        axes[j + 1].plot(8, 5, "*", color="lime", markersize=10, markeredgecolor="black")
    fig.suptitle("F11 — final walker distribution per condition (pooled over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
