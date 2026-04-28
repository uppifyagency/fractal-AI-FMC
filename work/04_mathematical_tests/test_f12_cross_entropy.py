"""F12 — Cross-entropy collapse / Gibbs equilibrium verification.

Sergio's claim (video, riga 528-537):
    "la inteligencia [...] va de que la probabilidad de que tú vayas a un sitio
    a otro sea proporcional a la recompensa".

Paper section 3 eq. (3): optimal scanning policy   P_S(x) ∝ R(x).
Deep-dive 01 Theorem 3: stationary distribution    π*(x) ∝ R(x)^α.

We test the GIBBS-EQUILIBRIUM form because it generalizes F12 to arbitrary balance α
and gives an analytic target curve to verify against. The α=1 case is exactly F12.

Two complementary experiments:

  TEST A — Unimodal landscape, α-scan.
    R(x) = single anisotropic Gaussian peak. We sweep α in {0, 0.5, 1, 2, 4}
    and verify that the empirical walker distribution matches R(x)^α / Z_α
    (Pearson on log-densities should be high; KL drops with N and T).

  TEST B — Multimodal landscape (3 peaks), with comparison to:
      (a) FMC canonical
      (b) FMC without relativize
      (c) Pure random walk (no FMC mechanics)
    Here we expect FMC to track P_R better than random, but to exhibit a
    well-known mode-collapse bias (visible in our deep-dive 01 Lemma 4 caveat).
    We document this honestly.

All metrics are computed against analytic targets via numerical integration on
a grid. Plots and JSON go to results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import xlogy

from fmc_core import FMCConfig, FMCSwarm
from toy_environment import BoundedDomain, gaussian_mixture_reward, grid_evaluate


HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------
def empirical_density(states_list: list[np.ndarray], L: float, n_grid: int) -> np.ndarray:
    """Pool walker positions across multiple snapshots into one PMF on the grid.

    Pooling across the *stationary window* reduces sampling noise and gives an
    unbiased estimator of the marginal walker distribution under the FMC kernel.
    """
    flat = np.concatenate(states_list, axis=0)
    H, _, _ = np.histogram2d(flat[:, 0], flat[:, 1],
                             bins=n_grid, range=[[0, L], [0, L]])
    H = H.T
    s = H.sum()
    return H / s if s > 0 else H


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(xlogy(p, p / q).sum())


def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p / p.sum(); q = np.clip(q, eps, None); q = q / q.sum()
    return float(-(p * np.log(q)).sum())


def shannon_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None); p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum(); q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def log_pearson(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    lp = np.log(np.clip(p, eps, None)).ravel()
    lq = np.log(np.clip(q, eps, None)).ravel()
    return float(np.corrcoef(lp, lq)[0, 1])


def gibbs_target(Z: np.ndarray, alpha: float) -> np.ndarray:
    """Analytic Gibbs target  pi*(x) ∝ R(x)^alpha , normalized as PMF."""
    pwr = np.power(Z, alpha)
    return pwr / pwr.sum()


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run_fmc(domain: BoundedDomain, R_fn, n_walkers: int, n_steps: int,
            seed: int, balance: float, use_relativize: bool) -> dict:
    rng = np.random.default_rng(seed)
    init = domain.random_init(n_walkers, rng)
    cfg = FMCConfig(n_walkers=n_walkers,
                    balance=balance,
                    use_relativize_reward=use_relativize,
                    use_relativize_distance=use_relativize,
                    rng_seed=seed)
    swarm = FMCSwarm(step_fn=domain.step_fn, reward_fn=R_fn,
                     init_states=init, config=cfg)
    snapshots = [swarm.states.copy()]
    for _ in range(n_steps):
        swarm.step()
        snapshots.append(swarm.states.copy())
    return {"snapshots": snapshots, "n_steps": n_steps}


def run_random(domain: BoundedDomain, n_walkers: int, n_steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    states = domain.random_init(n_walkers, rng)
    snapshots = [states.copy()]
    for _ in range(n_steps):
        states = domain.step_fn(states, rng)
        snapshots.append(states.copy())
    return {"snapshots": snapshots, "n_steps": n_steps}


# ---------------------------------------------------------------------------
# TEST A — alpha scan on unimodal landscape
# ---------------------------------------------------------------------------
def test_a_alpha_scan(n_walkers=400, n_steps=400, n_seeds=4, n_grid=80,
                      burn_in_frac=0.5,
                      alphas=(0.0, 0.5, 1.0, 2.0, 4.0)) -> dict:
    print("\n" + "=" * 60)
    print("TEST A — Gibbs equilibrium under alpha scan, unimodal R")
    print("=" * 60)
    domain = BoundedDomain(L=10.0, step_sigma=0.40)
    R = gaussian_mixture_reward(
        centers=[(5.0, 5.0)], sigmas=[1.5], weights=[1.0], baseline=0.05)
    X, Y, Z = grid_evaluate(R, domain.L, n_grid=n_grid)
    P_R = Z / Z.sum()

    out = {"alphas": list(alphas), "n_walkers": n_walkers, "n_steps": n_steps,
           "n_seeds": n_seeds, "n_grid": n_grid, "burn_in_frac": burn_in_frac,
           "domain_L": domain.L, "step_sigma": domain.step_sigma,
           "rewards_landscape": "unimodal Gaussian centered (5,5) sigma=1.5"}
    per_alpha: dict[str, dict] = {}
    grid_imgs = {}
    for a in alphas:
        kls, ces, corrs, tvs = [], [], [], []
        first_emp = None
        for seed in range(n_seeds):
            r = run_fmc(domain, R, n_walkers, n_steps, seed, a, use_relativize=True)
            burn = int(burn_in_frac * len(r["snapshots"]))
            P_W = empirical_density(r["snapshots"][burn:], domain.L, n_grid)
            P_target = gibbs_target(Z, a)
            kls.append(kl_divergence(P_W, P_target))
            ces.append(cross_entropy(P_W, P_target))
            corrs.append(log_pearson(P_W, P_target))
            tvs.append(total_variation(P_W, P_target))
            if seed == 0:
                first_emp = P_W
        per_alpha[str(a)] = {
            "kl_to_gibbs_mean": float(np.mean(kls)),
            "kl_to_gibbs_std": float(np.std(kls)),
            "ce_mean": float(np.mean(ces)),
            "log_pearson_mean": float(np.mean(corrs)),
            "log_pearson_std": float(np.std(corrs)),
            "tv_mean": float(np.mean(tvs)),
            "kl_to_gibbs_seeds": [float(v) for v in kls],
        }
        grid_imgs[a] = (first_emp, gibbs_target(Z, a))
        print(f"  alpha={a:>4}  KL(P_W||Gibbs)={np.mean(kls):.4f} ± {np.std(kls):.4f}  "
              f"log-Pearson={np.mean(corrs):.4f}  TV={np.mean(tvs):.4f}")

    out["per_alpha"] = per_alpha
    out["H_PR"] = float(shannon_entropy(P_R))

    # Plot: per-alpha empirical vs analytic Gibbs target
    fig, axes = plt.subplots(2, len(alphas), figsize=(3.0 * len(alphas), 5.4))
    for j, a in enumerate(alphas):
        emp, tgt = grid_imgs[a]
        axes[0, j].imshow(emp, origin="lower",
                          extent=[0, domain.L, 0, domain.L], cmap="viridis")
        axes[0, j].set_title(f"empirical α={a}")
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].imshow(tgt, origin="lower",
                          extent=[0, domain.L, 0, domain.L], cmap="magma")
        axes[1, j].set_title(f"target ∝ R^{a}")
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    fig.suptitle("Test A — empirical vs Gibbs target (R^α / Z) | unimodal R", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "f12A_alpha_scan.png", dpi=130)
    plt.close(fig)

    # Plot: log-Pearson and KL vs alpha
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    a_arr = np.array(alphas)
    pmean = np.array([per_alpha[str(a)]["log_pearson_mean"] for a in alphas])
    pstd = np.array([per_alpha[str(a)]["log_pearson_std"] for a in alphas])
    klmean = np.array([per_alpha[str(a)]["kl_to_gibbs_mean"] for a in alphas])
    klstd = np.array([per_alpha[str(a)]["kl_to_gibbs_std"] for a in alphas])
    axes[0].errorbar(a_arr, pmean, yerr=pstd, fmt="o-", capsize=3, color="#1f77b4")
    axes[0].set_xlabel("α (balance)"); axes[0].set_ylabel("Pearson(log P_W, log Gibbs)")
    axes[0].set_title("log-Pearson vs α")
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(a_arr, klmean, yerr=klstd, fmt="o-", capsize=3, color="#d62728")
    axes[1].set_xlabel("α (balance)"); axes[1].set_ylabel("KL(P_W ‖ Gibbs)  [nats]")
    axes[1].set_title("KL to Gibbs target vs α")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Test A — Gibbs equilibrium fidelity per α", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "f12A_metrics_vs_alpha.png", dpi=130)
    plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# TEST B — multimodal landscape: FMC vs no-relativize vs random
# ---------------------------------------------------------------------------
def test_b_multimodal(n_walkers=400, n_steps=400, n_seeds=4, n_grid=80,
                      burn_in_frac=0.5) -> dict:
    print("\n" + "=" * 60)
    print("TEST B — multimodal R: FMC canonical vs no-relativize vs random")
    print("=" * 60)
    domain = BoundedDomain(L=10.0, step_sigma=0.40)
    R = gaussian_mixture_reward(
        centers=[(2.5, 2.5), (7.5, 2.5), (5.0, 7.5)],
        sigmas=[0.7, 0.9, 1.1],
        weights=[1.0, 0.8, 1.2],
        baseline=0.05,
    )
    X, Y, Z = grid_evaluate(R, domain.L, n_grid=n_grid)
    P_R = Z / Z.sum()
    H_R = shannon_entropy(P_R)

    out = {"n_walkers": n_walkers, "n_steps": n_steps, "n_seeds": n_seeds,
           "n_grid": n_grid, "burn_in_frac": burn_in_frac,
           "H_PR": float(H_R), "domain_L": domain.L,
           "step_sigma": domain.step_sigma}

    conditions = ["FMC_canonical_a1", "FMC_no_relativize_a1", "Random_walk"]
    per_cond = {}
    sample_emp = {}
    for cond in conditions:
        kls, ces, corrs, tvs = [], [], [], []
        first_emp = None
        for seed in range(n_seeds):
            if cond == "Random_walk":
                r = run_random(domain, n_walkers, n_steps, seed)
            else:
                use_rel = (cond == "FMC_canonical_a1")
                r = run_fmc(domain, R, n_walkers, n_steps, seed,
                            balance=1.0, use_relativize=use_rel)
            burn = int(burn_in_frac * len(r["snapshots"]))
            P_W = empirical_density(r["snapshots"][burn:], domain.L, n_grid)
            kls.append(kl_divergence(P_W, P_R))
            ces.append(cross_entropy(P_W, P_R))
            corrs.append(log_pearson(P_W, P_R))
            tvs.append(total_variation(P_W, P_R))
            if seed == 0:
                first_emp = P_W
        per_cond[cond] = {
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls)),
            "ce_mean": float(np.mean(ces)),
            "log_pearson_mean": float(np.mean(corrs)),
            "log_pearson_std": float(np.std(corrs)),
            "tv_mean": float(np.mean(tvs)),
            "kl_seeds": [float(v) for v in kls],
        }
        sample_emp[cond] = first_emp
        print(f"  {cond:24s} KL={np.mean(kls):.4f}  log-Pearson={np.mean(corrs):.4f}  TV={np.mean(tvs):.4f}")

    out["per_condition"] = per_cond

    # Plot: empirical densities side-by-side with target
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    axes[0].imshow(P_R, origin="lower", extent=[0, domain.L, 0, domain.L], cmap="magma")
    axes[0].set_title("target P_R (∝ R)"); axes[0].set_xticks([]); axes[0].set_yticks([])
    for j, cond in enumerate(conditions):
        axes[j + 1].imshow(sample_emp[cond], origin="lower",
                           extent=[0, domain.L, 0, domain.L], cmap="viridis")
        axes[j + 1].set_title(cond.replace("_", " "))
        axes[j + 1].set_xticks([]); axes[j + 1].set_yticks([])
    fig.suptitle("Test B — empirical walker density vs P_R, multimodal landscape", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "f12B_multimodal_emp_vs_target.png", dpi=130)
    plt.close(fig)

    # Bar plot: KL/Pearson/TV across conditions
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    metrics = [("kl_mean", "KL(P_W ‖ P_R)"),
               ("log_pearson_mean", "Pearson(log P_W, log P_R)"),
               ("tv_mean", "TV(P_W, P_R)")]
    for ax, (key, title) in zip(axes, metrics):
        vals = [per_cond[c][key] for c in conditions]
        stds = [per_cond[c].get(key.replace("mean", "std"), 0.0) for c in conditions]
        bars = ax.bar(range(len(conditions)), vals, yerr=stds, capsize=4,
                      color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=8)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Test B — fidelity to P_R across conditions", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "f12B_metrics_bars.png", dpi=130)
    plt.close(fig)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    summary = {
        "test_A": test_a_alpha_scan(),
        "test_B": test_b_multimodal(),
    }
    out_json = RESULTS / "f12_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[F12] Wrote summary -> {out_json}")
    print(f"[F12] Plots -> {RESULTS}/f12*.png")

    # ----- automated assertion / reporting -----
    print("\n" + "=" * 60)
    print("Automated verification of F12 claim")
    print("=" * 60)
    a1 = summary["test_A"]["per_alpha"]["1.0"]
    print(f"  Test A α=1: log-Pearson(P_W, R) = {a1['log_pearson_mean']:.4f} ± {a1['log_pearson_std']:.4f}")
    print(f"  Test A α=1: KL(P_W ‖ R)         = {a1['kl_to_gibbs_mean']:.4f} ± {a1['kl_to_gibbs_std']:.4f}")
    pcurve = [summary["test_A"]["per_alpha"][str(a)]["log_pearson_mean"]
              for a in summary["test_A"]["alphas"]]
    is_monotonic = all(pcurve[i] <= pcurve[i + 1] + 0.05 for i in range(len(pcurve) - 1))
    print(f"  log-Pearson monotone-ish in α: {is_monotonic}  (curve: {pcurve})")

    cond = summary["test_B"]["per_condition"]
    fmc_kl = cond["FMC_canonical_a1"]["kl_mean"]
    rand_kl = cond["Random_walk"]["kl_mean"]
    no_rel_kl = cond["FMC_no_relativize_a1"]["kl_mean"]
    print(f"  Test B: FMC_canonical KL={fmc_kl:.4f}  vs Random={rand_kl:.4f}  vs no-relativize={no_rel_kl:.4f}")
    print(f"  FMC canonical beats random: {fmc_kl < rand_kl}")
    print(f"  FMC canonical beats no-relativize: {fmc_kl < no_rel_kl}")

    return summary


if __name__ == "__main__":
    main()
