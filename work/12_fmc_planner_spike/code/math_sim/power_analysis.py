"""Statistical power analysis on Round-1 results.

Questions answered:
1. Are the negative deltas (FMC - greedy) statistically significant?
2. What was the observed effect size?
3. Given Round-1 variance, how many seeds would be needed for Round-2 to detect:
   - delta = +0.05 (modest improvement)
   - delta = +0.10 (clear improvement)
4. Bootstrap CI95 for each delta
5. Power for the 5-seed sample we used
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.stats as st
from statsmodels.stats.power import TTestIndPower


HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = HERE.parent.parent / "03_round2_power_analysis.md"


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d."""
    nx, ny = len(x), len(y)
    s_pool = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if s_pool == 0:
        return float("inf") if x.mean() != y.mean() else 0.0
    return float((x.mean() - y.mean()) / s_pool)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of `diffs`."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = np.array([rng.choice(diffs, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def required_n(effect_size: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Two-sample t-test required n per group for given effect size."""
    if abs(effect_size) < 1e-6:
        return -1  # infeasible
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha,
                             power=power, alternative="two-sided")
    return int(math.ceil(n))


def observed_power(effect_size: float, n_per_group: int,
                  alpha: float = 0.05) -> float:
    analysis = TTestIndPower()
    return float(analysis.solve_power(effect_size=abs(effect_size),
                                      nobs1=n_per_group, alpha=alpha,
                                      alternative="two-sided"))


# ----------------------------- Reconstruct per-seed data -----------------------
# Round-1 JSON only stored mean+std, not raw seed values. We approximate the per-seed
# data from mean+std assuming Gaussian (n=5 seeds in Round-1).


def reconstruct_seeds(mean: float, std: float, n: int = 5,
                     seed: int = 7) -> np.ndarray:
    """Reconstruct n samples matching mean and std exactly."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    return raw * std + mean


# ----------------------------- Analysis tables ---------------------------------


def analyze_main_comparison() -> list[dict]:
    """Compare FMC vs greedy and FMC vs random."""
    data = json.loads((RESULTS / "01_main_comparison.json").read_text())
    fmc = reconstruct_seeds(data["fmc"]["final_coverage_mean"],
                            data["fmc"]["final_coverage_std"])
    greedy = reconstruct_seeds(data["greedy"]["final_coverage_mean"],
                              data["greedy"]["final_coverage_std"])
    random_b = reconstruct_seeds(data["random"]["final_coverage_mean"],
                                data["random"]["final_coverage_std"])

    rows = []
    for label, comparator in [("FMC vs greedy", greedy), ("FMC vs random", random_b)]:
        diffs = fmc - comparator
        d_mean, lo, hi = bootstrap_ci(diffs)
        d_cohen = cohen_d(fmc, comparator)
        t_stat, p_val = st.ttest_ind(fmc, comparator, equal_var=False)
        n_for_05 = required_n(0.05 / fmc.std(ddof=1)) if fmc.std(ddof=1) > 0 else -1
        n_for_10 = required_n(0.10 / fmc.std(ddof=1)) if fmc.std(ddof=1) > 0 else -1
        obs_pow = observed_power(d_cohen, n_per_group=5)
        rows.append({
            "comparison": label,
            "delta_mean": d_mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "cohen_d": d_cohen,
            "p_value": float(p_val),
            "observed_power_n5": obs_pow,
            "n_per_group_to_detect_d_0.05": n_for_05,
            "n_per_group_to_detect_d_0.10": n_for_10,
        })
    return rows


def analyze_noise_sweep() -> list[dict]:
    """Per-sigma analysis."""
    data = json.loads((RESULTS / "05_noise_sweep.json").read_text())
    rows = []
    for key, vals in data.items():
        fmc = reconstruct_seeds(vals["fmc_mean"], vals["fmc_std"])
        greedy = reconstruct_seeds(vals["greedy_mean"], vals["greedy_std"])
        diffs = fmc - greedy
        d_mean, lo, hi = bootstrap_ci(diffs)
        d_cohen = cohen_d(fmc, greedy)
        t_stat, p_val = st.ttest_ind(fmc, greedy, equal_var=False)
        rows.append({
            "sigma": vals["sigma"],
            "delta_mean": d_mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "cohen_d": d_cohen,
            "p_value": float(p_val),
        })
    return rows


def analyze_deception_sweep() -> list[dict]:
    """Per-deception-rate analysis."""
    data = json.loads((RESULTS / "06_deceptive_landscape.json").read_text())
    rows = []
    for key, vals in data.items():
        fmc = reconstruct_seeds(vals["fmc_mean"], vals["fmc_std"])
        greedy = reconstruct_seeds(vals["greedy_misled_mean"],
                                   vals["greedy_misled_std"])
        diffs = fmc - greedy
        d_mean, lo, hi = bootstrap_ci(diffs)
        d_cohen = cohen_d(fmc, greedy)
        t_stat, p_val = st.ttest_ind(fmc, greedy, equal_var=False)
        rows.append({
            "deception_rate": vals["deception_rate"],
            "delta_mean": d_mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "cohen_d": d_cohen,
            "p_value": float(p_val),
        })
    return rows


def design_round2_sample_sizes() -> dict:
    """Given Round-1 SD ~= 0.025 in coverage, compute n needed for various effects."""
    # Round-1 across all experiments: fmc_std ranged 0.013 - 0.032
    typical_sd = 0.025
    targets = [0.025, 0.05, 0.10, 0.15, 0.20]
    results = {}
    for delta in targets:
        d = delta / typical_sd  # Cohen's d
        n_80 = required_n(d, power=0.80)
        n_90 = required_n(d, power=0.90)
        n_95 = required_n(d, power=0.95)
        results[f"delta={delta}"] = {
            "delta_in_coverage_units": delta,
            "cohen_d_assumed_sd_0.025": d,
            "n_for_80%_power": n_80,
            "n_for_90%_power": n_90,
            "n_for_95%_power": n_95,
        }
    return results


def main() -> None:
    main_cmp = analyze_main_comparison()
    noise = analyze_noise_sweep()
    decep = analyze_deception_sweep()
    sample_design = design_round2_sample_sizes()

    json_out = {
        "main_comparison": main_cmp,
        "noise_sweep": noise,
        "deception_sweep": decep,
        "round2_sample_size_design": sample_design,
        "metadata": {
            "round1_n_per_cell": 5,
            "alpha": 0.05,
            "power_target": 0.80,
            "method": "Welch t-test, Cohen's d pooled-SD, bootstrap percentile CI",
        },
    }
    (RESULTS / "08_power_analysis.json").write_text(json.dumps(json_out, indent=2))

    print("=== MAIN COMPARISON ===")
    for r in main_cmp:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
              else "*" if r["p_value"] < 0.05 else " "
        print(f"  {r['comparison']:20} Δ={r['delta_mean']:+.3f}  "
              f"CI95=[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
              f"d={r['cohen_d']:+.2f}  p={r['p_value']:.4f}{sig}  "
              f"power@n5={r['observed_power_n5']:.2f}")
        print(f"    → n per group for d=0.05/SD: {r['n_per_group_to_detect_d_0.05']}, "
              f"for d=0.10/SD: {r['n_per_group_to_detect_d_0.10']}")

    print("\n=== NOISE SWEEP ===")
    for r in noise:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
              else "*" if r["p_value"] < 0.05 else " "
        print(f"  σ={r['sigma']:.1f}  Δ={r['delta_mean']:+.3f}  "
              f"CI95=[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
              f"d={r['cohen_d']:+.2f}  p={r['p_value']:.4f}{sig}")

    print("\n=== DECEPTION SWEEP ===")
    for r in decep:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
              else "*" if r["p_value"] < 0.05 else " "
        print(f"  decep={r['deception_rate']:.1f}  Δ={r['delta_mean']:+.3f}  "
              f"CI95=[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]  "
              f"d={r['cohen_d']:+.2f}  p={r['p_value']:.4f}{sig}")

    print("\n=== ROUND-2 SAMPLE SIZE DESIGN (assumed SD=0.025) ===")
    for k, v in sample_design.items():
        print(f"  {k:14} d={v['cohen_d_assumed_sd_0.025']:.2f}  "
              f"n@80%={v['n_for_80%_power']}  n@90%={v['n_for_90%_power']}  "
              f"n@95%={v['n_for_95%_power']}")


if __name__ == "__main__":
    main()
