"""gap2_statistical_test.py — Gap 2 of PAPER_HANDOFF.

Computes paired Wilcoxon, paired t-test, Mann-Whitney U, Cohen's d_z, and
bootstrap CI95 on the difference between exp17 and v4 baseline per-seed
Crafter scores.

Inputs:
    --exp17_json: per-seed JSON from evaluate_30seed.py (Gap 1 output)
    --v4_json:    per-seed JSON for v4 baseline (re-run with same seeds)
                  OR baseline_lock.json (10-seed fallback, unpaired only)

Output:
    JSON in results/statistical_validation.json with all test results.
    Markdown summary appended to paper/sec_results.md.

Usage:
    python gap2_statistical_test.py \
        --exp17_json results/exp17_30seed.json \
        --v4_json    results/v4_30seed.json \
        --out_json   results/statistical_validation.json
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path


# Re-derive Crafter score so we can recompute per-seed from raw_runs in case
# the JSON only contains aggregate fields.
CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]


def crafter_score_from_run(achievements_list: list) -> float:
    """Single-episode Crafter score (geometric mean over a single trial,
    treating each ach as binary 0/1 for that one episode).
    Per Hafner 2021, formal score is across-episode mean rates → so the
    *per-episode* version puts each rho_j in {0, 1}. The formula simplifies:
        Phi_episode = exp((K/J) * log(101)) - 1
    where K is the count of unlocked achievements. We retain the canonical
    formula in case future variants weight individual achievements.
    """
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for a in achievements_list:
        if a in rates:
            rates[a] = 1.0
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0


def per_seed_scores(json_path: Path) -> tuple[list[int], list[float], list[int]]:
    """Extract (seeds, per_seed_crafter_score, per_seed_n_ach) from a JSON
    that has raw_runs.
    """
    with open(json_path) as f:
        d = json.load(f)
    runs = d.get("raw_runs") or []
    if not runs:
        raise ValueError(f"{json_path} has no raw_runs")
    seeds = [r["seed"] for r in runs]
    scores = [crafter_score_from_run(r["achievements_list"]) for r in runs]
    n_ach = [int(r["achievements_unlocked"]) for r in runs]
    return seeds, scores, n_ach


def aggregate_crafter_score(per_seed_lists: list[list[str]]) -> float:
    """Standard Hafner 2021 formula across the seed bank."""
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    n = len(per_seed_lists)
    for ach_list in per_seed_lists:
        for a in ach_list:
            if a in rates:
                rates[a] += 1.0 / n
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return mu, math.sqrt(var)


def bootstrap_ci(xs: list[float], n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap CI on the mean."""
    import random
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        sample = [xs[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def run_tests(exp17_scores: list[float], v4_scores: list[float],
              paired: bool) -> dict:
    """Run the full battery of tests; safe to call with paired=False if only
    unpaired data is available."""
    from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind

    out = {
        "n_exp17": len(exp17_scores),
        "n_v4": len(v4_scores),
        "paired": paired,
    }

    mu_e, s_e = mean_std(exp17_scores)
    mu_v, s_v = mean_std(v4_scores)
    out["exp17_mean"] = round(mu_e, 4)
    out["exp17_std"] = round(s_e, 4)
    out["v4_mean"] = round(mu_v, 4)
    out["v4_std"] = round(s_v, 4)
    out["delta_mean_pp"] = round((mu_e - mu_v) * 100, 4)

    if paired and len(exp17_scores) == len(v4_scores):
        diffs = [a - b for a, b in zip(exp17_scores, v4_scores)]
        d_mu, d_s = mean_std(diffs)
        out["paired_diff_mean"] = round(d_mu, 4)
        out["paired_diff_std"] = round(d_s, 4)
        out["cohens_d_z"] = round(d_mu / d_s, 4) if d_s > 0 else float("inf")

        try:
            w_stat, w_p = wilcoxon(exp17_scores, v4_scores, alternative="greater")
            out["wilcoxon_W"] = round(float(w_stat), 4)
            out["wilcoxon_p_one_sided_greater"] = float(w_p)
        except ValueError as e:
            out["wilcoxon_error"] = str(e)

        t_stat, t_p = ttest_rel(exp17_scores, v4_scores, alternative="greater")
        out["paired_t"] = round(float(t_stat), 4)
        out["paired_t_p_one_sided_greater"] = float(t_p)

    # Always report unpaired as a robustness check
    u_stat, u_p = mannwhitneyu(exp17_scores, v4_scores, alternative="greater")
    out["mann_whitney_U"] = round(float(u_stat), 4)
    out["mann_whitney_p_one_sided_greater"] = float(u_p)

    t_ind, t_ind_p = ttest_ind(exp17_scores, v4_scores,
                               alternative="greater", equal_var=False)
    out["welch_t"] = round(float(t_ind), 4)
    out["welch_t_p_one_sided_greater"] = float(t_ind_p)

    # Cohen's d (unpaired) — pooled std
    pooled = math.sqrt((s_e ** 2 + s_v ** 2) / 2)
    out["cohens_d_unpaired"] = round((mu_e - mu_v) / pooled, 4) if pooled > 0 else float("inf")

    # Bootstrap CI on the delta of means
    deltas_boot = []
    import random
    rng = random.Random(0)
    n_boot = 10000
    n_e = len(exp17_scores)
    n_v = len(v4_scores)
    for _ in range(n_boot):
        sample_e = [exp17_scores[rng.randint(0, n_e - 1)] for _ in range(n_e)]
        sample_v = [v4_scores[rng.randint(0, n_v - 1)] for _ in range(n_v)]
        deltas_boot.append((sum(sample_e) / n_e) - (sum(sample_v) / n_v))
    deltas_boot.sort()
    out["delta_bootstrap_ci95_lo"] = round(deltas_boot[int(0.025 * n_boot)] * 100, 4)
    out["delta_bootstrap_ci95_hi"] = round(deltas_boot[int(0.975 * n_boot)] * 100, 4)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp17_json", required=True)
    ap.add_argument("--v4_json", required=True)
    ap.add_argument("--out_json", default="results/statistical_validation.json")
    args = ap.parse_args()

    print(f"[gap2] loading exp17 scores from {args.exp17_json}", file=sys.stderr)
    e_seeds, e_scores, e_nach = per_seed_scores(Path(args.exp17_json))
    print(f"[gap2] loading v4 scores from {args.v4_json}", file=sys.stderr)
    v_seeds, v_scores, v_nach = per_seed_scores(Path(args.v4_json))

    paired = (e_seeds == v_seeds)
    if not paired:
        print(f"[gap2] WARNING: seed banks differ (n_exp17={len(e_seeds)}, "
              f"n_v4={len(v_seeds)}); using unpaired tests", file=sys.stderr)

    results = run_tests(e_scores, v_scores, paired=paired)
    results["exp17_seeds"] = e_seeds
    results["v4_seeds"] = v_seeds
    results["exp17_per_seed_crafter"] = [round(s, 4) for s in e_scores]
    results["v4_per_seed_crafter"] = [round(s, 4) for s in v_scores]
    results["exp17_per_seed_n_ach"] = e_nach
    results["v4_per_seed_n_ach"] = v_nach

    # Aggregate Hafner score (recomputed)
    with open(args.exp17_json) as f:
        e_runs = [r["achievements_list"] for r in json.load(f)["raw_runs"]]
    with open(args.v4_json) as f:
        v_runs = [r["achievements_list"] for r in json.load(f)["raw_runs"]]
    results["exp17_aggregate_crafter_pct"] = round(
        aggregate_crafter_score(e_runs) * 100, 4)
    results["v4_aggregate_crafter_pct"] = round(
        aggregate_crafter_score(v_runs) * 100, 4)
    results["delta_aggregate_pp"] = round(
        results["exp17_aggregate_crafter_pct"] - results["v4_aggregate_crafter_pct"], 4)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print
    print("\n=== Gap 2 statistical validation ===")
    print(f"  exp17: n={results['n_exp17']}, mean={results['exp17_mean']:.4f}, "
          f"std={results['exp17_std']:.4f}")
    print(f"  v4:    n={results['n_v4']}, mean={results['v4_mean']:.4f}, "
          f"std={results['v4_std']:.4f}")
    print(f"  Δ mean (per-episode crafter): {results['delta_mean_pp']:.2f} pp")
    print(f"  Δ aggregate (Hafner formula): {results['delta_aggregate_pp']:.2f} pp")
    print(f"  bootstrap CI95 on Δ mean:    "
          f"[{results['delta_bootstrap_ci95_lo']:.2f}, "
          f"{results['delta_bootstrap_ci95_hi']:.2f}] pp")
    if paired:
        print(f"  Wilcoxon (paired, one-sided): "
              f"W={results.get('wilcoxon_W', 'NA')}, "
              f"p={results.get('wilcoxon_p_one_sided_greater', 'NA'):.6e}")
        print(f"  Cohen's d_z (paired):        {results['cohens_d_z']:.3f}")
    print(f"  Mann-Whitney U (one-sided):  "
          f"U={results['mann_whitney_U']}, "
          f"p={results['mann_whitney_p_one_sided_greater']:.6e}")
    print(f"  Welch t (one-sided):         "
          f"t={results['welch_t']}, "
          f"p={results['welch_t_p_one_sided_greater']:.6e}")
    print(f"  Cohen's d (unpaired):        {results['cohens_d_unpaired']:.3f}")
    print(f"  → saved {args.out_json}")


if __name__ == "__main__":
    main()
