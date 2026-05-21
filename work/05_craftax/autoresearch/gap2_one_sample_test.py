"""gap2_one_sample_test.py — Gap 2 alternative: exp17 18-seed vs v4
30-seed aggregate.

The orchestrator's v4 30-seed run hit per-seed slowness (single seed
finished in 174 min, then exhausted budget). Falling back to:
  - exp17 per-seed scores: 18 seeds with raw_runs
  - v4 reference: aggregate from `run007_top_cells_30seed.json` (N=512, M=40
    cell, mean_crafter = 29.27% over 30 seeds)

Tests run:
  - One-sample Wilcoxon on exp17 per-seed scores against H0: median = 0.2927
  - One-sample t against H0: mean = 0.2927
  - Bootstrap CI95 on exp17 mean
  - Cohen's d (one-sample) vs reference

Usage:
    python gap2_one_sample_test.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

CRAFTAX_CLASSIC_ACHIEVEMENTS = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_iron", "collect_coal",
    "place_furnace", "collect_diamond", "wake_up",
]


def crafter_score_from_run(achievements_list: list) -> float:
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for a in achievements_list:
        if a in rates:
            rates[a] = 1.0
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0


def aggregate_crafter(per_seed_lists: list[list[str]]) -> float:
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    n = len(per_seed_lists)
    for ach_list in per_seed_lists:
        for a in ach_list:
            if a in rates:
                rates[a] += 1.0 / n
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0


def main():
    from scipy.stats import wilcoxon, ttest_1samp

    # Load exp17 18-seed
    with open(RESULTS / "exp17_30seed.json") as f:
        d_e = json.load(f)
    runs = d_e["raw_runs"]
    e_seeds = [r["seed"] for r in runs]
    # Each run yields one Crafter score (per-episode, geometric mean over a
    # single binary trajectory). The aggregate score uses Hafner's
    # cross-episode formula. Both are reported.
    e_per_episode = [crafter_score_from_run(r["achievements_list"])
                     for r in runs]
    # aggregate_crafter returns 0-100 already, no need to multiply
    aggregate_crafter_pct = aggregate_crafter(
        [r["achievements_list"] for r in runs])

    # crafter_score_from_run returns 0-100 scale already (Hafner formula).
    # Keep everything in % to avoid scale confusion.
    n_e = len(e_per_episode)
    mu_e_pct = sum(e_per_episode) / n_e
    var_e = sum((x - mu_e_pct) ** 2 for x in e_per_episode) / (n_e - 1)
    sd_e_pct = math.sqrt(var_e)

    # Reference: v4 baseline 30-seed Crafter % (aggregate, run_007 top cells)
    H0_AGG_PCT = 29.27
    # v4 per-episode Crafter scores not stored in run007_top_cells_30seed.json.
    # We use the aggregate as the H0 reference and one-sample test on exp17.

    # One-sample t against H0 (everything in %)
    t_stat, t_p = ttest_1samp(e_per_episode, H0_AGG_PCT,
                              alternative="greater")
    # One-sample Wilcoxon against H0 (test if median > H0)
    diffs = [x - H0_AGG_PCT for x in e_per_episode]
    w_stat, w_p = wilcoxon(diffs, alternative="greater")

    # Cohen's d (one-sample)
    cohens_d = (mu_e_pct - H0_AGG_PCT) / sd_e_pct

    # Bootstrap CI95 on per-episode mean
    import random
    rng = random.Random(0)
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        sample = [e_per_episode[rng.randint(0, n_e - 1)] for _ in range(n_e)]
        boot_means.append(sum(sample) / n_e)
    boot_means.sort()
    ci_lo = boot_means[int(0.025 * n_boot)]
    ci_hi = boot_means[int(0.975 * n_boot)]

    # Bootstrap CI95 on the aggregate Crafter score (Hafner formula).
    # This is the headline metric — resample seeds with replacement and
    # compute the aggregate on each resample.
    e_ach_lists = [r["achievements_list"] for r in runs]
    boot_aggs = []
    for _ in range(n_boot):
        sample_idxs = [rng.randint(0, n_e - 1) for _ in range(n_e)]
        sample_lists = [e_ach_lists[i] for i in sample_idxs]
        boot_aggs.append(aggregate_crafter(sample_lists))
    boot_aggs.sort()
    agg_ci_lo = boot_aggs[int(0.025 * n_boot)]
    agg_ci_hi = boot_aggs[int(0.975 * n_boot)]
    # Probability that aggregate <= v4 reference (= bootstrap one-sided p)
    n_below_v4 = sum(1 for a in boot_aggs if a <= H0_AGG_PCT)
    bootstrap_p_aggregate = n_below_v4 / n_boot

    out = {
        "test_type": "one_sample_vs_v4_aggregate_baseline",
        "exp17_n_seeds": n_e,
        "exp17_seed_bank": e_seeds,
        "exp17_per_episode_mean_pct": round(mu_e_pct, 4),
        "exp17_per_episode_std_pct": round(sd_e_pct, 4),
        "exp17_per_episode_ci95_bootstrap_lo_pct": round(ci_lo, 4),
        "exp17_per_episode_ci95_bootstrap_hi_pct": round(ci_hi, 4),
        "exp17_aggregate_crafter_pct": round(aggregate_crafter_pct, 4),
        "exp17_aggregate_ci95_lo_pct": round(agg_ci_lo, 4),
        "exp17_aggregate_ci95_hi_pct": round(agg_ci_hi, 4),
        "exp17_aggregate_bootstrap_p_le_v4": bootstrap_p_aggregate,
        "v4_reference_aggregate_pct": H0_AGG_PCT,
        "delta_aggregate_pp": round(aggregate_crafter_pct - H0_AGG_PCT, 4),
        "delta_per_episode_mean_pp": round(mu_e_pct - H0_AGG_PCT, 4),
        "wilcoxon_one_sample_W": round(float(w_stat), 4),
        "wilcoxon_one_sample_p_one_sided_greater": float(w_p),
        "ttest_one_sample_t": round(float(t_stat), 4),
        "ttest_one_sample_p_one_sided_greater": float(t_p),
        "cohens_d_one_sample": round(cohens_d, 4),
        "exp17_per_seed_scores_pct": [round(s, 4) for s in e_per_episode],
        "notes": [
            "Per-episode Crafter score is computed per individual seed using",
            "the Hafner formula on that episode's binary achievement set.",
            "The cross-episode aggregate (50.6049%) is the standard reported",
            "headline score; per-seed mean (~50.5%) and std (~16) reflect the",
            "binary geometric-mean property: episodes with 14 vs 16 achievements",
            "differ enormously in Crafter score (highly nonlinear).",
            "v4 baseline reference: 29.27% aggregate from run_007 30-seed",
            "validation (file: work/05_craftax/results/run007_top_cells_30seed.json,",
            "cell N=512 M=40). v4 per-episode scores not stored in that JSON.",
            "Without paired v4 per-episode scores, we use a one-sample test",
            "against the aggregate baseline as the reference.",
        ],
    }

    out_path = RESULTS / "statistical_validation.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("=== Gap 2 (one-sample) statistical validation ===")
    print(f"  exp17 n_seeds:                   {n_e}")
    print(f"  exp17 per-episode mean:          {mu_e_pct:.4f}%  (std={sd_e_pct:.4f})")
    print(f"  exp17 per-episode CI95 boot:     [{ci_lo:.4f}, {ci_hi:.4f}] %")
    print(f"")
    print(f"  exp17 AGGREGATE Crafter:         {aggregate_crafter_pct:.4f}%   *** headline ***")
    print(f"  exp17 AGGREGATE CI95 boot:       [{agg_ci_lo:.4f}, {agg_ci_hi:.4f}] %")
    print(f"  bootstrap P(aggregate <= 29.27): {bootstrap_p_aggregate:.6e}")
    print(f"")
    print(f"  v4 reference (30 seeds):         {H0_AGG_PCT:.4f}%")
    print(f"  delta aggregate:                 {aggregate_crafter_pct - H0_AGG_PCT:+.4f} pp")
    print(f"  delta per-episode mean:          {mu_e_pct - H0_AGG_PCT:+.4f} pp")
    print(f"")
    print(f"  Wilcoxon (1-sample, alt=greater): "
          f"W={float(w_stat):.4f}, p={float(w_p):.6e}")
    print(f"  t-test (1-sample, alt=greater):   "
          f"t={float(t_stat):.4f}, p={float(t_p):.6e}")
    print(f"  Cohen's d (1-sample):             {cohens_d:.3f}")
    print(f"")
    print(f"  -> saved {out_path}")


if __name__ == "__main__":
    main()
