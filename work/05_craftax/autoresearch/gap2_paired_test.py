"""gap2_paired_test.py — Gap 2 with TRUE PAIRED Wilcoxon.

Now feasible because run007_top_cells_30seed.json contains v4 raw_runs
for seeds 42-71, matching exp17's seed bank. We pair on the 18 seeds
that exp17 completed (42-59).

Reports:
  - Paired Wilcoxon W test (signed-rank)
  - Paired t-test
  - Cohen's d_z (paired)
  - Bootstrap on aggregate Crafter score (per-seed paired resampling)

Usage:
    python gap2_paired_test.py
"""
from __future__ import annotations
import json
import math
import sys
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
    """Per-episode Crafter score in 0-100 scale (Hafner formula on
    binary achievement vector)."""
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    for a in achievements_list:
        if a in rates:
            rates[a] = 1.0
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0  # 0-100


def aggregate_crafter(ach_lists: list[list[str]]) -> float:
    """Cross-episode aggregate Crafter score in 0-100 scale."""
    rates = {a: 0.0 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS}
    n = len(ach_lists)
    if n == 0:
        return 0.0
    for ach_list in ach_lists:
        for a in ach_list:
            if a in rates:
                rates[a] += 1.0 / n
    log_terms = [math.log(1.0 + 100.0 * rates[a])
                 for a in CRAFTAX_CLASSIC_ACHIEVEMENTS]
    return math.exp(sum(log_terms) / len(log_terms)) - 1.0  # 0-100


def main():
    from scipy.stats import wilcoxon, ttest_rel, ttest_ind

    with open(RESULTS / "exp17_30seed.json") as f:
        d_e = json.load(f)
    with open(RESULTS / "v4_30seed.json") as f:
        d_v = json.load(f)

    e_runs = d_e["raw_runs"]
    v_runs = d_v["raw_runs"]

    e_by_seed = {r["seed"]: r["achievements_list"] for r in e_runs}
    v_by_seed = {r["seed"]: r["achievements_list"] for r in v_runs}

    paired_seeds = sorted(set(e_by_seed) & set(v_by_seed))
    print(f"[gap2] paired seeds: {paired_seeds}", file=sys.stderr)

    # Per-seed crafter scores (binary 0/1 ach vector → Hafner formula)
    e_pe = [crafter_score_from_run(e_by_seed[s]) for s in paired_seeds]
    v_pe = [crafter_score_from_run(v_by_seed[s]) for s in paired_seeds]

    # Aggregate crafter (full 18 paired seeds for exp17, full 30 for v4)
    agg_e_paired = aggregate_crafter([e_by_seed[s] for s in paired_seeds])
    agg_v_paired = aggregate_crafter([v_by_seed[s] for s in paired_seeds])
    agg_v_full30 = aggregate_crafter([r["achievements_list"] for r in v_runs])

    n = len(paired_seeds)

    # Paired Wilcoxon on per-episode (which is signed-rank on binary-vector
    # aggregate proxy — meaningful for direction even if low-power)
    diffs = [e - v for e, v in zip(e_pe, v_pe)]
    try:
        w_stat, w_p = wilcoxon(e_pe, v_pe, alternative="greater")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Paired t
    t_stat, t_p = ttest_rel(e_pe, v_pe, alternative="greater")

    # Cohen's d_z (paired)
    mu_d = sum(diffs) / n
    s_d = math.sqrt(sum((d - mu_d) ** 2 for d in diffs) / (n - 1))
    d_z = mu_d / s_d if s_d > 0 else float("inf")

    # Bootstrap on aggregate (resample seeds with replacement, recompute
    # paired aggregate delta on each resample)
    import random
    rng = random.Random(0)
    n_boot = 10000
    boot_deltas_paired = []
    boot_e_aggs = []
    boot_v_aggs = []
    for _ in range(n_boot):
        idxs = [rng.randint(0, n - 1) for _ in range(n)]
        sample_e = [e_by_seed[paired_seeds[i]] for i in idxs]
        sample_v = [v_by_seed[paired_seeds[i]] for i in idxs]
        ae = aggregate_crafter(sample_e)
        av = aggregate_crafter(sample_v)
        boot_e_aggs.append(ae)
        boot_v_aggs.append(av)
        boot_deltas_paired.append(ae - av)
    boot_deltas_paired.sort()
    boot_e_aggs.sort()
    boot_v_aggs.sort()
    delta_lo = boot_deltas_paired[int(0.025 * n_boot)]
    delta_hi = boot_deltas_paired[int(0.975 * n_boot)]
    e_lo = boot_e_aggs[int(0.025 * n_boot)]
    e_hi = boot_e_aggs[int(0.975 * n_boot)]
    v_lo = boot_v_aggs[int(0.025 * n_boot)]
    v_hi = boot_v_aggs[int(0.975 * n_boot)]
    n_below_zero = sum(1 for d in boot_deltas_paired if d <= 0)
    p_boot = n_below_zero / n_boot

    out = {
        "test_type": "paired_18_seed_with_run007_v4",
        "n_paired": n,
        "paired_seeds": paired_seeds,
        "exp17_aggregate_pct": round(agg_e_paired, 4),
        "exp17_aggregate_ci95_lo": round(e_lo, 4),
        "exp17_aggregate_ci95_hi": round(e_hi, 4),
        "v4_aggregate_paired_pct": round(agg_v_paired, 4),
        "v4_aggregate_paired_ci95_lo": round(v_lo, 4),
        "v4_aggregate_paired_ci95_hi": round(v_hi, 4),
        "v4_aggregate_full30_pct": round(agg_v_full30, 4),
        "delta_aggregate_paired_pp": round(agg_e_paired - agg_v_paired, 4),
        "delta_aggregate_paired_ci95_lo": round(delta_lo, 4),
        "delta_aggregate_paired_ci95_hi": round(delta_hi, 4),
        "bootstrap_p_delta_le_zero": p_boot,
        "exp17_per_episode_mean": round(sum(e_pe) / n, 4),
        "v4_per_episode_mean": round(sum(v_pe) / n, 4),
        "wilcoxon_paired_W": round(float(w_stat), 4) if not math.isnan(w_stat) else None,
        "wilcoxon_paired_p_one_sided_greater": float(w_p) if not math.isnan(w_p) else None,
        "paired_t_stat": round(float(t_stat), 4),
        "paired_t_p_one_sided_greater": float(t_p),
        "cohens_d_z_paired": round(d_z, 4),
        "exp17_per_seed_pct": [round(s, 4) for s in e_pe],
        "v4_per_seed_pct": [round(s, 4) for s in v_pe],
        "diff_per_seed": [round(d, 4) for d in diffs],
    }

    out_path = RESULTS / "statistical_validation_paired.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== Gap 2 PAIRED statistical validation ===")
    print(f"  n_paired:                   {n}")
    print(f"  exp17 aggregate:            {agg_e_paired:.4f}%   "
          f"CI95 [{e_lo:.4f}, {e_hi:.4f}]")
    print(f"  v4 aggregate (paired n=18): {agg_v_paired:.4f}%   "
          f"CI95 [{v_lo:.4f}, {v_hi:.4f}]")
    print(f"  v4 aggregate (full n=30):   {agg_v_full30:.4f}%")
    print(f"  Δ aggregate paired:         {agg_e_paired - agg_v_paired:+.4f} pp")
    print(f"  Δ aggregate CI95:           [{delta_lo:.4f}, {delta_hi:.4f}] pp")
    print(f"  bootstrap P(Δ ≤ 0):         {p_boot:.6e}")
    print(f"")
    print(f"  Wilcoxon paired (one-sided greater): "
          f"W={w_stat:.4f}, p={w_p:.6e}" if not math.isnan(w_stat) else
          f"  Wilcoxon paired: insufficient signed pairs")
    print(f"  Paired t (one-sided greater):       "
          f"t={float(t_stat):.4f}, p={float(t_p):.6e}")
    print(f"  Cohen's d_z (paired):               {d_z:.3f}")
    print(f"\n  -> saved {out_path}")


if __name__ == "__main__":
    main()
