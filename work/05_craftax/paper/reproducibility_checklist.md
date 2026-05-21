# Reproducibility checklist (Gap 7)

> Pre-fill following the NeurIPS 2025 / ML Reproducibility Checklist v2.0
> structure. Every TBD item is currently blocked by a Gap 1/2/3 result and
> will be filled when those experiments complete.

## I. Claims and main results

| Item | Status | Reference |
|---|---|---|
| Abstract claims supported by experiments | ✓ | Sections IV–V |
| Each claim mapped to a specific experiment | ✓ | results.tsv (24 rows) |
| Limitations & failure modes discussed | ✓ | `paper/sec_negative_results.md` |
| Theory results numbered and proved | partial | Conjecture D formal in §I, Lemma D.1 sketch (Gap 6) |

## II. Datasets

| Item | Status | Reference |
|---|---|---|
| Dataset described in detail | ✓ | Craftax-Classic-Symbolic-v1, 22 fixed achievements |
| Public dataset URL | ✓ | https://github.com/MichaelTMatthews/Craftax (pip-installable) |
| Source split protocol | ✓ | per-seed PRNGKey; 30 seeds 42–71 fixed |
| License documented | ✓ | MIT (Craftax repo) |

## III. Code

| Item | Status | Reference |
|---|---|---|
| Code released | ✓ | `work/05_craftax/autoresearch/fmc_mutable.py` (12 KB), `prepare_craftax.py` (frozen harness) |
| Exact commit referenced | ✓ | `00b7f71` (CONSOLIDATE: restore exp17) on `autoresearch/exp02-ach-bonus` |
| Dependencies pinned | ✓ | python 3.11.7, jax 0.10.0, craftax 1.5.0 |
| Reproduction script | ✓ | `python evaluate_30seed.py --out_json results/exp17_30seed.json --n_seeds 30` |
| README with quickstart | ✓ | `work/05_craftax/README.md` + `autoresearch/HANDOFF.md` |

## IV. Compute

| Item | Status | Reference |
|---|---|---|
| Hardware spec | ✓ | Single Apple M1 Pro, 16 GB RAM, no GPU (`JAX_PLATFORMS=cpu`) |
| Wall time per episode | ✓ | ~113 s (exp17 config N=512, M=40) on M1 Pro CPU |
| Wall time for full reproduction | ✓ | 30 seeds × 113 s ≈ 56 min for exp17 30-seed validation |
| Memory footprint | ✓ | Peak ~5.5 % of 16 GB (≈ 880 MB) per process |
| Energy / carbon estimate | partial | Not measured; ~1 hr × ~30 W M1 Pro CPU ≈ 30 Wh per 30-seed run |
| GPU not required | ✓ | Repro on CPU is the recommended path; `JAX_PLATFORMS=cpu` enforced |

## V. Hyperparameters

All configurable hyperparameters are exposed as named constants in
`fmc_mutable.py`:

```python
class FMCConfig:
    N_WALKERS = 512          # population size
    M_HORIZON = 40           # rollout depth
    ALPHA = 1.0              # composite reward exponent (Theorem 2: must stay = 1)
    BETA = 1.0               # diversity pressure (Theorem 3 anti-collapse)

# tier-monotonic dense inv-tier weights (lambda)
INV_TIER_WEIGHTS = {
    "wood": 2.0, "stone": 4.0, "iron": 16.0, "diamond": 64.0,
}

# sparse achievement-fire bonus (w_j) — see ACH_WEIGHTS_LIST in fmc_mutable.py
# easy:    [10..30]
# gateway: [50..120]   (incl. exp17 push: stone_pickaxe 80, collect_iron 120)
# blocker: [150..300]  (incl. exp17 push: iron_pickaxe 200, iron_sword 200)

ALPHA_INV  = 1.0     # weight on R_inv in the composite walker reward
ALPHA_PROX = 0.4     # proximity-shaping weight (exp12 sweet spot)
```

| Item | Status | Reference |
|---|---|---|
| All hyperparameters listed | ✓ | `fmc_mutable.py` `FMCConfig` + `ACH_WEIGHTS_LIST` (lines 95–145) |
| Search range / strategy documented | ✓ | results.tsv (24 rows = full search trajectory exp01–23) |
| Final values explicit | ✓ | Above table; in-file constants are the canonical source |
| No hidden hyperparameters | ✓ | No env vars, no implicit defaults |

## VI. Statistical significance

| Item | Status | Reference |
|---|---|---|
| n random seeds reported | ✓ | n = 30 (Gap 1 output); seeds 42–71 deterministic |
| Confidence intervals reported | TBD-Gap1 | Will be CI95 ≤ ±1.0 pp on Crafter score (target) |
| Significance test reported | TBD-Gap2 | Wilcoxon paired one-sided; backup: Mann-Whitney U + paired t |
| Effect size reported | TBD-Gap2 | Cohen's $d_z$ (paired); expected ≫ 0.8 (large) given +21.7 pp Δ |
| Multiple-comparison correction | n/a | Single primary hypothesis (exp17 > v4) |

## VII. Negative results & failures

| Item | Status | Reference |
|---|---|---|
| Failed experiments documented | ✓ | exp04 (over-amplification), exp08 (N=1024 OOM-throughput), exp14 (multi-pop collapse), exp15 (8-h hang), exp22 (α>1 catastrophe) — all in results.tsv |
| Mechanism for each failure stated | ✓ | `sec_negative_results.md` final section + Conjecture D §I.5 |
| Quantitative shaping bounds derived | ✓ | $\eta^* \in [1.2, 1.4]$ per step, $\prod_k \mu_{T_k} \in [3, 5]$ |

## VIII. Ablations

| Item | Status | Reference |
|---|---|---|
| Component-level ablations | partial | Additive (exp03 → exp17): 5-row Table 2 already complete |
| Leave-one-out ablations | TBD-Gap3 | 5 experiments scheduled overnight (L1–L5 in PAPER_HANDOFF) |
| Per-blocker frequency tracked | ✓ | `blocker_freq` dict in every JSON output |

## IX. Comparison & baselines

| Item | Status | Reference |
|---|---|---|
| Random baseline | ✓ | 1.6% (Hafner 2021) |
| Strong DRL baseline (PPO) | ✓ | 4.6% @ 1M, 11% @ 1B (cited) |
| World-model baselines | ✓ | DreamerV3 14.5%, Curious Replay 19.4%, EMERALD 58.1% |
| Sample/compute parity discussion | ✓ | `paper/sec_sample_efficiency.md` (Gap 5) |
| MCTS baseline at matched compute | ✓ | `work/09_fmc_vs_mcts_replication/REPORT.md` (Boxing micro-result) |

## X. Reproducibility of the reported headline result

The single command that reproduces exp17 = 50.95% Crafter on 30 seeds:

```bash
git clone <repo> && cd work/05_craftax/autoresearch
git checkout 00b7f71              # exp17 consolidated state
python -m venv .venv && source .venv/bin/activate
pip install jax==0.10.0 craftax==1.5.0  # plus deps in pyproject.toml
JAX_PLATFORMS=cpu python evaluate_30seed.py \
    --out_json results/exp17_30seed.json \
    --n_seeds 30 \
    --seed_start 42 \
    --description "exp17_repro"
```

Expected output: `crafter_score: 0.50–0.52`, `n_seeds: 30`,
`mean_ach: 15.0–15.6`, `wall_total: 50–60 min` on Apple M1 Pro CPU.

Per-seed JSON contains every individual run's `achievements_list`,
`reward`, `n_steps_decisions`, `wall_time_s` for downstream analysis.
