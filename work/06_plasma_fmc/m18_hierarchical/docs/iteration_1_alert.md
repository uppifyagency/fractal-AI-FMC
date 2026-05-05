# M18 iteration 1 — ALERT: Conjecture D transfer failed on linear plasma sim

**Date**: 2026-05-05
**Branch**: autoresearch/exp02-ach-bonus
**Scope**: first attempt at applying Craftax exp17 tier-stack compounding (Conjecture D) to TCV plasma shape control via FMC.

## What was built

- `m18_hierarchical/scripts/fmc_plasma_hierarchical.py`
  Extends `fmc_plasma_jax.py` with achievement-fire bonus:
  6 plasma achievements (plasma_present, shape_warm, shape_near, kappa_band,
  shape_close, shape_locked), tier weights [10, 30, 60, 100, 200, 300]
  matching Craftax exp03/exp17 schedule. Latched per walker, fires once.

- `m18_hierarchical/scripts/compare_baseline_vs_hierarchical.py`
  Closed-loop tracking comparison vanilla FMC vs hierarchical FMC,
  N_walkers=64, horizon=10, n_ticks=30, n_seeds=5, dt=1ms.

- `m18_hierarchical/scripts/rescue_voltage_std_horizon.py`
  Rescue test: voltage_std 50→200V, horizon 10→20.
  Tests at increasing displacement (S2, S3, S4 with err@root 10.6, 18.1, 28.0).

## What happened

Both runs produced **null results**: hierarchical FMC matches vanilla
within seed noise across all 4 scenarios tested.

| Scenario | err@root | Δ mean_err (hier−vanilla) | Δ steady |
|---|---|---|---|
| S1 near M16 | 0.65 | 0.000 | 0.000 |
| S2 far | 10.6 | 0.000 | 0.000 |
| S3 very_far | 18.1 | +0.113 | −0.074 |
| S4 extreme | 28.0 | −0.268 | −0.382 |

Achievement firing IS happening (ach_mean diagnostic = 2.81 to 6.0
depending on scenario) but does not propagate to control improvement.

## Root cause diagnosis

**The fire_bonus is uniform across walkers**, hence `relativize` standardizes
it to zero, hence cloning argmax is unchanged from vanilla.

Why? Two structural reasons:

1. **Plasma sim M2 is quasi-deterministic linear**. With voltage_std=200V
   walkers still converge on the same gradient trajectory within 3 ticks.
   Craftax (K=17 discrete actions, non-linear chain with rare events like
   "find diamond") naturally produces walker divergence; linear plasma sim
   does not.

2. **Threshold-based achievements on continuous error don't produce
   sparsity**. When err drops below 20 (a2 fires), it does so for ALL
   walkers near-simultaneously. Uniform fire_bonus → zero contribution
   to relativize-normalized cum_reward → no effect on cloning.

This is a **failure mode predicted by Conjecture D itself**: the sweet
spot 1.2-1.4× amplification only matters if the underlying signal is
already differentiated across walkers (the "outlier in log regime"
mechanism). Without walker divergence, no outlier exists to amplify.

## Honest interpretation

**Conjecture D does NOT trivially transfer** from Craftax (discrete actions,
sparse achievement chain, non-linear environment) to plasma shape control
(continuous actions, threshold-on-continuous achievements, linearized
simulator). This is informative — it bounds the conjecture's applicability.

Either:
- the plasma achievements need re-formulation to be genuinely sparse
  (path-dependent predicates), OR
- the simulator needs more non-linearity (use M11/M12 NN-shape sim), OR
- the transfer just doesn't apply at this level of FMC adaptation, and we
  should focus on other extensions (B = disruption precursor avoidance).

## Decision point for user

Three forward options proposed in main thread message; loop stopped
pending user direction.

## What this teaches us

The negative result is itself a contribution: it shows that "tier-stack
compounding amplification" requires a specific upstream condition —
**walker-level qualitative divergence under perturbation** — that the
linear plasma sim does not provide. This refines Conjecture D from
"general law for hierarchical reward shaping" to "specific to environments
where K (action space) and sim non-linearity produce walker divergence
within horizon T".

This is useful for paper-grade framing of Cong. D in MATH_CANON.md.
