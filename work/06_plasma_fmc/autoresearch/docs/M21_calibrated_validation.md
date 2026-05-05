# M21 — Sim-truth disconnect (calibrated vs uncalibrated)

> **Status**: definitive negative finding on in-sim metric reliability
> **Date**: 2026-05-05
> **Methodological discovery**: in-sim score is NOT a reliable autoresearch
> optimization target. The +1543% gain from M19 was real (exp39 IS a
> truth-good policy) but was found accidentally — uncalibrated sim score
> happened to align with truth direction better than calibrated.

## The four-point matrix

Run vanilla baseline AND exp39 best on BOTH sim variants, eval truth via
M14 freegs oracle on M16 real TCV-X21 65402 shape:

| controller | sim | in-sim score | mean_err | phys% | truth-err M16 |
|---|---|---|---|---|---|
| vanilla | uncalibrated | -0.0723 | 1.097 | 10.3% | 60.68 |
| vanilla | calibrated | **-2.2089** | 4.03 | 18.2% | 60.68 |
| exp39 | uncalibrated | **+1.0440** | 1.65 | 26.9% | 36.67 |
| **exp39** | **calibrated** | **-2.4648** | 6.99 | 45.2% | **12.47** ⭐ |

## Critical findings

### F9 — In-sim score INVERTS ranking on calibrated sim

Uncalibrated sim ranking (by score): exp39 (+1.04) > vanilla (-0.07).
Calibrated sim ranking (by score): vanilla (-2.21) > exp39 (-2.46).

If we had run M19 autoresearch on calibrated sim with same score formula,
we would have selected vanilla — a policy that scores 5× worse on TRUTH
(60.68 vs 12.47).

### F10 — Truth ranking is sim-invariant

On freegs oracle (real Grad-Shafranov physics), exp39 wins regardless of
which sim was used during training:
- exp39 truth M16: 12.47 (calibrated training) or 36.67 (uncalibrated training)
- vanilla truth M16: 60.68 (both sims produce similar truth-err)

The TRUTH oracle is the only authoritative metric.

### F11 — Calibrated sim makes exp39's mean_err 4× higher than uncalibrated

Same exp39 policy:
- Uncalibrated sim: mean_err 1.65 (sensitivity S × 1)
- Calibrated sim: mean_err 6.99 (sensitivity S × 10)

The calibrated sim has 10× shape sensitivity to coil currents (M9
finding). exp39 was tuned with aggressive driving on the gentler
uncalibrated sim → overshoots on calibrated → higher mean_err.

But calibrated sim also has 1.7× higher physicality (more time alive)
because the more responsive sim "rewards" sustained drive better.

### F12 — exp39 succeeds by accident, NOT by metric design

The autoresearch found exp39 on uncalibrated sim where the score formula
`-mean_err + 0.1*phys` happened to roughly correlate with truth. On
calibrated sim, the same formula REWARDS WRONG policies. This is not
a robust methodology.

**Truth oracle is the only reliable autoresearch target**.

## Implications for M22

### Option α — re-run autoresearch with truth oracle scoring

Modify `prepare_plasma.evaluate()` to score using freegs oracle truth-err
per tick instead of sim mean_err. Cost: ~15s overhead per experiment
(24ms × 30 ticks × 21 episodes = 15.1s) on top of current ~30-300s.
At 50% overhead, this is feasible for tight loop.

Expected: different optimum may emerge that's truly truth-best, beyond
exp39.

### Option β — accept exp39 as truth-good policy

exp39's truth-err 12.47 already beats M16 historical FMC (21.57). It is
the verified best-of-FMC. Stop here, write paper, M12 NN-shape remains
deployment-ready (3.47 truth-err).

### Option γ — distill exp39 into NN

Even if exp39 expert achieves 12.47 truth-err, distillation into NN
(student-teacher pattern, like M6→M12) might further improve. Cost: ~1
day work + GPU helpful.

## Recommendation

**Option α**: most informative scientifically — would establish whether
truth-oracle autoresearch finds a different/better optimum than sim-score
autoresearch. Confirms whether M19's +1543% was lucky or systematic.

The cost is bounded: existing harness needs only the score formula
swap. Mini-loop (5-10 experiments) to validate the methodology, then
expand if promising.

## Files

| Path | Content |
|---|---|
| `autoresearch/scripts/m20_calibrated.py` | Oracle-eval driver (calibrated sim) |
| `autoresearch/results/m20_calibrated_validation.json` | M20 calibrated truth-eval results |
| `autoresearch/prepare_plasma_calibrated.py` | Calibrated-sim eval harness |
| `autoresearch/evaluate_plasma_calibrated.py` | Calibrated-sim driver |
| `autoresearch/results/m21_exp39_calibrated.json` | exp39 in-sim score on calibrated |
| `autoresearch/results/m21_vanilla_calibrated.json` | vanilla in-sim score on calibrated |
| `autoresearch/docs/M21_calibrated_validation.md` | This document |

## Closing

**The in-sim metric was lying** — not in the M19 autoresearch sense, but
in a methodological sense. The +1543% gain was real but accidental. A
proper autoresearch needs truth-oracle scoring or no claim of
optimization. exp39 IS a verified truth-good policy (M20 calibrated
12.47 < M16 historical 21.57). The path to a paper-grade contribution
is M22 (oracle scoring) or accepting exp39 as the milestone result.
