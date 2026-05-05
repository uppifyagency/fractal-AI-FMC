# M21 — Calibrated-sim autoresearch (re-run on M9-calibrated TCV reference)

> **Status**: in progress (2026-05-05)
> **Trigger**: M20 oracle validation revealed M19 used uncalibrated sim
> (`build_jax_params`) instead of the M16-comparable calibrated sim
> (`build_calibrated_jax_params`). M21 re-runs the same axes on the
> calibrated sim to test whether exp39's tuning was sim-specific overfit.
> **European tokamak ground-truth**: TCV (Tokamak à Configuration
> Variable, EPFL Swiss Plasma Center) shot 65402 = TCV-X21 t=1.0s,
> target [R=0.889, Z=-0.056, κ=1.71, δ=0.12].

## Methodology

Karpathy autoresearch (`forrestchang/andrej-karpathy-skills` discipline):

1. **Baseline first**: exp00 = vanilla FMC (N=64, V=50, default W,
   M16-historical config) on calibrated sim. Establishes anchor.
2. **One axis change per experiment**: each exp mutates exactly one
   hyperparameter in `fmc_mutable_plasma_m21.py`. Description records
   what changed and the hypothesis.
3. **Auto-status gate**: `keep` if score > baseline + 1.0; else `discard`.
   Prevents committing noise as signal.
4. **Persistent results.tsv**: every run logged with commit hash.
5. **TCV-X21 65402** (the European real-shot benchmark) is included as
   the 7th scenario alongside 6 M15 published Degrave/Reimerdes targets.

## Key files

| Path | Role |
|---|---|
| `prepare_plasma_calibrated.py` | Eval harness (single change vs M19: `build_calibrated_jax_params()`) |
| `fmc_mutable_plasma_m21.py` | Mutable controller hyperparameters (agent edits this only) |
| `evaluate_plasma_m21.py` | Driver that imports calibrated harness, mutable file, logs to TSV |
| `results.tsv` | Append-only experiment log |
| `results/expNN.json` | Full per-seed/scenario data per experiment |

## Findings so far

### F9 — M19 exp39 does NOT transfer to calibrated sim (confirmed M20 hypothesis)

| run | N | V | W | score | mean_err | phys% | status |
|---|---|---|---|---|---|---|---|
| exp00 (calibrated baseline) | 64 | 50 | 1.0× | -2.21 | 4.03 | 18.2 | baseline |
| **exp01 (M19 exp39 config)** | **2048** | **120** | **1.6×** | **-2.46** | **6.99** | **45.2** | **discard** |

exp39 hyperparameters give *worse* score on calibrated sim despite better
physicality. The M19 in-sim optimum was sim-calibration-specific overfit,
as M20 hypothesised. Physicality up because larger walker swarm + heavier
weights does explore the action space more thoroughly, but the resulting
voltages overshoot under S × 10 sensitivity.

### F10 — V_STD optimum scales inversely with S sensitivity

| exp | V_STD | score | Δ vs baseline |
|---|---|---|---|
| exp00 | 50 | -2.21 | (baseline) |
| exp02 | 25 | -1.59 | +0.62 |
| exp03 | 12 | -1.19 | **+1.01** |
| exp04 |  8 | -1.18 | **+1.03** |
| exp05 |  5 | -1.19 | +1.02 |

Plateau at V≈[5,12]. Best operating point V=8 (mean_err 2.16). Scaling
relative to M19 (best V=120 on uncalibrated): **factor ~15× lower**, ≈ S
sensitivity factor (10×). Direct numerical confirmation of the
"perturbation amplitude must match feedback gain" intuition.

### F11 — N_WALKERS impact is sub-linear on calibrated sim (pending verify)

| exp | N_WALKERS | V | score | mean_err |
|---|---|---|---|---|
| exp04 | 64 | 8 | -1.18 | 2.155 |
| exp06 | 256 | 8 | -1.18 | 2.160 |
| exp07 | 2048 | 8 | (running) | (running) |

N=64 → N=256 produces no measurable improvement. Striking contrast with
M19 where N=64 → N=2048 was the largest single-axis gain. Hypothesis:
calibrated sim has lower decision-noise (stronger feedback), so the
"averaging" benefit of more walkers vanishes.

## Comparison with M19 (uncalibrated)

| metric | M19 baseline (uncal) | M19 best exp39 | M21 baseline (cal) | M21 best so far |
|---|---|---|---|---|
| score | -0.0723 | +1.0440 | -2.2089 | -1.18 (exp04) |
| mean_err | 1.10 | 1.65 | 4.03 | 2.155 |
| best V_STD | (50) | 120 | (50) | **8** |
| best N | (64) | 2048 | (64) | **64** (so far) |

## Path forward

Pending: SHAPE_WEIGHTS axis, HORIZON, P_AUX × 2, GAS_PUFF probes.
After hyperparameter sweep converges (~10-15 more experiments):
- Lock the M21 BEST configuration
- Run M21 freegs oracle validation (script: `m21_oracle_validation.py`)
- Compare directly with M16 historical FMC online (truth_err 21.57 on
  calibrated sim) and M12 NN-shape (truth_err 3.47).

If M21 BEST achieves truth_err < 10 on M16 real TCV target → genuine
calibrated-sim FMC tuning gain. If truth_err remains high → confirms
that distillation (M12 path) is the actual deployment lever, not FMC
hyperparameter tuning.
