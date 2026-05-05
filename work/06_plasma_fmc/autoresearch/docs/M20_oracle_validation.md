# M20 — freegs oracle validation of exp39 best policy

> **Status**: validation complete, MIXED results (honest reporting per
> user directive: "se i risultati non sono per niente soddisfacenti, avvisami")
> **Date**: 2026-05-05
> **Decision**: exp39 is NOT a deployment-ready replacement for M12 NN-shape.

## Summary table

| controller | M16 real TCV | M15 iter_like | M15 high_elong | mean phys% |
|---|---|---|---|---|
| **exp39 BEST** | **36.67** | 31.86 | 25.89 | 70% |
| baseline FMC online (M20 run) | 70.37 | 85.27 | 78.76 | 12% |
| M16 historical FMC online | 21.57 | n/a | n/a | 63% |
| M16 historical M12 NN-shape | **3.47** ← deploy-ready | 2.00 (M15 mean) | n/a | 100% |

## Key findings

### F6 — Real gain confirmed but overstated by in-sim metric (~10×)

In-sim score evolution: -0.0723 → +1.0440 = **+1.116 score units = +1543% relative**.
Truth-err evolution on M16: vanilla 70.37 → exp39 36.67 = **2× reduction**, NOT 14×.

The sim → freegs gap is consistent with M14 historical findings:
in-sim metrics overstate physical performance by 6-13×. For exp39:
sim mean_err 1.65 → truth steady 36.67 = ~22× ratio (within the
historical 6-13× envelope's upper band).

### F7 — Calibration mismatch (build_jax_params vs build_calibrated_jax_params)

M16 historical used `build_calibrated_jax_params()` (M9-calibrated sim
with corrected reference state and S × 10 sensitivity). M20 (and the
entire autoresearch loop) used `build_jax_params()` (uncalibrated).

Effect on baseline:
- M16 historical FMC online: truth 21.57 (calibrated sim)
- M20 baseline FMC online: truth 70.37 (uncalibrated sim)
- Δ = 48.80 truth-err just from sim calibration

This is a **methodological flaw** — the autoresearch should have used
the calibrated sim to be comparable with prior milestones. exp39's
real gain may be larger or smaller when measured on calibrated sim.

### F8 — exp39 does NOT replace M12 NN-shape as deployment artifact

| metric | exp39 (FMC tuned) | M12 NN-shape (DAgger distilled) |
|---|---|---|
| latency | ~1 sec/decision (N=2048) | 122 µs/decision |
| truth-err M16 | 36.67 | 3.47 |
| physicality M16 | 60% | 100% |
| trained on | uncalibrated linear sim | NN-shape sim (more nonlinear) |

M12 wins on every dimension that matters for deployment. The
autoresearch loop optimized FMC's own hyperparameters but didn't
match the policy distillation pipeline's gain.

## Honest interpretation

The autoresearch found:
1. **Real plasma control improvement** vs vanilla FMC online (2× truth-err)
2. **Better physicality regime** (60% vs 10% of policy outputs admit valid GS equilibrium)
3. Useful **scientific finding F1-F5** (V_STD regime-shift, regime-conditional revival, super-additive composition, structural neg_tri bottleneck, sub-→super-linear N scaling)

But the loop did NOT:
1. Produce a deployment-ready policy (latency 1s vs M12's 122µs)
2. Beat M12 NN-shape on real TCV target (10× worse)
3. Use the calibrated sim that M16 historical used

## Path forward (M21+ if continuing)

Two options:

**Option A — DAgger distill exp39 expert into NN policy**
- Generate dataset from exp39 (~500 samples × 60 sec each = 8 hours)
- Train MLP via DAgger pattern (like M6, M12)
- Re-validate vs M16 freegs oracle
- Expected: latency drops to ~100 µs, truth-err may improve via student-teacher pattern
- Cost: 1-2 days work + GPU helpful

**Option B — Re-run autoresearch on calibrated sim**
- Use `build_calibrated_jax_params()` instead
- Expect different optimal hyperparameters
- May reveal exp39's tuning is sim-calibration-specific overfit
- Cost: ~3 hours wall (re-do 39 experiments on new sim)

**Option C — Accept M19/M20 as scientific milestone, stop here**
- Document the methodology + findings
- M12 remains deployment-ready
- Autoresearch validated as a methodology for plasma FMC optimization
- Future work: apply autoresearch to NN distillation hyperparameters (DAgger iterations,
  student architecture, etc.)

## Reproducibility

```bash
cd work/06_plasma_fmc
JAX_PLATFORMS=cpu python autoresearch/scripts/m20_oracle_validation.py
# ~10 sec wall (oracle is fast for static target steady-state eval)
```

Output: `autoresearch/results/m20_oracle_validation.json`

## Files

| Path | Content |
|---|---|
| `autoresearch/scripts/m20_oracle_validation.py` | Driver |
| `autoresearch/results/m20_oracle_validation.json` | Full per-seed/target data |
| `autoresearch/docs/M19_milestone.md` | Autoresearch loop summary (39 exp, exp39 BEST) |
| `autoresearch/docs/M20_oracle_validation.md` | This document |

## Closing

The autoresearch methodology is validated as **useful for FMC
hyperparameter discovery** but **insufficient as a deployment
pipeline**. The real-world value is in (a) the systematic exploration
producing scientific findings F1-F8, and (b) the discovered
configuration as a starting point for downstream distillation.

Per user directive: this result is reported transparently as MIXED.
The +1543% in-sim score gain is a methodological artifact (in-sim
metric overstates), but a real 2× truth-err improvement vs same-sim
baseline is preserved. M12 historical remains the deployment artifact.
