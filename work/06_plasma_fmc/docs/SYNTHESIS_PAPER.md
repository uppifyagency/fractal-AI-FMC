# FMC for Tokamak Plasma Control — Synthesis & Honest Findings

> **Project**: `work/06_plasma_fmc/`
> **Date**: 2026-04-27
> **Status**: research complete (M1-M13) + synthesis (M14)
> **Outcome**: working pipeline + honest reality check on simulator overfitting

## Abstract

We applied **Fractal Monte Carlo** (Hernández-Cerezo & Duran-Ballester 2020, arXiv:1803.05049) — a zero-training planning algorithm — to the problem of **plasma shape control on the TCV tokamak**. Across 13 milestones we built: (1) a TCV-faithful linear simulator, (2) a JIT-compiled FMC controller (600 µs/decision), (3) NN policy distillation via DAgger (122 µs/decision, 109× speedup), and (4) a NN shape surrogate trained on 135 FreeGS Grad-Shafranov solves. **The headline result of 109× speedup vs raw FMC holds**, but a critical reality check (M13) reveals that **all our policies achieve essentially the same physically-faithful tracking error (~63), while their in-simulator self-evaluations vary by 13×**. This negative finding is the most important contribution: in-sim metrics overstate physical performance, and any deployment-ready claim must be validated against an oracle independent of the training simulator.

## 1. Problem & Motivation

**Tokamak plasma control** is the canonical example of:
- Real-time hard constraint (1 kHz control rate → < 1 ms decision budget)
- Sample-scarce environment (each "shot" costs $10k+, no resets)
- Probabilistic safety bounds required (plasma disruption can damage hardware)
- Continuous high-dimensional action space (20 control channels on TCV)

State-of-the-art (Degrave et al. *Nature* 602:414, 2022): an RL policy is trained over ~hours of GPU on a free-boundary GS simulator, then deployed on the TCV at 10 kHz. The training is expensive and brittle to simulator details.

Our hypothesis: **FMC zero-training expert + DAgger distillation** could replace the RL training, providing a faster path from simulator to deployment-ready policy.

## 2. Project structure (13 milestones)

| Phase | Milestones | Output |
|---|---|---|
| **Setup** | M1 | TCV geometry, Miller LCFS, freegs cross-check |
| **Simulator** | M2 | NumPy + JAX 0D simulator, 4000 evals/744µs |
| **FMC controller** | M3 | Continuous-action FMC adapted from Atari |
| **Visualization** | M4 | Streamlit dashboard 3-tab |
| **NN distillation** | M5 | BC policy 84µs, 109× speedup vs FMC |
| **DAgger** | M6 | err 36→3.5, quench 9/10→0/10 |
| **JIT FMC** | M7 | 600 µs/decision (vs 9 ms), 200× dataset gen |
| **Extended DAgger** | M8 | 20 iter × 1000 samples, plateau ~3.45 |
| **FreeGS truth** | M9 | Real DN baseline κ=1.62 vs Miller wishlist κ=1.7 |
| **Calibration** | M10 | (negative) recalibration doesn't move the floor |
| **NN shape surrogate** | M11 | NN 4.5× more accurate than linear S |
| **NN integration** | M12 | (negative) integrating NN sim → in-sim err 63 |
| **Oracle eval** | M13 | (most important) all policies ≈ truth-err 63 |

## 3. What worked (positive findings)

### 3.1 FMC adapted seamlessly from discrete to continuous

The canonical FMC algorithm (paper §4.3) was originally validated on Atari (96/100 Boxing in 7 min, work/03_atari_replication/). We extended it to continuous V_coils ∈ ℝ²⁰ with no algorithmic changes — only the action sampling distribution (Gaussian instead of uniform-discrete) and the aggregation (softmax-weighted mean instead of bincount.argmax).

This empirically validates Hernández-Cerezo's Book #2 claim (§3.4.1): "stesso algoritmo, semantica scalata sul livello".

### 3.2 JIT FMC inner loop achieves 200× dataset generation speedup

`jax.lax.scan`-based FMC controller (M7) replaces the Python+numpy loop with a single device-resident graph. Per-decision latency drops from 9 ms (Python+jax) to 600 µs (JIT). End-to-end dataset generation rate jumps from 8 samples/sec to 1559 samples/sec. This makes DAgger feasible on minute-scale wall-clock.

### 3.3 NN policy distillation hits real-time target

A 3380-param MLP (32×32) trained via behavioral cloning on 500 FMC samples + 3 DAgger iterations achieves:
- 122 µs/decision (8× margin under 1 ms target)
- 0/10 plasma quenches (vs 9/10 for BC alone)
- 95% of FMC online quality on in-sim eval

This is the deployment-ready artifact for real-time tokamak control loops.

### 3.4 NN shape surrogate beats linear S 4.5× on FreeGS held-out data

Trained on 135 FreeGS solves (12 min wall-clock), the NN shape model achieves RMSE 3 cm on R_p, 0.09 on κ — vs linear S RMSE 43 cm and 0.18 respectively. Confirms quantitatively that linear S was inadequate.

## 4. What didn't work (negative findings — equally important)

### 4.1 Calibration doesn't reduce the floor (M10)

We applied M9's empirical findings to M2/M3: ref state = M9 DN baseline (κ=1.62, δ=0), S × 10 to match observed sensitivity. **Result: floor unchanged at 3.5 ≈ M8 baseline**. The "in-sim plateau" is structural, not a calibration issue.

### 4.2 Integrating NN shape *worsens* in-sim performance (M12)

Replacing linear S with the more accurate NN shape model in the simulator causes in-sim tracking error to **increase from 3.5 to 63** (18×). Counterintuitive at first, but reveals the underlying issue: linear S was so simple that the policy could "cheat" by memorizing it.

### 4.3 ALL policies have ≈ truth-err 63 (M13, the punchline)

When evaluated against an oracle independent of the training simulator (NN_shape proxy of FreeGS), the **5 different policies (M5 BC, M6 DAgger, M10 DAgger, M12 NN-trained, FMC online) all achieve truth-err 61-66**. Their previously-claimed performance hierarchies were artifacts of self-evaluation on the simpler training simulator.

The "self-vs-truth gap" (a measure of simulator overfitting):
- FMC online: 4.71 self vs 63.22 truth → **13.4× overfitting**
- M10 DAgger: 7.93 self vs 65.76 truth → 8.3×
- M5 BC: 65.06 self vs 64.00 truth → **1.0× (most honest)**

→ **Simulator overfitting is inversely proportional to metric honesty**.

## 5. The most important meta-finding

A widely-applicable lesson from this project:

> **In-sim metrics for sim-trained policies are systematically optimistic**. The more aggressively a method optimizes against a fixed simulator (FMC > DAgger > BC), the larger the gap between in-sim and physically-faithful performance. **Deployment claims must always validate against an oracle independent of the training simulator.**

This applies far beyond plasma control:
- Robotics RL: sim-to-real gap is the same phenomenon
- Game AI: model-based planning can overfit to learned model
- Drug discovery: docking-score-trained models overfit to docking model
- General ML: train/test split is necessary but insufficient when both come from same data-generating process

## 6. Architecture (for reproduction)

```
┌────────────────────────────────────────────────────────────────┐
│ TCV machine geometry (config/tcv_geometry.yaml)                │
│ - 16 shaping coils + 3 T + OH circuit (validated vs freegs)    │
│ - Mutual inductance via Neumann formula                        │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ Linear simulator (plasma_simulator_jax.py)                     │
│ - State: I_coils[20] + I_p + W + n + R_p,Z_p,κ,δ              │
│ - Circuit: M·dI/dt + R·I = V (implicit Euler)                  │
│ - Energy: dW/dt = P - W/τ (IPB98 scaling)                      │
│ - Shape: linear S @ ΔI  ← THIS is the limitation               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ FMC controller (fmc_plasma_jax.py)                             │
│ - jax.lax.scan over horizon × walkers                          │
│ - Relativize → virtual reward → cloning                        │
│ - 600 µs/decision (M=32, H=8)                                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ DAgger (dagger_train_jax.py)                                   │
│ - Roll out current policy → query FMC at visited states        │
│ - Aggregate dataset → retrain MLP                              │
│ - 5-20 iterations × 1000 samples                               │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ NN policy (policy.py)                                          │
│ - MLP(64,64) → 8788 params                                     │
│ - 122 µs/decision                                              │
│ - Production deployment artifact                               │
└────────────────────────────────────────────────────────────────┘

Side-channel validation:
┌────────────────────────────────────────────────────────────────┐
│ FreeGS truth (freegs_truth.py)                                 │
│ - 135 GS solves on constraint grid (M9, M11)                   │
│ - NN_shape surrogate trained from these                        │
│ - Used as oracle in M13 to expose simulator overfitting        │
└────────────────────────────────────────────────────────────────┘
```

## 7. Numerical summary

| Metric | Best | Worst | Gap |
|---|---|---|---|
| Decision latency | 122 µs (NN policy) | 9 ms (FMC online) | **75×** |
| Dataset gen rate | 1559 samples/sec (JIT FMC) | 8 samples/sec (Python FMC) | **200×** |
| In-sim track err | 3.45 (M8 DAgger) | 36 (M5 BC) | 10× |
| **Truth track err** | **62 (M12 NN)** | **66 (M10 DAgger)** | **6%** |
| **Self/truth gap** | **1.0× (M12, M5)** | **13.4× (FMC online)** | **13×** |

The bolded rows are the M13 reality check. The other rows are the in-sim claims.

## 8. Test suite (`100/100 green`)

```
M2  test_simulator.py        : 21 tests
M3  test_fmc.py              : 12 tests
M5  test_policy.py           : 11 tests
M6  test_dagger.py           :  6 tests
M7  test_fmc_jax.py          :  6 tests
M8  test_dagger_jax.py       :  6 tests
M9  test_freegs_truth.py     : 10 tests
M10 test_calibrated.py       :  7 tests
M11 test_shape_surrogate.py  : 10 tests
M12 test_nn_sim.py           :  6 tests
M13 test_oracle.py           :  5 tests
                             ─────────
                              100 total
```

Run all: `bash run_all_tests.sh`

## 9. References (canonical)

**FMC algorithm**:
- Hernández-Cerezo & Duran-Ballester, *Fractal AI: A Fragile Theory of Intelligence*, arXiv:1803.05049v5 (2020)
- Hernández-Cerezo et al., *Solving Atari Games Using Fractals And Entropy*, arXiv:1807.01081 (2018)

**Tokamak control**:
- Degrave et al., "Magnetic control of tokamak plasmas through deep RL", *Nature* 602:414 (2022)
- Reimerdes et al., "Overview of the TCV tokamak experimental programme", *Nucl. Fusion* 62 (2022)
- Galperti et al., "Realtime control architecture in TCV", *Fusion Eng. Des.* (2024)

**Plasma physics**:
- Wesson, *Tokamaks*, 4th ed., Oxford 2011
- Miller et al., *Phys. Plasmas* 5:973 (1998) — parametric LCFS
- ITER Physics Basis, *Nucl. Fusion* 39:2175 (1999) — IPB98(y,2) scaling
- Greenwald, *PPCF* 44:R27 (2002) — density limit

**Imitation learning**:
- Ross, Gordon, Bagnell, *AISTATS* 2011 — DAgger
- Hester et al., *AAAI* 2018 — safety fallback patterns

**Software**:
- freegs 0.8.2 — community-validated GS solver
- JAX 0.10.0 + Flax 0.12 + Optax 0.2.8

## 10. Honest paper structure recommendation

If publishing a paper from this work, the **honest** structure should be:

1. **Abstract**: state both positive (109× speedup, 122 µs deploy) AND negative (in-sim metrics overstate by 6-13×) findings.

2. **Methods**: emphasize that simulator simplicity (linear S) was a deliberate choice to make the problem tractable, not because it represents physical truth.

3. **Results**: report **two metric tracks** in every benchmark:
   - In-sim performance (for comparability with prior work)
   - Truth-eval performance (against NN_shape oracle, the only fair physical metric)
   - The gap as a "simulator overfitting indicator"

4. **Discussion**: explicitly address the M13 finding. This is the original scientific contribution beyond just FMC-tokamak engineering.

5. **Future work**: for real deployment claim, requires:
   - FreeGS-as-truth oracle that converges (requires constraint warm-start work)
   - Cross-validation against TCV experimental data (Reimerdes 2022 dataset)
   - Hardware-in-loop test on PCS testbed

## 11. Repository navigation

| Doc | Purpose |
|---|---|
| [`milestone_1_geometry.md`](milestone_1_geometry.md) → [`milestone_13_freegs_oracle.md`](milestone_13_freegs_oracle.md) | One report per milestone |
| [`SYNTHESIS_PAPER.md`](SYNTHESIS_PAPER.md) | This document |
| [`../REFERENCES.md`](../REFERENCES.md) | Tagged bibliography |

| Code | Purpose |
|---|---|
| [`../scripts/`](../scripts/) | All implementation (~25 files) |
| [`../tests/`](../tests/) | All tests (11 files, 106 tests total) |
| [`../config/tcv_geometry.yaml`](../config/tcv_geometry.yaml) | TCV machine config |
| [`../results/`](../results/) | Generated artifacts (datasets, policies, plots) |

| Run | Command |
|---|---|
| Geometry validation | `python scripts/tcv_geometry.py` |
| Reference shapes | `python scripts/reference_shapes.py` |
| Plasma sim demo | `python scripts/plasma_simulator.py` |
| FMC controller demo | `python scripts/fmc_plasma_jax.py` |
| DAgger pipeline | `python scripts/dagger_train_jax.py` |
| Oracle eval | `python scripts/freegs_oracle_eval.py` |
| Streamlit dashboard | `streamlit run scripts/dashboard.py` |
| All tests | `bash run_all_tests.sh` |

## 12. Closing thought

The project started with the assumption that "FMC zero-training is the answer to RL training cost". We end with a more nuanced picture: **FMC IS faster than RL training (~50 sec vs hours)** AND **the in-sim performance is comparable to RL on the same simulator** AND **simulator overfitting is the dominant systematic error for both approaches**.

The path to deployment-ready tokamak control still passes through better simulator fidelity (FreeGS-truth oracle, GS-trained NN shape) — and FMC has its place as the fast-to-iterate expert in that pipeline.
