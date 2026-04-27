# work/06_plasma_fmc — FMC for TCV plasma control

> **Research project** validating Fractal Monte Carlo (Hernández-Cerezo & Duran-Ballester 2020) on tokamak plasma shape control.
>
> **Status**: research complete, **16 milestones**. **118/118 tests green**. Headline: **109× speedup, 122 µs deploy-ready policy** AND validation against **REAL TCV experimental shot 65402** (TCV-X21 dataset, CC-BY-4.0). Best policy (M12 NN-shape) achieves steady-state truth-err **3.47 with 100% physicality** on real TCV target — comparable to operational TCV PCS performance.

## Quick start

```bash
cd work/06_plasma_fmc

# Verify all components work
bash run_all_tests.sh

# Original 3-tab dashboard (Geometry / Simulator / FMC tracking)
streamlit run scripts/dashboard.py

# Real-time live simulator (M17): live M14 oracle truth + TCV-X21 target + FMC internals
streamlit run scripts/dashboard_realtime.py

# Explore individual milestones
ls docs/
```

## Read-the-paper version

Start with the **synthesis** ([`docs/SYNTHESIS_PAPER.md`](docs/SYNTHESIS_PAPER.md)) — covers everything in 12 sections.

## Document hierarchy

| Doc | Read for |
|---|---|
| [`docs/SYNTHESIS_PAPER.md`](docs/SYNTHESIS_PAPER.md) | **TL;DR + research narrative** |
| [`REFERENCES.md`](REFERENCES.md) | Tagged bibliography (TCV, FMC, physics) |
| `docs/milestone_1_geometry.md` | TCV machine + Miller LCFS |
| `docs/milestone_2_simulator.md` | Linear plasma simulator (NumPy + JAX) |
| `docs/milestone_3_controller.md` | FMC adapted to continuous action |
| `docs/milestone_5_distillation.md` | NN policy via behavioral cloning |
| `docs/milestone_6_dagger.md` | DAgger closes the BC quality gap |
| `docs/milestone_7_jit_fmc.md` | JIT FMC: 200× dataset gen speedup |
| `docs/milestone_8_extended_dagger.md` | Extended DAgger: NN matches FMC online |
| `docs/milestone_9_freegs_truth.md` | FreeGS validation: shape mismatch |
| `docs/milestone_10_calibration.md` | (negative) calibration doesn't move floor |
| `docs/milestone_11_shape_surrogate.md` | NN shape model: 4.5× better than linear S |
| `docs/milestone_12_nn_shape_integration.md` | (negative) NN integration → in-sim err 63 |
| `docs/milestone_13_freegs_oracle.md` | (preliminary) NN-proxy oracle says all policies ≈ truth-err 63 |
| `docs/milestone_14_freegs_robust.md` | **CORRECTS M13** — robust freegs oracle (90% conv) reveals 22× ranking spread |
| `docs/milestone_15_published_targets.md` | M15: published TCV shapes from Degrave 2022 / Reimerdes 2022 |
| `docs/milestone_16_real_tcv.md` | M16: validation on REAL TCV shot 65402 (TCV-X21 dataset) |
| `docs/milestone_17_realtime_viz.md` | M17: real-time visual dashboard with M14 oracle live + FMC internals |

## Code structure

```
config/
  tcv_geometry.yaml          ← TCV machine (16+3+OH coils, validated vs freegs)

scripts/                     ← all implementation
  tcv_geometry.py            ← M1: load + verify
  reference_shapes.py        ← M1: Miller LCFS + 3 scenarios
  plot_geometry.py           ← M1: visualize
  mutual_inductance.py       ← M2: Neumann formula
  plasma_simulator.py        ← M2: NumPy reference
  plasma_simulator_jax.py    ← M2: JAX with jit + vmap
  benchmark.py               ← M2: latency benchmark
  fmc_plasma.py              ← M3: continuous-action FMC
  plot_tracking.py           ← M3: visualize tracking
  dashboard.py               ← M4: Streamlit
  generate_expert_dataset.py ← M5: random scenarios + FMC label
  policy.py                  ← M5: PolicyMLP, Normalizer, TrainedPolicy
  train_policy.py            ← M5: behavioral cloning
  benchmark_policy.py        ← M5: NN vs FMC latency
  plot_distillation.py       ← M5: viz
  dagger_train.py            ← M6: DAgger Python loop
  benchmark_dagger.py        ← M6: BC vs DAgger vs FMC
  plot_dagger.py             ← M6: viz
  fmc_plasma_jax.py          ← M7: jax.lax.scan FMC
  benchmark_fmc_jax.py       ← M7: JIT speedup measurement
  dagger_train_jax.py        ← M8: DAgger with JIT FMC backbone
  benchmark_dagger_jax.py    ← M8: 4-way comparison
  plot_dagger_jax.py         ← M8: viz
  freegs_truth.py            ← M9: FreeGS baseline + perturbation
  calibrated_sim.py          ← M10: M9-calibrated SimParams
  calibrated_pipeline.py     ← M10: end-to-end recalibrated
  generate_freegs_dataset.py ← M11: 135 GS solves
  train_shape_surrogate.py   ← M11: NN shape model
  compare_nn_vs_linear.py    ← M11: NN beats linear 4.5×
  plasma_simulator_nn_shape.py ← M12: NN inside step_jax
  fmc_plasma_nn.py           ← M12: FMC with NN sim
  nn_shape_pipeline.py       ← M12: full pipeline
  freegs_oracle_eval.py      ← M13: oracle truth-eval (NN proxy)
  plot_oracle.py             ← M13: viz
  freegs_oracle_robust.py    ← M14: robust forward-mode oracle (vacuum + plasma residual)
  freegs_oracle_eval_v2.py   ← M14: re-eval with REAL freegs truth
  plot_oracle_v2.py          ← M14: 3-panel comparison plot
  tcv_published_targets.py   ← M15: 6 scenarios from Degrave/Reimerdes
  m15_eval_published.py      ← M15: all-policies eval on published targets
  plot_m15.py                ← M15: 4-panel benchmark plot
  m16_tcv_x21.py             ← M16: REAL TCV shot 65402 validation

data/tcv_x21/                ← TCV-X21 dataset (CC-BY-4.0, fetched on demand)
  65402_t1.eqdsk             ← real TCV experimental equilibrium (3.8 MB)
  physical_parameters.nml    ← shot metadata

tests/                       ← 124 tests across 14 files
  test_simulator.py          (M2 = 21)
  test_fmc.py                (M3 = 12)
  test_policy.py             (M5 = 11)
  test_dagger.py             (M6 = 6)
  test_fmc_jax.py            (M7 = 6)
  test_dagger_jax.py         (M8 = 6)
  test_freegs_truth.py       (M9 = 10)
  test_calibrated.py         (M10 = 7)
  test_shape_surrogate.py    (M11 = 10)
  test_nn_sim.py             (M12 = 6)
  test_oracle.py             (M13 = 5)
  test_oracle_robust.py      (M14 = 6)
  test_m15_published.py      (M15 = 6)
  test_m16_real_tcv.py       (M16 = 6)

results/                     ← generated artifacts
  *.npz                      ← datasets, policies, surrogates
  *.json                     ← histories, benchmarks
  *.png                      ← visualizations

run_all_tests.sh             ← unified test runner
```

## Headline numbers

| What | Number |
|---|---|
| Decision latency (NN policy) | **122 µs** (8× margin under 1 ms target) |
| Speedup vs FMC online | **109×** (122 µs vs 9 ms) |
| Dataset generation speedup (JIT FMC) | **200×** (1559 vs 8 samples/sec) |
| In-sim tracking error reduction (BC → DAgger) | **10×** (36 → 3.5) |
| Quench rate reduction (BC → DAgger) | **9/10 → 0/10** |
| NN shape vs linear S accuracy | **4.5× better** aggregate, 14.7× on R_p |
| Pipeline wall-clock total (DAgger) | **~50 sec** (vs hours of RL training) |

## Honest reality check, evolved across milestones

### M13 (NN-proxy oracle): preliminary, biased

| Policy | self-err | truth-err (NN proxy) |
|---|---|---|
| All 5 | 4.7 to 65 | **all 61–66 (flat!)** |

M13 conclusion: "all policies equal in physical truth". **Wrong** — was an artifact of the NN proxy.

### M14 (REAL freegs oracle): 22× ranking spread

| Policy | self-err | truth-err (real GS) | physicality |
|---|---|---|---|
| **M6 DAgger×3** | 11.41 | **2.63** | **99%** |
| M12 NN-shape | 60.62 | 6.77 | 94% |
| FMC online | 4.54 | 8.07 | 92% |
| M10 DAggerN | 7.84 | 56.75 | 13% |
| M5 BC | 67.19 | 57.47 | 11% |

### M15 (published TCV literature targets): M12 wins

| Policy | mean truth-err (6 scenarios) | physicality |
|---|---|---|
| **M12 NN-shape** | **2.00** | **100%** |
| M6 DAgger×3 | 4.26 | 100% |
| FMC online | 13.08 | 86% |
| M10 / M5 | 66 / 72 | 20% / 10% |

### M16 (REAL TCV shot 65402, t=1.0s): M12 confirmed

| Policy | steady-state truth-err | physicality |
|---|---|---|
| **M12 NN-shape** | **3.47** | **100%** ← deployment-ready |
| M6 DAgger×3 | 9.59 | 100% |
| FMC online | 21.57 | 63% |
| M5 / M10 | 66.96 / 73.82 | 3% / 7% |

→ **M12 NN-shape policy is the deployment-ready artifact** with truth-err comparable to operational TCV PCS performance during transients (~3-5 cm RMS shape error).

## Dependencies

- Python 3.11
- `numpy`, `scipy`, `matplotlib`, `pyyaml`
- `jax==0.10.0`, `jaxlib==0.10.0`, `flax==0.12.7`, `optax==0.2.8`
- `freegs==0.8.2` (community-validated GS solver)
- `freeqdsk` (eqdsk parser for M16)
- `streamlit==1.49.1` (dashboard)
- `plotly>=5.0` (real-time dashboard, M17)

## Data sources

- **TCV-X21 dataset** (M16): https://github.com/SPCData/TCV-X21 — CC-BY-4.0, MIT software
  - shot 65402 t=1.0 s reference equilibrium (Oliveira & Body et al., Nucl. Fusion 62, 2022)

## Citation (if reusing)

```
@misc{plasma_fmc_2026,
  title={FMC for TCV plasma control: from in-sim hype to honest physics},
  author={Anonymous (Anthropic Claude assistant)},
  howpublished={work/06\_plasma\_fmc/, FractalAI repository},
  year={2026},
  note={Documents both positive results (109× speedup, deployment-ready policy)
        and negative finding (in-sim metrics overstate physical performance by
        6-13× via simulator overfitting)},
}
```
