# work/06_plasma_fmc — FMC for TCV plasma control

> **Research project** validating Fractal Monte Carlo (Hernández-Cerezo & Duran-Ballester 2020) on tokamak plasma shape control.
>
> **Status**: research complete, 13 milestones + synthesis. **100/100 tests green**. Honest findings include both the headline result (**109× speedup vs raw FMC, 122 µs deploy-ready policy**) and the critical reality check (**all policies have similar physically-faithful tracking error; in-sim metrics overstate physical performance by 6-13×**).

## Quick start

```bash
cd work/06_plasma_fmc

# Verify all components work
bash run_all_tests.sh

# Interactive dashboard (Geometry / Simulator / FMC tracking)
streamlit run scripts/dashboard.py

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
| `docs/milestone_13_freegs_oracle.md` | (most important) all policies ≈ truth-err 63 |

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
  freegs_oracle_eval.py      ← M13: oracle truth-eval
  plot_oracle.py             ← M13: viz

tests/                       ← 100 tests across 11 files
  test_simulator.py
  test_fmc.py
  test_policy.py
  test_dagger.py
  test_fmc_jax.py
  test_dagger_jax.py
  test_freegs_truth.py
  test_calibrated.py
  test_shape_surrogate.py
  test_nn_sim.py
  test_oracle.py

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

## Honest reality check (M13)

| Policy | In-sim self-err | Truth-err (NN_shape oracle) | Overfitting |
|---|---|---|---|
| FMC online | 4.71 | 63.22 | **13.4×** |
| M10 DAgger×N | 7.93 | 65.76 | 8.3× |
| M6 DAgger×3 | 10.54 | 64.76 | 6.1× |
| M12 NN-shape | 61.68 | 61.68 | 1.0× |
| M5 BC | 65.06 | 64.00 | 1.0× |

→ **Simulator overfitting is inversely proportional to metric honesty**. In-sim metrics for sim-trained policies are systematically optimistic. Deployment claims must validate against an oracle independent of the training simulator.

## Dependencies

- Python 3.11
- `numpy`, `scipy`, `matplotlib`, `pyyaml`
- `jax==0.10.0`, `jaxlib==0.10.0`, `flax==0.12.7`, `optax==0.2.8`
- `freegs==0.8.2` (community-validated GS solver)
- `streamlit==1.49.1` (dashboard)

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
