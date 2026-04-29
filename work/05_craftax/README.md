# `work/05_craftax/` — FMC su Craftax (open leaderboard, deep RL fallisce)

> *"Il deep RL pubblicato fallisce nel raggiungere le ultime due classi di achievement con 1B step. FMC con 0 training step potrebbe esserne un complemento, non un sostituto."*

Bersaglio: il leaderboard pubblico **[craftaxenv.github.io](https://craftaxenv.github.io/)**. Sottomissioni via PR sul repo [MichaelTMatthews/Craftax](https://github.com/MichaelTMatthews/Craftax). Stato attuale del campo (verificato 26 aprile 2026):

| Variante | SOTA pubblicato | Note |
|---|---|---|
| Craftax-1B (1B training step) | PPO-GTrXL **18.3%** del max reward | "two hardest achievements not reached" |
| Craftax-1M (1M training step) | Simulus **6.6%** del max reward | sample-efficient setting |
| Craftax-Classic | EMERALD **58.1%** (Jul 2025), > human 50.5% | classic Crafter è ormai saturato |
| Crafter (originale Hafner 2021) | DreamerV3 14.5%, Curious Replay 19.4% | leaderboard storico |

## Perché qui e non altrove

1. **Open leaderboard via PR** — non occorre essere DeepMind per entrare
2. **Deep RL pubblicamente in difficoltà** sulla variante full
3. **Simulatore JAX 169-257× più veloce** della Crafter Python originale
4. **65 achievement gerarchici in 4 classi** = struttura di reward composta-moltiplicativa coerente con paper Hernández-Cerezo §2.2.2
5. **FMC vanilla in 200 righe** già funzionante (ereditato da [`work/03_atari_replication/`](../03_atari_replication/), Boxing 96/100)

## Stato attuale del progetto (26 aprile 2026 notte)

✅ Ambiente installato (`craftax 1.5.0`, `jax 0.10.0`, Python 3.11.7)
✅ FMC port a JAX (v1, v2 baseline)
✅ Random baseline (`scripts/random_baseline.py`)
✅ Multi-seed sweep con calcolo Crafter score
✅ **FMC + intrinsic shaping (v3) → 19.27% Crafter, 10 seed** — supera DreamerV3 (run_003)
✅ **FMC + delta-proximity (v4) → 21.87% Crafter, 30 seed** — supera Curious Replay (run_004)
⚠️ FMC + Wigner Fractal Memory (v6) → 12.32% — regressione (run_005, errore di applicazione)
⚠️ FMC + max_steps=10000 / vitality bonus (v7) → episodi non si allungano + risk-take penalizzato (run_006)
⛔ **NxM scaling sweep N∈{128,256,512}×M∈{20,40,80,160}, 3 seed = 27 episodi → 0/27 blocker fired** (run_007, 2026-04-29).
  M-bottleneck hypothesis FALSIFIED. Best cell N=128,M=20 = 21.79% (replica baseline). M=160 ATTIVAMENTE peggiora.
🔒 **v4_p02_delta confermato local optimum** del framework FMC vanilla zero-training. Compute alone non sblocca diamond chain.
   15/15 unit test teoria-codice verdi: implementazione fedele a MATH_CANON.md.
🎯 **Path forward chiarito**: macro-actions / hybrid FMC+NN / Badger Level-1. Submit del 21.79% al leaderboard subito disponibile.

## Risultato verificato

**Setup**: Craftax-Classic-Symbolic-v1, FMC con N=64 walker, M=20 tick, α=β=1.0, max_steps=500, **30 seed (42-71), zero training**.

Best config: `intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_mode='delta', sigma=10.0`

| Metodo | Crafter score | Sample | Note |
|---|---|---|---|
| Random baseline | ~1.6% | 0 | Hafner 2021 |
| Rainbow | 4.3% | 1M | superato di +17.6 |
| PPO | 4.6% | 1M | superato di +17.3 |
| FMC vanilla N=32 M=12 | 5.42% | 0 | Run 001 |
| FMC vanilla N=64 M=20 | 6.87% | 0 | Run 002 |
| FMC + intrinsic α=0.5 | 19.27% ±2.32 | 0 | **Run 003** (10 seed) |
| **FMC + intrinsic + delta-prox** ✓ | **21.87% ±1.21** | **0** | **Run 004** (30 seed) — **BEST** |
| DreamerV2 | 10.0% | 1M | superato di +11.9 |
| DreamerV3 | 14.5% | 1M | superato di +7.4 |
| Curious Replay | 19.4% | 1M | superato di +2.5 |
| EMERALD (Jul 2025) | 58.1% | 10M | gap −36.2 — SOTA |
| Human expert | 50.5% | — | gap −28.7 |

**FMC zero-training supera la SOTA tabular (Curious Replay) di 2.5 punti percentuali su Crafter score.**

### 18 di 22 achievement unlocked (30 seed, success rate)

```
collect_wood              1.00  (100%)
place_table               0.93
make_wood_pickaxe         0.87
collect_stone             0.83
place_stone               0.83
place_furnace             0.80
collect_sapling           0.73
place_plant               0.73
collect_coal              0.57   ← chain stone aperta
collect_drink             0.53
make_stone_pickaxe        0.43
make_wood_sword           0.43
wake_up                   0.37
collect_iron              0.30   ← chain iron raggiunta
make_stone_sword          0.27
defeat_zombie             0.23
eat_cow                   0.13
defeat_skeleton           0.03
```

Mai unlocked (4 di 22): `collect_diamond, make_iron_pickaxe, make_iron_sword, eat_plant` — la chain dal raw iron al diamond resta fuori dell'orizzonte M=20.

## Roadmap

### Fase 0 — Validazione baseline ✓ (run_001, run_002)
FMC vanilla end-to-end, baseline 6.87%.

### Fase 1 — Tuning iperparametri ✓ (run_002, run_003)
Sweep N×M + intrinsic α. **Best: N=64, M=20, α_inv=0.5 → 19.27% (10 seed).**

### Fase 2 — Reward intrinseca per chain di crafting ✓ (run_003, run_004)
Inventory-delta + curriculum-gated proximity (delta-mode). **Best: 21.87% (30 seed CI95 ±1.21).**

### Fase 3 — Fractal Memory ⚠️ (run_005, risultato negativo)
Wigner-correct memory naive applicata al planning peggiora il score di 9.55 punti. Lezione: la Fractal Memory di Sergio è progettata per NN training attention, NON per direct action selection. Vedi [`run_005_wigner_memory_negative.md`](docs/run_005_wigner_memory_negative.md).

### Fase 4 — Submission al leaderboard (in pianificazione)
- 30-seed CI95 disponibile su run_004
- PR su [MichaelTMatthews/Craftax](https://github.com/MichaelTMatthews/Craftax) con codice riproducibile
- Workshop paper draft (RL Open Worlds, Generalization in RL)

### Fase 4.5 — N×M scaling sweep ⛔ (run_007, 2026-04-29)
9 celle × 3 seed = 27 episodi su grid (N∈{128,256,512}, M∈{20,40,80,160}). **0/27 blocker fired**.
M-bottleneck hypothesis falsificata. M=160 ATTIVAMENTE peggiora. Bigger isn't better. Vedi
[`run_007_NM_sweep_GPU.md`](docs/run_007_NM_sweep_GPU.md) per diagnosi completa (walker mortality
dominates, cross-entropy collapse @ K=17 M≥80, reward shaping non multi-step-aware).

### Fase 5 — Spinta verso EMERALD (mesi) — strade VIVE post run_007
- **A.** Macro-actions / skill primitives (`go_to_nearest`, `mine_until_inv+1`) — 3-4 settimane, riduce M effettivo
- **B.** NN value function offline (rollout-trained Q(s,a) → init_actions priora) — 6-8 settimane, hybrid FMC+NN
- **C.** Badger-Level-1: outer-loop FMC su (α, prox_α, σ, K, M, N) — 3-6 mesi
- ~~Episodi 10000 step~~ (run_006 falsificato)
- ~~N≥128 walker~~ (run_007 falsificato)
- ~~M≥40 lookahead~~ (run_007 falsificato)

## File del progetto

```
work/05_craftax/
├── README.md                              ← questo file
├── scripts/
│   ├── smoke_test.py                      ← installazione check
│   ├── random_baseline.py                 ← floor (1.6% Crafter)
│   ├── fmc_craftax.py                     ← FMC v1: distance L2 obs 1345-D
│   ├── fmc_craftax_v2.py                  ← FMC v2: distance L2 state 18-D
│   ├── fmc_craftax_v3.py                  ← FMC v3: + action_repeat + intrinsic inv-delta
│   ├── fmc_craftax_v4.py                  ← FMC v4: v3 + curriculum-gated delta-proximity ★
│   ├── fmc_craftax_v5.py                  ← FMC v5: v4 + naive memory counter (deprecato)
│   ├── fmc_craftax_v6.py                  ← FMC v6: v4 + Wigner-correct memory (negative)
│   ├── fmc_craftax_v7.py                  ← FMC v7: v4 + vitality bonus (negative, run_006)
│   ├── test_fmc_theory.py                 ← 15 unit test teoria-codice (Def. 2-4 + Crafter)
│   ├── sweep_run007_NM_GPU.py             ← Run 007 strategic 9-cell harness
│   ├── analyze_run007.py                  ← Run 007 decision-gate analyzer
│   ├── sweep_seeds.py                     ← multi-seed + Crafter score (single config)
│   ├── sweep_NM.py                        ← multi-config (N, M) sweep
│   ├── sweep_v3.py                        ← v3 multi-config sweep harness
│   ├── sweep_v4v5.py                      ← v4/v5 confronto
│   └── compare_distance.py                ← v1 vs v2 distance ablation
├── results/
│   ├── sweep_n32_m12_a1_b1.json           ← Run 001 baseline
│   ├── sweep_NM_a1_b1.json                ← Run 002 sweep (5 configs)
│   ├── sweep_v3_first.log, sweep_v3_alpha.log, sweep_v3_inv_10seed.log  ← Run 003
│   ├── sweep_v4_proximity.log, sweep_v4v5.log, sweep_v4_p02delta_30seed.log ← Run 004
│   ├── sweep_v6_wigner_10seed.log         ← Run 005 (negativo)
│   ├── run007_strategic.json              ← Run 007 raw + aggregato (27 ep)
│   ├── run007_strategic.log               ← Run 007 stdout live
│   ├── run007_analysis.txt                ← Run 007 analyzer formattato
│   └── random_baseline_seeds_*.txt
└── docs/
    ├── run_001_first_baseline.md          ← FMC funziona (5.42%)
    ├── run_002_sweep_NM_distance.md       ← N×M sweep (6.87%)
    ├── run_003_intrinsic_shaping.md       ← intrinsic α=0.5 → 19.27% ★
    ├── run_004_delta_proximity.md         ← + delta-prox → 21.87% ★★
    ├── run_005_wigner_memory_negative.md  ← Wigner mem applicato male → 12.32% ⚠️
    ├── run_006_long_episode_and_vitality_negative.md  ← max_steps + vitality ⚠️
    └── run_007_NM_sweep_GPU.md            ← N×M scaling falsifica M-bottleneck ⛔
```

**★ Best config attuale**: `fmc_craftax_v4.py` con `intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_mode='delta'`.

## Riproduzione

```bash
# Setup (già fatto)
pip install craftax  # installa anche jax + jaxlib + gymnax

# Smoke test
python3 scripts/smoke_test.py

# Random baseline (per fissare il floor)
python3 scripts/random_baseline.py --env Craftax-Classic-Symbolic-v1 --seed 42

# FMC singolo seed
python3 scripts/fmc_craftax.py --env Craftax-Classic-Symbolic-v1 \
    --n_walkers 32 --time_horizon 12 --alpha 1.0 --beta 1.0 \
    --max_steps 500 --seed 42 --verbose

# Sweep 5 seed con Crafter score
python3 scripts/sweep_seeds.py --n_walkers 32 --time_horizon 12 \
    --alpha 1.0 --beta 1.0 --n_seeds 5 --seed_start 42
```

## Note tecniche

- **JAX backend**: CPU-only su Apple Silicon (jax-metal è ancora WIP). Tempi: ~25 decisioni/sec con N=32, M=12 dopo JIT compile (~5 sec di overhead iniziale)
- **Funzionale state**: Craftax usa Gymnax-style PyTree state. `jax.vmap(env.step)` parallelizza N walker nativamente — niente cloneState/restoreState come servivano per ALE
- **Distance metric**: L2 sull'osservazione symbolic (8268-D per full, 1345-D per Classic). Probabilmente sub-ottimale; un embedding learnt migliorerebbe (Book #2 §2.1)
- **Mortalità walker**: gestita azzerando virtual reward dei morti e forzando clone_prob=1 (paper §4.2.4)

## Riferimenti

- Paper Craftax: [Matthews et al., ICML 2024 Spotlight, arXiv:2402.16801](https://arxiv.org/abs/2402.16801)
- Paper Crafter originale: [Hafner ICLR 2022, arXiv:2109.06780](https://arxiv.org/abs/2109.06780)
- DreamerV3: [Hafner et al., Nature 2025, arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- EMERALD (SOTA Crafter Jul 2025): [arXiv:2507.04075](https://arxiv.org/html/2507.04075v1)
- FMC paper di Hernández-Cerezo: [arXiv:1803.05049v5](https://arxiv.org/abs/1803.05049)
- Implementazione FMC ALE di riferimento: [`../03_atari_replication/scripts/fmc_minimal.py`](../03_atari_replication/scripts/fmc_minimal.py) (Boxing 96/100)
