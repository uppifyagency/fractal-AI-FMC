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

## Stato attuale del progetto (26 aprile 2026)

✅ Ambiente installato (`craftax 1.5.0`, `jax 0.10.0`, Python 3.11.7)
✅ FMC port a JAX (`scripts/fmc_craftax.py`, ~150 righe attive)
✅ Random baseline (`scripts/random_baseline.py`)
✅ Multi-seed sweep con calcolo Crafter score (`scripts/sweep_seeds.py`)
✅ Primi numeri verificati su Craftax-Classic (vedi `results/`)

⏳ Prossimi step (vedi roadmap)

## Risultati preliminari verificati

**Setup**: Craftax-Classic-Symbolic-v1, FMC vanilla con N=32 walker, M=12 tick, α=β=1.0, max_steps=500, 5 seed (42-46), zero training.

| Metodo | Crafter score | Mean ach | Mean step | Note |
|---|---|---|---|---|
| Random baseline (5 seed) | ~1.6%* | 2.8 / 22 | 86 | * stima da Hafner 2021 |
| FMC vanilla N=32 M=12 | 5.42% | 5.4 / 22 | 136 | Run 001 |
| **FMC vanilla N=64 M=20** ✓ | **6.87%** | **6.4 / 22** | **176** | Run 002 — **best ad oggi** |
| Rainbow (Hafner 2021) | 4.3% | — | — | da paper originale |
| PPO (Hafner 2021) | 4.6% | — | — | da paper originale |
| DreamerV2 | 10.0% | — | — | da paper Hafner 2023 |
| DreamerV3 | 14.5% | — | — | SOTA stable model-based |
| Curious Replay | 19.4% | — | — | SOTA classic table |
| EMERALD (Jul 2025) | 58.1% | — | — | SOTA attuale, > human |
| Human expert | 50.5% | — | — | Hafner 2021 |

**FMC vanilla supera Rainbow e PPO con 0 training step.** Best config N=64, M=20 (sweep in [`docs/run_002_sweep_NM_distance.md`](docs/run_002_sweep_NM_distance.md)). Resta sotto DreamerV3 di ~7.6 punti — gap colmabile via reward intrinseca + Fractal Memory.

### 12 achievement uniche unlocked (config N=64, M=20)

`collect_drink, collect_sapling, collect_stone, collect_wood, eat_cow, make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone, place_table, wake_up`

Mai visti (10 di 22): `collect_coal, collect_diamond, collect_iron, defeat_skeleton, defeat_zombie, eat_plant, make_iron_pickaxe, make_iron_sword, make_stone_pickaxe, make_stone_sword`

Pattern: il planner trova le achievement entro un raggio τ = 20 tick × 1 azione, ma fallisce su catene profonde (collect_iron richiede make_stone_pickaxe richiede make_wood_pickaxe richiede pickup sticks…). La barriera è quella delle pickaxe stone+iron.

## Roadmap

### Fase 0 — Validazione baseline ✓ (completata oggi)
Verificare che FMC vanilla giri end-to-end e produca risultati ≥ random.

### Fase 1 — Tuning iperparametri (1-2 settimane)
- Sweep N ∈ {32, 64, 128, 256}, M ∈ {12, 20, 40}
- Sweep α/β ∈ {(0,1), (0.5,1), (0.5,1.5), (1,1), (1,2), (2,1)}
- Misurare scaling Crafter score vs sample-budget
- Output atteso: una configurazione ottima e una curva di scaling

### Fase 2 — Reward intrinseca per chain di crafting (2-3 settimane)
- Aggiungere reward intrinseca per "stare vicino" a obiettivi mancanti (e.g., near-tree, near-stone)
- Inspirazione: Curiosity-driven exploration (Pathak 2017) + Sergio §6.3 Common Sense Assisted Control
- Output: superare 10 ach mean (territorio DreamerV2)

### Fase 3 — Fractal Memory (3-4 settimane)
- Implementare Slide doc 2020: memoria persistente di trajectory chains
- Wigner-weighted sampling delle past trajectories vincenti
- Output: superare DreamerV3 (~14.5%) sulla Crafter score

### Fase 4 — Submission al leaderboard (1 settimana)
- Multi-seed (≥10) per CI95 affidabili
- PR su [MichaelTMatthews/Craftax](https://github.com/MichaelTMatthews/Craftax) con codice riproducibile
- Workshop paper draft (RL Open Worlds, Generalization in RL)

### Fase 5 (opzionale) — Craftax full (mesi)
- Affrontare il setup 1B con i 65 achievement gerarchici
- Entrare nella tabella Craftax-1B/1M ufficiale

## File del progetto

```
work/05_craftax/
├── README.md                              ← questo file
├── scripts/
│   ├── smoke_test.py                      ← installazione check
│   ├── random_baseline.py                 ← floor (1.6% Crafter)
│   ├── fmc_craftax.py                     ← FMC v1: distance L2 obs 1345-D
│   ├── fmc_craftax_v2.py                  ← FMC v2: distance L2 state 18-D
│   ├── sweep_seeds.py                     ← multi-seed + Crafter score (single config)
│   ├── sweep_NM.py                        ← multi-config (N, M) sweep
│   └── compare_distance.py                ← v1 vs v2 distance ablation
├── results/
│   ├── sweep_n32_m12_a1_b1.json           ← Run 001 baseline
│   ├── sweep_NM_a1_b1.json                ← Run 002 sweep (5 configs)
│   ├── compare_distance_n64_m20.json      ← Run 002 distance ablation
│   ├── sweep_NM_progress.log              ← log live dello sweep
│   └── random_baseline_seeds_*.txt
└── docs/
    ├── run_001_first_baseline.md          ← primo report (FMC funziona, 5.42%)
    └── run_002_sweep_NM_distance.md       ← sweep + distance (best 6.87%)
```

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
