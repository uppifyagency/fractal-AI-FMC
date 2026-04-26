# `work/06_fragile_replication/` — Replicazione FMC su tutti i simulatori del repo `fragile/`

> *Replicare cosa Sergio ha implementato in `fragile/`, dimostrando che lo stesso kernel FMC funziona su domain ontologicamente diversi: ottimizzazione continua + planning Atari + (futuro) Montezuma + (futuro) dm_control walker.*

## Cosa c'è qui

```
work/06_fragile_replication/
├── README.md                     ← questo file
├── scripts/
│   └── fmc_optimization.py       ← FMC NumPy per math benchmarks (~200 LOC)
├── results/
│   ├── math_benchmarks_3seeds.log ← 7 funzioni × 3 seed
│   ├── math_hard_funcs.log        ← Rastrigin/Rosenbrock hyperparam sweep
│   └── scipy_comparison.log       ← FMC vs scipy.optimize (DE/NM/COBYLA)
└── docs/
    └── run_007_multi_simulator_replication.md  ← report completo
```

Il lavoro Atari sta in [`../03_atari_replication/`](../03_atari_replication/) (preesistente). Questo modulo aggiunge il *paradigma optimization* (math) e estende l'inventario Atari.

## Risultato in una tabella

### Math benchmarks (FMC vs scipy.optimize, 5 seed, N=200, iters=1000)

| Function | dims | Known min | FMC_avg | FMC_best | scipy DE_best | NM_best |
|---|---|---|---|---|---|---|
| Sphere | 5 | 0 | 0.0001 | 0.0001 | 0 | 0 |
| Rastrigin | 5 | 0 | 0.83 | **0.013** | **0** | 46.76 |
| **EggHolder** | 2 | -959.64 | **-959.64 ★** | -959.64 | -956.92 | -566 |
| Styblinski-Tang | 5 | -195.83 | -195.83 | -195.83 | -195.83 | -153 |
| Rosenbrock | 5 | 0 | 0.87 | **0.020** | **0** | 0 |
| Easom | 2 | -1 | -0.80 | -1.00 | -1.00 | 0 |
| Holder Table | 2 | -19.21 | -19.21 | -19.21 | -19.21 | -8.10 |

**Highlight**: FMC vince su EggHolder (gradient-free, multimodale 2D estremo). Single-start methods crollano sui multimodali — atteso.

### Atari (single-seed conferma)

| Game | n_walkers | M | reward | n_steps | wall | paper target |
|---|---|---|---|---|---|---|
| **Boxing-v5** ✓ | 30 | 15 | **96/100** | 1342 | 412s | 100 |
| **MsPacman-v5** ✓ | 30 | 15 | **2050** | 647 | 150s | 29410 (RAM) |
| **Centipede-v5** (parziale) | 30 | 15 | **≥48919** | ≥2000 | 440s+ | 1351000 |

## Perché questo matters

1. **Conferma il "kernel FMC universale"** del paper Hernández §4: stesso algoritmo su ontologie diverse (continuous opt, discrete planning).
2. **Permette confronti fair vs literature standard** (scipy DE, Nelder-Mead, COBYLA) — chi voglia validare FMC ha numeri comparabili.
3. **Setup riproducibile in 200 LOC NumPy** — niente PyTorch/JAX per il math, niente plangym per l'Atari (gymnasium nativo).
4. **Estensibile**: stessa struttura di codice si applica a Montezuma (con visit-bonus) e dm_control walker (con plangym).

## Riproduzione

```bash
# Math benchmarks (singolo)
python work/06_fragile_replication/scripts/fmc_optimization.py \
    --func eggholder --dims 2 --n_walkers 200 --n_iters 1000 \
    --sigma 0.10 --sigma_decay 0.997 --balance 1.0 --seed 42 --verbose

# Atari (singolo, vedi work/03_atari_replication/)
python work/03_atari_replication/scripts/fmc_minimal.py \
    --game ALE/Boxing-v5 --n_walkers 30 --time_horizon 15 --fixed_steps 5 \
    --seed 42 --reward_limit 100 --max_steps 27000
```

## Cosa NON c'è (ancora)

- **MontezumaRevenge** — richiede visit-tracking exploration bonus (vedi `fragile/videogames.py:163`); estensione futura.
- **dm_control walker** — richiede `plangym` install; estensione futura.
- **Lennard-Jones** (math, n_atoms=10) — non testato perché 3·N dimensions diverge facilmente.
- **`fragile-rl` replication** — è un Dreamer-style framework, NON FMC. Replica richiede settimane (RSSM + manifold encoder + actor/critic).

Vedi [`docs/run_007_multi_simulator_replication.md`](docs/run_007_multi_simulator_replication.md) per i dettagli analitici.

---

*Iter 3 del /loop, 2026-04-27 mattina.*
