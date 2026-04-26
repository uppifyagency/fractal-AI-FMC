# 03 — Phase 1 della roadmap: replica controllata su 3-5 Atari

**Goal verificabile**: tabella di benchmark con 3-5 giochi Atari, ≥5 run per gioco, intervalli di confidenza al 95%, e confronto numerico col paper (Hernández-Cerezo & Duran-Ballester, 2020, §5.1).

## Selezione giochi (3-5)

Criterio di selezione: massimizzare la diversità di "personalità" del gioco mantenendo costo computazionale gestibile.

| Gioco | Categoria | Difficoltà | FMC paper score | Human Record | Razionale |
|---|---|---|---|---|---|
| **Boxing** | competitivo, win-condition | facile | 100 (cap) | 100 (cap) | smoke test rapido (cap basso) |
| **MsPacman** | esplorazione + sopravvivenza | media | 999 990 (immortalità) | 290 090 | "showcase" del paper |
| **Asteroids** | reattivo, continuous control | media | 12 575 000 | 10 004 100 | mostra che FMC supera l'HR |
| **Centipede** | rapido, spam-input | media | 1 351 000 | 1 301 709 | benchmark di sampling efficiency |
| **Montezuma Revenge** | sparse reward, exploration | molto alta | 5 600 | 1 219 200 | weak-spot di FMC, da analizzare |

**Path consigliato**: iniziare con **Boxing** (smoke test, ~5 min/run) → **MsPacman** (showcase) → **Centipede** (sampling).

## Setup sperimentale

### Parametri di base (dal paper, Tabella 5.1.3.3)

```yaml
# fragile/MsPacman.yaml
env:
  name: ALE/MsPacman-v5
  observation_mode: ram     # da paper: 61% migliore di IMG con stessi parametri

planner:
  algorithm: FMC
  n_walkers: 30             # paper: low setting per fair comparison vs SoTA
  time_horizon: 15          # in tick
  max_samples_step: 300     # cap per azione
  fixed_steps: 5            # skipframe
  balance: 1.0              # alpha = beta = 1
  
runs:
  n_runs: 5                 # per intervallo di confidenza
  seeds: [42, 137, 271, 314, 1729]
  max_episode_steps: 27000  # 30min @ 15fps
```

### Layout repository

```
03_atari_replication/
├── README.md                   ← questo file
├── configs/
│   ├── boxing.yaml
│   ├── ms_pacman.yaml
│   ├── asteroids.yaml
│   ├── centipede.yaml
│   └── montezuma_revenge.yaml
├── scripts/
│   ├── run_single.py           ← single (game, seed) → JSON result
│   ├── run_all.sh              ← orchestrazione 5 giochi × 5 seed
│   └── aggregate_results.py    ← JSONs → tabella + grafici
├── results/
│   └── (vuoto, popolato a runtime)
└── notebooks/
    └── analysis.ipynb          ← analisi finale
```

## Procedura

### Step 1 — Smoke test (Boxing)

```bash
cd work/03_atari_replication/scripts
python run_single.py --config ../configs/boxing.yaml --seed 42 --output ../results/boxing_seed42.json
```

**Criterio successo**: episodio termina in <10 min con `reward >= 99`.

### Step 2 — Run completo

```bash
bash run_all.sh
```

Tempo stimato: ~6 ore su CPU recente (8 core), ~2 ore su singola GPU (4090/A100).

### Step 3 — Aggregazione

```bash
python aggregate_results.py --input ../results --output ../results/SUMMARY.md
```

Output: tabella markdown con `mean ± 95% CI`, samples/azione medi, walltime.

## Tabella attesa di output

| Gioco | FMC (mean ± CI) | Paper score | Human Record | Δ paper | Samples/action |
|---|---|---|---|---|---|
| Boxing | 99.4 ± 1.2 | 100 | 100 | -0.6% | ~120 |
| MsPacman | 850K ± 80K | 999 990 | 290 090 | tbd | ~290 |
| ... | | | | | |

## Cosa verificare empiricamente (oltre al raw score)

1. **Sampling efficiency**: il paper dichiara ~360× meno sample di MCTS UCT. Il nostro setup deve mostrare <500 sample/action.
2. **RAM vs IMG (sezione 5.1.3.3)**: rilanciare MsPacman con `observation_mode: image` e verificare che il punteggio scenda di ~30-40%.
3. **Walker scaling**: lanciare Boxing con `n_walkers ∈ {10, 30, 100, 300}` e plotare reward vs N. Paper afferma scaling lineare.
4. **Time horizon scaling**: stesso test con `time_horizon ∈ {5, 15, 50, 150}` per Asteroids.

## Note di replicabilità

- **Determinismo**: settare `seed` su numpy, torch, gym/plangym, e l'ALE sticky-actions a 0.
- **Versioning**: pinned `gym==0.21` o `gymnasium==0.29.1` (incompatibili — scegliere uno).
- **Checksum delle ROM**: `md5sum *.bin` salvato in `results/rom_checksums.txt`.
- **Hardware**: registrare CPU + (GPU + driver CUDA) + RAM in ogni JSON di output.

## Differenze attese rispetto al paper

| Fattore | Effetto atteso |
|---|---|
| Hardware moderno (2024+) vs 2018 | walltime molto inferiore |
| ALE moderno (vs Stella diretto) | piccole discrepanze sui sticky-actions |
| Versione Python (3.11 vs 3.6) | comportamenti random diversi → riseed obbligatorio |
| `gym.AtariEnv` rimosso → `gymnasium.make("ALE/...-v5")` | rinomina ambienti |

→ aspettiamoci **±10% sui punteggi finali** rispetto al paper. Se Δ > 30%, c'è un bug.

## Limiti conosciuti

- **Montezuma's Revenge**: nemmeno il paper ha "risolto" il gioco con il setup standard; serve un time horizon molto più lungo (paper menziona 5600 punti, world record 1.2M)
- **Stocasticità intrinseca FMC**: la varianza walker-to-walker è alta. 5 seed potrebbero non bastare per CI strette
- **ROM legali**: dal 2022 le Atari ROM sono distribuibili via `autorom` per uso non-commerciale, ma controllare licenza in caso di pubblicazione
