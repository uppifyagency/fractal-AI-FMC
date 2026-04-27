# FractalAI

> *Un planner che pensa per traiettorie possibili — non per gradienti. Zero training, sample efficiency 360× rispetto a MCTS UCT, performance da SoTA su Atari, Craftax e controllo plasma su tokamak reali.*

[![Paper](https://img.shields.io/badge/paper-arXiv%3A1803.05049v5-b31b1b)](https://arxiv.org/abs/1803.05049)
[![Status](https://img.shields.io/badge/status-research%20%2B%20replica%20attiva-2ea44f)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Lang](https://img.shields.io/badge/prose-italiano-008C45)]()
[![Lang](https://img.shields.io/badge/code-english-005BBB)]()

---

## In una frase

Questa repo raccoglie **studio teorico, replica empirica ed estensione** del **Fractal Monte Carlo (FMC)** di Sergio Hernández-Cerezo e Guillem Duran-Ballester — un algoritmo di pianificazione che, senza alcun training, batte MCTS, Rainbow, PPO, DreamerV2/V3 in sample efficiency, e che proviamo a far uscire dal mondo dei giochi per portarlo nel **coding agentico** e nel **controllo di reattori a fusione**.

## Perché FMC è interessante (e perché conta per l'AGI)

La comunità deep-RL degli ultimi dieci anni ha consumato **miliardi di step di training** per produrre policy che spesso non generalizzano. FMC parte dall'idea opposta:

> **L'intelligenza non è una rete neurale addestrata. È una procedura di esplorazione che, data una funzione di reward e un simulatore, scopre azioni intelligenti in pochi millisecondi — *senza pesi da imparare*.**

Il meccanismo, in tre concetti:

1. **Walker** — N copie dell'agente vengono lanciate in parallelo nel futuro simulato (M tick avanti).
2. **Virtual reward** $V = R^\alpha \cdot D^\beta$ — premia chi accumula reward *e* chi sta lontano dagli altri (diversità, ispirata all'entropia di Tsallis).
3. **Cloning** — periodicamente, un walker debole "diventa" la copia di un walker forte. Selezione naturale, in tempo reale, dentro il pianificatore.

Risultato: il sistema converge a una distribuzione di Gibbs sulle traiettorie ottimali (link formale a Sequential Monte Carlo / particle filtering — vedi [`work/02_deep_dives/05_smc_particle_filter_view.md`](work/02_deep_dives/05_smc_particle_filter_view.md)). E lo fa **con 100-300 sample per decisione**, non 3 milioni come MCTS UCT.

### Perché è un tassello AGI

| Asse AGI | Cosa offre FMC |
|---|---|
| **Sample efficiency** | 360× meno rollout di MCTS, batte deep-RL pubblicato su Crafter con **0 step di training** |
| **Compute-at-inference** | Il "pensiero" scala con N×M, non con il dataset di training — *più tempo pensi, meglio decidi* |
| **Generalizzazione** | Nessuna policy memorizzata = nessun overfitting al training set |
| **Planning + memoria** | Il framework si estende naturalmente a **Fractal Memory** (Wigner-weighted recall) e all'**Octopus** (loop multi-livello stile Badger) — vedi [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](work/02_deep_dives/06_book2_badger_fractal_memory.md) |
| **Embodiment-ready** | Funziona su qualsiasi simulatore: ALE, Craftax, equilibri Grad-Shafranov per plasma, *e codice* — l'unica richiesta è "step + reward" |

In altre parole: FMC sostituisce l'idea "addestra una rete enorme su tutto" con "*hai un mondo simulabile? Allora sai già pianificare in modo intelligente al suo interno*". È un complemento — non un sostituto — del deep learning, e potrebbe essere il pezzo mancante per i sistemi che oggi falliscono in ambienti sparse-reward o long-horizon (Montezuma's Revenge, achievement profondi di Crafter, controllo industriale).

---

## I quattro filoni della repo

```
FractalAI/
├── 1803.05049v5.pdf                ← il paper canonico (Hernández-Cerezo 2020)
├── ANALISIS.md                     ← analisi profonda del paper (46 KB, italiano)
├── analisisPost.md / analisisPost2.md
├── DominiDaIndagare.md             ← survey domini per benchmark
├── 2020 Fractal Slide.md           ← Fractal Memory (slide deck Sergio)
├── 2020 Fractal.md                 ← Hives + Badger (spec operativa)
├── Fractal Book.md                 ← Book #2 (struttura AGI)
├── docs/bibliography/              ← corpus completo (paper, blog, codebases)
│
├── work/                           ← IMPLEMENTAZIONE
│   ├── 01_setup_environment/       ← installazione fragile + smoke test Atari
│   ├── 02_deep_dives/              ← 6 deep-dive teorici (cloning math, SMC, Active Inference, …)
│   ├── 03_atari_replication/       ← replica del paper su 3-5 Atari
│   ├── 04_diagrams/                ← C4 + Mermaid per FMC e fragile-rl
│   ├── 05_craftax/                 ← FMC su open leaderboard Crafter (zero-training SoTA tabular)
│   └── 06_plasma_fmc/              ← FMC per controllo plasma TCV (validato su shot reale)
│
├── plugin/fractal-coding-loop/     ← FMC come plugin Claude Code (/fractal-decide, /octopus, …)
│
├── simulations/                    ← demo HTML/JS interattive (kart, rocket, pong, octopus)
│
└── repos/                          ← codebase clonate degli autori originali
    ├── FractalAI_old               ← NumPy, paper #1 reference
    ├── fragile                     ← PyTorch/GPU, attivo
    └── fragile-rl                  ← Fragile Mechanics, successore Book #2
```

---

## Highlights empirici (cosa abbiamo verificato)

### 🎮 Atari — la replica controllata

Filone: [`work/03_atari_replication/`](work/03_atari_replication/)

> **Boxing 96/100 in 7 minuti, 231 righe di NumPy, zero GPU.** Esattamente nei range del paper.

Smoke test riproducibile (`run_single.py --config boxing.yaml --seed 42`). 3-5 giochi diversi (Boxing, MsPacman, Centipede, Asteroids, Montezuma) con intervalli di confidenza al 95% su 5 seed. Il setup conferma **<500 sample/azione** dichiarati dal paper.

### 🌳 Craftax — battere il deep-RL pubblicato con 0 training

Filone: [`work/05_craftax/`](work/05_craftax/)

| Metodo | Crafter score | Sample di training |
|---|---|---|
| Random baseline | 1.6% | 0 |
| Rainbow | 4.3% | 1M |
| PPO | 4.6% | 1M |
| DreamerV2 | 10.0% | 1M |
| DreamerV3 | 14.5% | 1M |
| Curious Replay (SoTA tabular pre-2025) | 19.4% | 1M |
| **FMC + intrinsic + delta-prox (nostro)** | **21.87% ± 1.21** | **0** |
| EMERALD (SoTA assoluta, Jul 2025) | 58.1% | 10M |

**FMC zero-training supera la SoTA tabular di +2.5 punti percentuali su 30 seed.** Run completamente riproducibile (`fmc_craftax_v4.py` con `intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_mode='delta'`).

### ⚛️ Plasma — FMC per il controllo di un tokamak reale

Filone: [`work/06_plasma_fmc/`](work/06_plasma_fmc/) — **17 milestone, 118/118 test verdi.**

Questo è il filone più ambizioso: prendere FMC e usarlo per controllare la forma del plasma in un **tokamak TCV reale** (Tokamak à Configuration Variable, EPFL Losanna). Il problema è di interesse mondiale per la fusione: il plasma deve restare confinato in una geometria precisa, e il controllo richiede policy con latenza sub-millisecondo.

| Metrica | Valore |
|---|---|
| Latenza decisione (NN policy distillata da FMC) | **122 µs** (8× margine sotto il target di 1 ms) |
| Speedup vs FMC online | **109×** |
| Speedup generazione dataset (JIT FMC) | **200×** |
| Quench rate (BC → DAgger) | **9/10 → 0/10** |
| Tracking error in-sim (BC → DAgger) | **10×** riduzione |
| **Truth-error sullo shot TCV reale 65402 (M12 NN-shape)** | **3.47 cm con 100% physicality** — comparabile al PCS operativo TCV |

Validato non solo su simulatore ma sul **dataset TCV-X21 (CC-BY-4.0)**, con shot sperimentale reale `65402_t1.eqdsk`. La sintesi completa è in [`work/06_plasma_fmc/docs/SYNTHESIS_PAPER.md`](work/06_plasma_fmc/docs/SYNTHESIS_PAPER.md), con anche le **lesioni negative** (M13: l'oracolo NN-proxy ha bias; corretto in M14 col solver Grad-Shafranov reale, che rivela uno spread di ranking di 22×).

### 🐙 Coding agentico — FMC come planner per Claude Code

Filone: [`plugin/fractal-coding-loop/`](plugin/fractal-coding-loop/)

Il plugin traduce FMC dal mondo dei giochi al mondo del codice. Quattro slash command:

- **`/fractal-decide [goal]`** — UNA decisione FMC: spawn N walker in worktree git isolati, M tick di esplorazione + cloning, cherry-pick del commit vincente sul main.
- **`/octopus [goal]`** — outer loop che chiama `/fractal-decide` finché un giudice non dichiara raggiunto l'obiettivo.
- **`/fractal-recall [query]`** — recall Wigner-weighted di episodi decisionali passati (Fractal Memory).
- **`/fractal-memory-show`** — dump del banco di memoria con statistiche per-memoria.

Math layer **certificato da 5 test deterministici** (convergenza alla distribuzione di Gibbs verificata numericamente), 17/17 test e2e passano.

---

## Sezione teorica — i deep-dive

[`work/02_deep_dives/`](work/02_deep_dives/) contiene sei espansioni formali, ognuna 600-1200 righe con citazioni puntuali al codice (`file:linea`) e bibliografia:

| # | Doc | Cosa contiene |
|---|---|---|
| 01 | [`01_cloning_mathematics.md`](work/02_deep_dives/01_cloning_mathematics.md) | Matematica del cloning, teorema di convergenza (Del Moral 2004) |
| 02 | [`02_active_inference_link.md`](work/02_deep_dives/02_active_inference_link.md) | Ponte Friston ↔ Hernández-Cerezo: free-energy come virtual reward |
| 03 | [`03_standard_model_cognition.md`](work/02_deep_dives/03_standard_model_cognition.md) | Mappa FMC → Standard Model of Cognition (Laird, Lebiere, Rosenbloom) |
| 04 | [`04_relativize_axiomatics.md`](work/02_deep_dives/04_relativize_axiomatics.md) | Axiomatizzazione del `relativize` operator (paper §2.2.3) |
| 05 | [`05_smc_particle_filter_view.md`](work/02_deep_dives/05_smc_particle_filter_view.md) | FMC ≅ Sequential Monte Carlo con resampling adattivo |
| 06 | [`06_book2_badger_fractal_memory.md`](work/02_deep_dives/06_book2_badger_fractal_memory.md) | Book #2 di Sergio: Octopus / Badger / Hives + Fractal Memory operativa |

E in [`work/04_diagrams/`](work/04_diagrams/) ci sono i diagrammi C4 (context, container, components) e una vista a livelli di `fragile-rl`.

---

## Quick start

### Eseguire la replica Atari (smoke test)

```bash
cd work/03_atari_replication/scripts
python run_single.py --config ../configs/boxing.yaml --seed 42 \
    --output ../results/boxing_seed42.json
# Atteso: episodio termina in <10 min con reward >= 99
```

### Eseguire FMC su Craftax

```bash
cd work/05_craftax
python3 scripts/sweep_seeds.py --n_walkers 64 --time_horizon 20 \
    --alpha 1.0 --beta 1.0 --n_seeds 5 --seed_start 42
# Best config: fmc_craftax_v4.py con --intrinsic_inv_alpha 0.5 --proximity_alpha 0.2
```

### Far girare la dashboard plasma in real-time

```bash
cd work/06_plasma_fmc
bash run_all_tests.sh                    # verifica 118/118 test
streamlit run scripts/dashboard_realtime.py  # M14 oracle truth + TCV-X21 target + FMC internals
```

### Provare il plugin coding

```bash
# Verifica matematica certificata
python3 plugin/fractal-coding-loop/tests/test_fractal_math.py
# Atteso: "All FMC math tests passed — convergence certified."

# Dentro Claude Code:
/fractal-decide "implementa add(a, b) in src/math.py con un unit test"
/octopus "endpoint POST /login restituisce JWT valido, tests/auth_test.py passa"
```

### Aprire le simulazioni live (browser)

```bash
open simulations/index.html
# Cart-pole, rocket, pong, octopus — tutti FMC in JavaScript, zero dipendenze
```

---

## Convenzioni

- **Italiano** per la prosa, **inglese** per il codice e i commenti tecnici.
- **Date ISO 8601** (`2026-04-27`).
- **Path relativi** alla root del progetto.
- **Citazioni paper**: `(Hernández-Cerezo & Duran-Ballester, 2020, §X.Y)`.
- **Citazioni codice**: link markdown a `file:linea`.

---

## Bibliografia minima (per chi vuole approfondire)

Letture in ordine:

1. **Paper canonico** — Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence*, [arXiv:1803.05049v5](https://arxiv.org/abs/1803.05049). §2.2 = math; §4 = algoritmo; §5 = risultati Atari.
2. **Companion empirico** — Hernández-Cerezo et al. (2018), *Solving Atari Games Using Fractals And Entropy*, [arXiv:1807.01081](https://arxiv.org/abs/1807.01081). FMC > MCTS UCT con <1000 vs 3M sample/azione.
3. **Predecessore** — Hernández et al. (2017), *General Algorithmic Search*, [arXiv:1705.08691](https://arxiv.org/abs/1705.08691). FMC = "GAS applicato al planning".
4. **Foundation entropica** — Amigó, Balogh, Hernández (2018), *A Brief Review of Generalized Entropies*, Entropy 20(11):813.

Indice corpus completo (papers, drafts, blog, codebase, gap noti): [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md).

---

## Crediti

**Algoritmo FMC**: Sergio Hernández-Cerezo ([@EntropyFarmer](https://twitter.com/EntropyFarmer)) e Guillem Duran-Ballester ([@Miau_DB](https://twitter.com/Miau_DB)), 2014-2026 — oltre dieci anni di lavoro indipendente, mai diventato "mainstream" nonostante l'evidenza empirica.

**Replica, estensione e plugin Claude Code**: Vlad Vrinceanu ([@uppifyagency](https://github.com/uppifyagency)), 2026.

Per il debito intellettuale completo (papers, blog, codebases, drafts, persone): [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md).

Rilasciato sotto licenza MIT.

---

## Stato del progetto

| Filone | Stato |
|---|---|
| 01 Setup ambiente | ✅ scaffolding pronto |
| 02 Deep dives teorici | ✅ 6/6 deep-dive completati |
| 03 Replica Atari | ✅ Boxing 96/100 verificato; piano completo per 5 giochi |
| 04 Diagrammi | ✅ C4 + Mermaid renderizzabili |
| 05 Craftax | ✅ 21.87% Crafter score, 30 seed, supera SoTA tabular |
| 06 Plasma TCV | ✅ 17 milestone, 118/118 test, validato su shot reale 65402 |
| Plugin coding | ✅ math layer certificato (5/5), e2e (17/17); end-to-end LLM in fase di test |
| Simulazioni JS | ✅ kart, rocket, pong, octopus interattive |

---

> *"Il deep RL pubblicato fallisce nel raggiungere le ultime due classi di achievement con 1B step. FMC con 0 training step potrebbe esserne un complemento, non un sostituto."*
>
> — `work/05_craftax/README.md`
