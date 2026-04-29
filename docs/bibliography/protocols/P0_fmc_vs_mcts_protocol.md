# Protocollo P0 — Replica FMC vs MCTS-UCT con protocollo controllato

> **Status**: 🚨 **ESCALATED** — task compute-heavy non eseguibile in-session di un agente. Specifica completa per esecuzione umana o pipeline di lungo termine.
>
> **Origine**: [`paper_fmc_dhdna_audit.md`](../paper_fmc_dhdna_audit.md) — risoluzione discrepanza D2 in [`CLAUDE.md`](../../../CLAUDE.md).
>
> **Priority**: 🚨 **Blocker** per qualunque submission accademica del paper FMC empirico.
>
> **Stima costo**: ~2-3 settimane focused work + cluster GPU.

---

## Obiettivo

Risolvere D2: il claim *"FMC è X volte più sample-efficient di MCTS-UCT"* dove $X$ varia da 100 a 10 000 a seconda della fonte. Produrre un numero singolo, blindato, con error bars e protocollo riproducibile.

## Domande di ricerca

1. **D2-Q1**: A parità di *samples per action* (ovvero rollouts × depth), FMC e MCTS-UCT raggiungono performance comparabili o no?
2. **D2-Q2**: Qual è il *minimum samples-per-action budget* a cui ciascun algoritmo *solva* (raggiunge soglia score-target) un dato gioco?
3. **D2-Q3**: Il rapporto $B^{\mathrm{MCTS-min}} / B^{\mathrm{FMC-min}}$ è coerente cross-task o specifico al gioco?

## Protocollo sperimentale

### Stack tecnico (già disponibile nel repo)

- **Environment**: `plangym` (Atari Boxing, Q-Bert, MsPacman come baseline)
- **FMC implementation**: `fmc-core/` (NumPy reference) + `fragile/` (PyTorch, GPU per scaling)
- **MCTS-UCT implementation**: **DA AGGIUNGERE**. Candidati:
  - `mctx` (DeepMind, JAX-based) — production-grade
  - `gym_atari_mcts` repository (Python) — più semplice, baseline
  - Custom port di `MCTS UCT` da [P1-7] del paper FMC (riferimenti citati in §5.1.1)

### Hardware target

- 1 GPU mid-tier (V100 / A100 / 4090) per FMC parallelizzato
- ~20 CPU cores per MCTS rollout parallelo
- ~100 GB storage per logs e checkpoints

### Setup controllato

| Variabile | Valore | Razionale |
|---|---|---|
| Games | Atari Boxing, Q-Bert, MsPacman | 3 task con dynamics distinti dal paper §5.1 (sticky actions disabilitati per determinismo) |
| Seeds | n=10 per ogni cella | Significance test richiede $n \geq 10$ |
| Sticky actions | OFF (deterministic) | Paper FMC v5 lavora in deterministic; per fairness con MCTS UCT che assume det-env |
| Frame skip | 4 | Standard Atari benchmark |
| Episodes per cell | 30 | Variance Atari richiede n moderato |
| **Sample budgets** | $B \in \{300, 1{,}000, 3{,}000, 10{,}000, 30{,}000, 100{,}000, 300{,}000\}$ | Logaritmica, copre il range citato (300 ≈ FMC paper, 150 000 ≈ MCTS) |
| FMC parametri | Match paper §5.1.3.3: `fixed_steps=5, time_limit=15, Max_walkers=30`, scale `Max_samples` per match $B$ | Fedele al setup originale |
| MCTS parametri | UCB constant $c = \sqrt{2}$ (canonical), depth limit dinamico fino a $B$ esaurito | Configurazione standard letteratura |
| Metric | Mean episode score $\pm$ std, success rate (game-specific threshold) | Replica metrica paper |

### Protocollo run-by-run

```
for game in [Boxing, QBert, MsPacman]:
  for B in budgets:
    for seed in 1..10:
      result_fmc[game,B,seed] = run_fmc(game, B, seed)
      result_mcts[game,B,seed] = run_mcts_uct(game, B, seed)
    mean, std = aggregate(seed)
    log(game, B, "FMC", mean, std)
    log(game, B, "MCTS", mean, std)
```

### Output atteso

Per ogni game, una **curva di performance vs budget**:

```
score
 │              ┌─── FMC plateau
 │      ╱──────┘
 │    ╱
 │   ╱   ┌── MCTS plateau
 │  ╱   ╱
 │ ╱   ╱
 │ │  ╱
 └─┴─╱──────────────── budget B (log scale)
```

I numeri di interesse:

- **$B^{\mathrm{FMC-min}}$**: budget minimo per FMC per raggiungere soglia score-target
- **$B^{\mathrm{MCTS-min}}$**: idem per MCTS-UCT
- **Ratio $r = B^{\mathrm{MCTS-min}} / B^{\mathrm{FMC-min}}$** ± CI95 via bootstrap

### Decisione

Confrontare $r$ con i claim del paper:

| Range $r$ misurato | Verdetto | Azione |
|---|---|---|
| $r > 100$ | Claim conservativo "$2$-$3$ OoM" del v5 §7 confermato | Scrivere paper v6 con questo numero, citando questo protocollo |
| $10 < r \leq 100$ | Claim "$359\times$" del v5 §5.1.2 era ottimistico ma directionally giusto | Riformulare claim con il numero corretto |
| $r \leq 10$ | Vantaggio FMC esiste ma è modesto | **Riposizionare il paper**: FMC come *competitive alternative* a MCTS, non *replacement* |
| $r \approx 1$ o $r < 1$ | Claim sample-efficiency falsificato | Crisi del program — riposizionare su altri vantaggi (parallelismo, continuous actions, etc.) |

### Caveat metodologici da documentare

- **Simulator-perfect access**: entrambi gli algoritmi assumono `env.set_state()` (via plangym). Questo è advantage rispetto a model-free RL ma equo nel confronto FMC vs MCTS.
- **Branching factor**: per Atari $K=18$ (azione joystick × button) ma effective $K_{\mathrm{useful}} \leq 6$ in pratica. MCTS-UCT esplora azioni una alla volta, FMC tutte simultaneamente — questo è un'asimmetria reale, non un protocollo bug.
- **Wall-clock vs sample-count**: riportare entrambi separatamente. FMC è embarrassingly parallel su GPU, MCTS richiede tree-shared-state.

## Deliverable atteso

1. `work/09_fmc_vs_mcts_replication/`:
   - `REPORT.md` con plot, tabelle, conclusioni
   - `runs/{game}_{algo}_{B}_{seed}.jsonl` raw logs
   - `notebooks/analysis.ipynb` per regenerate plots
   - `figures/` PNG/PDF publication-ready
2. Update di [`paper_fmc_dhdna_audit.md`](../paper_fmc_dhdna_audit.md) §"il claim headline" con risultato D2 risolto
3. Update di [`CLAUDE.md`](../../../CLAUDE.md) D2 a stato CLOSED
4. Section §5 del paper v6 con protocollo full-spec

## Trigger di esecuzione

Questo protocollo va eseguito quando:
- Il team ha 2-3 settimane focused di un developer (Vlad o Guillem)
- Cluster GPU disponibile per ~50-100 GPU-hours
- Decisione di iniziare il paper v6 / paper accademico empirico è presa

## Status corrente (2026-04-28)

- ✅ **MCTS-UCT baseline scritto** ([`work/09_fmc_vs_mcts_replication/scripts/mcts_uct.py`](../../../work/09_fmc_vs_mcts_replication/scripts/mcts_uct.py)) — implementa Kocsis & Szepesvári (2006) contro lo stesso protocollo `fmc.envs.base.Environment` di FMC.
- ✅ **plangym → fmc adapter** ([`fmc-core/src/fmc/envs/atari.py`](../../../fmc-core/src/fmc/envs/atari.py)) — bridge funzionante per RAM e RGB obs.
- ✅ **Atari runner end-to-end** ([`work/09_fmc_vs_mcts_replication/scripts/atari_episode.py`](../../../work/09_fmc_vs_mcts_replication/scripts/atari_episode.py)) — un comando, una partita.
- ✅ **Boxing micro-sweep, n=3 seed × 2 budget × 2 algo, in-session, CPU-only**: FMC mean +91 (B=80) e +100 (B=240) vs MCTS −5 entrambi i budget. Δ ~96-105 raw points. Vedi [`work/09_fmc_vs_mcts_replication/REPORT.md`](../../../work/09_fmc_vs_mcts_replication/REPORT.md).
- 🟡 **Caveat**: n=3 è sotto-campionato per pubblicazione; MCTS hyperparams non tunati; un solo gioco.
- 🟢 **Hardware revision**: il protocollo originale stimava ~50-100 GPU-h. **Misurazione effettiva: ~50 s / episodio FMC su CPU singola → ~7 ore single-CPU per il protocollo full.** Cluster GPU **non necessario**.
- 🟡 **Calendar**: full P0 (3 games × 7 budgets × 10 seeds × 2 algos = 420 episodi) eseguibile in un singolo sprint di 1-2 giorni su workstation.

## Riferimenti

- Audit DHDNA: [`paper_fmc_dhdna_audit.md`](../paper_fmc_dhdna_audit.md) §critical claim
- Paper origine: 1803.05049v5 §5.1.2, §6.2.1, §7
- Claim discrepanze: [`CLAUDE.md`](../../../CLAUDE.md) D2
- Companion paper: 1807.01081 (Atari companion) — può fornire numeri di benchmark per validazione