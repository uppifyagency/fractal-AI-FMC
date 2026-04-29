# Protocollo P1a — Replica Atari results con n=10 seed + error bars

> **Status**: 🟡 **ESCALATED** — task compute-heavy, pianificabile in parallelo a P0.
>
> **Origine**: [`paper_fmc_dhdna_audit.md`](../paper_fmc_dhdna_audit.md) §5.1.1 — i numeri Atari del paper sono single-seed, no error bars, no n-trials.
>
> **Priority**: P1 (high) — necessario per credibilità statistica del paper v6.
>
> **Stima costo**: ~2 settimane focused work + GPU time.

---

## Obiettivo

Replicare la tabella score di Atari 50 games del paper §5.1.1 (pp. 41-42) con:
1. **n=10 seed** per ogni gioco (paper attuale: single seed)
2. **Error bars** mean ± std (paper attuale: nessuno)
3. **Significance test** vs baseline (paper attuale: nessuno)
4. **Per-game baseline esplicito** (paper attuale: lump categorie "Best Planning SoTA"/"Best Learning SoTA")

## Approccio in due fasi

### Fase A — Replication audit (1 settimana)

Riprodurre i numeri del paper con seed singolo per validare lo stack `fragile` + `plangym`. Se non riproduciamo i numeri originali, non possiamo costruirci sopra.

```
for game in 50_atari_games:
  fmc_score = run_fmc(game, params=paper_params, seed=42)
  diff_pct = (fmc_score - paper_score[game]) / paper_score[game]
  if abs(diff_pct) > 0.10:
    flag_for_investigation(game)
```

**Soglia di accettazione**: 90% dei giochi entro ±10% del paper score. Sotto questa soglia, blocker per Fase B — qualcosa non è settato giusto.

### Fase B — Multi-seed con error bars (1-2 settimane)

```
for game in 50_atari_games:
  scores = []
  for seed in 1..10:
    scores.append(run_fmc(game, params=paper_params, seed=seed))
  mean[game] = np.mean(scores)
  std[game] = np.std(scores)
  ci95[game] = bootstrap_ci(scores, n=1000)
```

## Setup

### Stack tecnico

- **Core**: `repos/fragile/` (PyTorch GPU FMC)
- **Env wrapping**: `repos/plangym/` (Atari ALE backend)
- **Logging**: `repos/flogging/` (JSON structured) + W&B optional

### Parametri base (matching paper §5.1.3.3)

```python
fmc_params = {
    "fixed_steps": 5,
    "time_limit": 15,
    "max_walkers": 30,
    "max_samples": 300,
    "alpha": 1.0,
    "beta": 1.0,
}
```

### Hardware

- **GPU**: 1× A100 (preferred) o 4× T4
- **Time**: ~30 min/game/seed × 50 games × 10 seeds = ~250 GPU-hours total
- Possibile parallelizzazione via batch on GPU → ~50-80 GPU-hours

## Statistical methodology

Per ogni gioco, riportare:

| Metric | Formula | Note |
|---|---|---|
| Mean ± std | $\bar{x} \pm s$ con $s = \sqrt{\frac{1}{n-1}\sum(x_i - \bar{x})^2}$ | Standard |
| CI95 | Bootstrap percentile, $n_{\mathrm{boot}} = 1000$ | Robusto a non-normalità |
| vs paper | $z = (\bar{x} - x_{\mathrm{paper}}) / s$ | Quantifica deviation |
| Coverage of paper claim | "% di seed che superano `Standard Human`" | Riformulazione del claim "98% wins" |

### "Solved" criterion

Il paper §5.1.1 usa criteri ad hoc per "solved" (raggiunge ending score, raggiunge bug-induced limit, etc.). Per il v6:

- **"Solved" tier 1**: tutti i 10 seed superano la soglia di "ending score" / "human record" → claim forte
- **"Solved" tier 2**: media dei 10 seed supera la soglia ± std → claim moderato
- **"Solved" tier 3**: media supera soglia ma alcune run falliscono → claim debole, riportare percentile

## Decision matrix

Confronto risultati Fase B con tabella paper §5.1.1:

| Risultato | Lettura | Azione |
|---|---|---|
| Means matchano paper, std stretto (CV < 10%) | Risultati robusti | Aggiornare tabella v6 con error bars, claim sostanzialmente uguale |
| Means matchano paper, std largo (CV > 30%) | Variability sottostimata | Tabella v6 con error bars + caveat onesto |
| Means significativamente sotto paper | Non riproducibili come scritti | **Crisis** — riformulare in modo conservativo, indagare causa |
| Means sopra paper (rare ma possibile) | Stack v2026 batte v2020 | Bonus, ma indagare se è da frame-skip / sticky-actions diversi |

## Caveat to document

- **Sticky actions**: paper usa probabilmente det-env. ALE moderno usa sticky actions di default. **Disabilitare** per riproducibilità.
- **Frame-skip**: 4 (standard). Verificare match.
- **Score limit bugs**: 16/50 giochi del paper sono "solved due to 1M bug" — questi sono limit di bit precision dell'engine, non claim di algoritmo. Documentare quali sono e separarli nella tabella v6.
- **Best Planning SoTA / Best Learning SoTA come lump**: nel v6 vogliamo per ogni gioco la *specifica* baseline. Cercare papers [P1-7] e [L1-9] del paper §5.1.1 e riferirsi al numero per gioco.

## Deliverable atteso

1. `work/10_atari_replication/`:
   - `REPORT.md` con tabella paper-style + error bars + CI95
   - `runs/{game}_{seed}.jsonl` raw
   - `notebooks/atari_analysis.ipynb`
   - `figures/atari_table_with_errorbars.pdf`
2. Update audit DHDNA: §5.1.1 status da 🟡 PLAUSIBLE a 🟢 VERIFIED (con caveat)
3. Tabella v6 ready-to-paste con: game, FMC mean ± CI95, n_seeds, baseline per gioco

## Trigger di esecuzione

Eseguibile in parallelo a P0 — entrambi usano lo stesso stack `fragile + plangym`. Buon candidato per un focused 2-week sprint.

## Status corrente (2026-04-28)

- ✅ **Atari adapter** ([`fmc-core/src/fmc/envs/atari.py`](../../../fmc-core/src/fmc/envs/atari.py)) — RAM + RGB.
- ✅ **Multi-seed sweep con bootstrap CI95** ([`work/10_atari_replication/scripts/atari_seed_sweep.py`](../../../work/10_atari_replication/scripts/atari_seed_sweep.py)).
- ✅ **Boxing slice in-session** (n=5 seed × N=30 × M=15 paper params): mean +100.0, std 0.0, CI95 [100, 100]. 5/5 seed in knockout. ~82 s / seed CPU-singolo.
- 🟢 **Hardware revision**: il protocollo originale stimava ~250 GPU-h. **Misurazione effettiva: ~80 s / episodio FMC su CPU singola → ~11 ore single-CPU per i 50 giochi × 10 seed.** Cluster GPU **non necessario**.
- 🟡 **Open**: 49 giochi rimanenti + sticky-action verification + per-game baseline lookup.

## Riferimenti

- Paper §5.1.1 (pp. 40-42): tabella originale
- Paper §5.1.3.3 (p. 44): parametri usati per tabella RAM vs IMG (parametri base)
- `repos/fragile/src/fragile/fractalai.py`: implementazione FMC PyTorch
- `repos/plangym/`: ambiente wrapping