# Protocollo P3 — Ablation parametri RAM vs IMG

> **Status**: 🟡 **ESCALATED** — task compute-light ma non eseguibile in-session.
>
> **Origine**: [`paper_fmc_dhdna_audit.md`](../paper_fmc_dhdna_audit.md) §5.1.3.3 — la tabella RAM vs IMG del paper usa parametri ad hoc senza ablation.
>
> **Priority**: P3 (low) — nice-to-have, sblocca un claim secondario del paper.
>
> **Stima costo**: ~3-5 giorni focused work + GPU time modest.

---

## Obiettivo

Validare che il claim *"RAM observations battono IMG observations del 161.47% in media"* (paper §5.1.3.3, p. 44) sia robusto a varying budget di FMC, non un artefatto della specifica configurazione (`fixed_steps=5, time_limit=15, Max_walkers=30, Max_samples=300`).

## Domande di ricerca

1. **P3-Q1**: Il vantaggio RAM > IMG cresce, decresce, o rimane stabile con $N$ (walker count)?
2. **P3-Q2**: Lo stesso, con $M$ (time horizon)?
3. **P3-Q3**: Il vantaggio è uniforme tra giochi, o concentrato in alcuni (es. quelli con score-flicker tipo Pinball)?

## Setup sperimentale

### Game subset

Riprendere gli 8 giochi del paper §5.1.3.3:

```
games = [atlantis, bank_heist, boxing, centipede, ice_hockey,
         ms_pacman, qbert, video_pinball]
```

### Sweep parametrico

Sweep su 3 dimensioni:

| Parametro | Valori | Razionale |
|---|---|---|
| **Observation type** | RAM, IMG | Variabile primaria |
| **N (max_walkers)** | 30, 60, 120, 240 | Lo paper usa solo 30 |
| **M (time_limit)** | 10, 15, 30, 60 | Paper usa solo 15 |
| **Seeds** | 1..5 | n=5 sufficient per ablation (non publication-ready) |

**Cells totali**: 8 × 2 × 4 × 4 × 5 = **1280 run**.

A ~5 min/run = ~107 hours single-GPU. Parallelizzabile a ~25 GPU-hours.

### Parametri tenuti fissi

```python
base_params = {
    "fixed_steps": 5,
    "alpha": 1.0,
    "beta": 1.0,
    "max_samples": 300,
    "frame_skip": 4,
    "sticky_actions": False,
}
```

## Output atteso

Per ogni gioco, una **superficie 3D** di RAM/IMG ratio in funzione di $(N, M)$:

```
       M=10  M=15  M=30  M=60
N=30   1.6   1.7   1.4   1.1
N=60   1.5   1.6   1.5   1.3
N=120  1.4   1.5   1.6   1.5
N=240  1.3   1.4   1.5   1.6
```

## Decision matrix

| Pattern osservato | Lettura | Azione paper v6 |
|---|---|---|
| RAM/IMG ratio stabile a $\approx 1.6$ across all $(N, M)$ | Vantaggio robusto | Mantieni il claim, generalizza |
| RAM/IMG ratio $> 1$ ma decresce con $N$ | Vantaggio è artefatto del low-budget regime | Specifica claim: "for small N, RAM dominates; for large N, the gap closes" |
| RAM/IMG ratio collassa a 1 al crescere di $N$ | Il claim è artifact del setting `Max_walkers=30` | Riformulare: l'observation type matters solo a low budget |
| Pattern game-dependent (es. solo Pinball) | Selection bias nel paper §5.1.3.3 | Onestamente: documentare game-dependence |

## Deliverable atteso

1. `work/11_ram_vs_img_ablation/`:
   - `REPORT.md` con superfici 3D per gioco
   - `runs/{game}_{obs}_{N}_{M}_{seed}.jsonl`
   - `notebooks/ram_vs_img.ipynb`
   - `figures/ram_vs_img_surfaces.pdf`
2. Update audit DHDNA: §5.1.3.3 status da 🟡 PLAUSIBLE a 🟢 VERIFIED o 🟡 CONTEXT-DEPENDENT
3. Update paper v6 §5.1.3.3 con ablation che giustifica il claim

## Caveats

- **n=5 per cella** è sotto-campionato per publication (servirebbe n≥10). Sufficiente solo per ablation interna; per il paper finale, usare n=10 sul subset interessante che emerge dall'ablation.
- **No simulation perfetta su Atari per RAM**: tecnicamente la RAM dump è bit-perfect, l'IMG ha aliasing/flicker. Questo *spiega* il vantaggio RAM ma non lo *quantifica* — l'ablation è ancora utile.

## Trigger di esecuzione

Buon candidato per "spike" di 1 settimana se P0 e P1a non sono ancora pronti. Compute leggero, può girare su una singola workstation.

## Status corrente (2026-04-28)

- ✅ **Ablation driver** ([`work/11_ram_vs_img_ablation/scripts/ram_img_sweep.py`](../../../work/11_ram_vs_img_ablation/scripts/ram_img_sweep.py)) + aggregator.
- ✅ **Boxing micro-cell** (RAM vs RGB, n=2, paper params N=30, M=15): entrambi cap-bound a +100. RGB raggiunge il cap leggermente più veloce (104 vs 119 actions, n=2 troppo piccolo per significatività).
- 🟢 **Hardware revision**: 1280 celle × ~80 s = **~28 ore single-CPU per il protocollo full**. Cluster GPU non necessario.
- 🟡 **Boxing è cap-bound**, non differenzia. **Il claim §5.1.3.3 va testato sui giochi non-cap-bound** (Atlantis, Centipede, MsPacman, QBert, VideoPinball).

## Riferimenti

- Paper §5.1.3.3 (p. 44): tabella RAM vs IMG originale
- `plangym/` Atari backend supports both RAM and IMG observations