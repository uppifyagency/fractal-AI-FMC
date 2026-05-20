# E2 — Disegno pre-registrato (2026-05-20)

> **Pre-registrazione**: questo documento è scritto **prima** della raccolta dati.
> Distingue le analisi confermative (qui sotto) da quelle esplorative
> (best practice #1, skill `statistical-analysis`). Le ipotesi, la griglia, la
> dimensione campionaria e il piano d'analisi sono fissati e non vanno cambiati
> a posteriori.

## Domanda

[Congettura E](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale),
proposizione **E2**: gli esponenti FMC **α** e **β** si separano funzionalmente
in *desiderio di azione* (goal-seeking) e *preservazione di sé* (sopravvivenza)?
Esiste una banda Pareto-ottimale (α\*, β\*)?

## Disegno fattoriale

- **Fattore α** (esponente reward, `relativize(R)^α`): {0, 0.25, 0.5, 1, 2, 4} — 6 livelli.
- **Fattore β** (esponente distanza, `relativize(D)^β`): {0, 0.5, 1, 2} — 4 livelli.
- **Fattore layout**: {gauntlet, lake, scatter} — 3 livelli (covariata; E1-base ha
  mostrato eterogeneità per layout, va modellata).
- 6 × 4 × 3 = **72 celle**, n = **60 episodi/cella** → **4320 episodi FMC**.
- Kernel: `fmc-core/plan()` invariato, N = 64 walker, M = 20 tick. Identico a E1-base.
- Outcome per episodio (`lava`/`goal` mutuamente esclusivi): `died` ∈ {0,1},
  `reached_goal` ∈ {0,1}, altrimenti `timeout`. `survive` ≡ 1 − `died`.

## Ipotesi pre-registrate (direzionali, da E2)

| ID | Ipotesi | Test primario |
|----|---------|---------------|
| H1 | P(died) **cresce** con α (a β, layout controllati) | Cochran-Armitage trend, α ordinato |
| H2 | P(goal) **cresce** con α | Cochran-Armitage trend, α ordinato |
| H3 | P(died) **decresce** con β | Cochran-Armitage trend, β ordinato |
| H4 | P(goal) **decresce** con β | Cochran-Armitage trend, β ordinato |
| H5 | **Separazione funzionale**: α domina la varianza di `goal`; β domina la varianza di `survive` | decomposizione η² a due vie + GLM logistico |

**Falsificazione di E2**: se α e β **non** si separano (es. muovono lo stesso
asse, o nessuno dei due muove `survive`), oppure se non esiste una frontiera di
Pareto non-banale tra `survive` e `goal`.

**Nota a priori** (no HARKing): l'evidenza Craftax (exp22, α=1.5 → collasso
catastrofico) predice che H1/H2 possano **non** essere monotone su tutto il
range — possibile *rise-then-fall* ad α alto. Il test di trende lineare lo
rileverebbe come segnale debole; per questo riportiamo **anche** le proporzioni
per livello e un controllo esplicito di monotonicità (vedi punto 3 sotto).

## Piano di analisi (pre-specificato)

1. **Descrittiva** — proporzioni per cella con IC 95% di **Wilson**
   (robusto a celle 0%/100%, attese da E1-base). Heatmap α×β per `died` e `goal`.
2. **Trend (ipotesi primarie H1–H4)** — test di **Cochran-Armitage** per trend su
   `died` e `goal` vs α ordinato e vs β ordinato, dati aggregati. CA è un test di
   score, **robusto alla separazione perfetta** (valido con celle a 0 successi).
3. **Controllo di monotonicità** — proporzioni per livello con IC; verifica se il
   trend è monotono o *rise-then-fall* (worry exp22). Un CA nullo + shape a
   campana è una falsificazione *informativa* di H1/H2 "come monotone".
4. **Correzione multipla** — **Holm-Bonferroni** sulla famiglia primaria {H1,H2,H3,H4}.
5. **Separazione funzionale (H5)** — decomposizione **η²** a due vie sulle 24
   medie di cella (η²_α, η²_β, η²_interazione) per `goal` e per `survive`;
   robusta alla separazione. In supporto: GLM logistico `outcome ~ α + β + C(layout)`,
   odds ratio con IC 95%.
6. **Frontiera di Pareto** — scatter (`survive_rate`, `goal_rate`) per cella (α,β);
   insieme non-dominato; config Pareto-ottimali. Baseline E1-base come riferimento.
7. **Effect size** — odds ratio con IC, differenze di rischio. **Mai solo p-value.**
8. **Consistenza per layout** — punti 1–2 ripetuti per layout. E2 è "legge" solo se
   i segni di H1–H4 sono consistenti sui 3 layout (criterio analogo a Cong. A).

## Potenza / sensibilità (a priori)

Livello di significatività **0.05**, bilaterale (NB: distinto dall'esponente FMC α).

- **Test di trend (H1–H4, primari)**: N ≈ 4320 osservazioni/outcome. La potenza è
  > 0.99 per qualunque odds ratio per-unità praticamente rilevante (MDE OR ≈ 1.05–1.1).
  Le ipotesi primarie sono **ampiamente potenziate**.
- **Per-cella (n = 60)**: IC 95% Wilson semi-ampiezza ≈ **±0.12** (caso peggiore
  p = 0.5; più stretto agli estremi). MDE per confronto pairwise fra due celle ≈
  **0.26** (80% potenza). ⇒ Le differenze **fini** cella-cella **non** sono
  risolte individualmente *by design*: le ipotesi sono sui **trend aggregati**.
  La frontiera di Pareto è identificata nella struttura grossa (E1-base mostra
  estremi a distanza ≫ 0.26), non nelle differenze fini lungo la frontiera.
- n = 60 è un *first-pass*; per una frontiera publication-grade servirebbe n ≥ 150/cella.

## Riproducibilità

Semi deterministici per episodio. Codice: `e2_sweep.py` (raccolta) +
`e2_analysis.py` (analisi). Dati grezzi: `results/e2_raw.csv` (un episodio/riga).
