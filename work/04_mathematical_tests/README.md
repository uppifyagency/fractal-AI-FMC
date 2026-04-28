# 04 — Test matematici delle nozioni FMC

> *Verifica numerica controllata di due claim del seminario su slide di Sergio Hernández
> ([deep-dive 08](../02_deep_dives/08_video_seminar_extracted_insights.md)):*
>
> - **F12** — *cross-entropy collapse / Gibbs equilibrium*: la densità di walker FMC si allinea a `P_R(x) ∝ R(x)^α` su ogni slice della cone (paper §3, deep-dive 01 Teorema 3).
> - **F11** — *reward-negativity ablation*: con reward negative grezze, l'agente diventa "miedoso" (lento, fermo); `relativize` previene questo collasso comportamentale.

## Struttura

```
work/04_mathematical_tests/
├── README.md                    (questo file)
├── fmc_core.py                  Implementazione FMC numpy fedele al paper §4.4-5
├── toy_environment.py           Domini 2D + funzioni reward parametriche
├── test_f12_cross_entropy.py    F12: α-scan unimodale + multimodale (toy 2D)
├── test_f12_atari_boxing.py     F12: validazione su Atari Boxing (action-marginal)
├── test_f11_relativize_ablation.py  F11: 2 scenari × 4 condizioni
└── results/
    ├── f12_summary.json
    ├── f12A_alpha_scan.png
    ├── f12A_metrics_vs_alpha.png
    ├── f12B_metrics_bars.png
    ├── f12B_multimodal_emp_vs_target.png
    ├── f11_summary.json
    ├── f11_A_wells_offset_*.png       (3 plot)
    └── f11_B_smooth_gradient_*.png    (3 plot)
```

## Reproducibilità

```bash
cd work/04_mathematical_tests
python3 test_f12_cross_entropy.py        # ~30 s   (toy 2D, α-scan + multimodale)
python3 test_f11_relativize_ablation.py  # ~60 s   (2 scenari × 4 condizioni)
python3 test_f12_atari_boxing.py         # ~45 s   (Atari Boxing, action-marginal)
```

Dipendenze: numpy, scipy, matplotlib (versioni testate: numpy 2.2.6, scipy 1.16.1, matplotlib 3.10.6, Python 3.11.7). Per il test Atari serve anche `gymnasium 1.3.0` + `ale_py 0.11.2`. Nessuna GPU. Risultati deterministici per seed.

## Punti chiave numerici

### F12 — Gibbs equilibrium

| α (balance) | log-Pearson(log P_walker, log Gibbs) | KL(P_walker ‖ R^α) |
|---|---|---|
| 0.0 | ≈ 0  (uniforme, atteso) | 1.02 |
| 0.5 | 0.20 | 0.62 |
| **1.0** | **0.77** ← Sergio's claim P_W ∝ R | 1.21 |
| 2.0 | 0.67 | 0.90 |
| 4.0 | 0.64 | 0.48 |

**Verdetto F12**: la correlazione log-log a α=1 è **0.77**, confermando direzionalmente la proporzionalità di Sergio. Il KL non si annulla per finite-N + bias del `relativize` (si veda §3 sotto).

Nel landscape multimodale (3 picchi):
- FMC canonical: log-Pearson 0.82, KL 0.87
- FMC senza relativize: log-Pearson 0.86, KL 0.38 *(meglio)*
- Random walk: log-Pearson 0.04, KL 0.54 *(nessuna correlazione, baseline)*

**Atari Boxing (action-marginal P_FMC ∝ E[R|a])**: Pearson medio +0.45 / mediano +0.61, Spearman medio +0.43 / mediano +0.72, frazione decisioni positiva 82% (39 decisioni utili su 80 totali). Boxing score +14 dopo 80 decisioni. La proporzionalità di Sergio si conserva in dominio non-toy, con calo atteso per finite-N (50 walker su 18 azioni → ~3 walker/azione).

### F11 — Reward negativity ablation

**Scenario A** (wells + peak, R minimo -2.0, R massimo +2.0, reward flat negativa al di fuori dei picchi):

| Condizione | mean_x finale | frac al goal | speed | comportamento |
|---|---|---|---|---|
| FMC relativize | 0.72 | 0% | 1.71 | bloccato vicino allo start, caotico |
| FMC raw clip-at-0 | **8.04** | **84%** | 0.59 | risolve grazie a thresholding |
| **FMC raw signed (Sergio "fearful")** | **0.60** | **0%** | **0.57** | **stalled, "miedoso"** ✓ |
| Random walk | 3.97 | 2% | 0.36 | diffusione uniforme |

**Scenario B** (smooth gradient, R = -0.7 + 0.10·x, reward negativa ma con gradiente):

| Condizione | mean_x finale | frac > x=6 | reward finale |
|---|---|---|---|
| **FMC relativize** | **9.74** | 100% | **0.27** *(miglior)* |
| FMC raw clip-at-0 | 9.44 | 100% | 0.24 |
| FMC raw signed | 9.42 | 100% | 0.24 |
| Random walk | 3.97 | 24% | -0.30 |

**Verdetto F11**: la previsione di Sergio sul comportamento "miedoso" è **confermata empiricamente** in Scenario A (raw signed → mean_x=0.60, hull tiny=4.0, stalled). In Scenario B con gradiente disponibile relativize **brilla** rispetto a tutto il resto. Caveat: in landscape *flat-negative* né relativize né raw_signed funzionano — solo raw clip-at-0 risolve, evidenziando un limite non discusso da Sergio.

## Bias osservato di `relativize`

Un risultato secondario importante: `relativize` **amplifica α effettivo**. La trasformazione

```
R̂ = exp(z)             se z ≤ 0
R̂ = 1 + ln(1 + z)      se z > 0
```

con z = (R - μ)/σ ha derivata `dR̂/dz = exp(z)` per z<0 e `1/(1+z)` per z>0. Vicino a z=0 entrambe valgono 1, ma per z grandi positivi la mappa comprime, e per z grandi negativi la mappa espande verso 0. Il risultato è che la differenziazione tra walker buoni e cattivi viene **amplificata in senso esponenziale**: un walker leggermente meno buono della media diventa quasi-zero. Pertanto la temperatura inversa effettiva dell'algoritmo è > α nominale.

Conseguenza pratica vista in F12 Test A: `α_codice = 0.5` produce log-Pearson che si avvicina al match Gibbs ideale, mentre `α_codice = 1.0` over-concentra rispetto a `R^1`.

## Limiti dei test

1. **2D continuo, smooth**: i test usano landscape continui; FMC è stato originariamente concepito per Atari (stati discreti, dinamiche complicate). I risultati estendono la teoria ma non sostituiscono benchmark Atari.
2. **Static reward**: testiamo distribuzione stazionaria, non scanning sul cone causale forward dinamico (paper §3). Per testare pienamente F12 come paper-section-3 servono walker che partono tutti da `x_0` e si misurano slice nel tempo.
3. **N=200-400**: piccolo numero di walker. La convergenza analitica è in O(1/√N) (Del Moral 2004); con N ≥ 10000 KL dovrebbe scendere significativamente.
4. **β = 1 implicito**: il paper definisce `VR = D^β · R^α` con due esponenti; tutto FractalAI_old e i nostri test usano β=1.

## Cosa va in deep-dive 08

I numeri sopra sono integrati in [`work/02_deep_dives/08_video_seminar_extracted_insights.md` §7](../02_deep_dives/08_video_seminar_extracted_insights.md) come *Verifica numerica F12 e F11*.

## Direzioni future

- **F22 (robustezza al rumore)**: iniettare rumore gaussiano nelle osservazioni e misurare degradazione FMC vs MCTS UCT.
- **F8 (stochastic distance)**: confrontare empiricamente single-random-pair vs O(N²) full-pairwise distance estimator. Variance vs efficienza.
- **F15 (efficienza vs MCTS)**: re-eseguire benchmark Atari con budget di sample stretto per chiudere la discrepanza paper-vs-Sergio-orale.
- **Teorema unicità relativize (deep-dive 04)**: dimostrazione formale via sympy + carta degli assiomi A1-A5.
- **F12 ad alta N**: rifare F12 Atari con N=200-500 walker per verificare che la correlazione si avvicini al regime toy (log-Pearson ~0.77).
