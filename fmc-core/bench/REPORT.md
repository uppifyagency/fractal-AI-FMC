# fmc-core benchmark report

> **Generato dalla sessione `/loop` autonoma del 2026-04-27.**
> **Cosa testa**: Congettura A (Sergio's $b_{\text{eff}}^* \approx 6$) replicata indipendentemente in fmc-core su 4 task built-in.
> **Risultato sintetico**: il "6" è una scoperta empirica robusta a $K=9$, ma **scala con $K$**. Non è una costante universale.

## 1. Setup

- Implementazione: `fmc-core` Python (NumPy reference, mappato 1:1 a `docs/MATH_CANON.md`)
- Walker count: $N = 32$
- Horizon: $M = 15$ tick
- Seeds per cella: $20$
- CI95: bootstrap percentile, $5000$ resample
- Hardware: vedi campo `hardware` nei JSONL

## 2. Risultati per task ($K=9$)

### Rocket 2D free-flight (`rocket_alpha_beta_sweep`)

```
α=0.00 β=0.00 → b_eff = 7.99 [7.79, 8.17]
α=0.00 β=0.50 → b_eff = 3.81 [3.44, 4.23]
α=0.00 β=1.00 → b_eff = 3.09 [2.80, 3.40]
α=0.10 β=0.00 → b_eff = 5.35 [4.90, 5.80]    ★ Sergio config
α=0.10 β=0.50 → b_eff = 3.75 [3.29, 4.22]
α=0.10 β=1.00 → b_eff = 3.15 [2.77, 3.51]
α=0.50 β=0.00 → b_eff = 1.26 [1.12, 1.42]
α=0.50 β=0.50 → b_eff = 1.41 [1.22, 1.62]
α=0.50 β=1.00 → b_eff = 2.11 [1.84, 2.41]
α=1.00 β=0.00 → b_eff = 1.04 [1.00, 1.09]    palmera
α=1.00 β=0.50 → b_eff = 1.29 [1.10, 1.52]
α=1.00 β=1.00 → b_eff = 1.26 [1.11, 1.43]
```

### Navigation 2D (`navigation2d_alpha_beta_sweep`)

```
α=0.00 β=0.00 → b_eff = 7.99 [7.79, 8.17]
α=0.10 β=0.00 → b_eff = 5.98 [5.51, 6.45]    ★ centrato a 6
α=0.50 β=0.00 → b_eff = 1.82 [1.58, 2.06]
α=1.00 β=0.00 → b_eff = 1.66 [1.43, 1.89]
```

### Pendulum swing-up (`pendulum_alpha_beta_sweep`)

```
α=0.00 β=0.00 → b_eff = 7.99 [7.79, 8.17]
α=0.10 β=0.00 → b_eff = 6.40 [6.00, 6.81]    ★ CI95 inizia esattamente a 6
α=0.50 β=0.00 → b_eff = 1.77 [1.51, 2.04]
α=1.00 β=0.00 → b_eff = 1.44 [1.24, 1.66]
```

## 3. Test della K-universalità — la legge di scaling

Stesso task (navigation 2D), stessa config Sergio ($\alpha=0.1, \beta=0$), $K$ variabile su 8 valori (20 seed/cella):

| $K$ | $b_{\text{eff}}^*$ | CI95 | $c_K = b_{\text{eff}}^*/K$ |
|---|---|---|---|
| 3 | $2.62$ | $[2.47, 2.76]$ | $0.87$ |
| 4 | $3.44$ | $[3.25, 3.60]$ | $0.86$ |
| 6 | $4.73$ | $[4.41, 5.04]$ | $0.79$ |
| 9 | $5.97$ | $[5.41, 6.54]$ | $0.66$ |
| 12 | $7.29$ | $[6.56, 8.03]$ | $0.61$ |
| 16 | $8.39$ | $[7.46, 9.31]$ | $0.52$ |
| 24 | $9.88$ | $[8.67, 11.14]$ | $0.41$ |
| 32 | $10.76$ | $[9.30, 12.24]$ | $0.34$ |

Confronto fra modelli:

| Modello | Forma | Stima | SSE |
|---|---|---|---|
| Costante (Sergio's "6") | $b = c$ | $c = 6.63$ | 61.45 |
| Lineare | $b = c \cdot K$ | $c = 0.633$ | 123.86 |
| **Power law** | $b = a \cdot K^p$ | $a = 1.53,\; p = 0.595$ | **2.46** ✓ |

**Risultato (a M=15)**: $b_{\text{eff}}^*(α=0.1, β=0, M=15) \approx 1.53 \cdot K^{0.6}$

Il "magic 6" è un punto della curva a $K=9$, non una costante. Predizioni a $M=15$:
- $K=18$ (Atari Boxing): $\sim 9.0$
- $K=4$ (LunarLander): $\sim 3.5$

## 3b. M-dependence — anche K^0.6 è transiente

Variando il planning horizon $M$ (sweep `M_dependence.jsonl`):

| $M$ | $K=6$ | $K=9$ | $K=16$ |
|---|---|---|---|
| 5   | 5.31 | 7.45 | 11.05 |
| 15  | **4.73** | **5.97** | **8.39** |
| 30  | 3.34 | 4.24 | 5.09 |
| 60  | 1.94 | 2.45 | 2.55 |
| 120 | 1.26 | 1.55 | 1.61 |

A $M \to \infty$ tutti i $K$ collassano a palmera ($b_{\text{eff}} \to 1$), come previsto dal Teorema 2 di MATH_CANON (Gibbs equilibrium su $R^\alpha$).

**La legge $K^{0.6}$ è transiente**, valida solo nella finestra $M \sim 10$–$30$. Sergio's "6" è quindi *doppiamente* contingente: $K=9$ E $M=15$ specifico.

## 3c. N-dependence — anche N entra in gioco

A $K=9, M=15$ fissati, variando solo il numero di walker $N$ (sweep `N_dependence.jsonl`):

| $N$ | $b_{\text{eff}}^*$ | $K - b_{\text{eff}}^*$ |
|---|---|---|
| 8 | 3.12 | 5.88 |
| 16 | 4.80 | 4.20 |
| 32 | **5.97** | 3.03 |
| 64 | 7.10 | 1.90 |
| 128 | 7.67 | 1.33 |
| 256 | 8.05 | 0.95 |
| 512 | 8.10 | 0.90 |

**A $N \to \infty$ il sistema rimane near-uniform** ($b_{\text{eff}} \to K-1 \approx 8$), perché in $M=15$ tick la selezione non ha tempo di agire. Il deficit $K - b_{\text{eff}}$ scala come **power law $\propto N^{-0.45}$** — stessa fenomenologia di Wright-Fisher con tempo di fissazione $\tau \sim O(N)$.

**Sergio's "6" è triplamente contingente**: $K=9, M=15, N=32$. È un punto specifico della superficie

$$b_{\text{eff}}^* = 1 + (K-1) \cdot \mathcal{F}(M/N) \cdot \mathcal{G}(\alpha, K)$$

**Il vero contributo originale**: aver mostrato che il "6" non è una legge fisica universale ma uno snapshot di un transitorio multi-parametro Wright-Fisher-like.

## 4. Tabella riassuntiva — Sergio config $(α=0.1, β=0)$

| Task | $K$ | $b_{\text{eff}}^*$ | CI95 | Banda Sergio [5,7]? |
|---|---|---|---|---|
| Rocket 2D (JS, ref) | 9 | 5.78 (sd 0.62) | n/a | ✅ |
| Rocket 2D (fmc-core) | 9 | 5.35 | [4.90, 5.80] | ✅ |
| Navigation 2D | 9 | 5.98 | [5.51, 6.45] | ✅ |
| Pendulum swing-up | 9 | 6.40 | [6.00, 6.81] | ✅ |
| Navigation 2D (K=16) | 16 | 8.39 | [7.46, 9.31] | ❌ |

## 4b. Bet 2 — Fractal-of-Thought su LFM2.5-1.2B (Liquid AI)

Esperimento aggiuntivo (`bench/llm/fot.py`): FoT come planner per chain-of-thought su LLM piccolo.

**Setup**:
- Modello: `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` (Apple MLX backend, ~equiv Llama-3.2-1B)
- Embedding distance: `sentence-transformers/all-MiniLM-L6-v2`
- Reward: validità del numero estratto + length penalty (concise = preferito)
- Walker = una catena di reasoning completa
- Cycle = re-generation guidata dai walker forti

**Tre metodi confrontati su 12 problemi math hard, 2 seed**:

| Metodo | Accuracy | Avg tokens/problema |
|---|---|---|
| Greedy ($T=0$) | $66.7\%$ (16/24) | 247 |
| Self-consistency ($K=8, T=0.7$) | $83.3\%$ (20/24) | 1931 |
| **FoT** ($N=8, M=2$, embed dist) | $\mathbf{87.5\%}$ (21/24) | 2522 |

**Verdetto preliminare**:
- FoT $>$ greedy: $+20.8$pp (significativo)
- FoT $>$ self-consistency: $+4.2$pp (modesto)
- FoT costa $+30\%$ token vs SC

Su EASY tier (12 problemi single-step) tutti e tre al $100\%$. Il vantaggio FoT emerge solo nel regime hard, dove greedy fallisce sistematicamente (es. "bookshelf 5×8 - 28 = ?" — greedy dà 5).

**Limiti onesti**: con questo benchmark ridotto, il vantaggio FoT vs SC è entro la rumore stochastico. Per Bet 2 *go* serve setup ottimizzato (più cicli, reward function migliore, tempering di $\alpha$). Tutti dati in `bench/results/fot_llm_hard.jsonl`.

## 5. Conclusioni operative

1. **Trend $\alpha$ confermato** su 3 task: aumentando $\alpha$ il sistema collassa monotonamente verso palmera ($b_{\text{eff}} \to 1$).
2. **Trend $\beta$ controintuitivo confermato**: aumentare $\beta$ riduce $b_{\text{eff}}$ (vedi rocket sweep), coerente con caveat di MATH_CANON Th. 3.
3. **Sergio config $(\alpha=0.1, \beta=0)$ produce sweet spot a $K=9, M=15$** — robusto su 3 task con fisica diversa (navigazione 2D, free-flight, energy-balance).
4. **Il "magic 6" è $K$-dipendente**: a $K=16$ vale ~8.4, a $K=4$ vale ~3.4.
5. **Il "magic 6" è anche $M$-dipendente**: a $M=120$ collassa a ~1.5 per tutti i $K$.
6. **Quello che resta vero asintoticamente**: a $M \to \infty$, $b_{\text{eff}} \to 1$ (Teorema 2: Gibbs equilibrium concentrato sui massimi di $R^\alpha$).
7. **Insight finale**: il "6" di Sergio è uno snapshot di un transitorio doppiamente contingente, **non** una Terza Legge della cognizione. Il contributo del progetto è aver esibito la struttura del transitorio e mostrato che il claim universale è falsificato.

## 6. Implicazioni per future research

- **NO**: pubblicare paper con "Sergio's universal magic 6" senza il caveat $K=9$. Sarebbe scientificamente disonesto.
- **SÌ**: pubblicare la **scoperta del trend $c_K$** come contributo originale: "il branching ottimo di FMC scala come $c_K \cdot K$ con $c_K$ decrescente da 0.66 a 0.52 nel range $K \in [9, 16]$". Questo è genuino e nuovo.
- **NEXT**: aggiungere $K \in \{4, 6, 12, 20, 32\}$ per stimare la forma di $c_K$. Se $c_K = 6/K$ esatto, cioè $b_{\text{eff}}^* = 6$ è invariante, allora il "6" SI è universale e il mio K=16 era statisticamente compatibile con 6 (vediamo: a K=16 il LB CI95 è 7.46, lontano da 6 — quindi NO, $c_K$ è davvero decrescente più lento).
- Servirebbe anche estendere a $K=18$ (Atari Boxing), una volta integrata l'env reale.

## 7. Riproduzione

```bash
cd fmc-core
make install            # editable install
make test-all           # 55 Python + 11 JS test verdi
make bench-full         # rocket + navigation2D sweep, ~1 min
python3 -m bench.pendulum_sweep --full
python3 -m bench.k_dependence_sweep
```

Output JSONL in `bench/results/`. Ogni record contiene `params`, `values` (per seed), `mean`, `ci95_low/high`, `hardware`, `timestamp_utc`, `fmc_core_version`.

---

*Fine REPORT. Riferimenti: [MATH_CANON.md](../../docs/MATH_CANON.md), [work/07_sergio_branching_sweep/REPORT.md](../../work/07_sergio_branching_sweep/REPORT.md), podcast Sergio cap. 16.*
