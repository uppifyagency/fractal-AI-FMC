# W4B — FMC su logic-synthesis operator sequencing (phase-ordering)

Spike applicativo di Fractal Monte Carlo (FMC-base) sul **candidato #1** del
landscape survey: la scelta per-istanza della *sequenza* di operatori di
ottimizzazione technology-independent su And-Inverter Graph (AIG), con obiettivo
la minimizzazione del node-count (AND-gate). Baseline forti da battere:
**resyn2** (lo script canonico di ABC) e un **greedy** size-oriented.

- Motore: `fmc-core` (`fmc.core.plan`, gate E2 di wave3 riusato — nessuna
  reimplementazione di FMC).
- Tooling: `aigverse` 0.1.1 (binding mockturtle, gli stessi operatori concettuali
  di ABC: rewrite/resub/refactor/balance/cleanup).
- Script: [`w4b_logic_synthesis.py`](w4b_logic_synthesis.py).
- Data: 2026-07-09/10. Hardware: CPU singola, no GPU. Wall time totale: **1121.7 s**.

---

## TL;DR — verdetto onesto

**NON è una svolta.** Il candidato #1 del landscape **ha fallito il proprio gate
E2**: tutti e 10 i circuiti risultano `COLLAPSE (no-fit)`. In head-to-head FMC
**pareggia** greedy sui circuiti piatti e lo **peggiora leggermente** sui due soli
circuiti con struttura reale (cla_16 −3.49%, cla_32 −1.66% vs greedy), il tutto a
**37× il costo** in valutazioni di operatore. Wilcoxon FMC vs greedy p = 0.5000
(pari), bootstrap CI di `mean(FMC − greedy)` = **[0.00, +3.25] gate** (0 = pari,
<0 = FMC meglio) → **FMC pari-o-peggio di greedy**.

Il valore reale emerso dallo spike **non è FMC > baseline**, ma la **validità
predittiva del gate E2**: E2 ha diagnosticato il no-fit *prima* dell'head-to-head,
e l'head-to-head lo ha confermato. La causa è il **reward-plateau / quasi-determinismo**
degli operatori: lo swarm libero non diverge, quindi `relativize` degenera e il
cloning non porta informazione (FMC ≈ random/greedy).

---

## 1. Setup e API aigverse

**Operatori** (action set discreto esposto a FMC / greedy / E2), tutti che
ritornano un *nuovo* `Aig` (`inplace=False`), quindi snapshot = `Aig.clone()`
funge da `get_state`/`set_state` in modo pulito:

| azione | funzione aigverse | tempo/op (cla_16) | effetto tipico su size |
|---|---|---|---|
| `b`   | `balancing`               | 3.0 ms  | **depth-oriented — aumenta il node-count** |
| `rw`  | `aig_cut_rewriting`       | 61 ms   | riduce/pareggia (bottleneck di costo) |
| `rwz` | `aig_cut_rewriting(allow_zero_gain=True)` | 61 ms | riduce/pareggia |
| `rf`  | `sop_refactoring`         | 2.2 ms  | riduce/pareggia |
| `rs`  | `aig_resubstitution`      | 0.7 ms  | riduce (spesso il più efficace) |
| `cl`  | `cleanup_dangling`        | 0.06 ms | rimuove nodi morti (raramente da solo) |

`equivalence_checking` (SAT) ≈ 0.95 ms → usato come **gate di correttezza sull'output
finale** di ogni metodo (verificare ogni singolo passo di ogni walker, N·M·H SAT-call,
sarebbe stato proibitivo e non necessario).

**Reward** (per-stato, pre-relativize): `r(s) = g0 − num_gates(s)` = riduzione di
nodi accumulata dallo start (più alta = meglio). **Observation** (termine di
diversità FMC + dispersione E2): `[num_gates, num_levels, size]`.

### Nota di soundness: `balancing` × `random_aig`

`random_aig` è stato **escluso** dalla suite: `balancing` produce circuiti
**non equivalenti** su alcuni AIG random (verificato: seed 1 e 3 a `num_pis=8,
num_gates=100` → `equivalence_checking = False`). È un bug/limite di `balancing`
di aigverse su AIG con struttura random, non un problema di FMC. Sulla suite
aritmetica strutturata `balancing` è invece sound (10/10 output SAT-equivalenti).
Nota collaterale: con i soli operatori sound (senza `balancing`), `random_aig` ha
un landscape a **un solo scalino** — `resub` da solo fa 120→50/54 in un passo, poi
plateau: greedy lo cattura interamente in 1-2 step, nulla da pianificare.

---

## 2. Circuiti benchmark (suite aritmetica strutturata, equiv-safe)

| circuito | generatore | g0 (gates) | livelli | pis/pos |
|---|---|---:|---:|---|
| mult_3 | `ripple_carry_multiplier(3)` | 39 | 10 | 6/6 |
| mult_4 | `ripple_carry_multiplier(4)` | 84 | 16 | 8/8 |
| mult_5 | `ripple_carry_multiplier(5)` | 145 | 22 | 10/10 |
| rca_8  | `ripple_carry_adder(8)`      | 52 | 16 | 16/9 |
| rca_16 | `ripple_carry_adder(16)`    | 108 | 32 | 32/17 |
| cla_8  | `carry_lookahead_adder(8)`  | 85 | 14 | 16/9 |
| cla_16 | `carry_lookahead_adder(16)` | 186 | 29 | 32/17 |
| cla_32 | `carry_lookahead_adder(32)` | 391 | 60 | 64/33 |
| mux_8  | `multiplexer(8)`            | 24 | 2 | 17/8 |
| dec_5  | `binary_decoder(5)`         | 48 | 3 | 5/32 |

### Caratterizzazione del landscape (pre-analisi)

Una ricerca esaustiva depth-3 con dedup per firma strutturale mostra la natura
del problema **prima** di FMC:

- **mult_4, rca_8, rca_16, mult_5, mux_8, dec_5**: sono **punti fissi** dell'operator
  set — nessuna sequenza (fino a depth-3/4) riduce il node-count. Landscape **piatto**.
- **cla_8, cla_16, cla_32**: **discesa monotòna** — greedy scende passo-passo
  (cla_8: 85→52 in 5 step; cla_16: 186→108 in 9 step) senza minimi locali ingannevoli.
- mult_3: best raggiungibile 37 (−2 gate), già trovato da greedy.

Conclusione strutturale: **nessun circuito nello spazio dei generatori aigverse,
con questo operator set, esibisce i minimi locali ingannevoli che rendono
attraente FMC.** I landscape sono o piatti (niente da cercare) o greedy-risolvibili.

---

## 3. Gate E2 (free-swarm divergence) — ESEGUITO PRIMA

`N=32, M=8, 4 seed`. Gate primario: `disp_ratio ≥ 3.0` (calibrato in wave3 su 6
env, gap netto [2.39, 4.66]). Warn reward-degenere: `reward_cv_M < 0.02`.

| circuito | g0 | disp_ratio | reward_cv_M | ess_ratio | verdetto |
|---|---:|---:|---:|---:|---|
| mult_3 | 39 | 1.800 | 2.6344 | 0.578 | **COLLAPSE (no-fit)** |
| mult_4 | 84 | 1.693 | 0.7789 | 0.562 | **COLLAPSE (no-fit)** |
| mult_5 | 145 | 1.710 | 0.7760 | 0.561 | **COLLAPSE (no-fit)** |
| rca_8  | 52 | 1.836 | 0.9181 | 0.599 | **COLLAPSE (no-fit)** |
| rca_16 | 108 | 2.298 | 0.9862 | 0.596 | **COLLAPSE (no-fit)** |
| cla_8  | 85 | 1.298 | 0.6099 | 0.565 | **COLLAPSE (no-fit)** |
| cla_16 | 186 | 1.611 | 1.0516 | 0.554 | **COLLAPSE (no-fit)** |
| cla_32 | 391 | 1.805 | 1.8075 | 0.581 | **COLLAPSE (no-fit)** |
| mux_8  | 24 | **0.000** | **0.0000** | 1.000 | **COLLAPSE (no-fit)** |
| dec_5  | 48 | **0.000** | **0.0000** | 1.000 | **COLLAPSE (no-fit)** |

**Verdetto E2: 10/10 COLLAPSE.** disp_ratio 1.30–2.30, tutti sotto il gate 3.0.
mux_8/dec_5 hanno disp_ratio = 0.000 e reward_cv = 0.000: gli operatori non
cambiano *nulla* (già ottimi), lo swarm è un punto singolo. reward_cv_M NON è
degenere in senso stretto sui multiplier/cla (0.6–2.6): il canale reward *avrebbe*
segnale, ma il canale **dispersione dinamica** collassa perché gli operatori sono
quasi-deterministici → walker che scelgono la stessa sequenza atterrano sullo
stesso stato, e i pochi stati raggiungibili si sovrappongono in `[gates,levels,size]`.
È esattamente il fallimento previsto: `relativize` su un cloud che non diverge →
virtual reward ~uniforme → cloning argmax senza informazione → **FMC ≈ random**.

---

## 4. FMC vs resyn2 vs greedy (head-to-head)

`FMC: H=12, N=16, M=4, 4 seed`. resyn2 = `b;rw;rf;b;rw;rwz;b;rfz;rwz;b` (canonico
ABC mappato su aigverse). Confronto **appaiato per circuito**. `resyn2_f` = stato
finale dello script; `resyn2_b` = min lungo la traiettoria ("resyn2 + keep-best").

| circuito | g0 | resyn2_f | resyn2_b | greedy | g_ev | FMC_med | FMC_best | f_ev | FMC vs greedy | equiv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| mult_3 | 39 | 49 | 39 | 37 | 18 | 37.0 | 37 | 780 | +0.00% | Y |
| mult_4 | 84 | 128 | 84 | 84 | 6 | 84.0 | 84 | 780 | +0.00% | Y |
| mult_5 | 145 | 226 | 145 | 145 | 6 | 145.0 | 145 | 780 | +0.00% | Y |
| rca_8 | 52 | 69 | 52 | 52 | 6 | 52.0 | 52 | 780 | +0.00% | Y |
| rca_16 | 108 | 148 | 108 | 108 | 6 | 108.0 | 108 | 780 | +0.00% | Y |
| cla_8 | 85 | 73 | 64 | **52** | 36 | 52.0 | 52 | 780 | +0.00% | Y |
| cla_16 | 186 | 167 | 143 | **108** | 60 | 114.5 | 111 | 780 | **−3.49%** | Y |
| cla_32 | 391 | 366 | 313 | **261** | 60 | 267.5 | 223 | 780 | **−1.66%** | Y |
| mux_8 | 24 | 24 | 24 | 24 | 6 | 24.0 | 24 | 780 | +0.00% | Y |
| dec_5 | 48 | 48 | 48 | 48 | 6 | 48.0 | 48 | 780 | +0.00% | Y |

**Riduzione media nodi vs g0**: resyn2_final **−17.29%** (peggiora! il `balancing`
in coda gonfia la size dei multiplier), resyn2_best **+6.78%**, greedy **+11.91%**,
FMC (median) **+11.40%**.

### Statistica appaiata

- `FMC(median) − greedy` per circuito: `[0, 0, 0, 0, 0, 0, +6, +6, 0, 0]`
  → FMC pareggia su 8/10, **peggiora** su cla_16 e cla_32 (+6 gate ciascuno).
- **Wilcoxon** FMC vs greedy: W = 0.000, **p = 0.5000** (n_nonzero = 2/10) → nessuna
  differenza significativa, e i due segni non-nulli sono entrambi a **sfavore** di FMC.
- **Bootstrap 95% CI** di `mean(FMC − greedy)` = **[0.00, +3.25] gate** (0 = pari,
  <0 = FMC meglio) → l'intervallo è interamente ≥ 0: **FMC non batte mai greedy in media**.
- **FMC_best** (min sui 4 seed) su cla_32 raggiunge 223 < greedy 261: esiste una
  sequenza migliore di greedy, ma FMC la trova solo occasionalmente e non in modo
  affidabile (la mediana resta peggiore) → non è un vantaggio robusto.

### Costo

- Operator-eval: greedy totale **210**, FMC totale (media per-seed) **7800** →
  **FMC/greedy = 37×**. Ogni `plan` FMC costa H·N·M = 780 valutazioni/circuito/seed.
- Correttezza: **10/10 circuiti** con output SAT-equivalente all'originale per
  tutti e tre i metodi (resyn2, greedy, FMC).

---

## 5. Perché è collassato, e cosa servirebbe

**Perché COLLAPSE.** Gli operatori aigverse su questi circuiti sono
**quasi-deterministici e a bassa varianza**: (a) i multiplier/adder ripple e mux/dec
sono già punti fissi (0% di riduzione possibile), (b) i cla adder hanno una discesa
**monotòna** perfettamente coperta da greedy. In entrambi i casi lo swarm libero
non diverge (disp_ratio ≪ 3), quindi il meccanismo di selezione di FMC (relativize →
virtual reward → cloning) non ha traiettorie diverse tra cui scegliere. La
NP-hardness teorica del phase-ordering **non si manifesta** su queste istanze
piccole e pulite con questo operator set: mancano i minimi locali ingannevoli.

**Cosa servirebbe per un claim pubblicabile (fit reale).**
1. **Landscape più rugoso**: benchmark EPFL/ISCAS/EEP grandi e "difficili"
   (es. `hyp`, `log2`, `div`, `sqrt`, multiplier ≥ 16-bit) letti da AIGER, dove
   il phase-ordering ha realmente minimi locali e greedy si blocca.
2. **Obiettivo con trade-off** (area×delay, o technology mapping su LUT/standard-cell
   con `map`): un reward non-monotòno crea la frontiera dove le scelte di sequenza
   contano e greedy fallisce — è lì che FMC può divergere.
3. **Operator set più ricco/stocastico** (varianti con parametri randomizzati:
   cut_size, finestre resub, priority-cuts diversi) per iniettare divergenza nello swarm.
4. **Orizzonti lunghi** (H, M grandi) e istanze dove l'esaustivo mostra
   `best_reachable ≪ greedy` (gap ingannevole verificato *prima*).
5. **Baseline honest-hard**: ABC reale (`resyn2`, `resyn3`, `&dch`, `compress2rs`)
   e DRiLLS/AlphaSyn (RL) come riferimenti, non solo greedy/resyn2 mappati.
6. **Ri-eseguire E2 su questi**: procedere solo se E2 passa `DIVERGE`. Su questa
   suite E2 ha correttamente detto no — spendere compute FMC qui è stato dominato.

**Il risultato positivo dello spike** è metodologico: il gate E2 è **predittivo**.
Ha marcato no-fit a costo trascurabile e l'head-to-head da 1121 s lo ha confermato
(FMC pari-o-peggio a 37× costo). Questo estende la casistica di validità di E2
(già Cong. su MATH_CANON) a un dominio combinatorio discreto, non solo control.
