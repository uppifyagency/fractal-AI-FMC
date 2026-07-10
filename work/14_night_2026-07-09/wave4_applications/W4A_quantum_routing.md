# W4A — FMC su qubit routing / SWAP insertion (vs SABRE)

**Data:** 2026-07-10 · **Branch:** `autoresearch/exp02-ach-bonus`
**Codice:** [`w4a_quantum_routing.py`](w4a_quantum_routing.py) · **Stack:** fmc-core (in-repo) + qiskit 2.5.0 (SABRE) · CPU-only
**Riproduci:** `python3 w4a_quantum_routing.py` (~30 s) · pilot: `--pilot`

Primo spike applicativo di FMC su un problema HW 2026 concreto: **routing** di un
circuito logico su una coupling map di device inserendo gate **SWAP**, minimizzando
il numero di SWAP. Baseline forte: **SABRE** (`SabreSwap`, euristica `decay`).

---

## 1. Design dell'env (protocollo `fmc.envs.base.Environment`, stato immutabile/atomico)

- **Stato** = `(pos: logico→fisico, idx: prossimo gate 2q, swaps)`. `clone_state` fa
  deep-copy, `step` è puro (ritorna nuovo stato) → reversibilità/atomicità by design.
- **Gate processati in ordine di programma** (l'`idx` della spec). L'azione "applica il
  gate se adiacente" è ripiegata in **greedy execution**: dopo ogni SWAP (e al reset)
  si applicano gratis tutti i gate consecutivi già adiacenti. Identico al front-layer
  di SABRE → l'unica decisione reale è *quale SWAP inserire*.
- **Azioni** = gli archi del device (K = |E| SWAP candidati), **fisse** → FMC-compatibili.
- **Reward** (per-stato, pre-relativize):
  `r(s) = 10·gate_done − 0.1·swaps − 1.0·dist_next`, dove `dist_next` = distanza
  shortest-path sul coupling graph tra i due qubit fisici del *prossimo* gate.
  La distanza è **la stessa informazione che usa l'euristica SABRE** → darla a FMC è
  equo, non barare. Minimizzare gli SWAP ≡ finire in meno step (1 SWAP = 1 step).
- **FMC come planner closed-loop**: ad ogni step reale `plan(N,M)` → SWAP scelto →
  applica → ripeti fino a `done`. Uso standard di `core.plan`/`decide`.
- **Fairness**: layout iniziale **identità** per entrambi (SabreSwap su circuito già
  su qubit fisici). Isola la sola decisione di routing.
- **Validità**: ogni routing FMC è **ri-verificato in modo indipendente**
  (`build_physical_circuit`) ricostruendo il circuito fisico e controllando che ogni
  gate 2q cada su una coppia adiacente e che tutti i gate siano processati. **28/28 +
  28/28 circuiti validi, tutti i seed.**

Coupling maps: **linear-5** (n=5, |E|=4) e **grid-3×3** (n=9, |E|=12).
Circuiti: 24 random (depth 3n) + GHZ + QFT + 2 QFT-shuffled = **28 per mappa**.

---

## 2. Gate E2 PRIMA (il free swarm diverge sul routing?)

`e2_divergence` riusato tale quale (N=64, M=30, α=β=1, 5 seed). Gate primario:
`disp_ratio ≥ 3.0`.

| Mappa | K | disp_ratio | reward_cv_M | ess_ratio | b_eff/K | **Verdetto** |
|---|---|---|---|---|---|---|
| linear5 | 4 | **3.05** | 0.288 | 0.948 | 1.20/4 | **DIVERGE (FMC-fit)** |
| grid3x3 | 12 | **2.94** | 0.492 | 0.641 | 1.34/12 | **COLLAPSE (no-fit)** |

- **linear5 passa** (di poco) il gate: il canale reward non è degenere (cv 0.29).
- **grid3x3 fallisce** (di poco, 2.94 < 3.0). Nota metodologica: su env discreti ad
  alto branching, **un solo** SWAP casuale su 12 archi già disperde molto lo swarm
  (`disp_1`=3.95 vs 1.79 su linear5), gonfiando il denominatore e deprimendo il
  ratio. Il gate è quindi *pessimista* per K grande — ma la sua chiamata "no-fit" è
  poi **confermata dall'esito reale** (§4).

---

## 3. FMC vs SABRE — risultati appaiati (stesso circuito, 5 seed FMC / 8 SABRE, best-of-seeds)

### linear5 (28 circuiti) — E2: FMC-fit

| Metrica | SABRE | FMC best-of-5 | FMC mean-over-5 |
|---|---|---|---|
| SWAP medi | 9.04 | **8.46** | 8.80 |
| mean(SABRE−FMC) | — | **+0.57** | +0.24 |
| win/tie/loss (FMC) | — | **8/19/1** | 8/11/9 |
| bootstrap 95% CI (SABRE−FMC) | — | **[+0.21, +0.96] esclude 0** | [−0.09, +0.60] include 0 |
| Wilcoxon p | — | **0.011** | 0.32 |
| depth media | 17.6 | **16.5** | — |
| costo | **0.40 ms/circ** | 62 ms/circ (7 ms/decisione) | ~158× SABRE |

### grid3x3 (28 circuiti) — E2: no-fit

| Metrica | SABRE | FMC best-of-5 | FMC mean-over-5 |
|---|---|---|---|
| SWAP medi | **13.0** | 15.93 | 19.26 |
| mean(SABRE−FMC) | — | **−2.93** | −6.26 |
| win/tie/loss (FMC) | — | **3/0/25** | 1/1/26 |
| bootstrap 95% CI (SABRE−FMC) | — | **[−3.79, −2.00] esclude 0** | [−7.18, −5.24] esclude 0 |
| Wilcoxon p | — | **5.9e-5** | 6.3e-6 |
| depth media | 25.1 | 24.8 | — |
| costo | **0.48 ms/circ** | 144 ms/circ (7.5 ms/decisione) | ~303× SABRE |

Casi strutturati notevoli: **GHZ** FMC vince su entrambe le mappe (linear 3 vs 3;
grid 3 vs 4). **QFT** (all-to-all denso) è dove FMC crolla (grid: 23–30 vs 19–24
SABRE): FMC processa i gate in **ordine stretto** e non sfrutta la commutazione/riordino
che SABRE ottiene dal DAG front-layer.

---

## 4. Verdetto onesto

**Il gate E2 ha predetto correttamente l'esito su entrambe le mappe.** Dove E2 dice
FMC-fit (linear5) FMC pareggia/batte marginalmente SABRE; dove dice no-fit (grid3x3)
FMC perde in modo netto e significativo. Questo è il risultato scientificamente più
solido dello spike: **E2 come pre-filtro predittivo tiene anche su un dominio nuovo
(quantum routing), non solo sui fixture di calibrazione.**

- **linear5 → PARI (con edge marginale best-of-seeds).** Best-of-5 seed: −0.57 SWAP
  medi vs SABRE, Wilcoxon p=0.011, CI esclude 0 — statisticamente reale ma minuscolo
  (8 vittorie, 19 pari, 1 sconfitta su 28). Il *run tipico* (mean-over-seeds) è un
  **pareggio** (p=0.32). FMC anche leggermente più shallow (depth 16.5 vs 17.6).
- **grid3x3 → NO.** FMC perde 25/28 (best-of-5), p<1e-4. Concorrono due cause **nella
  stessa direzione**: (a) E2-collapse (swarm poco divergente); (b) svantaggio
  strutturale — l'env in-order non riordina i gate, mentre SABRE sì (evidente su QFT).
- **Costo:** FMC **158–303× più lento** di SABRE (7 ms/decisione, 62–144 ms/circuito
  vs 0.4–0.5 ms). Nessun vantaggio di qualità che giustifichi il costo.

**Sintesi: né svolta né sconfitta totale — dipendenza dalla struttura.** FMC è
*competitivo solo sul regime che E2 marca come fit* (chain lineari, poco densi), e lì
solo alla pari, a costo molto maggiore. Su topologie/carichi densi perde.

## 5. Cosa servirebbe per un claim pubblicabile

1. **Front-layer DAG nell'env FMC** (riordino gate + commutazione) per togliere lo
   svantaggio strutturale e isolare il contributo puro del planner FMC — la QFT dice
   che questo pesa molto.
2. **Confronto vs SABRE completo (SabreLayout+routing, `optimization_level=3`)**, non
   solo routing a layout identità: baseline di produzione più forte.
3. **Device reali** (IBM heavy-hex 27/127q, coupling asimmetrici, error-aware cost) e
   **circuiti benchmark** (QAOA, chimica, QASMBench) — non solo random/QFT toy.
4. **Budget FMC molto più alto** (N,M grandi, o il core GPU `fragile`) per vedere se la
   ricerca stocastica supera l'euristica greedy quando si può permettere il compute —
   e quantificare il trade-off qualità/tempo (SABRE è ~300× più veloce).
5. **Obiettivo depth/error-rate**, non solo SWAP-count, dove il vantaggio "shallow"
   marginale osservato su linear5 potrebbe contare di più.

**Conclusione operativa:** il gate E2 si conferma un filtro low-cost affidabile
*prima* di investire in un adapter. Per il routing, allo stato attuale FMC-in-order
non batte SABRE; il percorso non è chiuso ma richiede l'env DAG + budget maggiore
prima di rifare le misure.
