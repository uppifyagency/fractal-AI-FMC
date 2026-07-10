# W7 — FMC su landscape a rugosità sintonizzabile (NK di Kauffman): dove FMC-base *non* funziona, e perché E2 lo sapeva

> **Data**: 2026-07-10 (continuazione night_2026-07-09, Fase 2 applicativa).
> **Direttiva utente**: "applicazioni breakthrough con simulazione reale, test, adversarial review per la pubblicazione."
> **Regola d'oro rispettata**: smoke test E2 **prima** di ogni investimento (lezione W4).
> **Script** (seed derivati da 20260710/11): [`w7_nk_fmc.py`](w7_nk_fmc.py) (N=16), [`w7b_nk_hard.py`](w7b_nk_hard.py) (N=20 regime duro + FMC-EA + E2 reward-coupled). fmc-core in-repo, numpy 2.2.6.

---

## 0. Verdetto in una riga (onesto, come da protocollo)

**Non è un breakthrough — ed è un risultato pulito.** Su landscape NK (il modello canonico di rugosità sintonizzabile), **FMC-base non batte greedy-restart né simulated annealing** a budget di valutazioni identico, in *nessun* regime di rugosità $K$, con *due* incarnazioni di FMC (planner e EA generazionale). Negativo **robusto** alla review avversariale: tuning aggressivo di $\alpha,\beta,N_{\rm walk},M$, i "vincitori" in-sample evaporano out-of-sample; budget equo in entrambi i sensi (eval totali *e* unici); nessun bug nell'ottimo/baseline. Il contributo pubblicabile: **una frontiera di applicabilità** — FMC-base *sembra* essere un pianificatore per sistemi dinamici a dinamica rugosa, **non** un ottimizzatore combinatorio black-box (interpretazione meccanicistica, §3). ⚠️ **Nota di onestà (post-review, §6)**: su NK il gate E2 **non "predice" il no-fit** — il suo `disp_ratio` in coordinate di configurazione è *strutturalmente* bloccato sotto 3 per *qualunque* landscape a flip statici (non spara mai), quindi qui è vacuo come predittore *entro-dominio*. Il valore predittivo di E2 è **cross-dominio** (separa env dinamicamente-divergenti da env che collassano), non su NK.

---

## 1. Perché NK, e cosa testa

Il modello **NK di Kauffman (1989)** è la classe canonica di landscape a rugosità *sintonizzabile*: $N$ geni binari, la fitness del gene $i$ dipende da sé stesso + $K$ vicini epistatici, fitness totale = media delle $N$ contribuzioni. $K{=}0$ → picco unico liscio (greedy risolve); $K{=}N{-}1$ → massimamente rugoso (molti ottimi locali). È esattamente ciò che i due spike W4 dicevano mancasse (landscape piatti/monotoni). Permette il test causale che i 5 domini della notte non permettevano: **fissare tutto e girare solo la manopola di rugosità**.

**Fairness (critica per non barare, come in W4):** ogni ottimizzatore riceve lo **stesso budget** $B=H\cdot N_{\rm walk}\cdot M$ di valutazioni di fitness. Metrica = **rapporto di approssimazione** trovato/ottimo-globale, con l'ottimo globale calcolato per **forza bruta** su tutti i $2^N$ stati (vettorizzato). Baseline: random, **greedy steepest-ascent con restart**, **simulated annealing** (raffreddamento geometrico). FMC usato come ottimizzatore = miglior stato visitato dallo swarm (traccia il vero meccanismo relativize→virtual-reward→cloning+diversità).

---

## 2. Risultati

### 2.1 Regime facile (N=16, $2^{16}$ stati, budget 4608 ≈ 7% dello spazio)

| K | loc_opt% | disp_ratio (E2) | random | greedy | SA | FMC | FMC−greedy |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 2.11 | 0.989 | **1.000** | 1.000 | 1.000 | +0.000 |
| 2 | 0.1 | 2.11 | 0.964 | **1.000** | 1.000 | 0.986 | −0.014 |
| 8 | 1.1 | 2.11 | 0.969 | **0.995** | 0.988 | 0.968 | −0.027 |
| 15 | 5.7 | 2.11 | 0.959 | 0.963 | 0.938 | 0.970 | +0.007 |

### 2.2 Regime duro (N=20, $2^{20}$≈1.05M stati, budget 9600 ≈ 0.92% dello spazio)

| K | loc_opt% | disp_cfg | disp_rew | random | greedy | SA | FMC | FMC-EA | EA−greedy |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 2.31 | 2.55 | 0.972 | **1.000** | 1.000 | 1.000 | 0.997 | −0.003 |
| 2 | 0.0 | 2.31 | 2.03 | 0.962 | **1.000** | 1.000 | 0.991 | 0.992 | −0.008 |
| 4 | 0.0 | 2.31 | 1.60 | 0.946 | **1.000** | 0.999 | 0.989 | 0.978 | −0.022 |
| 8 | 0.4 | 2.31 | 1.23 | 0.930 | **0.988** | 0.970 | 0.981 | 0.953 | −0.035 |
| 12 | 1.4 | 2.31 | 1.11 | 0.927 | **0.958** | 0.953 | 0.923 | 0.934 | −0.024 |
| 19 | 4.8 | 2.31 | 1.06 | 0.926 | 0.903 | 0.874 | 0.895 | 0.905 | +0.002 |

**Lettura.**
1. **La manopola di rugosità funziona**: `loc_opt%` (frazione di ottimi locali a 1-flip) sale monotonicamente 0%→4.8% con $K$ — comportamento NK canonico, la simulazione è corretta.
2. **greedy-restart domina o pareggia ovunque.** FMC (planner) e FMC-EA (generazionale) sono **sotto greedy** in tutti i $K$ tranne il pari a $K{=}19$ (dove il landscape è quasi-random e perfino `random`=0.926 batte greedy=0.903). Due incarnazioni di FMC, due regimi, tutto lo sweep: **nessun edge**.
3. **E2: no-fit strutturale, non predizione (correzione post-review).** `disp_cfg` (osservazione = configurazione) è **costante ≈2.3 per ogni K** — cieco alla rugosità perché i flip casuali diffondono i bitstring in modo identico (distanza euclidea su vettori binari = $\sqrt{\text{Hamming}}$, fitness-blind: dimostrabile, non artefatto). `disp_rew` (reward-coupled) **decresce** con K (2.55→1.06). **In nessuna coordinata `disp_ratio` raggiunge 3.0.** ⚠️ Ma questo *non* è una "predizione corretta" del fallimento di FMC entro NK: essendo `disp_cfg` strutturalmente $<3$ per *ogni* landscape a flip statici, E2 non avrebbe *potuto* dire "fit" per nessuna istanza NK, difficile o facile. Il gate **non spara mai** → è vacuo come discriminante *entro-dominio*. È coerente con l'assenza di edge di FMC, ma il contenuto predittivo reale di E2 è **cross-dominio** (Rocket/control divergono, quantum-grid/synthesis collassano), non su NK.

---

## 3. Perché FMC-base non funziona qui (interpretazione meccanicistica, non causalità testata)

> ⚠️ Questa è un'**interpretazione** coerente con il meccanismo e con i priori del progetto, **non** una causalità dimostrata in questo studio: non esiste (qui) un problema statico in cui FMC *vince* da contrastare, quindi "planning≠optimization" resta plausibile-ma-non-testato-in-isolamento.

NK è un problema di **ottimizzazione statica**, non di **pianificazione dinamica**. La "divergenza" che E2 rileva e da cui FMC trae vantaggio è **dinamica** — la separazione caotica di traiettorie sotto una dinamica forward ricca (Atari, control, Craftax). Flippare bit su un landscape di fitness *statico* non genera dinamica caotica: due sequenze di flip diverse raggiungono stati diversi, ma non c'è un *sistema dinamico* la cui sensibilità alle condizioni iniziali FMC possa sfruttare. Senza divergenza dinamica, `relativize`→virtual-reward→cloning si riduce a una selezione fitness-proporzionale con pressione di diversità — cioè un EA mutation-only, che su NK è **dominato** da steepest-ascent-con-restart e SA (fatto noto della letteratura NK). FMC **non ha ricombinazione**, quindi nemmeno il vantaggio dei GA a K intermedio.

Questo raffina la survey W1D, che aveva marcato "logic synthesis" e "chip placement" (combinatori-sequenziali) come FMC-fit forte: il regime **combinatorio-statico** non è dove FMC-base vince. Coerente con W4B (logic synthesis fallì il proprio E2) e ora con NK sotto condizioni controllate.

---

## 4. Il contributo (onesto e pubblicabile)

Non "FMC batte X". Piuttosto:
- **Un negativo rigoroso e robusto**: FMC-base (2 varianti) non batte greedy-restart/SA su NK a budget identico, sotto controllo (manopola di rugosità), sopravvissuto a un tentativo avversariale di farlo vincere via tuning. Un dominio combinatorio-statico in meno tra i candidati "breakthrough".
- **Frontiera di applicabilità** (interpretazione, §3): il regime combinatorio-*statico* non è dove FMC-base vince — raffina la survey W1D che aveva marcato logic-synthesis/chip-placement come fit-forte.
- **Raffinamento operativo di E2**: su problemi statici il `disp_ratio` in coordinate di configurazione è *strutturalmente* cieco alla rugosità (i flip diffondono identicamente); nemmeno le coordinate reward-coupled fanno scattare il gate. Implicazione onesta: **E2 va misurato in coordinate di divergenza *dinamica*; su un problema statico non ha potere discriminante** (non spara mai) — quindi il suo uso come screen deve essere ristretto ai domini con dinamica forward.

**Cosa NON è emerso**: un dominio in cui E2 dice "fit" *e* FMC-base batte un incumbent forte. Attraverso quantum, logic-synthesis, plasma e ora NK, il pattern è "at-par-at-best". Il candidato onesto per un vero edge resta un dominio di **pianificazione dinamica** con incumbent debole — non un problema di ottimizzazione statica.

---

## 5. Test di correttezza (soundness)

- **Ottimo globale**: forza bruta vettorizzata su tutti i $2^N$ stati (denominatore esatto del rapporto di approssimazione); coerente con `NK.fitness` per costruzione (stesso ordinamento di bit MSB-first — verificato).
- **Budget identico**: contatore `Budget` condiviso conta *ogni* chiamata di fitness per tutti i metodi (cache-hit inclusi) → confronto equo.
- **Manopola di rugosità validata**: `count_local_optima` sale con K come atteso dal modello NK.
- **Determinismo**: seed espliciti per ogni istanza/metodo; nessun `Math.random` non-seedato.

---

---

## 6. Log della review avversariale (falsificatore Opus con mandato di *ribaltare* il negativo)

Verdetto: **CONFERMATO**, con due qualificazioni di onestà (già integrate sopra). Il revisore ha scritto i propri probe (`w7_adv_*.py`) e ha provato a far vincere FMC:

- **(A) Equità del budget**: contando gli stati *unici* valutati — random 0.996, greedy 0.92–0.93, FMC **0.84**, FMC-EA 0.91, SA **0.46–0.57**. Il metodo che spreca più budget in cache-hit è **SA (una baseline)**, non FMC. Ri-run con budget in eval *unici*: margini quasi identici → lo svantaggio ~9% di FMC non chiude il gap. Attacco morto.
- **(B) Tuning per far vincere FMC**: sweep $\alpha\in\{0.1..10\}$, $\beta\in\{0,0.5,1,2\}$, 6 trade-off swarm/orizzonte, entrambe le varianti. Alcune celle battevano greedy *in-sample* (K=12 α2β1 +0.029) → ma su **16 seed out-of-sample** appaiati **ogni** config perde a K=8 e K=12 (best −0.0125, t=−2.4; peggiori t=−6). Winner's curse: il +0.029 era un massimo-di-griglia che si ribalta a −0.0125 su dati held-out. **Nessun setting ragionevole fa vincere FMC.**
- **(C) Baseline**: greedy steepest-ascent+restart corretto; SA con cooling sano. Nessuna baseline gonfiata a favore di FMC.
- **(D) Ottimo brute-force**: 3 calcoli indipendenti (loop, vettorizzato, `itertools.product`) coincidono a 1e-12, bit-order MSB-first 0 mismatch. Denominatore corretto.
- **(E) E2 "cieco a K"**: riprodotto da zero (disp_cfg costante 2.295 a N=64, senza il modulo E2) — reale, non artefatto.
- **(F) Framing**: le due correzioni che ho integrato (§0, §2, §3) vengono da qui — "E2 predice" era vacuo entro-NK; "planning≠optimization" è interpretazione, non causalità testata.

Nota del revisore sul proprio harness: il budget in eval-unici può far girare `opt_fmc` a vuoto quando lo swarm smette di scoprire stati nuovi (cache-hit gratis) — aggiunto uno stall-guard a 60·B; quello spin è di per sé un segnale che l'esplorazione di FMC è clusterizzata.

---

*Fine W7. Script: [`w7_nk_fmc.py`](w7_nk_fmc.py) (~73 s), [`w7b_nk_hard.py`](w7b_nk_hard.py) (~68 s) + probe indipendenti della review `w7_adv_*.py`. Ogni numero è prodotto dagli script.*
