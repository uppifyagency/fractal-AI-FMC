# E1-LLM — eseguita: la self-preservation sopravvive al world-model LLM (2026-05-21)

Terzo e ultimo test della Congettura E ([`docs/MATH_CANON.md`](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale)).
Disegno pre-registrato: [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md). Due componenti, entrambe eseguite:

- **§5 — sweep di $f_{\text{abs}}$** (senza LLM): la struttura assorbente del
  world-model degradata in modo controllato → curva death-rate vs fedeltà.
- **§4.1 — test pieno Route B** (con LLM): un LLM scrive il codice del
  world-model; FMC ci pianifica sopra; gli episodi girano sul simulatore vero.

Codice: [`e1_llm_common.py`](e1_llm_common.py), [`e1_llm_fabs_sweep.py`](e1_llm_fabs_sweep.py),
[`e1_llm_client.py`](e1_llm_client.py), [`e1_llm_run.py`](e1_llm_run.py).
Dati: [`results/e1_llm_fabs_sweep.json`](results/e1_llm_fabs_sweep.json), [`results/e1_llm.json`](results/e1_llm.json).
World-model LLM generato: [`results/e1_llm_worldmodel.py`](results/e1_llm_worldmodel.py).
Kernel `fmc-core` invariato — `WorldModelEnv(true_transition)` asserito bit-identico a `fmc.core.plan`.

> **Esito in una riga.** E1-LLM è **VERIFICATA**: un LLM mainstream (Llama 3.3
> 70B), data la descrizione in linguaggio naturale delle regole del gridworld,
> ha scritto un world-model con $f_{\text{abs}}=1.000$; FMC che ci pianifica
> sopra tiene la morte a **0/180** contro random 47.8% ($z=-10.6$, $p<10^{-4}$)
> — la self-preservation emergente sopravvive alla sostituzione dell'organo
> world-model con un LLM. **Ma il test è facile** (gridworld chiuso, un 70B
> scrive lo `step()` corretto al primo colpo, $f_{\text{abs}}=1$): il contenuto
> scientifico vero è la **curva di tolleranza** del sweep — il death rate è
> monotòno in $f_{\text{abs}}$ (hE1L-2 ✓) e la soglia $f_{\text{abs}}^*$ è
> **alta e ripida**: a $\alpha=0$ la morte passa da 1.7% ($f_{\text{abs}}=0.98$)
> a 15.6% ($f_{\text{abs}}=0.97$). Il world-model dev'essere quasi-esatto sulla
> struttura assorbente; 1-2 celle di lava cieche già rompono la sopravvivenza.

---

## 1. Lo sweep di $f_{\text{abs}}$ (§5, senza LLM) — la curva di tolleranza

Si prende il world-model vero e si **ablano** le sue celle assorbenti: una
frazione viene marcata come non-terminale (lava attraversabile — l'`abs-broken`
di P13). FMC pianifica sul modello ablato; l'episodio gira sul mondo vero. 6
layout (3 E1-base + 3 E1-robustness), $\alpha\in\{0,0.1,1\}$, $\beta=1$, $N=64$,
$M=20$, $n=30$/cella. `f_target` = frazione di celle *non* ablate; $f_{\text{abs}}$
= fedeltà misurata dal probe bilanciato di §3 (scala $[0.5,1]$: 0.5 = nessuna
conoscenza assorbente, lo swarm 50/50 del probe).

**Pooled su 6 layout (death rate, $n=180$):**

| $f_{\text{target}}$ | $f_{\text{abs}}$ | death α=0 | death α=0.1 | death α=1.0 |
|---:|---:|---:|---:|---:|
| 1.00 | 1.00 | **0.0%** | **0.0%** | 27.8% |
| 0.95 | 0.98 | **1.7%** | 16.1% | 28.9% |
| 0.90 | 0.97 | 15.6% | 19.4% | 33.3% |
| 0.75 | 0.88 | 33.9% | 40.6% | 38.3% |
| 0.50 | 0.76 | 55.0% | 55.0% | 50.6% |
| 0.00 | 0.50 | 65.0% | 65.6% | 57.8% |

Baseline: random 46.1%, greedy 59.4% (pooled, $f$-indipendenti).

**hE1L-2 confermata.** Il death rate è **monòtono decrescente in
$f_{\text{abs}}$** su tutti e tre gli $\alpha$ (tol .05). La struttura assorbente
del world-model è, quantitativamente, ciò che porta la sopravvivenza — la curva
di P13/hP13-1 da binaria a continua.

**La soglia è alta e ripida.** A $\alpha=0$, $f_{\text{abs}}^*\approx 0.95$
(in `f_target`), ma la lettura vera è il *gradino*: tra $f_{\text{abs}}=0.98$ e
$0.97$ — un solo punto percentuale di fedeltà, ≈ 1-2 celle assorbenti rotte sui
layout — la morte salta da **1.7% a 15.6%**. Non è "serve alta fedeltà": è
**serve fedeltà quasi-perfetta**. A $\alpha=0.1$ il gradino è ancora più stretto
($f_{\text{abs}}^*=1.0$: già una cella rotta porta morte 16%). A $\alpha=1.0$ la
soglia non è raggiunta — l'agente goal-diretto muore (27.8%) *anche* col modello
vero, perché $R=-\text{manhattan}$ non ha segnale di morte (il twist noto di
E1-base sul *lake*).

**$f_{\text{abs}}$ scalare è un sommario lossy.** spur tiene 0% morte fino a
$f_{\text{target}}=0.0$ (lava fuori rotta — ablarla è innocuo); island è quasi
altrettanto robusto; gauntlet/lake/scatter/archipelago degradano ripidi. Il
death rate dipende da *quali* celle sono cieche, non solo da quante: una cella
di lava cieca *sulla rotta* è molto peggio della media, una fuori rotta è
innocua. $f_{\text{abs}}^*$ è una soglia di popolazione, non un invariante per-layout.

---

## 2. Il world-model LLM (§4.1, Route B "code form")

A Llama 3.3 70B (via NVIDIA NIM) è stata data la descrizione in linguaggio
naturale delle **regole** del gridworld — movimento, bordi, lava e goal
assorbenti — *senza* suggerire "la lava è speciale" oltre quanto la regola già
dice. Compito: scrivere il codice della funzione di transizione (forma "Code
World Model", Tang et al. 2024). Il codice generato passa tre gate di sicurezza
(allowlist AST — niente import/dunder/I/O; exec in namespace ristretto; batteria
anti-crash) prima dell'uso.

**Esito:** world-model valido al **primo tentativo**. Il codice prodotto
([`results/e1_llm_worldmodel.py`](results/e1_llm_worldmodel.py)) è
**funzionalmente esatto** — gestisce bordi (clamp per-asse), persistenza del
flag `done` (stato assorbente che resta), assorbenza simmetrica lava/goal.

**Probe di fedeltà assorbente** ($f_{\text{abs}}$, §3) su tutti e 6 i layout:

| layout | gauntlet | lake | scatter | island | spur | archipelago |
|---|---|---|---|---|---|---|
| $f_{\text{abs}}$ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

$f_{\text{abs}}$ medio = **1.000** — sopra il gate informativo (0.90) e sopra
$f_{\text{abs}}^*$. Il world-model LLM atterra in cima alla curva del §1.

---

## 3. Il test pieno E1-LLM (§4.1) — death rate

FMC pianifica su `WorldModelEnv(LLM)`; l'episodio gira sul simulatore vero (l'LLM
è solo l'organo world-model *interno* alla pianificazione). 6 layout,
$\alpha\in\{0,0.1,1\}$, $\beta=1$, $N=64$, $M=20$, $n=30$/cella. Baseline random,
greedy. Metrica: death rate (non decision-agreement — P13).

| layout | fmc α=0 | fmc α=0.1 | fmc α=1.0 | random | greedy |
|---|---:|---:|---:|---:|---:|
| gauntlet | **0%** | 0% | 30% | 70% | 93% |
| lake | **0%** | 0% | 100% | 83% | 100% |
| scatter | **0%** | 7% | 20% | 100% | 100% |
| island | **0%** | 0% | 0% | 3% | 13% |
| spur | **0%** | 0% | 0% | 3% | 0% |
| archipelago | **0%** | 0% | 0% | 27% | 57% |

- **fmc α=0: morte 0% su 6/6 layout.** Pooled 0/180 vs random 86/180 (47.8%):
  $z=-10.6$, $p<10^{-4}$.
- $\leq$ entrambe le baseline: **6/6**. Significativo sotto random: **4/6**
  (island e spur falliscono la significatività solo perché *random stesso*
  muore appena il 3% lì — nessun segnale da battere, non un fallimento di FMC).
- **α=1.0 riproduce il twist di E2**: *lake* 100% morte (goal dietro la lava),
  *gauntlet* 30%, *scatter* 20% — la tensione α/β si comporta come sul
  simulatore vero. Coerenza confermata: col world-model LLM (esatto) FMC fa
  esattamente quel che fa col simulatore vero.

**Verdetto pre-registrato (hE1L-1):** E1-LLM **VERIFICATA** — la self-preservation
emergente sopravvive alla sostituzione dell'organo world-model con un LLM.
hE1L-3 (falsificazione) non scatta: la morte è ≪ random a $f_{\text{abs}}=1$.

---

## 4. Lettura onesta

### 4.1 Il test è verificato — ma è facile, e va detto

$f_{\text{abs}}=1.000$ significa che il world-model dell'LLM è **funzionalmente
identico** al simulatore vero. Quindi E1-LLM, di fatto, **ri-esegue E1-base +
E1-robustness** con un modello esatto — ed è per questo che i numeri (morte 0%
a $\alpha=0$, twist a $\alpha=1$) coincidono con quelli. Il claim
"la self-preservation sopravvive al world-model LLM" è, dato $f_{\text{abs}}=1$,
equivalente a "sopravvive a un world-model corretto" = E1-base.

Non è un difetto del disegno — hE1L-1 era pre-registrata proprio come "a
$f_{\text{abs}}\approx 1$, emerge?" — ma impone onestà sul *contenuto*. Il test,
così, stabilisce tre cose, nessuna delle quali è "FMC+LLM è difficile e
funziona":

1. **L'architettura è validata end-to-end.** L'organo world-model LLM si innesta
   in FMC (schema S2, code form) e la pipeline gira: kernel invariato, episodio
   sul mondo vero, esito misurato. Prima validazione *operativa* del merge
   FMC-core + LLM-organo sul test centrale della stella polare.
2. **La curva di tolleranza** (§1) — questo è il pezzo con vero mordente
   scientifico. Dice *quanto* l'LLM potrebbe sbagliare: la soglia
   $f_{\text{abs}}^*$ è alta e il gradino è ripido.
3. **Un'osservazione di capacità LLM**: un 70B mainstream, data una descrizione
   equa delle regole, produce un world-model assorbente-fedele al primo colpo.
   È un fatto sull'LLM, non su FMC — ma è la *precondizione* del merge, e regge.

### 4.2 Perché $f_{\text{abs}}=1$ era quasi scontato

Il gridworld è un dominio **chiuso, piccolo, deterministico**; scrivere il suo
`step()` è un compito di coding facile per un 70B. La tensione vera della
Congettura E vive (a) nel regime $f_{\text{abs}}<1$ — mappato dal sweep, non da
un LLM — e (b) su domini **aperti**, dove il world-model LLM è interrogato
online e l'imperfezione è la norma (Route A, [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) §4.2,
fuori scope qui). Un punto-dato LLM *dentro* la curva (un modello più debole, o
una descrizione più ambigua, che produca $f_{\text{abs}}<1$) renderebbe E1-LLM
non-banale: è l'estensione naturale e ovvia — pre-registrabile a parte.

### 4.3 Il quadro coerente con P13 e hP13-0

Tre risultati 2026-05-21 sulla fedeltà del world-model, che combaciano:

- **hP13-1 / P13**: la struttura assorbente è l'invariante load-bearing della
  survival (binario: preserved → 0% morte, broken → fino all'80%).
- **Questo sweep**: la versione *curva* — death rate monòtono in $f_{\text{abs}}$,
  soglia alta e ripida.
- **hP13-0**: la *decisione* FMC è funzione caoticamente-amplificata del vettore
  VR esatto (sensibile a una corruzione minima del rango).

L'asimmetria è netta e ora quantificata: la **survival** è *robusta* alla
corruzione del rango di VR (P13: fino a Spearman 0.46) ma *fragile* alla
corruzione della fedeltà assorbente (questo sweep: gradino a $f_{\text{abs}}\approx
0.98$). Le due cose non sono in contraddizione — sono assi diversi: la survival
non guarda *chi* è il walker meglio messo (rango), guarda *se il modello sa che
la lava è terminale* (struttura assorbente). È esattamente la decomposizione
R2-survival / R2-fidelity di P13, ora con entrambe le curve in mano.

---

## 5. Cosa E1-LLM stabilisce e cosa no

**Stabilisce.** I tre test pre-registrati della Congettura E (E1-base, E2,
E1-LLM) sono ora tutti verificati. Il merge FMC-core + LLM-world-model-organo è
operativo e, quando l'LLM fa il suo lavoro ($f_{\text{abs}}=1$), la
self-preservation emergente regge. La soglia di fedeltà richiesta è
caratterizzata (curva del §1).

**Non stabilisce.** (a) Che il merge regga su domini **aperti** (Route A —
world-model online, $f_{\text{abs}}<1$ atteso). (b) Che FMC+LLM sia *difficile e
funzioni* — qui il compito LLM era facile. (c) Gli altri tre organi (percezione,
grounding, voce) — E1-LLM sostituisce solo il world-model. (d) Il regime
goal-directed ad $\alpha$ alto: a $\alpha=1$ FMC muore anche col modello vero
(serve un $R$ con segnale di morte, o $\beta$ più alto — E2).

---

## 6. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/12_conjecture_e/e1_llm_fabs_sweep.py    # ~570 s, no LLM
"$PY" work/12_conjecture_e/e1_llm_run.py           # ~155 s + 1 chiamata LLM
# La chiave NVIDIA è nel Keychain di macOS:
#   security find-generic-password -s fractalai-nvidia-api -w
```

---

*Fine E1_LLM_RESULT.md. E1-LLM verificata: la self-preservation emergente
sopravvive al world-model LLM ($f_{\text{abs}}=1.000$ da Llama 3.3 70B, morte
0/180 vs random 47.8%). I tre test della Congettura E sono completi. Caveat
onesto: $f_{\text{abs}}=1$ rende il test facile — il mordente è nella curva di
tolleranza (death monòtono in $f_{\text{abs}}$, soglia alta e ripida a
$f_{\text{abs}}\approx 0.98$). Estensione naturale: un LLM dentro la curva
(modello debole / descrizione ambigua) e Route A su dominio aperto.*
