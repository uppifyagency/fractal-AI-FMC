# P13 — risultato del proxy: la sparsità degrada la ricerca FMC? (2026-05-21)

Esecuzione dell'esperimento proxy pre-registrato in [`P13_DESIGN.md`](P13_DESIGN.md) §6.
Codice: [`p13_proxy.py`](p13_proxy.py). Dati grezzi: [`results/p13_proxy.json`](results/p13_proxy.json).

> **Esito in una riga.** Il criterio R2 pre-registrato (§6.5) **non è passato da
> alcuno schema genuinamente sparso** — ma l'esperimento ha *decomposto* R2 in due
> sotto-rischi che il design aveva erroneamente fuso, e sotto la metrica che conta
> davvero per E1-LLM il verdetto è **GO-conditional**. hP13-1 confermata in modo
> netto; hP13-0 (keystone) **non testata** per un difetto della griglia di rumore.

---

## 1. Cosa è stato testato

P13 chiede se un'interrogazione sparsa $O(N)$ del world-model degrada la ricerca
FMC (rischio R2). Il proxy emula i 3 schemi sparsi sul kernel-gridworld vero —
**nessun LLM** — e misura se la decisione FMC tiene rispetto al controllo FULL.

15 bracci × 2 $\alpha$ ($\{0.0, 0.1\}$) × 3 layout avversariali (`island`, `spur`,
`archipelago`, riusati da E1-robustness) × $n=30$ episodi. $N=64$, $M=20$, $\beta=1$.

**Kernel `fmc-core` invariato.** `proxy_plan` replica il loop pubblico di `plan()`
e chiama `fmc.core.{virtual_reward, clone_step, decide}` immutati. Sanity check
verificato all'avvio: `proxy_plan(FullSchema)` è **bit-identico** a `fmc.core.plan()`
su 5 seed. Il controllo FULL dà **0% morte su tutti e 3 i layout** — riproduce
esattamente E1-robustness. L'esperimento è valido.

---

## 2. Risultati (pooled sui 3 layout, n=90/cella)

| braccio | death% α=0 | death% α=0.1 | agreement | R2 (pre-reg.) |
|---|--:|--:|--:|:--|
| FULL (controllo) | 0.0 | 0.0 | 1.00 | — |
| S1 η=0 abs-preserved | 0.0 | 0.0 | 1.00 | PASS *(degenere — vedi §3)* |
| S1 η=1 abs-preserved | **0.0** | **0.0** | 0.33 | FAIL (agreement) |
| S1 η=2 abs-preserved | **0.0** | **1.1** | 0.31 | FAIL (agreement) |
| S1 η=4 abs-preserved | 6.7 | 10.0 | 0.28 | FAIL |
| S1 η=0 **abs-broken** | **32.2** | **30.0** | 0.72 | FAIL (death) |
| S1 η=1 abs-broken | 30.0 | 27.8 | 0.34 | FAIL |
| S1 η=2 abs-broken | 23.3 | 25.6 | 0.32 | FAIL |
| S1 η=4 abs-broken | 26.7 | 30.0 | 0.29 | FAIL |
| S2 distill B=50 | 8.9 | 8.9 | 0.25 | FAIL |
| S2 distill B=150 | 12.2 | 11.1 | 0.24 | FAIL |
| S2 distill B=400 | 5.6 | 7.8 | 0.24 | FAIL |
| S2 distill B=1125 | 4.4 | 1.1 | 0.30 | FAIL |
| S3 macro k=3 +stay | 5.6 | 2.2 | 0.16 | FAIL |
| S3 macro k=3 −stay | 5.6 | 1.1 | 0.16 | FAIL |

**Per-layout — `archipelago` è il layout discriminante** (random death lì = 31.7%;
su `island`/`spur` random ≈ 5%, quasi tutto sembra a posto). Il dato decisivo:

| braccio | island | spur | **archipelago** |
|---|--:|--:|--:|
| S1 η=0 abs-**broken**, α=0.1 | 10.0% | 0.0% | **80.0%** |
| S1 η=0 abs-**broken**, α=0.0 | 33.3% | 6.7% | **56.7%** |
| S1 η≤2 abs-**preserved** | 0.0% | 0.0% | 0.0%–3.3% |

---

## 3. Il verdetto pre-registrato — e perché va corretto

**Lettura stretta di §6.5.** Il criterio R2 pre-registrato chiede *death ≤ FULL+5pp*
**E** *agreement ≥ 0.85*. **Nessun braccio genuinamente sparso lo passa.** L'unico
"PASS" è `S1 η=0 abs-preserved` — ma η=0 (zero rumore) + abs-preserved (assorbenza
intatta) significa che il surrogato **è identico al kernel vero**: non è uno schema
sparso, è FULL ri-etichettato. La logica di verdetto del §6.5 ("GO-full se un
braccio S1 abs-preserved passa") si è attivata su un braccio degenere. **Quel
"GO-full" è un artefatto della logica di verdetto pre-registrata, non un risultato.**

Verdetto stretto, onesto: **nessuno schema sparso passa R2 come scritto.**

**Ma il criterio era mal specificato.** R2 pre-registrato fondeva due metriche —
death rate e decision-agreement — e i dati le **disaccoppiano nettamente**:

- **R2-survival** — la self-preservation emergente (E1, il death rate) sopravvive
  all'interrogazione sparsa? → **SÌ**, in modo robusto, *a condizione che il
  surrogato preservi la struttura assorbente*. S1 abs-preserved tiene death ≈ 0%
  fino a η=2; degrada con grazia a η=4 (~10%).
- **R2-fidelity** — FMC prende le *stesse decisioni* sotto interrogazione sparsa?
  → **NO**. La decision-agreement crolla a ~0.30 sotto qualunque rumore di segnale
  (η≥1), e a 0.14–0.20 per S3.

La self-preservation di E1 è preservata *mentre* le decisioni specifiche non lo
sono. Il criterio R2 le aveva legate; vanno separate.

---

## 4. Scorecard delle ipotesi pre-registrate

- **hP13-0** (un surrogato VR-rank-preserving preserva la decisione) — **NON
  TESTATA.** Difetto della griglia: il rumore $\eta$ Gaussiano salta da rango
  *perfetto* (η=0, Spearman 1.00) a rango *distrutto* (η≥1, Spearman ≈ 0.15, tick
  peggiore ≈ −0.04). Nessun braccio ha raggiunto un rango parziale-ma-alto, quindi
  il claim keystone — "rango preservato → decisione preservata" — non è valutabile.
  Causa: il segnale (manhattan ~0–28, coordinate 0–14) è a piccoli interi; η=1 è
  già rumore grande. *Follow-up pre-registrato*: sotto-griglia η ∈ {0.1, 0.25, 0.5}.
- **hP13-1** (S1 preserva E1 ⟺ il surrogato preserva la proprietà assorbente) —
  **CONFERMATA, in modo netto.** È il risultato più forte dell'esperimento.
  abs-preserved → death ≈ 0% (η≤2); abs-broken → death 23–32% pooled, **fino a
  80% su `archipelago` a α=0.1**. La proprietà assorbente è l'asse dominante, di
  gran lunga sopra il rumore di segnale.
- **hP13-2** (S2 ha un budget $B^*$ finito di convergenza) — **parziale.** Il death
  di S2 resta limitato (≤13%) ed è minimo a B=1125, ma non converge mai pulito a
  FULL: il campionamento con rimpiazzo lascia ~37% di chiavi non viste anche a
  B=1125 (fallback "stay"). Direzionalmente coerente, non risolta.
- **hP13-3** (il macro-menu preserva E1 se include un'azione non-letale) — **non
  discriminante.** S3 +stay e S3 −stay danno death quasi identici (5.6%/5.6% a
  α=0): su questi layout il menu goal-ranked esclude di rado *tutte* le azioni
  sicure. Inconcludente.

---

## 5. La scoperta affilata: la struttura assorbente è tutto

Il contrasto **abs-preserved vs abs-broken** è il segnale più netto dell'intero
esperimento, e va oltre "la self-preservation si rompe":

> Un world-model che **non rappresenta gli stati assorbenti** non si limita a
> perdere la self-preservation — la **inverte**. Su `archipelago` a α=0.1, S1
> abs-broken η=0 (nessun rumore di segnale, solo cecità all'assorbenza) muore
> all'**80%** — *peggio del random* (31.7%). FMC pianifica con sicurezza
> traiettorie *attraverso* la lava perché il suo modello del mondo gli dice che la
> lava è spazio libero. Un planner forte su un modello cieco all'assorbenza è
> attivamente letale.

Nota controintuitiva: `S1 η=0 abs-broken` ha agreement **0.72** — concorda con
FULL nel 72% delle decisioni — eppure muore al 30–80%. Le decisioni che
*differiscono* sono precisamente quelle vicino alla lava, dove l'assorbenza conta;
e sbagliarne il 28% concentrato lì è sufficiente a uccidere. Un surrogato può
essere d'accordo con la verità per la maggior parte del tempo ed essere comunque
mortale, se i suoi errori sono concentrati dove la posta è la sopravvivenza.

Per converso: abs-preserved tiene death ≈ 0% anche quando il rumore di segnale
distrugge l'ordinamento di VR (Spearman 0.15 a η=1). Il meccanismo VR-sink di
E1-robustness (la cella assorbente come pozzo di VR) è **degradato ma non distrutto**
dal rumore di segnale — la struttura conta, il segnale molto meno.

---

## 6. Caveat di onestà

- **Griglia di rumore troppo grossa** — vedi hP13-0. È il limite principale: il
  test keystone di P13 non è stato eseguito davvero.
- **Modello di errore ottimista** — il rumore è Gaussiano *non strutturato*
  (P13_DESIGN §8 lo aveva già segnalato). Un LLM sbaglia in modo *strutturato*;
  abs-broken è un primo proxy di errore strutturale (un bias sistematico:
  "la lava è libera"), e mostra che l'errore strutturale è quello che uccide.
- **n=30/layout** — adeguato per il segnale netto qui (le differenze sono enormi),
  ma le celle marginali (es. S1 η=4) hanno IC95 di Wilson larghi.
- **Dominio chiuso e piccolo** — gridworld 15×15. S2 e l'argomento di copertura
  non si trasferiscono ai domini aperti (P13_DESIGN §5, §8).
- **α ∈ {0, 0.1}** — il regime Common Sense / E1. Il regime goal-seeking ad α alto
  non è stato testato e ha bisogno di un test a sé (la fedeltà decisionale, qui
  scartata, vi conterebbe).

---

## 7. Verdetto P13 e implicazione per E1-LLM

**Verdetto: GO-conditional.** Non GO-full (nessuno schema sparso passa R2 come
scritto); non NO-GO (la self-preservation sopravvive in modo robusto). La
pre-registrazione aveva fuso due metriche; i dati le separano; sotto la metrica
che E1-LLM richiede davvero — *R2-survival*, "la self-preservation emerge?" — il
semaforo è verde, con **una condizione concreta e azionabile**:

> **Requisito pre-registrato per E1-LLM**: l'LLM-world-model deve **identificare
> correttamente gli stati terminali/assorbenti**. È un requisito raggiungibile —
> un LLM sa di norma riconoscere "sei morto" / "questo stato è terminale" — ed è
> *l'unica* cosa load-bearing. La fedeltà metrica del segnale di reward/distanza è
> di second'ordine.

La **decision-agreement esce dal gate di E1-LLM**: era la metrica sbagliata per la
domanda sulla self-preservation. Resta rilevante per agenti **goal-directed**
(α alto), che richiedono un test separato.

Conseguenza per la Congettura E: E1-LLM diventa eseguibile. Lo schema operativo
più semplice per E1-LLM *come test* (dominio chiuso) resta S2 — ma la lezione vera
non è "scegli S2", è "qualunque schema, garantisci che il modello del mondo sappia
cosa è terminale".

---

## 8. Prossimi passi

1. **hP13-0 ridone** — sotto-griglia η ∈ {0.1, 0.25, 0.5} su S1 abs-preserved, per
   testare davvero il claim keystone "rango preservato → decisione preservata".
2. **E1-LLM** — eseguibile; pre-registrare il requisito "absorbing-structure
   correttamente modellata" e misurarlo sull'LLM-world-model prima del test pieno.
3. **Test di fedeltà decisionale a α alto** — separato, per il regime goal-directed.

---

## 9. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/12_conjecture_e/p13_proxy.py     # ~1370 s (CPU)
```

Kernel `fmc-core` invariato — `proxy_plan` riusa solo le funzioni pubbliche; il
sanity check `proxy_plan(FULL) == fmc.core.plan()` è verificato a ogni run.

---

*Fine P13_RESULT.md. Verdetto pre-registrato: nessuno schema sparso passa R2 come
scritto; R2 si decompone in survival (preservata) + fidelity (no); E1-LLM =
GO-conditional sul requisito di struttura assorbente. hP13-1 confermata; hP13-0 da
rifare con griglia η fine.*
