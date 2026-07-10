# W8 — FMC su pianificazione dinamica ingannevole: il dato positivo mancante (con caveat onesti)

> **Data**: 2026-07-10 (Fase 3). **Direttiva utente**: "un dominio di pianificazione dinamica con incumbent debole dove FMC può davvero vincere".
> **Esito**: **trovato** — ma solo *dopo* che la review avversariale ha scoperto un bug di equità nel mio confronto iniziale. Su un env dinamico fortemente E2-fit (disp_ratio 7.66), con read-out d'azione appaiato ai baseline, **FMC-base pareggia l'MPC non-tuned e lo batte significativamente con il tuning teoricamente predetto** (α basso, β alto). Prima riga di evidenza controllata "E2-fit → FMC vince vs un planner standard".
> **Script**: [`w8_deceptive_nav.py`](w8_deceptive_nav.py) (env + planner), [`w8b_budget_sweep.py`](w8b_budget_sweep.py) (sweep budget, decode a maggioranza — *confondato*), [`w8_confirm.py`](w8_confirm.py) (confronto corretto), probe della review `w8_adv_*.py`.

---

## 0. Verdetto in una riga (corretto post-review)

Ho costruito **DeceptiveNav** (punto-massa con momentum, muro con varco laterale, reward ingannevole −distanza) come test *dinamico* per FMC. L'env **spara E2 forte** (disp 7.66). Il mio primo confronto dava un **negativo** (FMC perde contro random-shooting/CEM) — ma era **confondato da un'asimmetria di decode**: `core.plan` di FMC decide per **voto di maggioranza** (`decide`), mentre i baseline MPC restituiscono la prima azione della *singola* sequenza a reward finale massima (argmax). Non era mele-vs-mele. Con read-out **argmax appaiato**:
- **non-tuned** (α=β=1): FMC **pareggia** l'MPC (B=396: 0.775 vs 0.800, z=−0.27; B=576: 0.850 vs 0.825, z=+0.30) — la perdita "significativa" era per metà artefatto di decode;
- **tuned** nella direzione predetta dalla teoria (α=0.5, β=2 = meno reward-following, più esplorazione entropica): FMC **vince** — B=396: **1.000 vs 0.800, z=+2.98, p=0.003**; B=576: 0.950 vs 0.825 (z=+1.77, p=0.077); robusto a n=80 (z=+4.45) e su 3 basi di seed.

A read-out identico e budget identico, FMC-arg(0.5,2)=1.00 vs random-shooting 0.625 = **+0.375**: isola il valore del meccanismo FMC (resampling SMC + dispersione a entropia causale) *sopra* il puro random-shooting. **E2 aveva flaggato correttamente una head-to-head vincibile.**

---

## 1. L'env (e perché è un test equo)

**DeceptiveNav** (`w8_deceptive_nav.py`): stato $[x,y,v_x,v_y]$, 9 azioni (8 spinte + no-op), dinamica con drag/clip, **muro** a $y{=}5$ con **varco** largo 1.4 offset di `offset`. Start $(5,1)$, goal $(5,9)$. Reward **densa ingannevole** $r=-\text{dist(goal)}$ (+100 al goal): tira contro il muro; per passare bisogna allontanarsi lateralmente. `offset` = manopola di deception. Momentum + collisioni → dinamica divergente → **E2 spara** (disp 7.66 a $M{=}30$).

**Budget identico** $N\cdot M$ sim-call/decisione per tutti (verificato dalla review: FMC=N·M, rand-shoot=N·M, CEM≤N·M, greedy=K floor — nessun baseline gonfiato). Stessa reward, stesso env, **stesso read-out** (argmax) nel confronto corretto.

---

## 2. Risultati corretti (read-out argmax appaiato, offset 1.5, n=40, `w8_confirm.py`)

| B_dec (N,M) | rand-shoot | CEM | FMC-maj(1,1) *(confondato)* | FMC-arg(1,1) *(fair, non-tuned)* | FMC-arg(0.5,2) *(fair, tuned)* |
|---|---|---|---|---|---|
| 396 (36,11) | 0.625 | 0.800 | 0.425 (z=−3.44) | **0.775** (z=−0.27, pari) | **1.000** (z=+2.98, **p=0.003**) |
| 576 (48,12) | 0.825 | 0.800 | 0.600 (z=−2.22) | **0.850** (z=+0.30, pari) | **0.950** (z=+1.77, p=0.077) |

**Dipendenza dal budget** (dalla review, n=40–80): la vittoria tuned è a **B≥396**; pari a B=240; **perde** a basso budget (B=72, 128) — l'*opposto* dell'intuizione (poi ritrattata come rumore, §5) che l'edge fosse a basso budget. A budget moderato-alto FMC ha abbastanza walker/orizzonte perché resampling+dispersione paghino.

---

## 3. Diagnosi onesta: due cause, una di equità e una di tuning

1. **Asimmetria di decode (bug del confronto, non di FMC)**: la regola canonica di FMC (`decide`=voto di maggioranza della prima azione sui walker sopravvissuti) è un read-out *diverso* dall'argmax-della-migliore-sequenza dell'MPC. Confrontarle è ingiusto. Il fix (dare a FMC lo stesso argmax) recupera ~metà del deficit **senza toccare α,β**. *Nota*: non è "argmax = random-shooting" — a read-out identico FMC-arg(1,1)=0.775 vs random-shooting 0.625 (+0.15 dal solo resampling SMC) e FMC-arg(0.5,2)=1.00 (+0.375).
2. **Direzione di tuning predetta dalla teoria**: su un task ingannevole serve **meno** reward-following e **più** esplorazione. α basso (0.5) + β alto (2) = esattamente questo, ed è coerente con Teorema 4 (α_eff↓ → meno pressione selettiva) e col termine anti-collasso β (Teorema 3). α alto (5) è **catastrofico** (0.1–0.3 successo: la selezione spinge FMC *dentro* il muro), β=0 è sempre il peggiore per ogni α — la direzione è meccanicisticamente coerente, non un grid-search cieco.

**Caveat di scope (onesto)**: la vittoria *forte* usa tuning di FMC mentre i baseline sono a default. Mitiganti: (a) il fix del decode dà già un **pari** senza tuning; (b) a read-out identico e default, FMC batte comunque il random-shooting (+0.15); (c) la direzione del tuning è predetta, non cercata. Ma resta vero che per battere *CEM* serve il tuning. Non è "FMC domina"; è "FMC, comparato equamente, è competitivo e con tuning teoria-driven vince a budget moderato-alto".

---

## 4. Implicazione per il gate E2 (positiva)

Questo è il dato che mancava: un env **E2-fit** (disp 7.66) dove FMC-base, comparato equamente, **vince** vs planner standard. Quindi:
- **E2 ha flaggato correttamente una head-to-head vincibile** — un "fit" che si traduce in vittoria (con read-out equo + tuning). Rafforza il valore predittivo di E2.
- **Ritiro la mia bozza precedente "E2-fit necessaria-non-sufficiente basata su W8"**: era fondata sul negativo confondato. Con il confronto corretto FMC *non* perde qui. (La "non-sufficienza" può valere altrove — es. quantum-linear5 pari — ma **non** è supportata da DeceptiveNav.) PAPER_SYSTEMS aggiornato di conseguenza: W8 diventa un **caso-fit positivo**, non un controesempio.

---

## 5. Verdetto per la Fase 3

**Trovato un dominio di pianificazione dinamica dove FMC-base, comparato equamente, batte planner standard (random-shooting, CEM) a budget identico** — la prima evidenza controllata "E2-fit → FMC vince", coerente con la teoria (esplorazione entropica su reward ingannevole). Con i caveat: (i) la vittoria su CEM richiede tuning teoria-driven (α=0.5, β=2); (ii) è a budget moderato-alto (B≥396); (iii) l'incumbent è un planner generico, non un'euristica industriale matura. **Lezione di processo**: il negativo iniziale era un artefatto di decode che *io* avevo introdotto; solo la review avversariale l'ha scoperto — conferma che l'adversarial pass è load-bearing, in entrambe le direzioni.

---

## 6. Log della review avversariale (falsificatore Opus con mandato di *ribaltare il negativo*)

Verdetto: **RIBALTATO** (con qualificazione di budget). Il revisore ha scritto i propri probe (`w8_adv_*.py`), verificato la parità di budget (`w8_adv_stepcount.py`: FMC=N·M, CEM≤N·M — nessuna inflazione), verificato che la sua replica di `plan` è **bit-identica** a `core.plan` sul decode a maggioranza (0 mismatch/200 seed), e girato lo sweep decisivo α×β×decode (`w8_adv_tune.py`) + robustezza a n=80 su 3 basi di seed (`w8_adv_robust.py`) + il check del rumore n=12→60 (`w8_adv_noise.py`). I numeri chiave sono stati **riconfermati indipendentemente da me** in [`w8_confirm.py`](w8_confirm.py) (tabella §2). Correzioni integrate: il decode-fix, la direzione di tuning, e il ritiro del claim "necessaria-non-sufficiente da W8".

> ⚠️ **Onestà di processo (rumore)**: un pilot a n=12 mostrava FMC vincente a *basso* budget — **rumore**, ribaltato a n=40/60. La vittoria reale (tuned, argmax) è a budget *alto*, direzione opposta. Le success-rate binarie a n≤12 non sono affidabili.

---

*Fine W8. Script: `w8_deceptive_nav.py`, `w8b_budget_sweep.py`, `w8_confirm.py`, `w8_adv_*.py`. Ogni numero è prodotto dagli script (n=40, riconfermato n=80).*
