# W1B — Mappa rigorosa dell'evidenza empirica FMC

> **Ruolo**: research associate scettico / falsificatore. Obiettivo: distinguere
> ciò che è *dimostrato con rigore statistico* da ciò che è *direzionale* da ciò
> che è *aneddotico*, con citazione `file:riga` per ogni claim.
> **Data analisi**: 2026-07-09. **Autore**: agente W1B (empirico).

---

## 0. Sintesi tassonomica (semaforo)

| Risultato | Verdetto | Perché |
|---|---|---|
| exp17 >> baseline v4 su Craftax-Classic-Symbolic | 🟢 **SOLIDO** | test appaiato a 18 seed, Wilcoxon p=0.0019, Cohen d_z=0.74 |
| exp17 = **50.95% / "human-expert 50.5%"** (valore puntuale) | 🟡→🔴 **FRAGILE come framing** | metrica aggregata con CI95 ±11-13pp; confronto NON like-for-like con il numero umano; per-episodio la media è 30% |
| Conjecture D (compounding tier-stack) | 🟡 **DIREZIONALE** | ablation LOO coerente ma non-significativa per-tier; "compounding" non distinto da shaping additivo; falsifiche a n=1 |
| Boxing knockout FMC (Atari) | 🟢 **SOLIDO ma su 1 gioco cap-bound** | 5/5 seed a +100; tabella 50-giochi mai eseguita |
| FMC >> MCTS (D2) | 🟡 **DIREZIONALE** | Boxing n=3, MCTS non tunato, 1 gioco, episodio troncato |
| RAM > IMG (Atari) | 🔴 **ANEDDOTICO / potenzialmente falsificato** | n=2, gioco cap-bound, segnale opposto (RGB più veloce) |

---

## 1. Il risultato faro: exp17 = 50.95% Crafter zero-training

### 1.1 Come è misurato — due metriche distinte, riportata quella alta

Esistono **due "Crafter score" diversi** e la differenza è il singolo punto
metodologico più importante di tutto il corpus:

1. **Aggregato (pooled)**: si calcola il success-rate ρ_j di ogni achievement
   su TUTTI gli episodi in pool, poi si applica la formula geometrica di Hafner
   Φ = exp(mean_j log(1+100·ρ_j)) − 1. → **50.60%** (n=18)
   (`results/exp17_30seed.json`: `crafter_score=50.6, n_seeds_completed=18`).
2. **Per-episodio**: si applica Φ a ogni episodio poi si media.
   → **30.04% ± std 20.37** (`results/statistical_validation.json:24-26`).

Il numero-faro pubblicizzato (50.95% a n=11 in `results.tsv:19`; 50.60% a n=18)
è **sempre l'aggregato**. Questa è la convenzione ufficiale del leaderboard
Crafter (Hafner 2021), quindi la scelta è *difendibile*, ma ha conseguenze pesanti
(vedi 1.3, 1.5).

### 1.2 Seed, budget e intervalli di confidenza

- **Target 30 seed, completati 18.** 12 seed hanno saturato il wall-budget
  (`sec_results.md:3-8`; `peer_review_self.md:153-160` "C.1 Sample size").
  Il tetto è dovuto a episodi FMC lunghi su alcuni seed (fino a >1h/seed).
- **CI95 sull'aggregato = [36.85%, 59.46%]** — semiampiezza ~±11-13pp
  (`results/statistical_validation.json:29-30`; `sec_results.md:15`). Questo è
  **10× più largo** del target "≤±1.0pp" preregistrato in
  `PAPER_HANDOFF.md:176`. L'ammissione è esplicita in `peer_review_self.md:157-160`.
- **La colonna `ach_ci95` in `results.tsv` NON è il CI del Crafter score**: è il
  CI95 sul *numero medio di achievement* (`evaluate.py:56,61`;
  `prepare_craftax.py:91-92,147`). Quindi tutte le righe di `results.tsv`
  (es. exp17 "ci95=1.93") **non riportano incertezza sul valore Φ pubblicizzato**.
  Il CI vero su Φ compare solo nei file Gap-1/Gap-2 ed è largo.

**Nota di robustezza a favore**: il punto-stima aggregato è *stabile* fra i due
banchi di seed (50.95% a n=11 → 50.60% a n=18). Se fosse puro rumore selezionato,
ci si aspetterebbe una forte regressione verso ~40%. La stabilità suggerisce che
il punto-stima ~50% non è un artefatto di cherry-picking del banco di seed. **Ma**
la larghezza del CI resta reale: l'affermazione statisticamente onesta è
"aggregato ~45-55%, nettamente sopra v4 ~29%", NON "esattamente 50.95%".

### 1.3 Perché l'aggregato è alto: dipendenza da pochi seed-eroe

I 18 punteggi per-seed (`statistical_validation.json:40-59`) sono fortemente
asimmetrici a destra: valori 4.36, 5.61, 9.05, ... fino a **80.89** e **65.39**.
L'aggregato geometrico premia gli achievement rari purché compaiano in *qualche*
episodio: un diamante raccolto in 1-2 episodi su 18 alza Φ aggregato molto più di
quanto la sua rarità (ρ≈0.056) suggerirebbe intuitivamente. Il bootstrap che
ricampiona i seed con reinserimento **fa crollare** l'aggregato quando i pochi
episodi-eroe non vengono pescati → ecco il CI ±11pp. **Conclusione: il 50.6%
è trainato dalla coda destra (pochi episodi che completano la catena iron/diamond),
non dalla prestazione tipica** (mediana per-episodio ≈ 27%).

### 1.4 Significatività vs baseline — QUI il risultato regge

Contrariamente al framing "human-expert", la parte **davvero solida** è il
confronto vs baseline v4:

- **Test appaiato a 18 seed** (v4 per-seed estratti da run_007,
  `statistical_validation_paired.json`):
  - exp17 aggregato 50.60% vs v4 28.46% → **Δ +22.15pp**, bootstrap CI95
    [8.81, 32.12], **p(Δ≤0)=0.0001** (`:24-34`).
  - Per-episodio: exp17 30.04% vs v4 15.35% (≈ **raddoppio**).
  - **Wilcoxon appaiato W=124, p=0.00189**; t-appaiato p=0.00295;
    **Cohen d_z=0.74** (`:37-41`).
- **Caveat**: `diff_per_seed` (`:82-101`) mostra che su **3/18 seed exp17 è
  PEGGIO** di v4 (−5.79, −5.24, −13.50) e su 2 seed identico. L'effetto è reale
  ma non uniforme. d_z=0.74 è medio-grande, **ben lontano** dal "d>5" fantasioso
  predetto in `PAPER_HANDOFF.md:220`.

Questo test appaiato **è pubblicabile as-is** come "il reward shaping proposto
migliora significativamente FMC vs baseline non-shaped, su Craftax-Classic-Symbolic".

### 1.5 Il confronto con "human-expert 50.5%" NON è like-for-like

Tre problemi gravi, tutti ammessi (parzialmente) nel self-review:

1. **Ambiente diverso.** exp17 gira su **Craftax-Classic-Symbolic-v1**
   (osservazioni simboliche, reimplementazione JAX). Il 50.5% umano è
   **Crafter-original** (Hafner 2021, osservazioni a pixel, giocatori umani reali).
   Ammesso in `peer_review_self.md:204-209` (C.7 "Single benchmark family").
2. **Test diretto sull'ambiente giusto → 3.77%.** Quando exp17 viene realmente
   portato su Crafter-original (`sec_crafter_smoke.md:29-31`), ottiene **3.77%**
   (vs 3.62% baseline, Δ+0.15pp, n=3), a compute 50× inferiore. Loro sostengono
   che il full-compute recupererebbe i ~50pp, ma **questo non è mai stato
   testato** (`sec_crafter_smoke.md:49-55`). L'unico datapoint sull'ambiente
   dove vive il numero umano dà 3.77%, non 50%.
3. **Nessuno studio umano proprio** (`peer_review_self.md:196-201`, C.6): il
   confronto è solo su una metrica presa da un'altra pubblicazione, su un altro
   ambiente. Confronto FMC (zero-training, 10.24M campioni di *inferenza* per
   episodio) vs umano (real-time, zero simulazione) è per natura apples-to-oranges.

**Verdetto 1.5**: la frase "matches/beats human-expert" è una **coincidenza di
punto-stima fra due numeri non commensurabili**, dentro una banda di ±13pp.
È il claim più a rischio-reviewer dell'intero lavoro.

### 1.6 Multiple comparisons / garden of forking paths

- La traiettoria exp03→exp17 è un **hill-climb greedy** su 23 esperimenti, gate
  "keep se aggregato migliora" (`README.md:11-16`; `HANDOFF.md:251-258`).
- Ogni stima aggregata durante il loop era a **n≈9-13** (`results.tsv`), con CI
  su Φ ancora più largo di ±11pp. Selezionare il massimo su ~23 estrazioni
  rumorose → **winner's curse** (bias verso l'alto).
- **Baseline mobile**: il gate confrontava con 29.27 storico, ma il baseline
  in-sessione era 27.44 (`results.tsv:2`) e quello appaiato 28.46
  (`statistical_validation_paired.json:27`). Il gate "+1pp" era applicato contro
  un riferimento che oscilla di ~2pp per solo rumore di seed. L'agente stesso
  segnala che il gate è "too generous" (`HANDOFF.md:166-172`).
- **Numero di seed variabile per esperimento** (9-13): dato che il budget è a
  parete-fissa, config più lente (N grande) completano meno seed E ottengono
  punteggi peggiori → **confondimento seed-count × config** (es. exp08 N=1024 a
  n=2; exp21 N=768 a n=9). I confronti fra esperimenti NON sono appaiati né a
  n costante.

**Mitigazione parziale**: l'endpoint (exp17) è stato ri-validato su un banco di
seed esteso (n=18) con stabilità del punto-stima → il *winner's curse sull'endpoint*
è limitato. Ciò che resta non-rigoroso è la **micro-traiettoria** (ogni +1pp).

---

## 2. Conjecture D — chain-tier compounding amplification

### 2.1 Traiettoria exp03→exp17

Numeri (`HANDOFF.md:16-18`, `PAPER_HANDOFF.md:98-111`, `results.tsv:6,12-14,18-19`):

| stadio | Φ agg | Δ | n |
|---|---|---|---|
| exp03 (ach-fire tier-weighted) | 40.96 | — | 13 |
| exp09 (+ iron inv) | 42.89 | +1.93 | 13 |
| exp10 (+ stone inv) | 44.14 | +1.24 | 13 |
| exp11 (+ wood inv) | 45.94 | +1.80 | 12 |
| exp16 (iron-ach 150→200) | 50.65 | +4.71 | 11 |
| exp17 (+ gateway-ach push) | 50.95 | +0.30 | 11 |

**Problemi statistici della traiettoria:**
- I passi +1.24 e +1.80 e +0.30 sono **dentro o comparabili al rumore** (CI su Φ
  ~±11pp a questi n). Solo il salto exp16 (+4.71pp) è di ampiezza plausibilmente
  reale, e comunque non testato con CI.
- exp17 su exp16 = **+0.30pp**: quasi certamente rumore. Che exp17 (non exp16) sia
  l'"headline" è arbitrario.
- **exp17=exp18=exp19 = 50.9524% IDENTICI a 4 decimali** (`results.tsv:19-21`;
  `HANDOFF.md:71-72,136-138`). Interpretato come "attrattore strutturale /
  argmax saturato". Lettura più parsimoniosa: sugli **11 seed fissi** i tweak di
  reward non hanno cambiato *nessuna* traiettoria → sono esperimenti a
  **informazione zero**, non conferme indipendenti. Non provano "saturazione
  strutturale", provano invarianza su un piccolo set fisso.

### 2.2 Ablation leave-one-out (Gap 3) — il test più pulito

Da `results/gap3_summary.json`, partendo da exp17 (~50.6%):

| ablation | Φ | n | Δ da exp17 |
|---|---|---|---|
| L1 (− iron inv) | 42.64 | **1** ⚠️ | −8 (INVALIDO, n=1) |
| L2 (− stone inv) | 44.32 | 30 | −6.3 |
| L3 (− wood inv) | 42.90 | 30 | −7.7 |
| L4 (− iron ach) | 43.31 | 30 | −7.3 |
| L5 (− gateway ach) | 45.84 | 30 | −4.8 |

- **Direzione coerente**: rimuovere qualunque tier abbassa il punteggio → coerente
  con "tutti i tier contribuiscono".
- **MA**: (a) **L1 è a n=1** (`gap3_summary.json:6`, wall 9595s, 0.03 dec/s → un
  singolo episodio bloccato; ammesso in `peer_review_self.md:170-175`, C.3);
  (b) **nessun CI o test** sui Δ delle ablation; i Δ (~5-8pp) sono comparabili
  alla semiampiezza CI dell'aggregato (~±8-11pp a n=30) → **la significatività
  per-tier NON è stabilita**.
- **Il difetto concettuale più serio** (ammesso in `peer_review_self.md:69-77`,
  A.4): l'ablation monotòna è coerente con *qualsiasi* shaping additivo monotono;
  **non identifica univocamente il "compounding"**. Anzi la traiettoria additiva
  exp03→exp11 (+1.93+1.24+1.80 = +4.97 totale) è approssimativamente **additiva**,
  non evidentemente super-additiva. Il termine "compounding amplification" rischia
  di sovra-descrivere ciò che è "shaping additivo monotono con sweet-spot sulla
  magnitudine".

### 2.3 Falsifiche dei multipli >1.4×

Casi di collasso (`HANDOFF.md:216-226`, `PAPER_HANDOFF.md:128-137`):
- exp04 (η≈6.67, diamond 1000): −4pp, n=11
- exp15 (diamond ×1.67 → 500): HUNG 8h, **n=0**
- exp22 (α 1.0→1.5): −23.7pp, n=13

**Problemi**:
- Sono **1 punto per caso** (`peer_review_self.md:18`, A.1 "n=1 per failure case").
- Enorme **gap non testato fra 1.4× e 6.67×**: si sa solo che 1.33× funziona
  (exp16) e 6.67× collassa (exp04). Il bound superiore "sweet-spot ≤1.4×"
  (`PAPER_HANDOFF.md:130-132`) è **interpolazione**, non misurato. exp15 (1.67×)
  non ha prodotto un punteggio, è andato in hang → non è una falsifica pulita del
  reward, è una patologia di processo.
- exp22 (α=1.5) è un confondente diverso (parametro di cloning, non magnitudine di
  reward) → falsifica il Teorema 2 (α>1 collassa), non direttamente il bound sui
  multipli.

**Verdetto Conjecture D**: 🟡 DIREZIONALE. Solido come "il reward shaping tier
proposto è necessario e ogni componente contribuisce (direzione)"; **non solido**
come "legge di compounding con bound quantitativo [1.2,1.4]". La replica
cross-benchmark (Gap 4) — che la renderebbe una *legge* — è solo uno smoke test
nullo (`sec_crafter_smoke.md`, n=3, Δ+0.15pp, 50× meno compute).

---

## 3. FMC vs MCTS (discrepanza D2)

Da `work/09_fmc_vs_mcts_replication/REPORT.md:57-66`:

| algo | B | seed | mean | std | min | max |
|---|---|---|---|---|---|---|
| FMC | 80 | 3 | +91.3 | 10.0 | 80 | 99 |
| MCTS | 80 | 3 | −5.0 | 5.0 | −10 | 0 |
| FMC | 240 | 3 | +100.0 | 0.0 | 100 | 100 |
| MCTS | 240 | 3 | −5.0 | 5.0 | −10 | 0 |

**Cosa regge**: a parità di budget campioni/azione, su Boxing, FMC vince e MCTS
gioca a livello random. Segnale direzionale forte e nella direzione predetta dalla
teoria (UCB1 collassa a uniforme se nessuna foglia è raggiunta).

**Cosa NON regge per un claim pubblicabile** (caveat elencati dal report stesso,
`REPORT.md:68-87`):
1. **n=3** (protocollo P0 richiede n=10 + bootstrap CI95).
2. **Un solo gioco**, Boxing = il più facile della tabella paper §5.1.1.
3. **MCTS non tunato** (c=√2 canonico ma non ottimizzato per la sparsità −1/0/+1;
   niente sweep di c né rollout_depth).
4. **max_actions=200** tronca l'episodio (Boxing dura ~600) → può favorire
   arbitrariamente l'uno o l'altro.
5. Determinismo frame-skip / sticky-action da riverificare.

**Cosa manca per pubblicare**: il **full P0 sweep** — 3 giochi × 7 budget × 2 algo
× 10 seed = 420 episodi, stimati **~7h su singola CPU** (`REPORT.md:90-102`), più
un adattatore MCTS che interroghi realmente `plangym` Atari via `set_state`.
Finché non c'è, D2 resta 🟡 **"direzionale, coerente col paper, non un numero
definitivo"**. La matrice di decisione (`REPORT.md:151-160`) è esplicitamente
*deferred*.

---

## 4. Atari replication (P1a)

Da `work/10_atari_replication/REPORT.md:22-31`:

| gioco | n | N | M | obs | mean | std | CI95 |
|---|---|---|---|---|---|---|---|
| Boxing | 5 | 30 | 15 | RAM | +100.0 | 0.0 | [100,100] |

**Cosa regge**: **5/5 seed raggiungono il knockout** (+100). Replica e *stringe* il
numero single-seed del paper (96/100). ~82 s/seed su CPU singola.

**Limiti severi**:
- **Un solo gioco cap-bound**: al cap +100, std=0 e il CI collassa a un punto
  (`REPORT.md:62-73`). La metrica media±std è degenere; l'unica lettura onesta è
  "frazione di seed che raggiunge il cap" (che il report ammette di non calcolare
  ancora sistematicamente, `REPORT.md:90-96`).
- **Tabella 50-giochi mai eseguita** (`REPORT.md:34-59`): il deliverable P1a
  (50×10=500 episodi, ~11h CPU) è *pending*. 16/50 giochi del paper sono
  segnalati come "solved due to 1M bug" → il cap-effect si ripeterebbe.
- Sticky-actions=False vs default ALE=True da verificare (`REPORT.md:74-79`).

**Verdetto**: 🟢 solido *per Boxing specificamente* (robusto multi-seed su gioco
facile); 🟡→🔴 **aneddotico per il claim generale** "FMC replica la tabella Atari
del paper con error bar", perché è 1 gioco su 50, il più facile, e cap-bound.

---

## 5. RAM vs IMG ablation (P3)

Da `work/11_ram_vs_img_ablation/REPORT.md:24-36`:

| gioco | N | M | RAM | RGB | Δ | RAM actions-to-win | RGB actions-to-win |
|---|---|---|---|---|---|---|---|
| Boxing | 30 | 15 | +100 | +100 | 0.0 | 119 | 104 |

- **n=2**, gioco cap-bound → entrambi saturano +100, il claim "RAM > IMG 161%" del
  paper è **invisibile** a questo gioco (`REPORT.md:30-37`).
- Il *segnale secondario* (actions-to-win) suggerisce **RGB più veloce** — direzione
  **opposta** al paper — ma n=2 è totalmente sotto-potenza.
- Full sweep (1280 celle, ~28h CPU) *pending* (`REPORT.md:39-58`). Il report stesso
  colloca il risultato tentativamente nell'ultima riga della matrice ("RGB≥RAM,
  claim paper falsificato") pur avvertendo che serve il full sweep.

**Verdetto**: 🔴 **aneddotico**. Non conferma né la propria replica né il claim del
paper; se qualcosa, punta nella direzione opposta, ma è sotto-potenza.

---

## 6. Rischi metodologici trasversali (checklist falsificatore)

| Rischio | Presente? | Dove |
|---|---|---|
| **p-hacking / multiple comparisons** | Sì, parziale | 23 esperimenti, gate greedy, CI su Φ mai loggato in `results.tsv`; mitigato dalla ri-validazione dell'endpoint |
| **Seed cherry-picking** | Basso sull'endpoint | punto-stima stabile n=11→n=18; ma micro-traiettoria a n variabile |
| **Baseline mobile/ambigua** | Sì | 27.44 vs 28.46 vs 29.27 a seconda del banco (`results.tsv:2`, `statistical_validation_paired.json:27`, `HANDOFF.md:9`) |
| **Budget non appaiato** | Sì | n seed varia 1-13 con la config; confronti non appaiati né iso-n |
| **Metrica ambigua (agg vs per-ep)** | Sì | headline aggregato 50.6% vs per-episodio 30%; scelta difendibile ma consequenziale |
| **Confronto non like-for-like (umano)** | Sì | ambiente diverso; datapoint sull'ambiente giusto = 3.77% |
| **n=1 spacciato per esperimento** | Sì | ablation L1 (`gap3_summary.json:6`); falsifiche exp04/15 |
| **Incoerenza interna nei file** | Sì | `statistical_validation.json:64-66` afferma "per-seed mean ~50.5%" mentre il campo reale `:24` è 30.04% — nota errata/aspirazionale |
| **Trasparenza / onestà** | ALTA | il self-review (`peer_review_self.md` C.1-C.7) ammette quasi tutti i limiti sopra; i REPORT Atari/MCTS elencano i propri caveat |

Nota importante a favore del progetto: **la disciplina di documentazione è
eccellente**. I limiti non sono nascosti — sono elencati esplicitamente. Il rischio
non è la frode, è l'**over-claiming nel titolo/abstract** ("human-expert",
"compounding law") rispetto a ciò che i dati sostengono.

---

## 7. Verdetto finale a tre livelli

### 7.1 🟢 SOLIDI / pubblicabili as-is
1. **exp17 migliora significativamente FMC vs baseline non-shaped su
   Craftax-Classic-Symbolic**: appaiato n=18, Wilcoxon p=0.0019, d_z=0.74,
   Δ per-episodio ≈ raddoppio (15%→30%). Sblocca 3/4 blocker
   (iron_pickaxe 33%, iron_sword 11%, diamond 5.6%).
   [`statistical_validation_paired.json`, `sec_results.md:73-92`]
2. **FMC risolve Boxing in modo robusto**: 5/5 seed a +100, CPU, ~82s/seed.
   [`work/10 REPORT.md:24-31`]

### 7.2 🟡 DIREZIONALI (serve altro lavoro — specificato)
1. **Conjecture D** — serve: (a) test di significatività sui Δ delle ablation LOO;
   (b) ri-run di L1 a n≥30 (ora n=1); (c) uno **sweep denso dei multipli 1.4×→6.67×**
   per stabilire davvero il sweet-spot; (d) soprattutto una **dimostrazione che
   l'effetto è super-additivo** e non semplice shaping additivo.
2. **FMC vs MCTS (D2)** — serve il **full P0**: 3 giochi × 7 budget × 10 seed +
   MCTS con sweep di c e rollout_depth + episodi a lunghezza piena (~600 azioni).
   ~7h CPU, già scriptabile.
3. **Cross-benchmark Conjecture D** — serve la replica su Crafter-original a
   **compute appaiato (N=512, M=40, 30 seed)**; lo smoke test attuale (n=3, 50×
   meno compute, Δ+0.15pp) è nullo.

### 7.3 🔴 FRAGILI / a rischio metodologico
1. **"exp17 = 50.95% = human-expert 50.5%"** come titolo: confronto non
   like-for-like (ambiente diverso; sull'ambiente umano FMC fa 3.77%); CI95
   aggregato ±11-13pp; punto-stima trainato da pochi episodi-eroe. Riformulare in
   "raggiunge ~45-55% aggregato zero-training su Craftax-Classic-Symbolic, che è
   dell'ordine del punteggio umano su Crafter-original, con le dovute cautele".
2. **RAM vs IMG (P3)**: n=2, cap-bound, segnale opposto al paper. Non usare come
   evidenza in nessuna direzione finché non c'è il full sweep.
3. **"Structural local optimum a 50.95%"** dedotto da exp18/19 identici: gli
   esperimenti identici a 4 decimali sono a informazione zero su 11 seed fissi, non
   una prova di attrattore strutturale.
4. **Bound quantitativi Conjecture D [1.2,1.4]×** e "prodotto stack ∈[3,5]":
   basati su falsifiche a n=1 con gap non testati.

---

*Fine W1B. Citazioni verificate su file al 2026-07-09.*
