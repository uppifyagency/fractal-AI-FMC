# W33 — Restatement onesto del claim exp17 (falsificazione dei NUMERI VERI)

> **Ruolo**: falsificatore empirico. Obiettivo: separare ciò che i dati
> sostengono da ciò che il titolo afferma. Ogni numero ha fonte `file:riga`
> o chiave JSON. Aggregato e per-episodio sono distinti **ovunque**.
> Data: 2026-07-09. Verdetto in una riga: **il risultato forte è reale ma
> il titolo attuale è insostenibile as-is** — su tre punti indipendenti.

---

## 0. Fonti lette (con path assoluti)

- `work/05_craftax/autoresearch/results/statistical_validation.json` (test one-sample)
- `work/05_craftax/autoresearch/results/statistical_validation_paired.json` (test appaiato — **il test forte**)
- `work/05_craftax/autoresearch/results/exp17_30seed.json` (ri-validazione headline, n=18)
- `work/05_craftax/autoresearch/results/gap3_summary.json` (leave-one-out L1–L5)
- `work/05_craftax/autoresearch/results/v4_30seed.json` (baseline v4 estratto da run_007)
- `work/05_craftax/autoresearch/HANDOFF.md`, `PAPER_HANDOFF.md`, `results.tsv`
- `work/05_craftax/paper/sec_crafter_smoke.md` (cross-benchmark su Crafter-original)
- `work/05_craftax/paper/peer_review_self.md` (self-review, ammette le debolezze)

---

## 1. I NUMERI VERI — aggregato vs per-episodio

### 1.1 Metrica headline (exp17)

| Quantità | Valore | Fonte |
|---|---|---|
| **Aggregato Crafter (headline pubblicato)** | **50.95 %** (n=11) | `results.tsv:19` (`f1c9ac2`, exp17) |
| **Aggregato Crafter (ri-validazione n=18)** | **50.6049 %** | `statistical_validation.json` → `exp17_aggregate_crafter_pct` (riga 28); `exp17_30seed.json:2` (`crafter_score: 50.6`) |
| CI95 aggregato (n=18, bootstrap) | **[36.85 %, 59.46 %]** (±~11–13 pp) | `statistical_validation.json` righe 29–30 (`..._ci95_lo/hi_pct`) |
| **Media PER-EPISODIO** | **30.0363 %** | `statistical_validation.json:24` (`exp17_per_episode_mean_pct`) |
| Dev. std per-episodio | **20.3706 %** | `statistical_validation.json:25` |
| CI95 per-episodio (bootstrap) | [21.57 %, 39.63 %] | `statistical_validation.json` righe 26–27 |
| **Mediana per-seed** | **≈ 27.7 %** | calcolata da `exp17_per_seed_scores_pct` righe 40–59 |
| n seed effettivi | **18** (target dichiarato 30) | `statistical_validation.json:3`; `exp17_30seed.json:6` |

**Firma dell'inflazione da aggregato**: l'aggregato (50.6 %) supera **15 dei 18
punteggi per-seed**. Solo 3 seed lo battono (80.89 %, 65.39 %, 52.83 % —
`exp17_per_seed_scores_pct`). La mediana è 27.7 %. La metrica di Hafner è una
media geometrica delle frequenze di sblocco *poolate* su tutti gli episodi:
premia l'**unione** delle capacità della popolazione, non il run tipico.
Non è cherry-picking illegittimo (è la metrica standard), ma un run singolo
tipico vale ~30 %, non ~51 %.

### 1.2 Confronto exp17 vs baseline v4 — APPAIATO, stesso ambiente (il dato forte)

| Quantità | Valore | Fonte (`statistical_validation_paired.json`) |
|---|---|---|
| exp17 aggregato | 50.6049 % | riga 24 |
| v4 aggregato (stessi 18 seed, appaiato) | 28.4564 % | riga 27 |
| **Δ aggregato appaiato** | **+22.15 pp**, CI95 **[8.81, 32.12]** | righe 31–33 |
| bootstrap p(Δ≤0) | **0.0001** | riga 34 |
| exp17 media per-episodio | 30.0363 % | riga 35 |
| v4 media per-episodio | 15.3523 % | riga 36 |
| **Wilcoxon appaiato** | W=124, **p = 1.89×10⁻³** | righe 37–38 |
| t appaiato | t=3.145, p = 2.95×10⁻³ | righe 39–40 |
| **Cohen dz (appaiato)** | **0.7413** (effetto medio-grande) | riga 41 |

### 1.3 Il test one-sample (contro l'aggregato v4 29.27 %) NON è significativo

| Test | Valore | Fonte (`statistical_validation.json`) |
|---|---|---|
| Δ per-episodio vs aggregato v4 | **+0.77 pp** | riga 34 (`delta_per_episode_mean_pp`) |
| Wilcoxon one-sample | W=75, **p = 0.677** | righe 35–36 |
| t one-sample | t=0.16, **p = 0.438** | righe 37–38 |
| Cohen d one-sample | **0.0376** (trascurabile) | riga 39 |

**Lettura**: confrontando la distribuzione per-seed di exp17 (media 30.04 %)
contro il *numero aggregato* v4 (29.27 %), non c'è quasi differenza (p≈0.68).
Il guadagno reale emerge SOLO nel confronto appaiato per-seed contro per-seed
(v4 per-episodio = 15.35 %). Questo è il cuore della fragilità: il "salto"
50.6 vs 29.27 è in gran parte l'artefatto aggregato-vs-per-episodio, **non**
un miglioramento del run tipico rispetto all'aggregato di riferimento.

### 1.4 Tassi di sblocco blocker (exp17, n=18) e like-for-like sull'ambiente umano

| Achievement | Freq. sblocco | Fonte |
|---|---|---|
| make_iron_pickaxe | 0.3333 (6/18) | `exp17_30seed.json:23` |
| collect_iron | 0.3889 | `exp17_30seed.json:32` |
| make_iron_sword | 0.1111 (2/18) | `exp17_30seed.json:25` |
| collect_diamond | 0.0556 (**1/18**) | `exp17_30seed.json:36` |
| eat_plant | 0.0 (strutturalmente inaccessibile) | `exp17_30seed.json:31` |

**Like-for-like sull'ambiente umano (Crafter-original, pixel)** —
`sec_crafter_smoke.md:29-31`:

| Metodo | Crafter-original | n | Fonte |
|---|---|---|---|
| FMC v4 (no shaping) | **3.62 %** | 3 | `sec_crafter_smoke.md:29` |
| **FMC exp17 (full shaping)** | **3.77 %** | 3 | `sec_crafter_smoke.md:30` |
| Δ | +0.15 pp | — | `sec_crafter_smoke.md:31` |

Caveat dichiarato dagli autori stessi: config **50× più piccola** (N=32 vs 512,
M=12 vs 40) e n=3 → "statistically meaningless" (`sec_crafter_smoke.md:35,52`).
Il 3.77 % **non** è un confronto equo del metodo pieno; è però l'unico numero
esistente sull'ambiente su cui l'umano segna 50.5 %.

---

## 2. Verifica incoerenza interna del JSON — ESITO: **VERO (confermato)**

In `statistical_validation.json`:

- **riga 24**: `"exp17_per_episode_mean_pct": 30.0363`
- **riga 25**: `"exp17_per_episode_std_pct": 20.3706`
- **riga 28**: `"exp17_aggregate_crafter_pct": 50.6049`
- **righe 63–64** (blocco `notes`): testo *"...per-seed mean (~50.5%) and
  std (~16) reflect the binary geometric-mean property..."*

**Contraddizione**: la nota (righe 63–64) afferma che la **media per-seed è
~50.5 %** con std ~16, ma il campo dati reale (`exp17_per_episode_mean_pct`,
riga 24) vale **30.0363 %** e la std (riga 25) vale **20.3706**, non ~16. La
nota confonde l'**aggregato** (50.6 %, che è quantità *cross-episodio*, non una
media di punteggi per-seed) con la **media per-seed** (30.04 %). La media
aritmetica reale dei 18 punteggi per-seed (righe 40–59) è ≈ 30.0 %, coerente
col campo, **non** con la nota. → L'incoerenza denunciata è **reale e
verificata**: campo `exp17_per_episode_mean_pct = 30.0363` (riga 24) vs nota
"per-seed mean (~50.5%)" (riga 63).

Incoerenza secondaria (headline): il titolo pubblicato è **50.95 %**
(`results.tsv:19`, n=11) mentre la ri-validazione n=18 dà **50.60 %**
(`exp17_30seed.json:2`). Il 50.95 % è baked-in anche nella stringa di config
`_mutation` (`exp17_30seed.json:81`) come label, non come misura del file.

---

## 3. RESTATEMENT ONESTO (forma abstract)

### (a) Il claim difendibile e forte che i dati sostengono

> Su Craftax-Classic-Symbolic-v1, con planning zero-training su singola CPU,
> il reward-shaping chain-tier (exp17) porta il punteggio Crafter aggregato da
> 28.46 % (baseline v4, no shaping) a **50.60 %** — un miglioramento appaiato
> di **+22.1 pp** (18 seed, stesso seed-bank; Wilcoxon appaiato p = 1.9×10⁻³;
> t appaiato p = 3.0×10⁻³; Cohen dz = 0.74). Sulla media per-episodio il
> guadagno è **+14.7 pp** (30.04 % vs 15.35 %). L'effetto è attribuibile allo
> stacking di componenti di shaping: la rimozione leave-one-out di ciascuna
> abbassa il punteggio di **−4.8 → −7.9 pp** (gap3 L2–L5, n=30). È, per quanto
> a nostra conoscenza, il primo planner senza addestramento a sbloccare la
> catena iron→diamond su questa suite (make_iron_pickaxe 33 %, make_iron_sword
> 11 %, collect_diamond 5.6 %).

Questo claim è vero, testato, appaiato, e non richiede alcun asterisco.

### (b) Cosa il titolo attuale eccede — e perché

| Titolo attuale | Cosa dicono i dati | Perché eccede |
|---|---|---|
| "**50.95 %** Crafter" | Ri-validazione n=18 = **50.60 %**; run n=11 = 50.95 % | Numero headline dal run più piccolo (11 seed); la conferma robusta è 50.60 % |
| "**matches/beats human-expert (50.5 %)**" | Aggregato-vs-aggregato **ma cross-ambiente**. Sull'ambiente umano reale (Crafter-original, pixel) exp17 = **3.77 %** (n=3, 50× meno compute) | Umano su Crafter-original/pixel; exp17 su Craftax-Classic-**Symbolic**. Non like-for-like sull'ambiente |
| Implica una performance "da 51 %" | Media per-episodio = **30.04 %**; mediana per-seed ≈ **27.7 %**; l'aggregato batte 15/18 seed | L'aggregato (media geometrica poolata) misura l'unione della popolazione, non il run tipico |
| "zero-training, single-CPU" | **VERO** — nessun claim da correggere | `JAX_PLATFORMS=cpu`, nessun peso appreso (peer_review_self A.1) |
| Cong. D come "legge generale" | Confermata **solo** su Craftax-Classic; cross-benchmark è n=3 non significativo | Nessuna replica like-for-like a compute pieno |
| n=30 (target) | **n=18** effettivi (budget wall esaurito) | CI95 aggregato ±~11–13 pp, ampio |

### (c) Cosa servirebbe per sostenere le affermazioni forti

1. **Like-for-like su Crafter-original a compute pieno** (N=512, M=40, 30 seed,
   shaping v4 vs exp17 appaiato). È l'unico test che può giustificare "matches
   human-expert": stesso ambiente, stessa metrica. Protocollo già scritto in
   `sec_crafter_smoke.md:88-100` (~30–90 h/config CPU, ~6–18 h su multi-core).
2. **Super-additività (sinergia) formale**: dimostrare che inv-tier ⊕ ach-fire
   > inv-tier + ach-fire separati, con un contrasto d'interazione 2×2 a n=30
   (ora è solo argomentato via leave-one-out, con L1 a n=1 — `gap3_summary.json:6`).
3. **Replica su Procgen** (Heist/altro) per elevare Cong. D da osservazione
   Craftax-specifica a legge cross-family (PAPER_HANDOFF §Gap 4).
4. **Chiudere n=18 → n=30** con budget wall esteso (~6 h) per stringere il CI95.

---

## 4. Mini-audit "garden of forking paths" (23 esperimenti hill-climb)

**Salvaguardie PRESENTI:**
- ✅ **Endpoint ri-validato** su seed-bank più ampio: headline hill-climb n=11
  (50.95 %, `results.tsv:19`) ri-testato a n=18 (50.60 %, `exp17_30seed.json`).
  I due valori sono vicini → l'headline non è un fluke degli 11 seed originali.
- ✅ **Test appaiato genuino** contro v4 sugli stessi 18 seed
  (`statistical_validation_paired.json`), non solo contro un numero storico.
- ✅ **Leave-one-out** delle 5 componenti (gap3): tutte fanno scendere il
  punteggio (−4.8→−7.9 pp) → coerente con contributo reale di ogni pezzo.
- ✅ **Controlli negativi** riportati (exp04, exp15, exp22 falliscono/collassano
  — `HANDOFF.md:57,68,75`); nessun cherry-picking degli esperimenti falliti.
- ✅ **Confounder controllati**: N/M/α/β fissi, stesso seed-bank (peer_review B.3).

**Salvaguardie MANCANTI (rischio di overfitting al seed-bank):**
- ❌ **Nessun test-set hold-out**: i seed di hill-climb e quelli di validazione
  partono entrambi da 42 (`exp17_30seed.json:41` `seed_start: 42`) → overlap.
  L'endpoint è selezionato E validato su seed sovrapposti → bias ottimistico.
- ❌ **Metrica di fitness = metrica riportata** (aggregato Crafter) su 23
  configurazioni → multiple-comparison non corretta. La fitness era l'aggregato,
  proprio la quantità gonfiata (vedi §1.1).
- ❌ **Optional stopping**: la regola era "a 50 % dichiara vittoria e fermati"
  (`HANDOFF.md:332`). Il loop si è fermato al PRIMO config che ha superato il
  target → bias verso l'alto sull'endpoint riportato.
- ❌ **Auto-status gate difettoso**: confronta con baseline storica 29.27, non
  con il best corrente (`HANDOFF.md:167-172`, "Insight 6") → override manuali.
- ❌ **L1 a n=1** (`gap3_summary.json:6`): un pilastro dell'argomento di
  sinergia poggia su un singolo seed.
- ❌ **Nessuna pre-registrazione** di metrica/stopping oltre "raggiungi 50 %".

**Verdetto audit**: la ri-validazione a n più alto e il paired test sono
salvaguardie serie e non banali; ma senza hold-out, con optional stopping al
target e fitness = metrica gonfiata, l'headline 50.95 %/50.60 % va trattato come
**stima ottimistica dell'endpoint selezionato**, non come misura non-distorta.
Il Δ appaiato +22 pp (§1.2) è invece robusto a questi difetti perché confronta
DUE configurazioni sugli STESSI seed.
