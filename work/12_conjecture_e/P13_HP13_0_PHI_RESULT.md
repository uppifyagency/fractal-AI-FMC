# hP13-0, finalmente testata: il keystone VR-rank è falsificato — la survival no (2026-05-21)

Chiude il fix pre-registrato in [`P13_HP13_0_RESULT.md`](P13_HP13_0_RESULT.md) §3:
testare la *keystone* di P13 — "un surrogato che preserva il rango di VR preserva
la decisione FMC" — con un knob **diretto** sul rango (frazione φ di inversioni a
coppie sul vettore VR), dopo che il rumore additivo si era rivelato all-or-nothing.

Codice: [`p13_hp13_0_phi.py`](p13_hp13_0_phi.py). Dati: [`results/p13_hp13_0_phi.json`](results/p13_hp13_0_phi.json).
Kernel `fmc-core` invariato — `proxy_plan` esteso con un `vr_hook` opzionale
(default `None` → bit-identico a `fmc.core.plan`, asserito).

> **Esito in una riga.** Il knob φ funziona: lo Spearman è ora **liscio** da 1.00
> a −0.02 su φ∈[0,1] — il regime intermedio che il rumore additivo non sapeva
> raggiungere è finalmente coperto. E lì la keystone **fallisce**: a Spearman
> **0.97** (rango quasi perfetto — 1 coppia di walker su 64 scambiata)
> l'agreement decisionale è **0.47**, contro la soglia pre-registrata 0.85.
> Preservare il rango di VR **non** preserva la decisione FMC. *Però*: il death
> rate resta **0% fino a Spearman 0.46** — la self-preservation è robusta alla
> corruzione massiccia del rango. La keystone (sufficiency) è **FALSIFICATA**;
> la survival no.

---

## 1. Risultati (α=0, β=1, N=64, M=20, pooled sui 3 layout avversariali, n=30)

| φ | VR-rank Spearman | decision-agreement | death% | goal% |
|---:|---:|---:|---:|---:|
| 0.00 | **1.00** | **1.00** | 0.0 | 0.0 |
| 0.05 | **0.97** | **0.47** | 0.0 | 0.0 |
| 0.10 | 0.90 | 0.39 | 0.0 | 0.0 |
| 0.15 | 0.84 | 0.37 | 0.0 | 0.0 |
| 0.20 | 0.81 | 0.35 | 0.0 | 0.0 |
| 0.30 | 0.71 | 0.32 | 0.0 | 0.0 |
| 0.40 | 0.59 | 0.31 | 0.0 | 0.0 |
| 0.55 | 0.46 | 0.30 | 0.0 | 0.0 |
| 0.70 | 0.30 | 0.27 | 1.1 | 0.0 |
| 0.85 | 0.14 | 0.25 | 4.4 | 0.0 |
| 1.00 | −0.02 | 0.22 | 13.3 | 0.0 |

(`goal% = 0` ovunque: a α=0 l'agente non insegue il goal — è il regime "Common
Sense" puro di E1, atteso. Random floor dell'agreement con ~5 azioni ≈ 0.20.)

---

## 2. Lettura onesta

### 2.1 Il fix ha funzionato — la keystone è finalmente testabile

Il rumore additivo saltava da Spearman 1.00 (η=0) a 0.44 (η=0.05) senza punti in
mezzo, perché lo swarm clusterizzato rende il vettore VR densamente *legato* e
qualunque rumore riordina il cluster in blocco. Le **inversioni di rango a
coppie** non hanno questa patologia: preservano il *multiset* di VR esatto (le
magnitudini non si toccano, i tie restano tie), permutano solo l'assegnazione
walker↔VR. Risultato: Spearman liscio e monotono — 1.00, 0.97, 0.90, 0.84, 0.81,
0.71, 0.59, 0.46, 0.30, 0.14, −0.02. **Il regime (0.5, 1.0) che decide la
keystone è coperto fittamente.** Il buco diagnostico di `P13_HP13_0_RESULT.md` è
chiuso.

### 2.2 La keystone fallisce — e fallisce in modo netto

La keystone (P13_DESIGN §4, §6.4) è un'implicazione di **sufficienza**: *se* il
surrogato preserva il rango di VR, *allora* la decisione FMC è preservata
(agreement ≥ 0.85). Il dato è un controesempio diretto:

- A **Spearman 0.97** — rango quasi perfetto, *una sola* coppia di walker su 64
  scambiata — l'agreement è **0.47**. Non 0.84, non 0.80: 0.47. Meno della metà.
- A Spearman 0.84 (φ=0.15) l'agreement è già 0.37 — vicino al floor caotico.
- L'agreement raggiunge 0.85 **solo** a Spearman = 1.00 esatto, cioè a corruzione
  **zero**. Qualunque corruzione non nulla del rango, per quanto minima, fa
  crollare l'agreement.

Confronto incrociato che irrobustisce il finding: il redo a rumore additivo
([`P13_HP13_0_RESULT.md`](P13_HP13_0_RESULT.md)) a η=0.05 dava Spearman 0.44 →
agreement 0.49; qui φ=0.05 dà Spearman 0.97 → agreement 0.47. **Che lo Spearman
sia 0.44 o 0.97, l'agreement è ~0.47.** L'agreement è di fatto piatto appena
sotto 0.5 su quasi tutto l'intervallo di rango — un *precipizio* a φ=0, non una
rampa. La monotonìa (tol 0.05) è tecnicamente vera ma fuorviante sulla forma.

### 2.3 Il meccanismo: la decisione è funzione caoticamente-amplificata del VR *esatto*

Perché una sola coppia scambiata distrugge metà delle decisioni? Il `clone_step`
è stocastico e lo swarm è un sistema dinamico caotico: una perturbazione
minima del vettore VR a un tick → `clone_idx` leggermente diverso → composizione
dello swarm diversa al tick successivo → amplificata su M=20 tick → azione
`decide(labels)` finale diversa. La decisione FMC non è una funzione *liscia* del
rango di VR: dipende dal vettore VR **esatto**, magnitudini comprese, in modo
amplificato dalla profondità del rollout.

Conseguenza metodologica: **"decision-agreement vs FULL" è la metrica di successo
sbagliata per un planner caotico-stocastico.** Misura la divergenza di traiettoria,
non la qualità della decisione. Due run dello *stesso* kernel vero con seed
diversi "disaccorderebbero" allo stesso modo. L'agreement vale 1.00 solo perché
arm e ref girano sullo stesso seed: è un test di bit-identicità, non di qualità.

### 2.4 Ciò che sopravvive — ed è la metà che conta

Il death rate resta **0% fino a φ=0.55, cioè Spearman 0.46** — rango degradato di
oltre metà — e si rompe solo quando il rango è *distrutto* (Spearman ≤ 0.30 →
death 1.1%, 4.4%, 13.3%). La self-preservation **non segue il rango di VR**: è
robusta a una sua corruzione massiccia.

Questo riconferma e affila il finding *R2-survival* di [`P13_RESULT.md`](P13_RESULT.md)
§3: l'invariante che porta la survival **non è il rango di VR ma la struttura
assorbente del world-model** (qui il kernel vero, dove la lava è sempre
terminale). La *identità* della decisione (quale azione esatta) è fragile al
rango; la *qualità* della decisione (non morire) è robusta finché il modello del
mondo tiene gli stati terminali.

---

## 3. Verdetto e conseguenze

**hP13-0: FALSIFICATA** (claim di sufficienza). Il rango di VR quasi
perfettamente preservato (Spearman 0.97) **non** preserva la decisione FMC
(agreement 0.47 ≪ 0.85). L'unico residuo vero è quasi-vacuo — "per la decisione
*identica* serve il VR *identico*". La keystone come la usava P13_DESIGN §4 è
morta.

> ⚠️ Lo script ha stampato in automatico "WEAKENED" (ramo *monotone ∧ ¬high_ok*).
> È troppo indulgente: un claim di **sufficienza** che fallisce a Spearman 0.97
> non è "indebolito", è **falsificato** — 0.47 non è "poco sotto 0.85", è un
> crollo al floor caotico. La logica di verdetto è stata corretta (aggiunto il
> ramo *near-perfect-rank falsifier*: Spearman ≥ 0.95 con agreement < 0.85 →
> FALSIFIED). Stessa classe di correzione del "SUPPORTED" degenere di
> `P13_HP13_0_RESULT.md`. I dati nel JSON non sono toccati; il campo `verdict`
> del JSON resta "WEAKENED" come da run, il verdetto onesto è qui.

**Conseguenze per P13 / E1-LLM:**

1. **L'argomento VR-rank di [`P13_DESIGN.md`](P13_DESIGN.md) §4 cade.** Non si
   può più giustificare l'interrogazione sparsa dicendo "basta che il surrogato
   LLM azzecchi il *ranking* di VR". Non basta: servirebbe il vettore VR quasi
   esatto, che è incompatibile con un surrogato sparso.
2. **Il gate di E1-LLM ne esce *rinforzato*, non indebolito.** Il gate
   pre-registrato in [`E1_LLM_DESIGN.md`](E1_LLM_DESIGN.md) — "l'LLM-world-model
   deve modellare correttamente la struttura assorbente/terminale" — è proprio
   l'invariante che empiricamente porta la survival (death 0% a Spearman 0.46).
   La VR-rank era un argomento *alternativo e più debole*: cade lui, non il gate.
3. **La metrica di E1-LLM dev'essere l'esito (death/goal rate), non
   l'agreement decisionale.** Per un planner caotico l'agreement-vs-controllo è
   privo di significato. `E1_LLM_DESIGN.md` già pre-registra il death rate come
   metrica: confermato come scelta giusta.
4. **Legame con la Congettura B.** La sensibilità caotica della decisione a una
   perturbazione infinitesima del VR *è* il comportamento edge-of-chaos che
   H-B1a sondava su λ₁. Il VR-rank-vs-decisione e λ₁-vs-δ₀ guardano la stessa
   instabilità dello swarm da due lati.

**Netto:** E1-LLM resta **GO-conditional** — gate = struttura assorbente,
metrica = death rate. Questo risultato non sposta il verdetto: rimuove un
argomento di fattibilità falso (VR-rank) e conferma quello vero.

---

## 4. Note

- **Corruzione a tutti i tick, incluso t=0.** `RankCorruptor` corrompe il VR a
  ogni tick; `S1Schema` del proxy originale esentava t=0. Irrilevante per la
  conclusione: il redo a rumore additivo *con* esenzione t=0 dava agreement 0.49
  a η=0.05, questo *senza* esenzione dà 0.47 — stessa cifra. Il gap dalla soglia
  0.85 è troppo ampio perché l'esenzione di un tick lo colmi.
- **Perché φ=0.05 → 1 sola coppia.** N=64, round(0.05·64)=3, reso pari → m=2 →
  una coppia. È il punto: *una* inversione su 64, Spearman 0.97, e l'agreement è
  già crollato.

## 5. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/12_conjecture_e/p13_hp13_0_phi.py     # ~726 s (CPU)
```

---

*Fine P13_HP13_0_PHI_RESULT.md. La keystone VR-rank di P13 è falsificata: a
Spearman 0.97 l'agreement è 0.47. La decisione FMC è funzione caoticamente-
amplificata del VR esatto, non del suo rango. La survival invece è robusta
(death 0% fino a Spearman 0.46) — l'invariante è la struttura assorbente del
world-model. E1-LLM: gate = struttura assorbente, metrica = death rate;
l'argomento VR-rank è rimosso.*
