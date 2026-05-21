# hP13-0 redo — risultato: la keystone resta non testata, ma ora sappiamo perché (2026-05-21)

Follow-up di [`P13_RESULT.md`](P13_RESULT.md) §8 punto 1: rifare hP13-0 — la
*keystone* di P13, "un surrogato che preserva il rango di VR preserva la
decisione FMC" — con una **griglia di rumore η fine**, dopo che il proxy
originale era saltato da rango perfetto (η=0, Spearman 1.00) a rango distrutto
(η≥1, Spearman ~0.15).

Codice: [`p13_hp13_0.py`](p13_hp13_0.py). Dati: [`results/p13_hp13_0.json`](results/p13_hp13_0.json).

> **Esito in una riga.** La griglia fine (fino a η=0.05) **non raggiunge comunque**
> il regime intermedio: Spearman salta 1.00 (η=0) → 0.44 (η=0.05), senza punti in
> mezzo. La keystone resta **non testata** — ma il *perché* è ora diagnosticato:
> il vettore VR è densamente *legato* (lo swarm si raggruppa), e il rumore
> additivo è perciò all-or-nothing per il rango. Serve un altro tipo di
> degradazione.

---

## 1. Risultati (α=0, S1 abs-preserved, pooled sui 3 layout, n=30)

| η | VR-rank Spearman | decision-agreement | death% |
|---:|---:|---:|---:|
| 0.00 | **1.00** | **1.00** | 0.0 |
| 0.05 | 0.44 | 0.49 | 0.0 |
| 0.10 | 0.39 | 0.45 | 0.0 |
| 0.20 | 0.32 | 0.41 | 0.0 |
| 0.35 | 0.27 | 0.38 | 0.0 |
| 0.50 | 0.23 | 0.37 | 0.0 |
| 0.75 | 0.18 | 0.35 | 0.0 |
| 1.00 | 0.15 | 0.33 | 0.0 |

---

## 2. Lettura onesta

**Il buco non è chiuso.** hP13-0 ha bisogno di un surrogato con rango di VR
*alto-ma-imperfetto* (Spearman ∈ (0.5, 1.0)) per testare il claim keystone — "se
il rango è ben preservato, la decisione è preservata (agreement ≥ 0.85)". La
griglia fine, anche al suo punto più piccolo η=0.05, dà **Spearman 0.44**. Tra
1.00 e 0.44 non c'è alcun punto dati. Il regime in cui la keystone si decide
**non è raggiunto**.

> ⚠️ Lo script `p13_hp13_0.py` stampava in origine "hP13-0 SUPPORTED" — un
> artefatto degenere: l'unico η con Spearman ≥ 0.80 è η=0, che è il kernel vero
> senza rumore (non un surrogato sparso). È la stessa classe di bug del "GO-full"
> di `p13_proxy.py`. La logica di verdetto è stata corretta (esclude η=0); il
> verdetto onesto è **INCONCLUSIVO**. I dati nel JSON non sono toccati.

**Perché il rumore additivo non basta — la diagnosi.** A α=0, VR $=\widehat{D}^\beta$,
e $D$ è la distanza a coppie tra walker. Il cloning **raggruppa lo swarm**: copia
i walker vincenti → molti walker finiscono in stati identici o quasi → le loro
distanze a coppie sono *legate* (tied) → il vettore VR, dopo `relativize`, è un
cluster fitto di valori quasi uguali con pochi outlier. Conseguenza: **qualunque**
rumore additivo non infinitesimo riordina il cluster — Spearman crolla da 1.00 a
0.44 con η=0.05 — mentre i pochi outlier mantengono il rango (da cui la
correlazione residua 0.44, non 0). Il rumore additivo è quindi *all-or-nothing*
per il rango di VR su questo task: non esiste un η che dia "rango per lo più
preservato".

**Cosa il dato mostra comunque.**
- Nel regime *basso-rango* (Spearman 0.44 → 0.15), la decision-agreement segue il
  rango in modo monotono (0.49 → 0.33). Il *conseguente* di hP13-0 ("rango basso →
  agreement bassa") è coerente — ma è il *ramo non interessante*.
- Il **death rate è 0% a ogni η.** La self-preservation è solidissima,
  indipendente dal rango di VR — riconferma il finding *R2-survival* di P13
  ([`P13_RESULT.md`](P13_RESULT.md) §3): il death rate non segue né l'agreement
  né il rango; ciò che conta è la struttura assorbente, qui sempre preservata.

---

## 3. Verdetto e fix

**hP13-0: ancora INCONCLUSIVO** — la keystone (rango alto → decisione preservata)
non è testata, perché su questo gridworld il rumore additivo non sa produrre un
surrogato a rango alto-ma-imperfetto.

**Il fix è ora preciso** (era "griglia η più fine" in P13_RESULT; ora sappiamo che
non basta): degradare il rango di VR con un knob *diretto* — una frazione
controllata $\phi$ di **inversioni di rango a coppie** sul vettore VR, lasciando
intatto il resto. Questo produce per costruzione Spearman $\approx 1-2\phi$ su
tutto l'intervallo $(0,1)$, e permette di mappare agreement vs Spearman nel
regime che conta. Si lega all'argomento dei tre livelli di P13_DESIGN §4 (errore
affine / monotono / non-monotono): $\phi$ è precisamente la *dose di errore
non-monotono*, l'unico modo di fallimento che conta.

> **Nota.** Questo è un esperimento eseguibile subito su `fmc-core` (nessun LLM):
> un terzo schema di degradazione accanto a S1/S2/S3, da aggiungere a
> `p13_proxy.py`. È il vero modo di chiudere hP13-0.

---

## 4. Riproducibilità

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/12_conjecture_e/p13_hp13_0.py     # ~450 s (CPU)
```

---

*Fine P13_HP13_0_RESULT.md. hP13-0 resta non testata: il rango di VR è
fragilissimo al rumore additivo (Spearman 1.00→0.44 con η=0.05) perché lo swarm
clusterizzato produce VR densamente legate. Fix: degradare il rango con una
frazione controllata di inversioni a coppie, non con rumore additivo. Death rate
0% a ogni η — R2-survival riconfermata.*
