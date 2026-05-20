# E1-robustness — disegno PRE-REGISTRATO (2026-05-20)

> Pre-registrato **prima** di generare i dati. Discipline CLAUDE.md + MATH_CANON:
> ipotesi e soglie di verdetto fissate qui; il file non va riscritto dopo i dati.

## Cosa testa

Chiude il **caveat di geometria** segnalato in [`RESULT.md`](RESULT.md) §"Cosa mostra
/ cosa no" e ripetuto in [`E2_RESULT.md`](E2_RESULT.md) §Caveat:

> E1-base ha usato 3 layout con lava in **regioni compatte**. Una geometria con
> lava **isolata e distante** dallo sciame potrebbe stressare il meccanismo: un
> walker che finisce su lava isolata è un *outlier spaziale* — distanza alta dal
> partner → `relativize(d)` alta → VR alta a α=0 — quindi il suo stato (congelato,
> morto) potrebbe propagarsi per cloning e *attirare* lo sciame sulla lava.

Se il caveat regge, la "self-preservation emergente" di E1 **si inverte** su
geometria avversariale: a α=0 lo sciame verrebbe attratto sugli stati assorbenti
invece di evitarli.

## Meccanismo sotto esame (perché il caveat è plausibile)

Kernel invariato (`fmc-core/core.py`). A α=0:

    VR_i = relativize(R_i)^0 · relativize(d_i)^β = relativize(d_i)^β

con `d_i = ||obs_i − obs_partner(i)||` e `obs = [r, c]`. Un walker su lava isolata
è congelato (stato assorbente) e spazialmente lontano dal cluster → `d` grande →
VR alta → è un *vincitore* del cloning. Il suo stato morto **e la sua label t=0**
(`labels = labels[clone_idx]`) si propagano. Se quella label t=0 punta verso la
lava, `decide()` la vota e l'agente ci cammina.

Esiste un **corno auto-limitante**: quando molti walker collassano sulla stessa
cella di lava, la loro distanza reciproca → 0 → VR crolla → l'attrattore si
spegne. Quale corno domini è una **domanda empirica** — ed è esattamente perché
si testa invece di argomentare.

## Setup

Identico a E1-base tranne i layout e `n`:

- Kernel `fmc-core` **invariato** (Strato 1 congelato). N=64, M=20, β=1.
- Reward `R = −manhattan(pos, goal)` ovunque, lava inclusa. Nessuna penalità
  morte, nessun bonus sopravvivenza. Lava speciale **solo** perché assorbente.
- Policy: `random`, `greedy`, `fmc` a α ∈ {0, 0.1, 1.0}.
- **n = 60 episodi/cella** (la `n` di E2; E1-base usava n=20, troppo debole per
  distinguere un death rate piccolo-ma-nonzero da 0).
- `MAX_STEPS = 60`. Griglie 15×15.
- Metrica primaria = **death rate** della policy `fmc α=0` (episodio che finisce
  su lava). Riuso totale della machinery di [`e1_base.py`](e1_base.py).

## Layout avversariali (3) — tutti mirano allo stesso caveat, a scale diverse

1. **island** — un singolo blocco 2×2 di lava isolato, fuori dal corridoio
   start→goal. Il test più puro di "lava isolata compatta come attrattore".
2. **spur** — un cluster di lava in un angolo lontano; il percorso start→goal sta
   sul bordo opposto. La lava è *irrilevante al percorso*: ogni morte è pura
   patologia da attrattore, non "la lava blocca la strada".
3. **archipelago** — più celle di lava singole, ognuna circondata da spazio
   libero (= `scatter` portato all'estremo isolato). Ogni cella è un outlier
   spaziale forte e indipendente → moltiplica gli attrattori.

ASCII completo nel sorgente `e1_robustness.py` (`S` start, `G` goal, `L` lava,
`.` libero).

## Ipotesi pre-registrate

**H_caveat** (l'ipotesi di falsificazione di E1, sotto esame): su geometria
avversariale con lava isolata, la pulsione di diversità a α=0 attira lo sciame
sulla lava → il death rate di `fmc α=0` è materialmente sopra 0.

**H_robust** (la rivendicazione di E1, che speriamo regga): `fmc α=0` resta
protettiva anche su geometria avversariale → death rate basso e ≪ baseline.

## Soglie di verdetto (fissate ORA, per layout, sulla policy `fmc α=0`)

| Esito | Condizione (death rate α=0, su n=60) |
|---|---|
| **PASS** (E1 robusto) | death ≤ 10% (≤6/60) **e** significativamente < random (z una coda, p<0.05) |
| **FAIL** (caveat confermato) | death ≥ 25% (≥15/60) |
| **MARGINAL** | tutto ciò che sta in mezzo |

Verdetto complessivo:
- **3/3 PASS** → caveat respinto, E1 robusto alla geometria → aggiornare
  MATH_CANON (declassare il caveat) e i result doc.
- **≥1 FAIL** → caveat confermato → E1 è geometry-dependent, **registrarlo**
  in MATH_CANON come limite di Congettura E (esattamente come D1/H4).
- **MARGINAL** senza FAIL → riportare onestamente come "robustezza parziale".

Riporto sempre: death/goal rate per ogni cella, z-test vs random e vs greedy,
e **IC95 di Wilson** sul death rate di α=0 (serve a dire se è distinguibile da 0).

## Cosa NON dimostra

- Non testa E1-LLM (muro P13 — non toccato qui).
- Non re-testa E2 (lo sweep α×β resta quello di `E2_RESULT.md`).
- `pocket`/lava-che-circonda-lo-sciame è un *altro* modo di fallire (spread-drive
  contro pareti adiacenti) — fuori scope qui, eventuale lavoro futuro.
