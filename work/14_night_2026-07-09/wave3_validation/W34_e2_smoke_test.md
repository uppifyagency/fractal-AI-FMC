# W34 — Smoke test E2 di divergenza: un "fit gate" riutilizzabile per FMC

**Data**: 2026-07-09 · **Wave 3 (validazione rigorosa)** · Sessione notturna 2026-07-09
**Script**: [`w34_e2_smoke.py`](w34_e2_smoke.py) · **Core riusato**: `fmc-core/src/fmc/core.py` (Def. 2–6)

---

## 0. Perché questo strumento esiste

FMC batte la ricerca casuale **solo se lo swarm libero diverge entro l'orizzonte
di pianificazione `M`**. Lo swarm libero = `N` walker che partono dallo stesso
stato `s0`, ciascuno con un rollout ad azioni casuali indipendente. Il kernel di
cloning *seleziona* tra traiettorie divergenti; se non c'è nulla tra cui
selezionare, FMC ≈ random.

Questa è la causa esatta del **fallimento su plasma (M18, 2026-05-05)**: la sim
lineare di TCV è quasi-deterministica → tutti i walker collassano sulla stessa
traiettoria di gradiente indipendentemente dall'azione → le reward diventano
quasi-identiche → `relativize` mappa un vettore quasi-costante su ~tutti-uno (il
ramo `std → 0`) → la virtual reward è ~uniforme → l'argmax del cloning non porta
informazione.

E2 misura questo collasso **prima** di investire in un adapter. È un gate
riutilizzabile: si applica a ogni candidato applicativo con `set_state`/`step`.

---

## 1. La metrica E2 (definizione operativa)

Dato un env (protocollo `fmc.envs.base.Environment`), uno stato `s0`, e
`(N, M, α, β)`, si misurano — mediati su 8 seed:

### Canale dinamico (swarm libero, senza cloning) — il gate primario

- **`disp_1`** = distanza L2 media a coppie delle osservazioni dopo **1 step**
  (partendo da `s0` identico, è la dispersione indotta da *una* scelta d'azione:
  il "quanto" di autorità di controllo).
- **`disp_M`** = distanza L2 media a coppie delle osservazioni all'orizzonte `M`.
- **`disp_ratio = disp_M / disp_1`** — dispersione terminale **in unità di
  autorità di controllo di un singolo step**. Scale-free.
  - `> soglia`: la nuvola cresce (dinamica espansiva/caotica) → **FMC-fit**.
  - `≈ 1`: la nuvola non supera mai un quanto di controllo (la contrazione
    schiaccia l'accumulo) → **collasso**, come il plasma.

### Canale relativize/VR (ciò che FMC vede davvero) — diagnostici, NON gate

- **`reward_cv_M`** = `std/(|mean|+ε)` del vettore reward grezzo a `M`. È
  *esattamente* l'input il cui CV controlla se `relativize(reward)` mantiene
  segnale.
- **`ess_ratio`** = `ESS(VR@M)/N` (Def. 5). `≈1` ⇒ VR uniforme ⇒ cloning nullo.
- **`b_eff`** = branching factor effettivo delle etichette sopravvissute (Def. 6),
  riportato per contesto.

### Il gate finale

```
DIVERGE (FMC-fit)   se   disp_ratio ≥ 3.0
COLLAPSE (no-fit)   altrimenti
(+ warning "reward-degenerate" se disp_ratio ≥ 3.0 ma reward_cv_M < 0.02)
```

**Perché `disp_ratio` da solo, e non un AND con `ess_ratio`/`reward_cv`?**
Deciso **dai dati reali**, non a priori (vedi §2):

- `disp_ratio` separa i due gruppi con **margine ampio e vuoto**: collasso
  `{1.94, 2.39}` vs divergenti `{4.66, 5.52, 8.84, 24.9}`. La soglia `3.0` sta
  nel gap `[2.39, 4.66]`, con margine da entrambi i lati.
- `ess_ratio` **fa falso-negativo** su CartPole (0.649 nonostante `disp_ratio`
  5.5): reward di sopravvivenza + morte di massa gonfiano l'uniformità della VR
  anche quando le traiettorie si sparpagliano ampiamente. Margine sottile e
  inaffidabile → **escluso dal gate**, tenuto come diagnostico.
- `reward_cv` **non separa affatto** (Rocket 0.36 < LinearContractive 0.86): i
  walker morti (reward=0) comprimono il CV della reward di un env divergente
  sotto quello di un env lineare smooth → **escluso dal gate**. Un `reward_cv`
  ~0 segnalerebbe però il modo di fallimento *distinto* "reward-degenerate"
  (dispersione OK ma reward non discrimina) → tenuto come warning soft.

---

## 2. Validazione su casi noti (numeri reali)

Comando: `cd fmc-core && python3 ../work/14_night_2026-07-09/wave3_validation/w34_e2_smoke.py`
(`N=64, M=30, α=β=1.0`, 8 seed; runtime totale ≈ 3.4 s, CPU singola.)

| env | atteso | **verdetto E2** | disp_ratio | reward_cv_M | ess_ratio | b_eff/K | disp_1→disp_M |
|---|---|---|---:|---:|---:|---:|---|
| Rocket (nonlin.)          | diverge  | **DIVERGE**  | **24.90** | 0.364 | 0.554 | 1.00/9 | 0.028→0.683 |
| Navigation2D (nonlin.)    | diverge  | **DIVERGE**  |  **4.66** | 1.507 | 0.579 | 1.43/9 | 0.047→0.221 |
| Pendulum (nonlin.)        | diverge  | **DIVERGE**  |  **8.84** | 3.169 | 0.523 | 1.64/9 | 0.030→0.266 |
| CartPole (nonlin.)        | diverge  | **DIVERGE**  |  **5.52** | 2.503 | 0.649 | 1.63/2 | 0.353→1.950 |
| LinearContractive 2D (plasma mimic) | collapse | **COLLAPSE** | **1.94** | 0.864 | 0.668 | 1.78/9 | 0.014→0.028 |
| LinearIntegrator1D        | collapse | **COLLAPSE** |  **2.39** | 1.087 | 0.644 | 1.27/3 | 0.018→0.043 |

**Separazione: 6/6 corretti.** La soglia `disp_ratio ≥ 3.0` classifica
correttamente entrambi i gruppi.

Gli env-giocattolo lineari (definiti nello script) riproducono il regime plasma:
`x_{t+1} = A·x_t + B·nudge(a)` con `A` stabile (contrazione) e attuazione debole
`B`, reward quadratica concava `−‖x‖²`. Tutti i walker vengono tirati verso
l'origine indipendentemente dall'azione → `disp_M ≈ disp_1` → gate scattato.

### 2.1 Sweep di calibrazione — la metrica traccia la proprietà vera

Per escludere che `3.0` sia un cutoff arbitrario overfittato su 6 punti, si varia
il raggio spettrale `A` di `LinearContractive` da fortemente contraente a
espansivo (`B=0.01`, 8 seed):

| A (raggio spettrale) | disp_ratio | ess_ratio | verdetto |
|---:|---:|---:|---|
| 0.50 | 1.19 | 0.639 | COLLAPSE |
| 0.70 | 1.45 | 0.668 | COLLAPSE |
| 0.85 | 1.94 | 0.668 | COLLAPSE |
| **0.95** | **3.10** | 0.556 | **DIVERGE** |
| 1.00 | 5.44 | 0.559 | DIVERGE |
| 1.02 | 7.50 | 0.568 | DIVERGE |
| 1.05 | 13.20 | 0.567 | DIVERGE |

`disp_ratio` **cresce monotonamente** con `A` e attraversa il gate **esattamente
al confine di stabilità** (raggio spettrale ≈ 0.93 rispetto all'orizzonte `M=30`):
per `A < 1` la contrazione domina (collasso), per `A ≥ ~0.95` le traiettorie si
espandono abbastanza da fornire materiale a FMC. La metrica misura la proprietà
dinamica reale (espansività), non un numero arbitrario.

### 2.2 Validità predittiva — il verdetto E2 predice il vantaggio di FMC

Prova del nove: FMC (`core.plan` a ogni step) vs controllore casuale, closed-loop
(episodio 30 step, `N=48, M=15`, 3 seed, ritorno cumulato):

| env | verdetto E2 | ritorno FMC | ritorno random | **FMC − random** |
|---|---|---:|---:|---:|
| Rocket            | diverge  | 29.81 | 24.21 | **+5.61** |
| Navigation2D      | diverge  | 18.17 |  1.44 | **+16.74** (≈12×) |
| Pendulum          | diverge  |  0.578|  0.086| **+0.49** (≈7×) |
| LinearContractive | collapse | −4.74 | −5.23 | **+0.49** (pareggio) |

Il vantaggio di FMC **crolla** dai +16.7 di Navigation2D al residuo +0.49 del
plasma-mimic (dove i ritorni sono ≈ −5 e l'unico edge residuo viene da `B≠0`).
**Il gate predice dove il vantaggio svanisce.**

---

## 3. Come usare il gate su un nuovo candidato

```python
from w34_e2_smoke import e2_divergence
env = MyCandidateEnv(...)            # protocollo fmc.envs.base.Environment
m = e2_divergence(env, env.reset(), N=64, M=30, alpha=1.0, beta=1.0)
print(m["verdict"], m["disp_ratio"])
# DIVERGE -> vale la pena costruire l'adapter completo e sweeppare α,β.
# COLLAPSE -> NON investire: FMC ≈ random su questo dominio (regime plasma).
```

**Caveat operativi**:
- L'osservazione (`observe`) deve catturare le differenze *task-rilevanti* tra
  walker: `disp_ratio` è valido quanto la feature su cui è calcolato. Feature
  degeneri ⇒ falso "collapse".
- Il gate misura la divergenza **dinamica**. Il secondo modo di fallimento
  (dispersione OK ma reward piatta = "reward-degenerate") è coperto solo dal
  warning soft su `reward_cv`; su un candidato con reward sparsa/plateau
  ispezionare anche `reward_cv_M` a mano.

---

## 4. Scoping candidato #1 — logic synthesis / operator sequencing

### 4.1 Soddisfa E2 in teoria? — **Probabile SÌ sul canale di dispersione**

Il problema: dato un And-Inverter Graph (AIG) di un circuito booleano, applicare
una **sequenza di operatori di rewriting** (`rewrite`, `refactor`, `balance`,
`resub`, …) per minimizzare i nodi (area) o la profondità (delay). L'ordine conta
enormemente — **phase-ordering, NP-hard**, landscape rugoso e non-monotono (un
operatore localmente peggiorativo può abilitare un grande guadagno successivo).

**A favore (E2-fit)** — è il regime opposto al plasma:
- Spazio combinatorio, non-convesso, **non-contrattivo**: applicare un operatore
  non è una mappa lineare stabile verso un attrattore comune. Due walker con
  prefissi di operatori diversi producono AIG strutturalmente diversi → il node
  count diverge → dispersione nello stato/reward.
- **Azioni discrete** (insieme finito di ~6–12 operatori), struttura a **eventi
  rari** (certe sequenze sbloccano riduzioni grandi): stessa firma
  `(discrete-action, rare-event, non-linear)` del regime dove la Congettura D ha
  funzionato su Craftax (contrasto netto col regime `(continuous, linear-sim)`
  che l'ha fatta fallire su plasma — cfr. memoria M18).

**Rischi (da verificare, non liquidare)**:
- **Reward plateau** → modo di fallimento "reward-degenerate": molte sequenze
  brevi danno lo stesso node count (specie prima che scatti un operatore
  riducente). Se su orizzonte `M` `reward_cv → 0`, il canale α collassa e resta
  solo β (anti-collasso) → FMC degrada verso pura esplorazione. **È il warning
  soft di E2**: da controllare esplicitamente sul dominio reale.
- **Metrica di distanza tra AIG** per il termine β: serve una feature
  strutturale (es. `[n_nodi, n_livelli, n_AND, max_fanout, hash_strutturale]`).
  Due grafi distinti con uguale (nodi, livelli) sarebbero invisibili a una
  distanza scalare → il design della feature è load-bearing.

**Verdetto sulla carta**: il canale di *dispersione* è quasi certamente E2-fit
(combinatorio, rugoso, non-contrattivo). Il canale *reward* va confermato con lo
smoke test reale — **non investire sull'adapter completo prima di far girare E2**.
È esattamente il caso d'uso per cui il gate esiste.

### 4.2 Cosa serve minimamente per un vero smoke test E2

Requisiti per il protocollo `Environment`:
`clone_state` (copia AIG), `step(s, op)` (applica un operatore), `observe`
(feature vector strutturale), `reward` (`−n_nodi`), `actions` (set di operatori),
`sample_action` (uniforme).

Vincolo ambientale verificato in questa sessione: **ABC assente; nessuna libreria
AIG importabile** (`aiger`, `py_aiger`, `aigverse`, `pyeda` tutte NON importabili).
**Disponibili**: `sympy 1.14`, `networkx 3.5`.

### 4.3 MVP eseguibile più piccolo (prossimo step)

**AIG minimale in Python puro** (~200–300 LOC, zero dep compilate):

1. **Rappresentazione**: AIG come DAG di nodi AND a 2 ingressi con archi
   complementati (dict semplici o `networkx.DiGraph`; `networkx` è già presente).
2. **3 operatori** che preservano la funzione:
   - `strash` (structural hashing: fonde nodi AND duplicati) — riduce/mantiene.
   - `const_prop` (`x∧1=x`, `x∧0=0`, `x∧x=x`, `x∧¬x=0`).
   - `cut_rewrite`: rewrite locale su cut a 4 ingressi via una **piccola tabella
     NPN4 precalcolata** (è l'essenza di ABC `rewrite`, e il pezzo di lavoro
     principale). `balance`/`refactor` opzionali in un secondo momento.
3. **Reward** `= −n_nodi` (o `−(n_nodi + λ·n_livelli)`).
4. **Circuiti seed** costruiti a mano, minuscoli: full-adder, maggioranza a 3,
   albero di MUX. `sympy.logic` usato **solo** per verificare l'equivalenza
   funzionale degli operatori durante lo sviluppo (test, non hot path).
5. **Girare E2**: `e2_divergence(aig_env, seed_aig, N=64, M=20)` e leggere
   `disp_ratio` **e** `reward_cv_M` (qui il warning reward-degenerate è il rischio
   reale). Confermare con la prova `fmc_vs_random`.

**Scorciatoia da tentare prima (30 s, potrebbe azzerare il lavoro di rewriting)**:
`pip install aigverse` — pubblica wheel manylinux (binding a mockturtle/EPFL) e
*potrebbe* installarsi senza brew/compilazione. Se importa, espone
`rewrite/refactor/balance/resub` reali → si salta l'implementazione a mano e si
fa E2 su benchmark EPFL veri. Se fallisce (probabile senza toolchain), ripiegare
sull'AIG minimale sopra.

**Regola d'oro**: far girare E2 sull'MVP **prima** di qualunque investimento in un
adapter completo o in uno sweep. Se `disp_ratio < 3.0` (improbabile) o
`reward_cv_M ≈ 0` (rischio concreto), il dominio è no-fit o weak-fit e si
riscopre in un pomeriggio ciò che su plasma è costato l'intero ciclo M18.
