# P13 — Design doc: interrogazione sparsa dell'LLM-world-model

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione.
> **Data**: 2026-05-21.
> **Stato Congettura E**: E1-base ✓, E2 ✓, E1-robustness ✓ (caveat geometria respinto).
> Manca **E1-LLM** — bloccata da P13. Questo documento scioglie il blocco a livello
> di progettazione: non *esegue* E1-LLM, ma la rende eseguibile-su-carta e isola
> l'unico esperimento economico che la sblocca davvero.

---

## 1. Cos'è P13

P13 è la **sotto-domanda critica di fattibilità** della Congettura E
([`docs/MATH_CANON.md`](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale), P13 nella tabella predizioni):

> **P13** — Esiste uno schema di interrogazione *sparsa* dell'LLM-world-model con
> costo $O(N)$ chiamate per decisione (anziché $O(N \cdot M)$) **senza degradare
> la ricerca FMC**.

Finché P13 è aperta, il test **E1-LLM** (ripetere E1 con il simulatore sostituito
da un world-model LLM) è *teoria non eseguibile*: lo swarm impone $N \cdot M$
chiamate-LLM per singola decisione. Questo documento non risolve P13 — la
**decompone** in due rischi indipendenti, ne mostra uno già delimitato e progetta
l'esperimento che attacca l'altro.

---

## 2. Contesto architetturale — l'inversione dello stack

La Congettura E inverte l'architettura agentica standard. Non "LLM-agente + tool",
ma **core FMC = agente** (la volontà: ricerca + pulsioni $\alpha,\beta$), **LLM =
organo** (interfaccia sensomotoria). L'LLM fornisce i quattro componenti che FMC
richiede ma non possiede su domini aperti:

| Organo LLM | Funzione FMC | Chiamate / decisione |
|---|---|---|
| percezione | osservazione → stato simbolico $x_0 \in E$ | **1** |
| modello del mondo | kernel $\mathcal{M}: (x,a)\mapsto x'$, branchable | **$N \cdot M$** ← il muro |
| grounding azione | $a^* \in A$ → comando eseguibile | **1** |
| voce | stato → linguaggio | **≤ 1** |

Tre organi su quattro costano $O(1)$ chiamate per decisione. **Tutto il muro di
compute è il singolo organo "modello del mondo"**: è l'unico interrogato dentro il
doppio loop walker × tick. P13 riguarda esclusivamente quell'organo. Gli altri tre
non sono in scope qui (sono un problema di prompt engineering, non di complessità).

---

## 3. Il muro di compute — modello di costo

Per una singola decisione FMC con $N$ walker e orizzonte $M$:

$$
C_{\text{decisione}} = \underbrace{N \cdot M}_{\text{world-model}} + \underbrace{O(1)}_{\text{percezione + grounding + voce}} \text{ chiamate-LLM.}
$$

Numeri concreti, configurazione E1-base ($N=64$, $M=20$):

| Quantità | Valore |
|---|---|
| chiamate-LLM / decisione | $64 \times 20 + 3 \approx 1283$ |
| decisioni / episodio (gridworld, $T\!\le\!100$) | $\sim 100$ |
| chiamate-LLM / episodio | $\sim 1.28 \times 10^5$ |
| token / chiamata world-model (stima: stato+azione in prompt, stato' in completion) | $\sim 800$ |
| token / episodio | $\sim 1.0 \times 10^8$ (100 M) |
| episodi per un E1-LLM con la $n$ di E1-robustness (3 layout × 60) | 180 |
| **token totali E1-LLM** | **$\sim 1.8 \times 10^{10}$ (18 miliardi)** |

A un prezzo *ottimistico* da modello piccolo ($\sim\$1$/Mtok blended) E1-LLM costa
$\sim\$18\,000$; a prezzi da modello di frontiera ($\$5\text{–}15$/Mtok),
$\$90\,000\text{–}270\,000$. **E1-LLM non è un "run overnight"** — non senza una
mitigazione. Il muro è reale e questo giustifica P13 come gate.

### 3.1 Decomposizione del muro — due rischi indipendenti

Il muro $N\cdot M$ confonde due problemi distinti. Separarli è il primo contributo
di questo design:

- **R1 — costo & latenza.** Quante chiamate, quanti token, quanti secondi. È una
  domanda di *ingegneria ed economia*. Mitigabile con leve note (batching,
  caching, modelli piccoli) e **delimitabile a priori** — vedi §3.2. Non richiede
  un esperimento di ricerca.
- **R2 — degradazione della ricerca.** Se interrogo l'LLM *sparsamente* (salto
  tick, salto walker, uso un surrogato per la parte profonda del rollout), la
  qualità della *decisione* FMC tiene? È una domanda *scientifica*. E — punto
  chiave di questo documento — **è testabile senza alcun LLM** (§5).

P13 enunciata correttamente è quasi tutta R2: "*senza degradare la ricerca FMC*".
R1 è l'inviluppo di fattibilità entro cui R2 va verificata.

### 3.2 R1 — inviluppo, non esperimento

Una leva ortogonale agli schemi del §4: dentro un singolo tick, le $N$ chiamate
world-model sono **indipendenti** (un walker non dipende dall'altro allo stesso
tick). Si collassano in **una richiesta batched** di $N$ item, o $N$ richieste in
continuous-batching. Effetto: i $N\cdot M$ round *sequenziali* diventano $M$ round
(la dipendenza residua è solo lungo i tick, $t\!+\!1$ dipende da $t$).

- **Latenza**: con batching, $\sim M$ round × $1\text{–}3$ s ≈ $20\text{–}60$ s per
  decisione. Risolvibile.
- **Costo in token**: il batching **non lo cambia** ($N\cdot M$ inferenze restano
  $N\cdot M$ inferenze). Se il collo di bottiglia è il costo in token, serve uno
  degli schemi del §4; se è solo la latenza, basta il batching.

Conclusione su R1: è un inviluppo stimabile (§3, tabella), non una domanda aperta.
**Il resto del documento è R2.**

---

## 4. Perché P13 può funzionare — l'argomento VR-rank

Prima degli schemi, l'argomento teorico per cui la sparsità *può* non degradare la
ricerca. È la ragione per cui P13 non è disperata.

La decisione FMC finale è (Def. 1, MATH_CANON):
$$
a^* = \arg\max_{a \in A} \#\{i : \ell^{(i)}_M = a\}.
$$
Le etichette $\ell^{(i)}$ evolvono **solo** via cloning (Def. 4), con rate
$\rho_{\text{clone}}(i\to k) = (\mathrm{VR}^{(k)}-\mathrm{VR}^{(i)})/\mathrm{VR}^{(i)}$.
E $\mathrm{VR} = \widehat{R}^\alpha \cdot \widehat{D}^\beta$ (Def. 3), dove
$\widehat{R},\widehat{D}$ sono output di `relativize` (Def. 2): z-score seguito da
una mappa monotona, **affine-invariante** ($\widehat{R}(a\mathbf{r}+b)=\widehat{R}(\mathbf{r})$
per $a>0$).

Conseguenza, in **tre livelli di tolleranza** all'errore del world-model:

1. **Errore affine** in $R$ o $d$ (il surrogato produce $a\cdot r + b$, $a>0$):
   per l'invarianza affine di `relativize`, la $\mathrm{VR}$ è **identica** →
   stesso $a^*$, stesso $b_{\text{eff}}$. Tolleranza piena, gratis.
2. **Errore monotono non affine** (rank-preserving ma non affine): l'*ordinamento*
   di $\mathrm{VR}$ è preservato, le *magnitudini* no. Il cloning preserva allora
   la **direzione** di ogni clone (segno di $\mathrm{VR}^{(k)}-\mathrm{VR}^{(i)}$)
   → la direzione del flusso di etichette → $a^*$ (l'argmax è robusto al rango);
   cambiano i *tassi* $\rho_{\text{clone}}$, quindi $b_{\text{eff}}$ e la velocità
   di concentrazione. Tolleranza sulla *decisione*, non sul transitorio.
3. **Errore non monotono** (viola il rango di $\mathrm{VR}$): cambia la direzione
   di qualche clone → il flusso di etichette diverge → $a^*$ può cambiare. **È
   l'unico modo di fallimento che conta.**

> **P13 non richiede un world-model traiettoria-accurato. Richiede un world-model
> il cui errore sia, al peggio, *monotono* in $R$ e $d$** — cioè rank-preserving
> sulla $\mathrm{VR}$. Requisito molto più debole della simulazione metrica
> esatta, e allineato a ciò in cui un LLM è bravo (giudizi relativi/ordinali) vs
> debole (predizione metrica). L'unico nemico è l'errore *rank-violating*.

Questo argomento è **esso stesso una predizione falsificabile** (hP13-0, §6) e
inquadra cosa deve preservare uno schema sparso: non la traiettoria, non le
magnitudini di $\mathrm{VR}$, ma l'**ordinamento di $\mathrm{VR}$** tra walker ai
tick di cloning.

---

## 5. I tre schemi candidati — formalizzati

Per ciascuno: meccanismo, dove cade la chiamata-LLM, complessità, assunzione
load-bearing, modo di fallimento.

### S1 — Root-expansion + surrogato

**Meccanismo.** L'LLM-world-model espande **solo il tick 0**: le $N$ transizioni
$(x_0, \ell^{(i)}) \mapsto x_1^{(i)}$. I tick $1\ldots M\!-\!1$ girano su un
surrogato economico $\widehat{\mathcal{M}}$ (simbolico, oppure una piccola NN, o
una versione approssimata del kernel).

**Costo.** $O(N)$ chiamate-LLM / decisione (le $N$ di root). Riduzione $\times M$.

**Assunzione load-bearing.** Il surrogato deve preservare le proprietà del kernel
che contano per il *re-weighting di VR* lungo il rollout — in particolare, la
struttura degli **stati assorbenti** (cf. converso locale del Teorema 3,
[`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md)): la self-preservation di E1
*non* è un fenomeno di tick-0 — la diagnostica E1-robustness mostra la frazione di
label "verso-lava" decadere $19.5\% \to 13\% \to 7.2\% \to 2.9\% \to 0.1\%$ lungo
*tutto* l'orizzonte. Se il surrogato non sa che "lava è assorbente", S1 perde il
meccanismo che fa funzionare E1.

**Modo di fallimento.** Surrogato troppo degradato → il rollout profondo è rumore
→ FMC decide su segnale tick-0 puro (miope) o su rumore.

### S2 — Distillazione offline

**Meccanismo.** Interroga l'LLM-world-model **una volta**, offline, con budget $B$
transizioni, per generare traiettorie; distilla in un surrogato veloce
$\widehat{\mathcal{M}}$. Forma forte: distillare in **codice eseguibile** (un
"code world-model", cf. Tang et al. 2024, arXiv:2405.15383 — un LLM genera il
codice che *è* il world-model, guidato da MCTS). FMC gira interamente su
$\widehat{\mathcal{M}}$: **zero chiamate-LLM nel loop**.

**Costo.** $O(1)$ ammortizzato online ($B$ offline, ammortizzato su tutte le
decisioni di tutti gli episodi).

**Assunzione load-bearing.** Il dominio è abbastanza **stazionario** perché un
$\widehat{\mathcal{M}}$ distillato offline resti valido a tempo di esecuzione.

**Modo di fallimento.** Domini aperti / non-stazionari — dove il surrogato va
stantìo — sono *esattamente* i domini per cui si voleva l'LLM. S2 **non risolve
P13**: scambia il muro di compute con un muro di generalizzazione. Resta valido
per domini chiusi (e per E1-LLM stessa, che è un gridworld chiuso — vedi §7).

### S3 — Gerarchico / macro-azioni

**Meccanismo.** L'LLM propone un **menu ristretto di macro-azioni** ("date le
osservazioni, quali sono le $k$ cose sensate da fare qui?", **1 chiamata**); FMC
cerca le *conseguenze* delle macro-azioni su un sim simbolico economico
(cf. HANDOFF Tier-2E).

**Costo.** $O(1)$ chiamate-LLM / decisione.

**Assunzione load-bearing.** Esiste un sim simbolico economico delle conseguenze
delle macro-azioni.

**Modo di fallimento.** Se hai già un sim economico delle conseguenze, l'organo
"modello del mondo" LLM è quasi superfluo — S3 degenera in "FMC su sim vero, LLM
sceglie solo il menu di azioni". Restringe il ruolo del world-model LLM a
$\sim 0$. È onesto solo chiamarlo *mitigazione del muro* quando il sim economico
è genuinamente disponibile e l'LLM aggiunge valore solo nel pruning del menu.

### Sintesi

| Schema | Chiamate-LLM/decisione | Risolve P13 su domini aperti? | Rischio principale |
|---|--:|---|---|
| baseline (full) | $N\cdot M \sim 10^3$ | — | costo |
| **S1** root + surrogato | $O(N)$ | sì, se il surrogato preserva gli assorbenti | drift del surrogato |
| **S2** distillazione | $O(1)$ amm. | **no** (solo domini chiusi) | non-stazionarietà |
| **S3** macro-azioni | $O(1)$ | sì, ma riduce il ruolo del world-model | richiede sim economico |

Lettura pre-registrata: **S1 è l'unico candidato che riduce il costo *e* mantiene
l'LLM come world-model su domini aperti.** S2 è la via per E1-LLM *come test*
(gridworld chiuso). S3 è una mitigazione architetturale, non una risposta a P13
in senso stretto.

---

## 6. L'esperimento proxy — testare R2 senza LLM

Il cuore operativo del design. R2 ("la sparsità degrada la ricerca?") **non
richiede un LLM per essere testato.** Richiede solo: un kernel vero noto + un modo
di degradarlo in modo controllato. Lo abbiamo già — è il gridworld di E1.

### 6.1 Idea

Sul `gridworld_terminal` (kernel vero, $\mathcal{M}$ noto e deterministico),
girare FMC sotto "budget di interrogazione" che **emulano ciascuno schema**, e
misurare se la decisione FMC tiene rispetto al controllo full. Nessun LLM, nessuna
GPU: gira su `fmc-core` NumPy come E1-base ($\sim$ minuti).

### 6.2 Bracci

- **FULL** (controllo). Kernel vero a ogni $(walker, tick)$ — è E1-base/E1-robustness.
- **S1-proxy — root + surrogato rumoroso.** Tick 0: kernel vero per tutti gli $N$
  walker. Tick $1\ldots M\!-\!1$: surrogato = kernel vero **corrotto** con rumore
  di livello $\eta$ (perturbazione dell'osservazione che entra in $R,d$) — *ma con
  gli stati assorbenti preservati* in un sotto-braccio, e *non preservati* in un
  altro. Sweep $\eta$. Misura: a quale fedeltà del surrogato la decisione FMC
  eguaglia FULL — e se la preservazione degli assorbenti è necessaria.
- **S2-proxy — distillazione.** Costruisci $\widehat{\mathcal{M}}$ campionando il
  kernel vero offline con budget $B$ transizioni (approssimazione tabellare o
  k-NN). FMC gira su $\widehat{\mathcal{M}}$. Sweep $B$. Misura: quante transizioni
  offline finché FMC-su-$\widehat{\mathcal{M}}$ ≈ FMC-su-vero.
- **S3-proxy — macro-menu.** Restringi $A$ a un menu di $k<K$ macro-azioni
  (menu-oracolo: top-$k$ per una euristica); FMC cerca sul sim vero. Misura: la
  restrizione del menu preserva lo $0\%$ morte di E1?

### 6.3 Metriche

1. **Esiti E1/E2**: death rate, goal rate (con IC95 di Wilson), sui 3 layout
   avversariali di E1-robustness (`island`, `spur`, `archipelago`).
2. **Decision-agreement**: frazione di passi-decisione in cui il braccio sparso
   sceglie lo *stesso* $a^*$ di FULL (dallo stesso stato, stesso seed).
3. **Diagnostica VR-rank** (collega §4): correlazione di rango (Spearman) tra il
   vettore $\mathrm{VR}$ del braccio sparso e quello di FULL. Si riportano **tre**
   numeri — media sui tick, **tick peggiore**, e media sui **tick decisivi** — per
   non far nascondere alla media un crollo di rango localizzato. È la misura
   diretta dell'argomento VR-rank del §4.

### 6.4 Ipotesi pre-registrate

Registrate *prima* dei dati, in linea con la disciplina del progetto
(CLAUDE.md §4; MATH_CANON).

- **hP13-0 (argomento VR-rank).** Un surrogato che preserva l'**ordinamento di
  VR** preserva la decisione FMC (decision-agreement ≥ 0.9), *anche se*
  traiettoria-inaccurato. La preservazione del rango va verificata **ai tick
  decisivi** (quelli in cui la frazione di label "verso-assorbente" cala di più —
  cf. la diagnostica E1-robustness), non solo in media: una Spearman media alta
  può mascherare un crollo di rango ai pochi tick che contano. — *Falsificata* se
  decision-agreement crolla pur con rango preservato ai tick decisivi.
- **hP13-1 (S1).** S1-proxy preserva lo $0\%$ morte di E1 **se e solo se** il
  surrogato preserva la proprietà assorbente delle celle di lava. Predizione:
  sotto-braccio "assorbenti preservati" PASS a $\eta$ moderato; sotto-braccio
  "assorbenti non preservati" FAIL anche a $\eta$ piccolo. — Sharp: lega P13 al
  meccanismo della converso-locale-Teorema-3.
- **hP13-2 (S2).** Esiste un budget $B^*$ finito oltre il quale FMC-su-$\widehat{\mathcal{M}}$
  è statisticamente indistinguibile da FULL sul gridworld chiuso. Predizione:
  $B^*$ scala con la copertura dello spazio stati ($\sim |E|\cdot|A|$). — Su
  domini aperti $B^*$ diverge: documentato come confine di scope, non testato.
- **hP13-3 (S3).** Il macro-menu preserva E1 **se** include almeno un'azione
  non-letale per stato (vero per costruzione se il menu contiene "resta"/direzioni
  sicure). — Quasi una tautologia: serve a *delimitare*, non a stupire.

### 6.5 Criterio go/no-go per P13

Uno schema **"passa R2"** se, sui 3 layout E1-robustness:

- il death rate resta dentro l'IC95 di Wilson del death rate di FULL, **e**
- decision-agreement ≥ 0.85, **e**
- non introduce attrazione verso gli assorbenti (death rate ≤ FULL + 5 pp).

**Decisione P13:**

- **GO-test** se S2 (distillazione) passa R2 sul gridworld chiuso. Sufficiente per
  rendere **E1-LLM eseguibile** come test (dominio chiuso, §7) — *ma non risolve
  P13 in senso pieno*, perché S2 scambia il muro di compute con un muro di
  generalizzazione (§5).
- **GO-full** se S1 (root + surrogato, con assorbenti preservati) passa R2 a un
  costo dentro l'inviluppo R1 del §3 ($\le\$1000$ per E1-LLM completo). Solo
  questo chiude **P13 come predizione** — uno schema $O(N)$ valido anche su domini
  aperti.
- **NO-GO** se nessuno schema passa R2: P13 falsificata — la sparsità degrada
  intrinsecamente la ricerca FMC. E1-LLM resta non eseguibile a costo ragionevole;
  la Congettura E si chiude su E1-base+E2 (risultato comunque pubblicabile) e si
  documenta il limite.

Nota: passare il proxy del §6 è **necessario non sufficiente** — il proxy testa
R2 con un modello di errore ottimista (§8). Un GO del proxy autorizza E1-LLM, non
la dichiara già vinta.

---

## 7. Percorso a E1-LLM dopo P13

Se P13 è GO, E1-LLM diventa eseguibile. Nota di scope: **E1-LLM è essa stessa un
gridworld chiuso** — il test sostituisce il simulatore con un LLM-world-model *sul
medesimo gridworld di E1*, per misurare se la self-preservation emerge ancora
quando il modello del mondo è un LLM invece di un simulatore esatto. Essendo
chiuso, **S2 (distillazione) è sufficiente per E1-LLM come test**, anche se non
risolve P13 in generale. La distinzione conta:

- **E1-LLM (il test)** — domani, dominio chiuso → S2 basta, costo $\sim\$80$ offline.
- **Congettura E su domini aperti** (la stella polare vera) → serve S1, ed è dove
  il rischio "drift del surrogato" va davvero affrontato.

---

## 8. Cosa questo design NON copre

- **Non risolve R1 oltre la stima.** L'economia in token del §3 è un inviluppo, non
  un benchmark di serving.
- **Non testa l'accuratezza di un LLM come world-model.** Il proxy del §6 usa un
  surrogato *derivato dal sim vero*: misura "la sparsità degrada la ricerca?", non
  "un LLM è un buon modello del mondo?". Quest'ultima è E1-LLM proper.
- **Il modello di errore del proxy è ottimista.** S1-proxy corrompe il kernel con
  rumore $\eta$ *non strutturato*. Un LLM-world-model sbaglia in modo **strutturato
  e correlato** (stati plausibili-ma-sbagliati, bias sistematici) — ed è l'errore
  strutturato che può essere *rank-violating* (§4) in modo coerente, il modo di
  fallimento che conta. Un PASS del proxy con rumore non strutturato è perciò un
  **limite inferiore di difficoltà**: necessario, non sufficiente. Un braccio
  S1-proxy con corruzione *strutturata* (bias direzionale dipendente dallo stato)
  va aggiunto prima di fidarsi del verdetto per E1-LLM su dominio aperto.
- **Non copre i domini non-stazionari.** hP13-2 li marca come confine di scope.
- **Non tocca gli altri tre organi** (percezione/grounding/voce) — $O(1)$, fuori
  dal muro, problema di prompt engineering.

---

## 9. Costi e prossimi passi

| Passo | Costo | Output |
|---|---|---|
| `p13_proxy.py` — i 3 bracci proxy sul gridworld | $\sim$ minuti CPU (come E1-base) | `P13_RESULT.md`, decisione GO/NO-GO |
| Se GO: E1-LLM via S2 (distillazione, gridworld chiuso) | $\sim\$80$ offline + compute trascurabile | chiude la stella polare su dominio chiuso |
| Congettura E su dominio aperto: S1 + studio del drift | progetto a sé | fuori dallo scope di P13 |

**Prossimo passo immediato**: implementare ed eseguire `p13_proxy.py` secondo §6,
poi scrivere `P13_RESULT.md` con il verdetto pre-registrato del §6.5.

---

## 10. Riproducibilità (quando il proxy sarà eseguito)

```bash
cd "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI"
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
"$PY" work/12_conjecture_e/p13_proxy.py    # da implementare secondo §6
```

Kernel `fmc-core` invariato (Strato 1 congelato): il proxy riusa solo le funzioni
pubbliche di `plan()`, come `e1_robustness_diag.py`.

---

## Riferimenti

- **MATH_CANON** §Congettura E, P13; Def. 1–4 (decisione, relativize, VR, cloning).
- **Hao, S. et al.** (2023). *Reasoning with Language Model is Planning with World
  Model*. arXiv:2305.14992. — LLM ri-prompato come world-model + MCTS (RAP).
- **Zhao, Z. et al.** (2023). *Large Language Models as Commonsense Knowledge for
  Large-Scale Task Planning*. NeurIPS 2023 (LLM-MCTS). — LLM come world-model +
  policy euristica dentro MCTS.
- **Tang, H. et al.** (2024). *Generating Code World Models with LLMs Guided by
  Monte Carlo Tree Search*. arXiv:2405.15383. — distillazione del world-model in
  **codice eseguibile** (forma forte di S2).
- [`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md) — converso locale del
  Teorema 3 (stati assorbenti = pozzi di VR); meccanismo che S1 deve preservare.
- [`HANDOFF.md`](HANDOFF.md) — programma research-partner, Tier-2E.

---

*Fine P13_DESIGN.md. Design pre-registrato — nessun dato ancora raccolto. Il
verdetto GO/NO-GO si decide in `P13_RESULT.md` secondo i criteri del §6.5.*
