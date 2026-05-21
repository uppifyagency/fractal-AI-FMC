# E1-LLM — design doc pre-registrato: self-preservation con un world-model LLM

> **Tipo**: design doc pre-registrato. Scritto *prima* di qualsiasi esecuzione.
> **Data**: 2026-05-21.
> **Stato**: design completo. **Esecuzione bloccata su accesso a un'API LLM** —
> ogni cosa qui è eseguibile nel momento in cui l'API è disponibile; la
> componente *f_abs-sweep* (§5) è invece eseguibile **subito, senza LLM**.

---

## 1. Cos'è E1-LLM

E1-LLM è il terzo e ultimo test della Congettura E ([`docs/MATH_CANON.md`](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale))
— quello che chiude la stella polare. E1-base ed E2 hanno verificato la
self-preservation emergente e la separazione α/β **sul simulatore vero**. E1-LLM
chiede:

> La self-preservation emergente di E1 **sopravvive** quando il simulatore è
> sostituito da un world-model fornito da un LLM?

Se sì, il merge FMC+LLM (core agentico FMC + LLM-organo, [deep dive 02](../02_deep_dives/02_active_inference_link.md))
ha la sua prima validazione empirica sul test centrale. Se no, FMC resta un
planner che ha bisogno di un simulatore vero, e la Congettura E si chiude su
E1-base+E2.

Setup invariante rispetto a E1-base: gridworld con lava **assorbente**,
reward $=-\text{manhattan}$ ovunque, **nessuna penalità lava, nessun bonus
sopravvivenza**, kernel `fmc-core` invariato. L'unica differenza: il kernel
$\mathcal{M}$ non è il simulatore vero ma un **world-model derivato da un LLM**.

---

## 2. Cosa ha insegnato P13 — e perché cambia il design di E1-LLM

Il proxy P13 ([`P13_RESULT.md`](P13_RESULT.md), 2026-05-21) ha già risposto, *senza
LLM*, alla domanda generale "la sparsità/imperfezione del world-model degrada la
ricerca FMC?". Due lezioni che questo design **incorpora**:

1. **Il requisito load-bearing è la struttura assorbente** (hP13-1, confermata
   netta). Un surrogato che non sa quali stati sono terminali non perde solo la
   self-preservation — la *inverte*: FMC pianifica con sicurezza traiettorie
   *attraverso* la lava (death fino all'80% su `archipelago`, *peggio del
   random*). Un surrogato che preserva la struttura assorbente tiene death ≈ 0%
   anche sotto forte rumore di segnale.

2. **La metrica giusta è il death rate, non la decision-agreement.** P13 ha
   decomposto R2 in *survival* (preservata) e *fidelity* (no). E1-LLM chiede se la
   self-preservation *emerge* — è una domanda di *survival*. La decision-agreement
   con una FMC-su-simulatore-vero **non entra nel verdetto** di E1-LLM (resta
   rilevante solo per il regime goal-directed ad α alto).

Di conseguenza il design di E1-LLM **pre-registra il requisito di struttura
assorbente** come gate, e misura il **death rate** come metrica primaria.

---

## 3. Il requisito pre-registrato: fedeltà di struttura assorbente

Prima del test pieno, E1-LLM **misura** se il candidato LLM-world-model soddisfa
il requisito hP13-1. Definiamo la **fedeltà di struttura assorbente**:

$$
f_{\text{abs}} := \frac{\#\{(x,a) : \widehat{\mathcal{M}}_{\text{LLM}} \text{ predice correttamente il flag terminale di } x'\}}{\#\{(x,a) \text{ testati}\}} \in [0,1].
$$

**Probe di fedeltà assorbente** (pre-registrato). Si interroga il world-model LLM
su una batteria bilanciata di coppie $(x,a)$ che atterrano su celle lava / goal /
libere, e si misura $f_{\text{abs}}$ — la frazione in cui l'LLM predice
correttamente *se lo stato risultante è assorbente*. È un test economico (poche
centinaia di query) e si esegue **prima** del test pieno.

Soglia pre-registrata: il test pieno è informativo per la Congettura E solo se
$f_{\text{abs}}$ è alta. Dal risultato binario di P13 (abs-preserved vs
abs-broken) ci si attende che la soglia critica $f_{\text{abs}}^*$ sia alta
($\geq 0.9$) — ma il valore preciso non è noto, ed è ciò che il §5 misura.

---

## 4. Protocollo

### 4.1 Route B — world-model LLM distillato (primaria)

Schema **S2** di [`P13_DESIGN.md`](P13_DESIGN.md), raccomandato per E1-LLM *come
test* (dominio chiuso): il gridworld è chiuso e stazionario, quindi la
distillazione offline è valida (P13_DESIGN §5, §7).

1. **Costruzione del world-model.** Si fornisce a un LLM la descrizione in
   linguaggio naturale delle regole del gridworld di E1 (movimento, bordi, lava
   assorbente, goal assorbente — *senza* dire all'LLM "la lava è speciale" più di
   quanto lo dica la descrizione delle regole). Si ottiene $\widehat{\mathcal{M}}_{\text{LLM}}$
   in una di due forme:
   - **forma tabellare**: si interrogano le transizioni $(x,a)\mapsto x'$;
   - **forma codice** (preferita — "Code World Model", Tang et al. 2024): l'LLM
     *scrive il codice* del world-model dalla descrizione; FMC gira sul codice.
2. **Probe di fedeltà assorbente** (§3) → $f_{\text{abs}}$.
3. **Test pieno**: FMC con $\mathcal{M}=\widehat{\mathcal{M}}_{\text{LLM}}$,
   $\alpha\in\{0,0.1,1.0\}$, $\beta=1$, $N=64$, $M=20$, nessuna reward di
   sopravvivenza. Layout: i 6 di E1 (3 E1-base: `gauntlet`/`lake`/`scatter`;
   3 E1-robustness: `island`/`spur`/`archipelago`). Baseline: random, greedy.
   $n\geq 30$ episodi/cella. **L'episodio gira sul simulatore vero** — l'LLM è
   solo il world-model *interno* alla pianificazione FMC; l'esito (morte/goal) è
   misurato sul mondo vero.

### 4.2 Route A — world-model LLM online (estensione open-domain)

L'LLM interrogato a ogni tick (batched) come kernel $\mathcal{M}$. È il test
"pieno" della Congettura E su domini aperti, e richiede uno schema sparso $O(N)$
(S1, P13_DESIGN). **Fuori dallo scope di E1-LLM come test** — qui il gridworld è
chiuso e S2 basta. Route A è il passo successivo, su un dominio aperto.

---

## 5. Il sweep di $f_{\text{abs}}$ — eseguibile *subito*, senza LLM

P13 ha testato la struttura assorbente in modo **binario** (preservata / rotta).
E1-LLM la pre-registra come **curva**: degradando in modo controllato la fedeltà
assorbente del world-model e misurando il death rate.

Si prende $\widehat{\mathcal{M}}$ (in forma tabellare) e si **ablano** le sue voci
di celle assorbenti: una frazione $1-f_{\text{abs}}$ delle celle lava/goal viene
marcata come non-terminale. Sweep $f_{\text{abs}} \in \{1.0, 0.95, 0.9, 0.75, 0.5, 0.0\}$.
Si mappa death rate vs $f_{\text{abs}}$ → si trova la soglia $f_{\text{abs}}^*$.

> **Nota operativa importante.** Questo sweep **non richiede un LLM**: la
> degradazione controllata della fedeltà assorbente si applica a una tabella
> world-model *derivata dal kernel vero* (lo `S2Schema` di [`p13_proxy.py`](p13_proxy.py)
> + un knob di ablazione assorbente). È quindi **eseguibile subito** su `fmc-core`
> come follow-up diretto di P13 — raffina hP13-1 da binario a curva. È il pezzo
> di E1-LLM che non aspetta l'API.

---

## 6. Ipotesi pre-registrate

- **hE1L-1 (la self-preservation emerge con un world-model LLM).** A
  $f_{\text{abs}}\approx 1$, FMC a basso $\alpha$ con $\mathcal{M}=\widehat{\mathcal{M}}_{\text{LLM}}$
  evita gli stati assorbenti a un tasso $\leq$ random **e** $\leq$ greedy,
  statisticamente significativo, su $\geq 3$ layout. → la self-preservation
  emergente di E1 sopravvive alla sostituzione dell'organo world-model.
- **hE1L-2 (soglia di fedeltà assorbente).** Il death rate è monotono
  decrescente in $f_{\text{abs}}$; esiste una soglia $f_{\text{abs}}^*$ (attesa
  alta, $\geq 0.9$) sopra la quale il death rate resta $\approx 0$ e sotto la
  quale sale verso il random. — *testabile subito*, §5.
- **hE1L-3 (falsificazione).** Se a $f_{\text{abs}}=1$ il death rate è $\approx$
  random, E1-LLM è **falsificata**: la self-preservation *non* sopravvive alla
  sostituzione con un world-model LLM — qualcosa oltre la struttura assorbente è
  load-bearing, e va identificato.

---

## 7. Criteri di successo / falsificazione

- **E1-LLM verificata**: hE1L-1 vale (death $\leq$ random e $\leq$ greedy,
  significativo) su $\geq 3$ layout, a $f_{\text{abs}}\approx 1$.
- **E1-LLM falsificata**: death rate $\approx$ random a $f_{\text{abs}}=1$
  (hE1L-3) — risultato negativo, comunque pubblicabile: delimita il merge.
- **Decision-agreement esplicitamente NON nel verdetto** (P13_RESULT §3): E1-LLM
  misura se la self-preservation *emerge*, non se FMC-con-LLM replica le decisioni
  di FMC-con-simulatore-vero.

---

## 8. Cosa serve per eseguire, e costi

| Componente | Richiede LLM? | Costo |
|---|---|---|
| §5 sweep di $f_{\text{abs}}$ | **no** — tabella da kernel vero + ablazione | minuti CPU — **eseguibile subito** |
| §4.1 costruzione $\widehat{\mathcal{M}}_{\text{LLM}}$ | sì (offline, una volta) | $\sim\$80$ (stima P13_DESIGN §3, dominio chiuso) |
| §3 probe di fedeltà assorbente | sì (poche centinaia di query) | trascurabile |
| §4.1 test pieno (FMC su $\widehat{\mathcal{M}}_{\text{LLM}}$) | no (gira sul surrogato distillato) | minuti CPU |

L'esecuzione del cuore di E1-LLM (Route B) è **bloccata solo sull'accesso a
un'API LLM** per la costruzione offline del world-model — la stessa decisione
compute/API pendente nel programma. Tutto il resto è eseguibile su `fmc-core`.

---

## 9. Cosa E1-LLM (Route B) NON testa

- **Domini aperti / non-stazionari** — Route B distilla offline; vale per il
  gridworld chiuso. La Congettura E su domini aperti è Route A, un progetto a sé.
- **Gli altri tre organi LLM** (percezione, grounding, voce) — E1-LLM sostituisce
  solo il world-model. Percezione/grounding/voce restano da validare a parte
  (cf. il "buco dei 3 organi gratis" — la percezione richiede una metrica $d$
  canonica).
- **Il regime goal-directed ad α alto** — la fedeltà decisionale, scartata qui,
  vi conterebbe; serve un test separato.

---

## Riferimenti

- [`P13_RESULT.md`](P13_RESULT.md) — hP13-1 (struttura assorbente load-bearing),
  decomposizione R2-survival/R2-fidelity, verdetto GO-conditional.
- [`P13_DESIGN.md`](P13_DESIGN.md) — schemi S1/S2/S3, argomento VR-rank, §3 costi.
- [`p13_proxy.py`](p13_proxy.py) — `S2Schema` riusabile per lo sweep di $f_{\text{abs}}$ (§5).
- [deep dive 02](../02_deep_dives/02_active_inference_link.md) — perché l'LLM è
  l'organo modello-del-mondo e FMC il motore di inferenza.
- **Tang, H. et al.** (2024). *Generating Code World Models with LLMs Guided by
  MCTS*. arXiv:2405.15383 — forma "codice" di $\widehat{\mathcal{M}}_{\text{LLM}}$.

---

*Fine E1_LLM_DESIGN.md. Design pre-registrato — nessun dato raccolto. Il requisito
di struttura assorbente (hP13-1) è il gate; il death rate è la metrica; lo sweep
di $f_{\text{abs}}$ (§5) è eseguibile subito senza LLM; il cuore di E1-LLM attende
un'API LLM. Verdetto in un futuro `E1_LLM_RESULT.md`.*
