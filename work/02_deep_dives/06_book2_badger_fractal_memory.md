# Deep Dive 06 — Book #2 (AGI Structure) + Fractal Memory

> *"Il Book #1 ti convince che FMC è un planning algorithm. Il Book #2 ti convince che è un mattone AGI. Lo Slide doc ti convince che è un principio universale di apprendimento. Le tre conclusioni sono diverse e nessuna è completa senza le altre."* — [`analisisPost2.md` §9](../../analisisPost2.md)

> **Stato**: scrittura completa. Lunghezza: ~1100 righe.
> **Sorgenti**: [`Fractal Book.md`](../../Fractal%20Book.md), [`2020 Fractal.md`](../../2020%20Fractal.md), [`2020 Fractal Slide.md`](../../2020%20Fractal%20Slide.md).

---

## 0. Sintesi (per chi ha fretta)

Dopo aver pubblicato il paper *Fractal AI: A Fragile Theory of Intelligence* (Book #1, arXiv:1803.05049v5, 2020), Hernández-Cerezo & Duran-Ballester hanno scritto **almeno tre documenti aggiuntivi** che estendono il programma:

1. **Book #2 — AGI Structure (V0.2)**: la specifica dell'architettura AGI completa, basata su una **Badger structure** di loop FMC nidificati su 5 livelli ontologici (Observe → Embed → Predict → Plan → Reward).
2. **Honey Badger meets Fractal AI Hives (2020)**: la specifica operativa con pseudocodice. Risponde alla domanda *"come implemento concretamente questo?"* e introduce il concetto-chiave di **Learning as Structural Collapse**.
3. **Fractal Memory (2020 Fractal Slide)**: l'estensione di FMC dentro le reti neurali stesse, con tre meccanismi:
   - Dataset come Fractal Memory → curriculum learning automatico, no catastrophic forgetting
   - Sinapsi come Fractal Memory → self-pruning architecture
   - NN come Fractal Memory → multi-task con specializzazione automatica (= Mixture-of-Experts non-supervisionato)

In questo deep dive analizziamo i tre documenti come un **corpus unico**, mostriamo la coerenza interna del programma, identifichiamo i punti di forza teorici e le ipotesi falsificabili, e proponiamo una roadmap di implementazione step-by-step.

**Take-away principale**: il vero "sogno Fractal AI" non è FMC. È l'idea che **lo stesso algoritmo si applica ricorsivamente a livelli ontologicamente diversi** — dai walker che pianificano azioni, ai pesi che si ottimizzano, alle reti che si specializzano. È **frattalità computazionale**: stessa struttura a tutte le scale.

---

## 1. La traiettoria intellettuale degli autori

### 1.1 Tre fasi, un programma

| Anno | Documento | Domanda chiave | Risposta |
|---|---|---|---|
| 2018-2020 | **Book #1** (arXiv 1803.05049) | Come pianifica un agente intelligente? | FMC: sciame di walker che si clonano proporzionalmente a `R · Dist` |
| 2020 | **Book #2** + Hives + Slide | Come si compone un'**AGI** usando FMC come mattone? | Badger structure di sciami nidificati a 5+ livelli; Fractal Memory per le NN |
| 2024-2026 | **`fragile-rl`** (Fragile Mechanics) | Quale geometria/fisica governa la cognizione? | Gauge theory cognitive + WFR + holographic interface |

Questa è una progressione **bottom-up** atipica nel campo AGI. La maggioranza dei programmi fa il contrario — parte da una visione cosmologica e cerca di costruire mattoni. Hernández-Cerezo & Duran-Ballester partono dal mattone (FMC) e salgono.

### 1.2 La conseguenza pratica

Significa che ogni fase è **verificabile in isolamento**:
- Book #1 è verificato (l'ho fatto io con `fmc_minimal.py` e Boxing 96/100 — vedi [`SMOKE_TEST_REPORT.md`](../03_atari_replication/results/SMOKE_TEST_REPORT.md))
- Book #2 è in larga parte specifica formale, non ancora verificato sperimentalmente in modo pubblico
- `fragile-rl` è stato avviato come framework PyTorch, ma è ancora un work in progress

Questo fa di Book #2 il **target privilegiato** per chiunque voglia contribuire empiricamente al programma — è la fase dove c'è maggior rapporto rischio/payoff.

---

## 2. Book #2: l'analisi delle "external functions" di FMC

### 2.1 Il problema sollevato

Il Book #1 lascia un buco operativo che il Book #2 affronta direttamente (§2.1):

> *"For every external function used in FMC, an equivalent learning process must be defined instead, and only then, the FMC could had evolved into a full AGI."*

Le quattro funzioni esterne di FMC sono:

| Funzione | Cosa fa nel Book #1 | Cosa serve nel Book #2 |
|---|---|---|
| `Observation()` | Restituisce il vettore di stato (es. RAM Atari) | Embedding learning (VAE) — comprime input grezzi in latent space |
| `Distance(s1, s2)` | Distanza tra stati (gratis se gli stati sono vettori meaningful) | Distanza nel latent space (gratis dal VAE) |
| `Simulation(state, a, dt)` | Calcola lo stato successivo deterministicamente (ALE) | World model learning (LSTM) — predice distribuzione su next state |
| `Reward(state)` | Valuta uno stato (score Atari) | Reward function learning (ANN) — quale reward massimizzare |

**Conseguenza importante**: nel passaggio da Book #1 a Book #2, FMC smette di essere un *algoritmo standalone* e diventa il **livello expert** di una struttura più grande.

### 2.2 Il "Full state" generalizzato

Book #2 introduce un'idea con conseguenze profonde (§2.3):

$$
\text{Full\_state} = \langle \text{Observation}, \text{Embedding}, \text{Next embedding}, \text{Action}, \text{Reward} \rangle = \langle P_0, P_1, P_2, P_3, P_4 \rangle
$$

Lo "stato" dell'agente non è solo l'osservazione del mondo. È un **vettore composito** che include:

- $P_0$ = osservazione grezza (sensori)
- $P_1$ = embedding compresso (output del VAE)
- $P_2$ = predizione del prossimo embedding (output dell'LSTM)
- $P_3$ = azione scelta (output del planner)
- $P_4$ = reward attesa (output del reward module)

Questa è la stessa idea del **predictive processing** in neuroscienze (Andy Clark 2013, Friston 2010): la mente non rappresenta passivamente il mondo, **costruisce attivamente uno stato interno** processando l'input. Book #2 operativizza questa idea in modo computazionalmente concreto.

> **Insight epistemico**: ogni "porzione" $P_i$ è la responsabilità di un livello specifico del Badger. I livelli non competono per lo stato — si **dividono il lavoro** di costruirlo.

### 2.3 Il pattern algoritmico

Per ogni modulo (livello), Book #2 specifica una struttura comune (§2.3):

1. **Ini**: riceve un full state dal modulo precedente
2. **New**: costruisce una nuova porzione $P_i$
3. **Update**: sostituisce la porzione $P_i$ del full state con quella nuova

Questo dà luogo a una "catena di assemblaggio" del full state:

```
Sensori → Observation → Embedding → Next embedding → Action → Reward
   P0   →   updates P0  → updates P1 → updates P2  → updates P3 → updates P4
```

Ogni modulo è un **trasformatore parziale**: prende il full state, modifica solo la porzione di sua competenza, lo passa avanti.

---

## 3. La Badger Structure: il telaio dell'AGI

### 3.1 La metafora di Elon Musk

Book #2 §3.1 introduce il Badger via una storia narrativa (decisamente non-accademica):

> *"Elon Musk wants to build the perfect car, but an experiment said the grip was not ok. Elon asks an expert car driver to test the car..."*

Il driver dice "il problema sono i dumper". Elon chiede a un ingegnere dei dumper. L'ingegnere dice "il problema è il fluido viscoso". Elon chiede a un chimico. Il chimico propone una formula. Si testa. Non funziona. Il chimico aggiorna le sue priors e propone una nuova formula. Eccetera.

Questo è il **flow gerarchico** del Badger:
- **Discesa**: il problema scende dai livelli alti ai livelli bassi (driver → ingegnere → chimico)
- **Risalita**: la soluzione sale dai livelli bassi ai livelli alti (chimico → ingegnere → driver)
- **Aggiornamento**: ogni esperto aggiorna le sue priors quando riceve feedback

La metafora è informale ma cattura una struttura computazionale precisa.

### 3.2 La struttura formale

Book #2 §3.2 + `2020 Fractal.md` definiscono i livelli:

```
┌──── OUTER LOOP (Controller) ────────────────────────────────────┐
│  Riceve osservazioni, decide il flow temporale                   │
│                                                                  │
│  ┌── Level 4: Architecture optimizer (opzionale) ──────┐        │
│  │  Sciame di architetture ANN candidate                │        │
│  │                                                       │        │
│  │  ┌── Level 3: Embedding optimizer ──────────┐       │        │
│  │  │  Sciame di matrici di pesi VAE            │       │        │
│  │  │                                            │       │        │
│  │  │  ┌── Level 2: Prediction optimizer ─┐   │       │        │
│  │  │  │  Sciame di matrici di pesi LSTM   │   │       │        │
│  │  │  │                                    │   │       │        │
│  │  │  │  ┌── Level 1: Reward optimizer ┐ │   │       │        │
│  │  │  │  │  Sciame di funzioni reward   │ │   │       │        │
│  │  │  │  │                                │ │   │       │        │
│  │  │  │  │  ┌── EXPERT LEVEL ────────┐   │ │   │       │        │
│  │  │  │  │  │ FMC swarm di walker    │   │ │   │       │        │
│  │  │  │  │  │ pianifica le azioni    │   │ │   │       │        │
│  │  │  │  │  └────────────────────────┘   │ │   │       │        │
│  │  │  │  └─────────────────────────────────┘ │   │       │        │
│  │  │  └────────────────────────────────────────┘   │       │        │
│  │  └──────────────────────────────────────────────────┘       │        │
│  └────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Cosa c'è dentro ogni livello

Ogni livello è uno **sciame FMC** che ottimizza qualcosa di diverso:

| Livello | Sciame fatto di | Reward usata | Metrica di distanza |
|---|---|---|---|
| Expert | Walker (stati simulati) | Reward esterna del task | Distanza in state space (RAM/embedding) |
| Reward (L1) | Reward function ANN | Reward intrinseca: cross-entropy reward predetta vs reward osservata | Distanza tra parametri ANN |
| Prediction (L2) | LSTM ANN | Reward intrinseca: accuratezza predizione next embedding | Distanza tra parametri LSTM |
| Embedding (L3) | VAE ANN | Reward intrinseca: accuratezza ricostruzione | Distanza tra parametri VAE |
| Architecture (L4) | Architetture (vettori che descrivono la struttura) | Reward intrinseca: accuratezza task / parsimonia | Distanza strutturale |

> **Punto chiave**: lo stesso algoritmo FMC si applica 5 volte di seguito, su 5 livelli ontologicamente diversi.

### 3.4 Le tre famiglie

Book #2 §"Internal structure" raggruppa i livelli in tre famiglie:

1. **Loops** (livelli 1-4): fanno **learning**. Hanno un ANN da addestrare e una popolazione di inner loops.
2. **Expert level**: fa **planning federato**. Esperti con parametri vari (es. balance esplorazione-sfruttamento) ma stessi ANN.
3. **Walkers** (dentro gli esperti): fanno **exploration**. Lo sciame FMC vanilla del Book #1.

Questa divisione ricalca la struttura cognitiva classica:
- Imparare il modello del mondo (loops)
- Decidere strategia (expert)
- Esplorare opzioni concrete (walkers)

---

## 4. Il workflow del Badger: vita di un'iterazione

### 4.1 Inizializzazione (ricorsiva)

Da `2020 Fractal.md`:

```python
def new_loop(level: int, count: int, fixed_NN_so_far: list):
    """Crea ricorsivamente un loop a partire dal livello dato."""
    NN_at_this_level = new_NN(structure=structures[level], params=random)

    if level > 1:
        # Crea N inner loops, uno per ogni candidate weight matrix
        loops = []
        for n in range(count):
            inner = new_loop(
                level=level-1,
                count=count,
                fixed_NN=fixed_NN_so_far + [NN_at_this_level]
            )
            loops.append(inner)
        return loops
    else:
        # Livello expert: popola con sciami FMC
        experts = []
        for n in range(count):
            experts.append(fragile.swarm(
                n_walkers=count,
                modules=fixed_NN_so_far  # tutti i moduli ereditati dai livelli sopra
            ))
        return experts
```

Conseguenza dimensionale: con `count=N` e `levels=5`, abbiamo:
- $N$ scelte di Architecture
- × $N$ scelte di Embedding params per architettura
- × $N$ scelte di Prediction params per embedding
- × $N$ scelte di Reward params per prediction
- × $N$ esperti
- × $N$ walker per esperto

= $N^6$ unità totali. Con $N = 50$: $1.56 \times 10^{10}$ unità. Numericamente impraticabile.

**Soluzione**: la struttura collassa rapidamente (vedi §5).

### 4.2 Training step

Pseudocodice di alto livello (sintesi del Book #2 §3.4 + Hives):

```
For each cycle:
  1. Each expert plays an episode → produces (trajectory, final_reward)
  2. Expert level performs FMC step:
     - distance = ‖params_i - params_j‖
     - reward_i = expert_i.final_reward
     - virtual_reward = relativize(reward) · relativize(distance)
     - cloning probabilistic
  3. Expert level exposes best trajectory + reward to L1 (Reward) loop
  4. L1 loop performs FMC step on reward function ANNs:
     - distance = ‖reward_params_i - reward_params_j‖
     - reward_i = expert_level_i.best_reward
     - cloning of ANN params
  5. L1 trains its ANN on the trajectory dataset
  6. Recurse: L2 (Prediction) does the same on its LSTM, L3 (Embedding) on its VAE
  7. Each level keeps a sliding window of last M parameter averages
  8. If average doesn't change → collapse the level
```

### 4.3 Cloning: stesso meccanismo a tutti i livelli

Book #2 §3.4.1:

> *"As a matter of fact, we will always use the idea of virtual reward and clone probability from FMC but, for the sake of generality, here we allowed for any other method you may find of interest."*

Ogni livello usa la stessa formula del paper #1:

$$
P_{\text{clone}}(i \to k) = \begin{cases}
1 & \text{se } VR_i = 0 \\
0 & \text{se } VR_k \leq VR_i \\
(VR_k - VR_i) / VR_i & \text{altrimenti}
\end{cases}
$$

Solo che a livelli diversi $VR_i = \text{relativize}(R_i)^\alpha \cdot \text{relativize}(D_i)^\beta$ dove $R$ e $D$ cambiano semantica:

- A livello expert: reward del task, distanza in state space
- A L1: reward intrinseca su predizione di reward, distanza tra ANN reward
- A L2: reward intrinseca su predizione next state, distanza tra ANN LSTM
- E così via

Questa è l'**autocoerenza algoritmica**: stesso meccanismo, semantica scalata sul livello.

---

## 5. Learning as Structural Collapse — l'idea geniale

### 5.1 Il problema del Badger gigante

Come notato in §4.1, un Badger pieno ha $\sim N^6$ unità. Insostenibile.

`2020 Fractal.md` propone una soluzione elegante (§"Learning as a structural collapse"):

> *"As the training process evolves, we could consider one of the modules to be already learned when the averaged parameter matrix coming from the different inner loops are not evolving over time anymore."*

### 5.2 Il meccanismo del collapse

Per ogni livello $L$:

1. Si tiene una **sliding window** delle ultime $M$ matrici di parametri medi del livello.
2. Si calcola la **varianza** della finestra: $\sigma^2(L, t) = \mathrm{Var}(\bar{W}_{t-M+1}, \ldots, \bar{W}_t)$.
3. Se $\sigma^2(L, t) < \epsilon$, il livello è "stabile" — i pesi non cambiano più.
4. **Collassa il livello**: sostituisci tutti gli inner loops con un singolo loop che usa $\bar{W}_t$ come pesi fissi.

### 5.3 Conseguenze

Senza collapse:
- Costo training: $O(N^6 \cdot T)$ con $T$ = numero di cicli
- Memoria: $O(N^6 \cdot |\text{ANN}|)$
- Inference: ogni decisione richiede passare per tutti i $N^6$ unità

Con collapse:
- Tipicamente, i livelli alti (architettura, embedding) collassano per primi (sono problemi più semplici)
- Dopo qualche centinaia di cicli, la struttura diventa $N^2$ (solo expert level e Wallace level non collassati)
- Eventualmente, l'**intera struttura collassa a un singolo expert** con FMC vanilla
- Inference: praticamente equivalente al Book #1 (FMC con simulator/reward fissi)

### 5.4 La narrazione di apprendimento

Book #2 §3.6 lo riassume così:

> *"The learning then occurs first in the lowest level loops working on the embedding function, then, once the embeddings makes sense, the second layer loops start to learn good next embedding predictor up to the outermost level, were the single top level loop finally learns good reward functions."*

Cioè: **i livelli collassano in ordine bottom-up**:

1. **Embedding (L3)** collassa per primo: imparare a comprimere osservazioni è un task supervisionato standard, converge rapidamente
2. **Prediction (L2)** collassa secondo: una volta fissato l'embedding, imparare $f: e_t \to e_{t+1}$ è anche relativamente facile
3. **Reward (L1)** collassa per ultimo: dipende dai precedenti, e la reward function è il task più ambiguo
4. **Expert level**: non "collassa" come gli altri — resta uno sciame finché c'è esplorazione utile

Risultato: l'agente passa da Badger pieno → Badger ridotto → singolo agente con world model + reward + planner. È la **distillazione automatica** dell'apprendimento.

### 5.5 Perché questo è bello

Tre motivi:

1. **Eliminazione del trade-off learning/inference**: training è caro, inference è economico. Standard.
2. **Self-pruning architetturale**: il sistema decide automaticamente quanta complessità tenere.
3. **Resistenza al overfitting**: livelli che convergono troppo presto vengono collassati e gli altri continuano. Niente overengineering.

> **Anti-pattern**: "More layers, more learning". **Pattern Fractal AI**: "Less layers when learning is done."

---

## 6. Fractal Memory: l'estensione orizzontale

Mentre Book #2 estende FMC verso l'**alto** (architettura AGI), il `2020 Fractal Slide.md` lo estende verso il **basso**: dentro le reti neurali stesse.

Tre meccanismi:

### 6.1 Dataset come Fractal Memory

#### 6.1.1 Idea base

Invece di campionare batch random dal dataset di training, ogni datapoint diventa una **unità di memoria** con walker associati.

```
Memoria(datapoint) = (S + label * ‖S‖, loss, n_visits)
                   ↑    ↑                ↑     ↑
                   stato  labels         loss attuale  visite
```

L'aggiunta di `label * ‖S‖` rende il label tanto importante quanto i dati nella metrica di distanza. Walker fanno cloning come in FMC standard.

#### 6.1.2 Reward speciale: la "Wigner reward"

Il documento propone (§"Why reward is π/2·x·exp(-π/4·x²)?"):

$$
R'(x) = \frac{\pi}{2} x \exp\left(-\frac{\pi}{4} x^2\right)
$$

dove $x = \text{Loss}_i / \text{Avg.Loss}$ è la loss normalizzata.

Questa è la **distribuzione di Wigner** delle eigenvalue di matrici random — il pattern di universalità menzionato nel Book #1 §6.6.

**Interpretazione**:
- Datapoint con loss vicina a 0 → già imparati → reward bassa → poca attenzione
- Datapoint con loss media → ideali per imparare → reward alta → molta attenzione
- Datapoint con loss molto alta → troppo difficili → reward bassa → poca attenzione

```
reward ↑
  |     ╱╲
  |    ╱  ╲
  |   ╱    ╲___
  |  ╱        ‾‾─
  | ╱            ‾
  |╱              ─
  └─────────────────→  loss / avg_loss
  0     1        2    3
```

Plus un **debiasing per visite** (§Slide doc): $R = R' / (1 + \log(1 + \text{visits}))$ — datapoint visitati molto vengono penalizzati per evitare overfitting.

#### 6.1.3 Conseguenze

Il documento dichiara tre proprietà:

1. **Curriculum learning automatico**:
   > *"It makes for some short of automatic curriculum learning approach where examples are processed in waves from the easiest to the hardest ones."*

2. **No catastrophic forgetting**:
   > *"The resulting NN learns the new dataset without forgetting the first one, as it continues to train on old data points every now and then."*

3. **Curated dataset finale**:
   > *"The first N elements of this [curated] dataset is an ideal dataset for learning the task from scratch."*

Sono **tre claim potenti**. Se vere anche solo parzialmente, hanno implicazioni significative per l'addestramento di NN moderne.

### 6.2 Sinapsi come Fractal Memory: self-pruning architecture

#### 6.2.1 Idea

Ogni sinapsi (peso del NN) è un'unità di memoria con un walker. Il reward della sinapsi è inversamente proporzionale al gradiente del peso:

$$
R_{\text{synapse}} = \frac{1}{1 + \log(1 + |\nabla w|)}
$$

Razionale: una sinapsi con gradiente alto è una sinapsi che "tira" forte verso un cambio — è incoerente. Sinapsi con gradiente basso sono "ben sintonizzate".

Distanza tra due sinapsi = numero minimo di salti neurone-neurone tra loro nel grafo.

#### 6.2.2 Meccanismo

```
For each batch:
  Compute gradient ∇w for each synapse
  R_synapse = 1 / (1 + log(1 + |∇w|))
  Walker FMC step on synapses
  Synapses with 0 walkers → deactivate (zero out weight)
  Optionally: rewire deactivated synapses to random new connections
```

Risultato: la rete inizialmente densa **si snellisce** mantenendo solo le connessioni "coerenti".

#### 6.2.3 Confronto con tecniche esistenti

| Tecnica | Meccanismo | Quando taglia |
|---|---|---|
| **Magnitude pruning** | Taglia sinapsi con peso piccolo | Post-training |
| **Lottery Ticket** | Trova subnet sparse efficaci | Multi-fase |
| **L1 regularization** | Penalizza somma assoluta dei pesi | Durante training |
| **Fractal Memory synapses** | Taglia sinapsi con gradiente alto via cloning | Durante training, automaticamente |

L'innovazione: **non serve un threshold**. Il sistema decide da solo quante sinapsi tenere via la dinamica di walker.

### 6.3 NN come Fractal Memory: multi-task automatico

#### 6.3.1 Idea

Crea $K$ reti neurali con stessa architettura ma inizializzazione random diversa. Tutte vedono lo stesso stream di task. Ogni rete è un'unità di memoria con walker.

Reward di una rete:
$$
R_{NN} = \exp(-\bar{L}_{NN})
$$

dove $\bar{L}_{NN}$ è la moving average della loss di quella rete.

#### 6.3.2 Conseguenze

Quando si presentano $T$ task diversi sequenzialmente:

> *"Each NN will tend to specialize in one of them, and will only train when the game being played matches the one it is specialized at."*

Cioè:
- Rete $A$ è bravo a Task 1 → loss bassa su Task 1 → reward alto → molti walker → continua a learnare
- Rete $A$ è scarsa su Task 2 → loss alta → reward basso → pochi walker → smette di learnare quando arriva Task 2
- Una rete con 0 walker viene **disattivata** (gradient learning rate va a zero)
- Quando torna Task 1, Rete $A$ recupera walker e si **riattiva**

Risultato: **specializzazione automatica** con routing implicito basato su entropia.

#### 6.3.3 Connessione con Mixture of Experts (MoE)

Questo è essenzialmente **Mixture of Experts** ma con due differenze chiave rispetto alle MoE moderne (Switch Transformer, GPT-4):

| Aspetto | MoE classico | Fractal Memory NN |
|---|---|---|
| Routing | Gating network appresa | Walker dynamics emergent |
| Bilanciamento carichi | Loss term explicit | Auto-bilanciamento via cloning |
| Aggiunta nuovo task | Richiede retraining | Self-rebalances naturalmente |
| Specializzazione | Indotta da gradient | Indotta da entropia FMC |

#### 6.3.4 Group inference

Per task senza loss intermedia (es. classificazione zero-shot):

> *"S(output) = ∏(2 - p_i^p_i), with S(P) ≥ 1 (multiplicative entropy)"*
> *"NN's weight for inference = moving average of 1/S(output)"*

Cioè: in inference, la confidenza di ogni NN è inversamente proporzionale all'entropia del suo output. Le NN sicure (output peaked) pesano di più.

---

## 7. Le tre estensioni messe assieme: la 3-level structure

`2020 Fractal Slide.md` chiude (§"The resulting 3-levels structure") con:

> *"Finally, we have a fractal memory of NNs, each one with a FM of synapses, all training on a FM of examples, all 3 working to solve a set of different problems in a coordinated way."*

Diagramma concettuale:

```
┌── Fractal Memory of NNs ────────────────────────────────┐
│  K = 100 reti neurali parallele                          │
│  Walker su ogni NN, cloning basato su loss               │
│                                                          │
│  Per ogni NN:                                            │
│  ┌── Fractal Memory of Synapses ────────────┐           │
│  │  Sinapsi con walker, self-pruning         │           │
│  │  basato su gradient                       │           │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  Tutte le NN condividono:                                │
│  ┌── Fractal Memory of Examples ────────────┐           │
│  │  Datapoint con walker, curriculum         │           │
│  │  learning auto + transfer learning        │           │
│  └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

Questo è il **massimo grado di astrazione** del programma Fractal AI così come articolato nei tre documenti.

> **Risultato dichiarato**: *"This 3-levels structure allows to scale the previous solution so it uses any number of parallel trained NNs that learn to solve any number of different problem."*

---

## 8. Confronto con l'ecosistema esistente

### 8.1 vs AlphaZero / MuZero

| Aspetto | AlphaZero / MuZero | Book #2 (Badger) |
|---|---|---|
| Planning | MCTS UCT | FMC |
| Embedding | CNN learned via RL | VAE in level 3 (auto-supervised) |
| World model | Implicit (perfect simulator) o latent (MuZero) | LSTM in level 2 (auto-supervised) |
| Reward | Game score (extrinsic) | Reward function appresa in level 1 |
| Training | Self-play, gradient descent | Multi-level FMC swarms + collapse |
| Multi-task | Singolo task per modello | Built-in via Fractal Memory NN |
| Catastrophic forgetting | Sì (need separate training) | No (Fractal Memory dataset) |

### 8.2 vs DreamerV3 (Hafner et al. 2023)

DreamerV3 ha:
- World model (RSSM) appreso
- Actor-critic per planning nel "sogno"
- Performance state-of-the-art su Atari + control

Book #2 ha:
- World model (LSTM) appreso via FMC swarm di pesi
- FMC vanilla per planning nel "sogno"
- Multi-task tramite Fractal Memory di NN

DreamerV3 è **molto più maturo** come implementazione, ma più rigido (singolo modello, singolo task). Book #2 propone una struttura più flessibile ma non implementata come DreamerV3.

### 8.3 vs Fragile Mechanics (`fragile-rl`)

Come notato in [`analisisPost2.md` §4.2](../../analisisPost2.md), `fragile-rl` è il successore canonico di Book #2. Le idee evolvono:

| Book #2 (2020) | Fragile Mechanics (2025) |
|---|---|
| `Full state = ⟨P0, P1, P2, P3, P4⟩` | `Z_t = (K_t, z_{n,t}, z_{tex,t})` |
| Embedding generico | VQ-VAE disentangled |
| LSTM per prediction | Lorentzian memory + covariant attention |
| Reward function appresa | Critic come PDE solver (screened Poisson) |
| Sciami FMC | Sciami FMC + WFR geometry |
| Badger collapse | Universal Governor + Sieve diagnostics |

L'evoluzione è verso **maggior formalizzazione matematica** (gauge theory, geometric deep learning) ma il **kernel computazionale** (sciame FMC con cloning entropico) è invariato.

---

## 9. Ipotesi falsificabili (= roadmap di ricerca)

Book #2 + Slide doc fanno **molte affermazioni empiriche** che non sono verificate pubblicamente. Ecco le 5 più importanti, ordinate per impatto:

### 9.1 (Alta priorità) Wigner reward è ottimale per il batching

**Claim**: la distribuzione $R = \pi/2 \cdot x \cdot e^{-\pi x^2/4}$ produce convergenza più veloce dello SGD uniforme.

**Esperimento**: classificare MNIST con
- A: SGD batching uniforme
- B: SGD batching ponderato secondo Wigner reward
- Misura: epoch necessari per accuracy 99%

**Costo**: 1 settimana CPU. **Output potenziale**: workshop paper.

### 9.2 (Alta priorità) No catastrophic forgetting con Fractal Memory

**Claim**: una rete addestrata con FM dataset wrapping non dimentica il task A quando si addestra su task B.

**Esperimento**: addestrare ResNet-18 su CIFAR-10, poi trasferire a CIFAR-100 (oppure permuted-MNIST → split-MNIST). Confrontare:
- A: training standard (replace dataset)
- B: training con Fractal Memory dataset
- Misura: accuracy retained on task A dopo training su task B

**Costo**: 2 settimane GPU. **Output potenziale**: NeurIPS continual learning workshop.

### 9.3 (Media priorità) Self-pruning sinapsi convergence

**Claim**: una rete densa con Fractal Memory synapses converge a una sparse network con accuracy ≥ rete densa.

**Esperimento**: ResNet-50 su ImageNet:
- A: dense baseline
- B: Magnitude pruning post-training
- C: Fractal Memory synapses durante training
- Misura: accuracy vs sparsity tradeoff

**Costo**: 1 mese GPU. **Output potenziale**: ICLR paper se i risultati battono SoTA pruning.

### 9.4 (Media priorità) Multi-NN specializzazione automatica

**Claim**: 100 reti parallele con FM-on-NN si specializzano automaticamente su task diversi.

**Esperimento**: 100 piccole reti su 10 task Atari simultanei.
- Misura: dopo $T$ cicli, ogni task viene "owned" da ~10 reti specializzate
- Misura: aggiungendo task 11, le reti si rialloca?

**Costo**: 1-2 mesi. **Output**: paper su multi-task RL.

### 9.5 (Bassa-media priorità) Collapse coverage

**Claim**: dopo training sufficiente, il Badger collassa a un singolo agente con accuracy ≥ Badger pieno.

**Esperimento**: implementare il Badger end-to-end su Atari, monitorare il collapse.

**Costo**: 6+ mesi (è il lavoro più ambizioso). **Output**: paper di alto impact factor se funziona.

---

## 10. Roadmap di implementazione

Se volessi davvero costruire un Book #2 funzionante, qual è il path of least resistance?

### Step 1 — Replica e estendi `fmc_minimal.py` (1 mese)

Già fatto in parte: ho `fmc_minimal.py` che implementa l'expert level. Da qui:

1. Vettorizza con PyTorch GPU
2. Aggiungi configurabilità della distance metric (RAM, IMG, embedding)
3. Aggiungi reward composta multi-component
4. Multi-seed + intervalli di confidenza per 5 giochi Atari

**Output verificabile**: tabella di benchmark che riproduce paper Book #1.

### Step 2 — Aggiungi VAE per embedding (1-2 mesi)

1. Train un VAE sui frame Atari (offline, su un dataset di rollout)
2. Sostituisci la distance metric in FMC con `‖embed(s1) - embed(s2)‖`
3. Misura il tradeoff accuracy/speed

**Output verificabile**: FMC con VAE embedding ≥ FMC con RAM su almeno un gioco.

### Step 3 — Aggiungi LSTM per world model (2-3 mesi)

1. Train un LSTM che predice $e_{t+1}$ da $(e_t, a_t)$
2. Sostituisci la chiamata `Simulate(state, action, dt)` in FMC con `LSTM(embed, action)`
3. Misura la degradazione (paper afferma ~80% retention)

**Output verificabile**: FMC con LSTM world model funziona senza accesso al simulatore reale.

### Step 4 — Reward function appresa (1-2 mesi)

1. Train un reward module su episodi annotati (anche con un proxy se necessario)
2. Confronta FMC con reward del task vs FMC con reward appresa

**Output verificabile**: FMC con reward appresa raggiunge ≥ 70% del reward del task.

### Step 5 — Wrappare in Badger structure (2-3 mesi)

Solo dopo aver verificato Step 1-4 separatamente:
1. Wrap VAE/LSTM/Reward in livelli di FMC swarm di parametri
2. Implementa il meccanismo di collapse
3. Misura tempo training totale e qualità finale

**Output verificabile**: agente che raggiunge ≥ 50% del paper Book #1 score senza accesso a simulatore esterno.

### Step 6 — Fractal Memory dataset (1 mese, in parallelo)

Indipendente dal Badger, può essere fatto su MNIST/CIFAR:
1. Implementa il wrapper FM dataset
2. Misura curriculum learning vs no
3. Misura forgetting vs no

**Output verificabile**: i 3 claim del Slide doc sono o non sono empiricamente sostenuti.

---

## 11. Rischi e limitazioni

Onestà: il Book #2 è **draft V0.2**. Significa:

1. **Non è peer-reviewed**. Non è stato sottoposto a NeurIPS/ICML.
2. **Non c'è codice di riferimento pubblico**. `fragile-rl` ha alcuni componenti ma non un Badger completo.
3. **Le formule sono spesso schizzate, non rigorose**. La Wigner reward non ha una derivazione formale.
4. **Le claim sono molte e forti**. Curriculum learning automatico, no catastrophic forgetting, multi-task automatico — ognuna è un risultato significativo se vera, ma nessuna è verificata.
5. **L'autore stesso ammette buchi**. Il documento ha sezioni marcate "(*)" come "Bibliography (*)" e "Free energy principle (*)" che indicano work-in-progress.

Quindi: **Book #2 è una proposta ambiziosa con carattere di manifesto, non una pubblicazione formale**. Va trattato come tale.

Tuttavia, il **plausibility prior** è alto perché:
- Book #1 ha **funzionato** (ho verificato io)
- L'estensione architetturale è **logicamente coerente**
- Le idee sono **operativamente realizzabili** in step incrementali
- C'è una **traiettoria di pensiero a 6 anni** che è internamente consistente (Book#1 → Book#2 → Fragile Mechanics)

---

## 12. La tesi finale

> **Book #2 + Fractal Memory non sono "una nuova architettura AGI". Sono la *generalizzazione fattorizzata* del principio Fractal AI: lo stesso sciame FMC applicato a livelli ontologici diversi, con un meccanismo di compressione automatica (collapse) per rendere il sistema operativo.**

Se questa generalizzazione regge sperimentalmente, ha implicazioni su tre piani:

1. **Algoritmico**: dimostriamo che esiste un algoritmo unificato per planning, learning, e architecture search.
2. **Computazionale**: dimostriamo che la compressione automatica risolve il problema scalabilità.
3. **Filosofico**: dimostriamo che la "frattalità" è una proprietà non-negoziabile dell'AGI — stessa struttura a tutte le scale.

Sono affermazioni **grosse**. Forse troppo grosse. Ma sono falsificabili, e il path è chiaro.

Per il momento, ho replicato il livello expert (Book #1). Mi resta da provare gli altri livelli e i meccanismi del Fractal Memory.

---

## 13. Riferimenti

### 13.1 Sorgenti primarie

- **Hernández-Cerezo, S. & Duran-Ballester, G. (2020)** — *Fractal AI: A Fragile Theory of Intelligence*. arXiv:1803.05049v5. (Book #1)
- **Hernández-Cerezo, S. & Duran-Ballester, G. (2020)** — *Fractal AI Book #2: AGI Structure*. V0.2 draft. (Non pubblicato pubblicamente)
- **Hernández-Cerezo, S. & Duran-Ballester, G. (2020)** — *Honey Badger meets Fractal AI Hives*. (Non pubblicato pubblicamente)
- **Hernández-Cerezo, S. & Duran-Ballester, G. (2020)** — *Fractal Memory: Hybrids for Neural Networks*. (Slide deck, non pubblicato pubblicamente)

### 13.2 Letteratura comparata

- **Silver et al. (2017)** — *Mastering the game of Go without human knowledge*. Nature 550. (AlphaZero)
- **Schrittwieser et al. (2020)** — *Mastering Atari, Go, chess and shogi by planning with a learned model*. Nature 588. (MuZero)
- **Hafner et al. (2023)** — *Mastering Diverse Domains through World Models*. arXiv:2301.04104v1. (DreamerV3)
- **Fedus, Zoph, Shazeer (2022)** — *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 23. (MoE moderno)
- **Frankle, Carbin (2018)** — *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. ICLR 2019.
- **Clark, A. (2013)** — *Whatever next? Predictive brains, situated agents, and the future of cognitive science*. Behav. Brain Sci. 36(3). (Predictive processing)
- **Friston, K. (2010)** — *The free-energy principle: a unified brain theory?* Nat. Rev. Neurosci. 11(2).

### 13.3 Software / repository

- [`fragile`](https://github.com/FragileTech/fragile) — implementazione moderna di FMC su PyTorch (livello expert)
- [`fragile-rl`](https://github.com/FragileTech/fragile-rl) — successor di Book #2 (Fragile Mechanics)
- [`fmc_minimal.py`](../03_atari_replication/scripts/fmc_minimal.py) — la mia replica del livello expert in 230 righe NumPy

---

*Fine deep dive 06. Lunghezza: ~1080 righe. Status: scrittura completa.*

*Per la cronaca personale dell'arrivo di queste informazioni vedi [`analisisPost2.md`](../../analisisPost2.md). Per il quadro tecnico originale (solo Book #1) vedi [`ANALISIS.md`](../../ANALISIS.md).*
