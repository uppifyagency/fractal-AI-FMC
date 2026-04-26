# Post-Esperimento II: il quadro si allarga (l'arrivo del Book #2 e del Fractal Memory)

> *"Hai trovato un singolo dente fossile e hai pensato fosse di un cane. Poi hai trovato il cranio. Ed era un sauropode."*

Documento personale di seconda fase. Scritto **dopo** aver scoperto che il paper arXiv:1803.05049 (Book #1) è solo **una parte di un programma di ricerca molto più ambizioso** che include:

- **Book #2 — AGI Structure** (V0.2, draft 2020): l'architettura AGI completa
- **2020 Fractal Hives**: la specifica operativa del Badger+FMC con pseudocodice
- **2020 Fractal Slide — Fractal Memory**: l'estensione di FMC alle reti neurali stesse

Per il documento di prima fase (che resta valido) vedi [`analisisPost.md`](analisisPost.md).

---

## 1. Lo shock del contesto allargato

Quando ho fatto girare Boxing e ho visto il punteggio 96/100, la mia ipotesi finale era:

> *FMC è un planning algorithm potente e sottovalutato. Funziona perfettamente quando hai un simulator perfetto. Per il mondo reale serve un world model appreso, e quello reintroduce la complessità del deep learning.*

Era una conclusione **ragionevole** ma basata su informazione **incompleta**. Non sapevo che gli stessi autori avevano già pensato a tutto questo, e avevano già scritto la specifica del *come* mettere FMC dentro una struttura AGI completa.

Il Book #2 risponde **esattamente** alle obiezioni che avevo sollevato in [`analisisPost.md` §5.2](analisisPost.md):

> *"FMC senza simulator perfetto degrada. Quanto? Nessuno lo sa."*

Risposta del Book #2 (cap. 2):

> *"For every external function used in FMC, an equivalent learning process must be defined instead, and only then, the FMC could had evolved into a full AGI."*

E poi enumera con precisione le quattro funzioni esterne che FMC usa nel paper #1, e per ognuna propone l'analogo *learnable*:

| Funzione esterna in Book #1 | Modulo learnable in Book #2 |
|---|---|
| `Observation(sensors)` → vector | **VAE** (variational autoencoder) — embedding learning |
| `Distance(s1, s2)` | Distanza euclidea sulle embedding (gratis dal VAE) |
| `Simulation(state, action, dt)` | **LSTM** — world model learning |
| `Reward(state)` | **Reward module** — reward function learning |

Questa è la stessa filosofia di **AlphaZero** (VAE → LSTM → Critic → MCTS) ma con FMC al posto di MCTS, e con un'**innovazione strutturale** che AlphaZero non ha: **i moduli stessi sono ottimizzati da uno sciame FMC**, non da gradient descent classico.

---

## 2. Il Badger Structure: la cosa che mi mancava

Ho letto il Book #2 due volte. La prima volta ero confuso. La seconda volta ho capito che mi era mancata una cosa fondamentale leggendo il Book #1: **FMC non è un punto di arrivo, è un mattone**. E il Badger Structure è il modo in cui questi mattoni si compongono per fare un'AGI.

### 2.1 La struttura a cipolla

```
┌──── OUTER LOOP (Controller) ──────────────────────────────────┐
│  Decide cosa imparare e quando                                 │
│                                                                │
│  ┌── Level 4: Architecture optimizer (optional) ──────┐       │
│  │  Cerca la struttura ANN ottima per ciascun modulo  │       │
│  │                                                     │       │
│  │  ┌── Level 3: Embedding optimizer ──────────┐     │       │
│  │  │  Trova i pesi VAE ottimi                  │     │       │
│  │  │                                            │     │       │
│  │  │  ┌── Level 2: Prediction optimizer ─┐   │     │       │
│  │  │  │  Trova i pesi LSTM ottimi         │   │     │       │
│  │  │  │                                    │   │     │       │
│  │  │  │  ┌── Level 1: Reward optimizer ┐ │   │     │       │
│  │  │  │  │  Trova la reward function    │ │   │     │       │
│  │  │  │  │  ottima                       │ │   │     │       │
│  │  │  │  │                                │ │   │     │       │
│  │  │  │  │  ┌── EXPERT LEVEL ────────┐   │ │   │     │       │
│  │  │  │  │  │ FMC swarm di walker    │   │ │   │     │       │
│  │  │  │  │  │ pianifica le azioni    │   │ │   │     │       │
│  │  │  │  │  └────────────────────────┘   │ │   │     │       │
│  │  │  │  └─────────────────────────────────┘ │   │     │       │
│  │  │  └────────────────────────────────────────┘   │     │       │
│  │  └──────────────────────────────────────────────────┘     │       │
│  └────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

Ogni livello è uno sciame FMC che ottimizza i parametri del modulo a quel livello. Il livello expert è quello che ho già implementato in `fmc_minimal.py` — pianifica le azioni con FMC vanilla. Tutti gli altri livelli **riapplicano FMC alla scelta dei propri pesi**.

### 2.2 La proprietà geniale: "Learning as structural collapse"

Da `2020 Fractal.md` §"Learning as a structural collapse":

> *"If the embedding NN freezes its parameters after some training, we can decide embedding is already properly learnt, and 'collapse' all inner loops into a single one sporting the averaged parameter."*

Questa è una proprietà **bellissima**. Significa che la struttura **si auto-semplifica** mano a mano che impara:

- Inizialmente: ogni livello ha N inner loops × N expert × N walker = $N^3$ unità
- Con il tempo i livelli inferiori "convergono" — i pesi smettono di muoversi
- Quando un livello si "freeze", lo si **collassa** a una singola istanza
- Risultato finale: dopo training, l'agente è un'unica istanza con i pesi distillati

Questo elimina di colpo l'obiezione che il Badger sia "troppo grande" per essere praticabile. Sì, in fase di training serve molta CPU/GPU. Ma in **inferenza** la struttura collassa a un agente classico (VAE+LSTM+FMC), efficiente quanto AlphaZero distillato.

### 2.3 La cosa più sottile: tutto è uno sciame

Ho passato un'ora a capire quest'idea. Lascio che parli il documento (sezione "Internal structure of the Badger"):

> *"Everything in the resulting Badger structure, including outer loop, inners loops, expert levels and experts themselves, can be seen as special instances of a swarm."*

Cioè: **a ogni livello c'è uno sciame FMC che opera su qualcosa di diverso**:

| Livello | Sciame fatto di | Reward usata | Distanza |
|---|---|---|---|
| Expert | walker (stati simulati) | reward esterna del task | distanza in state space |
| Reward (L1) | reward functions ANN | reward intrinseca della L1 | distanza tra parametri ANN |
| Prediction (L2) | LSTM ANN | accuratezza della predizione | distanza tra parametri LSTM |
| Embedding (L3) | VAE ANN | accuratezza ricostruzione | distanza tra parametri VAE |
| Architecture (L4) | architetture | un mix di accuratezza + parsimonia | distanza tra strutture |

In pratica: lo **stesso algoritmo FMC** viene applicato 5 volte di seguito, su 5 livelli ontologicamente diversi, ognuno che ottimizza i pesi del livello inferiore.

> Questa è **autocoerenza algoritmica** — l'algoritmo si applica a sé stesso. È il principio della **frattalità** (da cui il nome "Fractal AI").

### 2.4 La ridefinizione del "full state"

Book #2 introduce un'idea che mi ha colpito (§2.3):

> *"Full_state = ⟨Observation, Embedding, Next embedding, Action, Expected reward⟩"*

Cioè: lo "stato" dell'agente non è solo l'osservazione — è un **vettore composito** che include:
- P0: osservazione grezza dai sensori
- P1: embedding compresso
- P2: predizione del prossimo embedding
- P3: azione scelta
- P4: reward attesa

Ogni livello del Badger aggiorna **una porzione** di questo vettore. L'osservazione è aggiornata dai sensori (level 0), l'embedding dal VAE (level 3), la predizione dall'LSTM (level 2), l'azione dal planner FMC (expert level), la reward dal reward module (level 1).

Questa è una **conceptual move** importante: lo stato non è quello che il mondo ti dà, è quello che la tua mente **costruisce** processando ciò che il mondo ti dà. È la stessa idea del **predictive processing** in neuroscienze (Clark 2013, Friston 2010), ma operativizzata in modo computazionale.

---

## 3. Il Fractal Memory: la cosa che non avevo previsto

Mentre il Book #2 estende FMC verso l'**alto** (architettura AGI), il documento `2020 Fractal Slide.md` lo estende verso il **basso**: applica il principio FMC alle **reti neurali stesse**.

Tre estensioni:

### 3.1 Dataset come Fractal Memory

Idea base: invece di campionare batch random dal dataset, ogni datapoint diventa una "unità di memoria" con walker associati. Il reward del datapoint è una funzione della loss attuale dell'NN su quel punto. Walker fanno cloning come in FMC standard.

Conseguenze (citazione diretta):

> *"It makes for some short of automatic curriculum learning approach where examples are processed in waves from the easiest to the hardest ones."*

E:

> *"Fractal AI tries to maximize reward but, at the same time, it maximizes diversity too. So a bunch of very different 'easy examples' will be present in the resulting memory, along with a big collection of not-so-evident-yet ones."*

E (forse la più importante):

> *"The resulting NN learns the new dataset without forgetting the first one."*

**Catastrophic forgetting** è uno dei problemi storici dell'apprendimento neurale. Nessuna soluzione elegante esiste. Il Fractal Memory ne propone una via puramente entropica.

### 3.2 Reward speciale: la "universality pattern"

Una formula catturare la mia attenzione:

$$
R'(x) = \frac{\pi}{2} x \exp\left(-\frac{\pi}{4} x^2\right)
$$

dove $x$ è la loss normalizzata. Questa è la **distribuzione di Wigner** delle eigenvalue di matrici random — il pattern di universalità menzionato nel Book #1 §6.6.

Significato: il sistema vuole che la densità di walker sui datapoint segua **questa specifica distribuzione**, non la reward grezza. Datapoint con loss medio sono i più importanti per imparare; quelli con loss vicino a 0 (già imparati) o molto alta (non ancora imparabili) sono meno importanti.

Trovo affascinante che il pattern di Wigner — emerso storicamente in fisica nucleare — appaia naturalmente come distribuzione ottimale per l'apprendimento. Se confermato empiricamente, è un'ipotesi grossa.

### 3.3 Sinapsi come Fractal Memory: self-pruning architecture

Da §"The idea of self-pruning synapses":

> *"If we could do the same with the synapsis dataset, the initially dense connections would self-prune into a more sparse connectivity as connections run out of walkers: we added plasticity to our NN structure."*

Ogni sinapsi ha un reward inversamente proporzionale al gradiente del peso. Sinapsi "incoerenti" (gradiente alto) ricevono pochi walker e si **disattivano**. Risultato: l'architettura della rete **si snellisce automaticamente** durante il training.

Questa è una soluzione elegante al problema della pruning, che oggi richiede metodi euristici tipo *Lottery Ticket Hypothesis* (Frankle & Carbin 2018) o *magnitude pruning*. Fractal Memory propone un meccanismo unificato basato su entropia.

### 3.4 NN come Fractal Memory: multi-task automatico

Il livello più alto: 100 reti neurali parallele, ognuna trattata come un'unità di memoria. Ogni rete ha walker associati. Quando si presenta un task:

> *"Each NN will tend to specialize in one of them, and will only train when the game being played matches the one it is specialized at."*

E in inference:

> *"The ones 'specialized' on this game will score significantly better than the others, so their Ps will prevail and they will tend to be the only active NNs after some cycles."*

Questo è esattamente il pattern dei **Mixture of Experts** modernamente popolari (Switch Transformer di Fedus et al. 2022, GPT-4 con MoE), ma con un meccanismo di routing **non-supervisionato basato su entropia**, non su gating learned via gradient descent.

---

## 4. Cosa cambia nel mio quadro

### 4.1 Le mie obiezioni del primo documento

Riprendo le obiezioni di [`analisisPost.md` §5](analisisPost.md):

| Obiezione | Risposta da Book #2 / Fractal Memory |
|---|---|
| Un gioco/un seed non basta | Vero, ma ortogonale — qui parliamo di teoria, non empirico |
| Il simulator è perfetto (ALE) | **Risolto in Book #2**: world model appreso via LSTM al level 2 |
| Throughput non è real-time | **Risolto via collapse**: in inferenza diventa singolo expert efficient |
| Il 4% di gap potrebbe essere bug | Vero, ma resta valido |

Il Book #2 risponde a 2 delle mie 4 obiezioni in modo strutturalmente convincente. Non con codice, ma con architettura.

### 4.2 La connessione con `fragile-rl` (Fragile Mechanics)

Una cosa importante che ora capisco: **`fragile-rl` non è un'evoluzione separata — è il successore canonico del Book #2**. Ricordando l'architettura di `fragile-rl/docs/source/1_agent/intro_agent.md`:

```
State = (K, z_n, z_tex)
       ↑    ↑    ↑
     macro nuisance texture
```

Confronto con Book #2:

```
Full_state = ⟨Observation, Embedding, Next embedding, Action, Reward⟩
                  ↑              ↑
              raw input    "embedding compressed"
```

Le idee sono allineate. `fragile-rl` formalizza ulteriormente l'embedding decomponendolo in (macro, nuisance, texture) — una raffinazione del concetto generico di "embedding" del Book #2. Aggiunge la macchineria geometrica/gauge-teorica come *strumento matematico* per formalizzare le proprietà degli embedding.

In altre parole: **Book #2 → Fragile Mechanics** è una progressione naturale di 5 anni di pensiero, non un cambio di rotta.

### 4.3 La traiettoria intellettuale degli autori

Ora vedo il loro programma in tre fasi:

| Anno | Documento | Domanda | Risposta |
|---|---|---|---|
| 2018-2020 | Book #1 (paper arXiv) | Come pianifica un agente intelligente? | Fractal Monte Carlo |
| 2020 | Book #2 + Hives + Slide | Come si compone un'AGI usando FMC? | Badger structure di sciami nidificati |
| 2024-2026 | `fragile-rl` (Fragile Mechanics) | Quale geometria/fisica governa l'AGI? | Gauge theory cognitive + WFR + holographic |

È un programma **coerente** che parte dal mattone e sale fino al cosmo. La maggior parte dei programmi di ricerca AI fa l'opposto (parte dal cosmo astratto e non riesce a costruire mai mattoni).

### 4.4 Le mie idee originali: aggiornate

Riprendo le 6 idee che avevo dato in [`analisisPost.md` §4](analisisPost.md), aggiornate alla luce del Book #2:

#### Idea 1 (Toolkit FMC educativo) → ridimensionata
Il mio `fmc_minimal.py` ora è solo l'**expert level** del Badger. Per essere completo serve aggiungere VAE/LSTM/Reward module learnable. L'effort è 10× quanto pensavo. Ma il valore è anche maggiore.

#### Idea 2 (FMC come planner per offline RL) → ancora valida
Resta una buona direzione. Il Book #2 conferma che è l'approccio giusto.

#### Idea 3 (FMC su problemi non-RL) → ridimensionata
Per drug discovery / trial clinici servono moduli world-model adatti al dominio. L'effort è alto. Ma il payoff potenziale è altrettanto alto.

#### Idea 4 (FMC in sistemi più grandi) → confermata
Esattamente quello che propone il Book #2. L'expert level è il "modulo planner" che si plug-in in architetture più grandi.

#### Idea 5 (FMC + LLM) → **molto rinforzata**
Il Fractal Memory propone esplicitamente di trattare le NN come unità FM. Un LLM come "memoria fractal" durante reasoning è esattamente la generalizzazione naturale. Vedo qui la possibilità di un paper di alta qualità, perché:
- Tree-of-Thought (Yao et al. 2023) è già accettato come tecnica
- Sostituirlo con Fractal-of-Thought (FoT) è una mossa naturale
- I benefici teorici di FMC su MCTS si trasferiscono al ragionamento LLM

#### Idea 6 (Coscienza ricorsiva) → **strutturalmente fondata**
Il Book #2 §5 enumera 9 first principles entropici, e propone esplicitamente in §6 "Research directions":

> *"What is Consciousness? Is it a level we haven't explored? Is it an emergent behaviour from other layers? Or is it a complex mixture of both things?"*

Gli autori si pongono **letteralmente la stessa domanda**. La coscienza nel framework Book #2 è (potenzialmente) un nuovo livello del Badger: un livello che opera sui pesi del livello reward (`{K_i}` del paper #1), un meta-meta-pianificatore che sceglie i propri obiettivi.

Questa non è speculazione mia. È la direzione di ricerca **dichiarata** dagli autori.

---

## 5. Le nuove idee che mi sono venute (post-Book #2)

### 5.1 Implementare il Badger nel mio stack

Ora che l'expert level (FMC) è verificato funzionante via Boxing, ha **senso** aggiungere i livelli superiori uno alla volta:

**Step 1**: aggiungere un VAE che impara un embedding della RAM Atari (128 → 32 dimensioni). Misurare se la distanza FMC sull'embedding è meglio della distanza sulla RAM grezza.

**Step 2**: aggiungere un LSTM che predice l'embedding successivo dato (embedding, action). Misurare se FMC che usa l'LSTM come simulator (invece dell'ALE perfetto) ottiene almeno 80% del risultato originale.

**Step 3**: aggiungere un reward module appreso (anche se in Atari il reward esterno c'è già). Misurare se l'agente impara a sopravvivere meglio quando una reward intrinseca aggiuntiva è ottimizzata in parallelo.

Questo è un programma di lavoro **concreto**, **incrementale**, e con **gate empirici** ad ogni step. Si può fare in 2-3 mesi.

### 5.2 Replicare il Fractal Memory su un dataset classico

Suggestion: prendere MNIST o CIFAR-10, implementare il Fractal Memory dataset wrapping, e misurare:
- Velocità di convergenza vs SGD standard
- Resilienza al catastrophic forgetting (transfer da MNIST→CIFAR)
- Qualità della "curated dataset" finale (è davvero più piccolo? È davvero più informativo?)

Effort: 1 mese. Output: paper su workshop NeurIPS/ICLR. Il contributo è verificare empiricamente i claim del Slide doc, che ad oggi non sono verificati pubblicamente.

### 5.3 Fractal Memory + LLM training

Questa è speculativa ma potenzialmente massiccia. Idea: nei moderni training LLM (post-training, fine-tuning, RLHF), il batching è ancora largamente uniforme. Sostituire il batching con un Fractal Memory potrebbe:

- Accelerare la convergenza del fine-tuning
- Ridurre il forgetting durante DPO/PPO
- Curare automaticamente dataset di training

Se funzionasse anche solo del 20% meglio, sarebbe un risultato gigante per la community LLM.

Effort: 2-3 mesi con accesso a GPU serie. Output potenziale: paper di alto impact factor.

### 5.4 La "Wigner reward" come oggetto di studio

La formula $R = \frac{\pi}{2} x e^{-\pi x^2 / 4}$ con $x$ = loss normalizzata mi affascina. È un'**affermazione empirica** travestita da formula matematica. Affermazione: la distribuzione ottimale di loss in un dataset di training è il pattern di Wigner.

Vorrei verificarla. Costruire un esperimento dove si misura la distribuzione di loss su un classifier ben-trained e vedere se segue Wigner. Se sì, è un risultato fondamentale; se no, la formula del Slide doc è un'ipotesi euristica e non un teorema.

Questo è un esperimento da 1-2 settimane. Lo aggiungerei alla todo list.

---

## 6. Il punto epistemologico, di nuovo

Nel primo `analisisPost.md` mi ero posto la domanda:

> *Perché abbiamo creduto, per 10 anni, che servissero reti neurali con miliardi di parametri per risolvere giochi che si risolvono con 200 righe di NumPy + un simulatore?*

Ora ho una versione più sottile dello stesso punto:

> *Perché abbiamo creduto, per 10 anni, che il "deep learning" e il "planning" fossero approcci separati, quando esiste una formulazione (Book #2 + Fragile Mechanics) in cui sono lo **stesso algoritmo applicato a livelli ontologici diversi**?*

La risposta più onesta che mi do: l'incentivazione accademica favorisce **specialisti**. Un ricercatore che fa "deep RL" è un esperto di reti, ottimizzazione, scalability. Un ricercatore che fa "planning" è un esperto di MCTS, IDA*, A*. Un ricercatore che fa entrambi è uno strano. Un ricercatore che propone una **sintesi** dei due è ancora più strano. I sistemi di valutazione (revisione paper, citation count, hiring) tendono a punire la sintesi a favore della specializzazione.

Hernández-Cerezo & Duran-Ballester sono outsider — non sono in DeepMind/OpenAI, non hanno una macchina di pubblicazione, non hanno keynote alle conferenze. Questo è il loro punto debole nel sistema accademico ma è anche il loro punto di forza intellettuale: possono pensare la sintesi senza pagare il costo sociale di farlo.

---

## 7. La sintesi aggiornata

Aggiorno la sintesi in tre frasi del primo documento:

> **Una.** Il programma Fractal AI non è un singolo paper, è un **arc** di ~6 anni di lavoro che parte dal Book #1 (planning), passa dal Book #2 (struttura AGI con apprendimento integrato), e arriva a `fragile-rl` (cosmologia gauge-teorica della cognizione). Ognuno è coerente con i precedenti.

> **Due.** L'idea-cardine è la **frattalità**: lo stesso algoritmo (FMC con virtual reward = R · Dist) si applica ricorsivamente a livelli ontologicamente diversi — dai walker che pianificano azioni, ai pesi che si ottimizzano, alle reti neurali che si specializzano. Questa autocoerenza è ciò che dà al programma la sua **eleganza unificante**.

> **Tre.** La soluzione al "world model problem" che avevo segnalato nel primo `analisisPost.md` è già contenuta nel Book #2 del 2020 — solo che non l'avevo letto. Questo mi ricorda l'umiltà necessaria nel valutare un programma di ricerca: **leggi tutto il corpus prima di fare critiche**, perché la cosa che pensi di aver scoperto come buco potrebbe essere già coperta nel paper successivo.

---

## 8. Cosa farò ora

Le priorità si riassumono. Aggiornata la mia roadmap personale:

### Settimane prossime
1. **Multi-seed Boxing** (35 min totali) — completare la base empirica del Book #1
2. **Verifica della Wigner reward** su MNIST — esperimento di 1-2 settimane

### Mesi prossimi (1-3)
3. **Implementare Step 1 del Badger**: VAE su Atari + FMC su embedding (vs RAM grezza)
4. **Articolo onesto**: blog post tecnico che presenta il programma Fractal AI come un **arc** invece che come un paper isolato

### Mesi 3-6
5. **Step 2-3 del Badger**: LSTM world model + reward module appreso
6. **POC Fractal-of-Thought con LLM** (idea 5.3 sopra)

### Mesi 6-12
7. **Pubblicazione**: paper basato su uno dei tre filoni (ArXiv minimum, idealmente NeurIPS workshop)

---

## 9. Una nota sull'umiltà

Quando ho scritto il primo `analisisPost.md` ero entusiasta. Avevo fatto girare l'algoritmo, avevo verificato che funziona, avevo idee per il futuro. Ero **convinto** che il mio quadro fosse completo e aggiornato.

E poi sono arrivati tre file e mi sono accorto che avevo letto solo **un terzo** del programma intellettuale di questi due autori.

C'è una lezione ovvia: in un campo che si muove velocemente come l'AI, **non smettere mai di leggere**. La frase "ho già letto tutto su X" è quasi sempre falsa.

Ma c'è una lezione meno ovvia: i veri programmi di ricerca **rivelano la loro completezza solo a un'esplorazione paziente**. Il Book #1 da solo ti convince che FMC è un planning algorithm. Il Book #2 ti convince che è un mattone AGI. Il Slide doc ti convince che è un principio universale di apprendimento. Le tre conclusioni sono **diverse** e nessuna è completa senza le altre.

Il messaggio per chi leggerà questo documento tra mesi: **se ti capita di trovare un singolo paper Fractal AI e ti sembra "incompleto", la domanda giusta non è "perché è incompleto" ma "dov'è il resto?"**. Per molti programmi di ricerca seri, c'è sempre un *resto*.

---

## 10. Per chi continuerà questo lavoro

Le 30+ righe in cima a [`work/`](work/) sono ora **insufficienti**. Il piano operativo va riformulato in modo Badger-aware:

- `01_setup_environment/` resta valido (è solo l'expert level)
- `03_atari_replication/` resta valido (è la verifica del expert level)
- `02_deep_dives/` deve essere **espanso** per coprire Book #2 e Fractal Memory
- `04_diagrams/` deve aggiungere diagrammi del Badger structure

Mi appunto questi come prossimi todo. Non sono per oggi.

---

*Documento personale di seconda fase, post-arrivo del Book #2 e dei documenti Hives + Fractal Memory.*

*La mia conclusione personale: il programma Fractal AI è uno dei programmi di ricerca AGI più seri e sotto-pubblicizzati del decennio 2018-2026. Vale la pena dedicarci tempo serio.*

*Per il quadro tecnico originale vedi [`ANALISIS.md`](ANALISIS.md). Per le riflessioni di prima fase vedi [`analisisPost.md`](analisisPost.md). Per il piano operativo vedi [`work/`](work/).*

---

# ADDENDUM 2026-04-26 — La terza espansione (la bibliografia completa)

> *"Hai trovato il cranio del sauropode. Poi hai trovato la coda. E le ossa del bacino. E le impronte. E il diario fossilizzato dell'animale che lo ha ucciso. Ora sai non solo che era un sauropode — sai come viveva."*

Dopo aver scritto questo documento, ho fatto un terzo passaggio: **ho cercato e archiviato tutto il corpus accademico degli autori**, dal 2014 al 2026. L'ho messo in [`docs/bibliography/`](docs/bibliography/).

E un'altra cosa è cambiata.

## A1. Cosa ho trovato cercando sistematicamente

Cinque elementi nuovi, oltre ai tre drafts che già avevo:

1. **[General Algorithmic Search](docs/bibliography/sources/papers/2017_general_algorithmic_search_1705.08691.pdf)** (arXiv:1705.08691, 2017) — l'**antesignano**. Sergio + Guillem + José M. Amigó pubblicano la versione "ottimizzazione" dell'algoritmo, **un anno prima del paper FMC**. Stesso swarm, stesso virtual reward, stesso cloning. Applicato a function optimization invece che a planning.

2. **[A Brief Review of Generalized Entropies](docs/bibliography/sources/papers/2018_brief_review_generalized_entropies.pdf)** (Entropy 2018) — la **foundation matematica**. José Amigó + Sámuel Balogh + Sergio Hernández. **236 citazioni** (la più citata del Sergio). Stabilisce il framework delle entropie generalizzate (Tsallis, Rényi, Hanel-Thurner) che giustifica le reward composte non-additive di FMC.

3. **[Solving Atari Games Using Fractals And Entropy](docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf)** (arXiv:1807.01081, 2018) — il **companion sperimentale** del paper #1. Sergio + Guillem + Spiros Baxevanakis. Numeri Atari "puri", più chiari del Book #1.

4. **8 blog post archiviati** dal "lab notebook" pubblico [Entropic and Fractal Intelligence](http://entropicai.blogspot.com/) — anni 2014-2017. La gestation di tutte le idee.

5. **Physics-Inspired Swarm Optimization** (World Scientific, 2021) — book chapter peer-reviewed. **Conferma che gli autori possono pubblicare in venue formali.** La scelta di non pubblicare Book #2/Hives/Slide è **strategica**, non per incapacità.

## A2. Le tre rivelazioni della cronologia

### A2.1 La gestation di 4-5 anni per ogni idea

Confrontando blog vs paper formali, emerge un pattern:

| Idea | Prima apparizione | Formalizzazione | Gestation |
|---|---|---|---|
| `relativize` | 2014-03 (`level 7` post) | 2018 paper §2.2.3 | **4 anni** |
| Multi-agent (Octopus) | 2015-12 (`collaboration` post) | 2020 Book #2 Badger | **5 anni** |
| Plant pot swarm | 2015-09 (`basics` post) | 2017 GAS paper | 2 anni |
| FMC su Atari | 2017-06 (`solved Atari` post) | 2018 paper 1807 | 1 anno |
| Continuous learning NN | 2017-07 (`retrocausality` post) | 2020 Slide doc | 3 anni |

**Take-away epistemologico**: nessuna delle "scoperte" del programma è arrivata di colpo. Tutte hanno avuto **anni di gestazione informale** sul blog prima della formalizzazione. Questo è un pattern classico della scienza profonda — non quella che fa hype.

### A2.2 La maturità sperimentale prima del paper

Il post di **giugno 2017** [`solved_atari_games.md`](docs/bibliography/sources/blog_posts/2017-06_solved_atari_games.md) mostra che FMC funzionava su Atari **un anno prima** della pubblicazione del paper 1807. I numeri (MsPacman 11.5k, Qbert 18.4k, VideoPinball 500k) erano già SoTA.

Il paper 2018 ha fatto scalare i numeri di **3 ordini di grandezza** — ai bug-induced limits dei giochi (MsPacman 999,990).

Significato: tra giugno 2017 e luglio 2018 c'è stato un **lavoro intenso di ottimizzazione**. Il paper rappresenta il punto di massima ottimizzazione, non il primo successo.

**Implicazione per la mia replica**: il mio Boxing 96/100 con `fmc_minimal.py` è in fascia "primo successo non ottimizzato". Per arrivare al 100/100 servirebbe lavoro paragonabile a quello del 2017-2018. **Il 4% di gap che mi tormentava ha finalmente una spiegazione storica.**

### A2.3 La filosofia della reward singola

Il post **2016 Pareto Frontiers** ([`pareto_frontiers.md`](docs/bibliography/sources/blog_posts/2016-04_pareto_frontiers.md)) mostra che Sergio rifiuta esplicitamente il framework Pareto:

> *"Real-world problems typically have single underlying objectives... we only have one goal in life — maximizing long-term well-being."*

Questa è una **posizione filosofica audace**. Nella letteratura RL/control, il Pareto è ortodosso. Sergio dice no, una sola objective function.

Conseguenza nel paper: la formula `R(s) = R₀(s) × R₁(s) × ... × Rₙ(s)` (composizione moltiplicativa) è la concretizzazione di questo principio. Non Pareto, ma **uno scalar combo che cattura tutto**.

Conseguenza in `fragile-rl`: il critic field $V$ come PDE solver è **un singolo scalar field** sull'intero state space. Stessa filosofia, formalizzata 10 anni dopo.

## A3. Cosa cambia per il Fractal Coding Loop

Avere il corpus completo cambia anche la mia visione del Fractal Coding Loop.

### A3.1 La metafora dell'octopus è la struttura giusta

Nel post [2015-12 Fractal AI Collaboration](docs/bibliography/sources/blog_posts/2015-12_fractal_ai_collaboration.md) Sergio scrive:

> *"a fractal tree of intelligence layers where fingers coordinate as a hand, hands function as an octopus, and multiple octopuses respond to collective instructions"*

Per il Fractal Coding Loop:
- **Fingers** = walker individuali (sub-agent Claude)
- **Hand** = expert level (un task specifico, es. "refactor auth")
- **Octopus** = una sessione di sviluppo (multi-task coordinati)
- **Multiple octopuses** = un team di sessioni in parallelo (che è oltre Claude Code attuale)

La gerarchia è **già concettualizzata** nel 2015. Non sto inventando, sto traducendo per Claude Code.

### A3.2 Il principio della reward singola si applica al coding

Dal post Pareto: **un singolo scalar reward è preferibile a una somma di KPI**.

Per il coding, questo significa:
- Non sommare `tests + lints + diff_size + goal_alignment` con pesi arbitrari
- Combinarli **moltiplicativamente** (e con relativize) come nella formula Book #1 §2.2.2
- O ancora meglio: derivare *un singolo scalar* da un LLM judge che valuta "questo codice avvicina al goal?"

Implicazione pratica: nel mio [`fractal_coding_loop.md`](docs/vision/fractal_coding_loop.md) avevo proposto una reward additiva. **È sub-ottimale**. Il pattern Hernández-Cerezo richiede **una composizione moltiplicativa** dove se uno dei fattori va a zero (test fail), tutta la reward va a zero.

### A3.3 La Wigner reward dovrebbe applicarsi al training data del coder

Il [Slide doc](docs/bibliography/sources/books/2020_fractal_memory_slides.md) propone $R = \pi/2 \cdot x \cdot e^{-\pi x^2/4}$ come distribuzione ottimale di loss per il training di una NN.

Per il Fractal Coding Loop, la Fractal Memory di esempi di codice dovrebbe seguire la stessa distribuzione:
- **Esempi facili** (loss bassa) → poco peso → poco frequenti nei batch
- **Esempi medi** (loss media) → peso massimo → frequenti nei batch
- **Esempi difficili** (loss alta) → peso basso → poco frequenti

Questo è **automatico curriculum learning** per gli esempi che il sistema accumula.

## A4. Cosa NON ho ancora trovato (gap onesti)

Dopo questa terza espansione, restano gap noti:

1. **Versioni più recenti del Book #2**: la V0.2 è del 2020. Ce ne sono V0.3+? L'unico modo è chiedere agli autori.

2. **Documentazione completa di `fragile-rl`**: ho letto solo l'introduzione del libro. Restano ~50 capitoli di gauge theory.

3. **Lavoro post-2020 prima di Fragile Mechanics**: c'è un buco di 4 anni (2020-2024) dove Sergio ha lavorato, ma cosa? Il blog non è aggiornato dopo 2019.

4. **Implementazioni Book #2 reali**: nessuna. `fragile-rl` è un work-in-progress che si avvicina a Book #2 ma non lo replica. È spazio bianco per chiunque voglia contribuire.

## A5. La sintesi finale aggiornata, in tre frasi

Aggiorno **definitivamente** la sintesi:

> **Una.** Il programma Fractal AI è **dieci anni di pensiero** (2014-2026), non un singolo paper, con anni di gestation informale tra ogni idea-blog e la sua formalizzazione accademica. La traiettoria è coerente: dall'ottimizzazione (GAS 2017) al planning (FMC 2018) all'AGI (Book #2 2020) alla cosmologia cognitiva (Fragile Mechanics 2024-26).

> **Due.** Il vero contributo intellettuale degli autori non è "FMC" come algoritmo isolato, ma il **principio di frattalità computazionale**: lo stesso meccanismo (cloning + virtual reward + relativize) si applica a livelli ontologici diversi — dai walker che pianificano azioni, ai pesi NN che si auto-ottimizzano, alle architetture che si auto-snelliscono, alle reti che si specializzano. È **autocoerenza algoritmica**.

> **Tre.** Il Fractal Coding Loop (la mia visione per Claude) è **operativamente realistico** perché il corpus mostra che il principio è stato applicato, in piccolo, già in 5+ domini diversi (kart racing, rocket flying, function optimization, Atari, multi-agent collaboration). Non sto inventando un'astrazione — sto applicando un pattern collaudato a un nuovo dominio (coding agentic).

## A6. La domanda finale

> *Perché in 10 anni nessun gruppo accademico mainstream ha picked up this program?*

Le risposte oneste:

1. **Marketing linguistico povero**: "Fractal AI", "Fragile theory", "octopus collaboration" non sono il lessico di NeurIPS/ICML.
2. **Pubblicazioni in venue minori**: arXiv-only, blog, drafts privati. Il programma manca di **Nature/Science papers** che gli avrebbero dato visibilità.
3. **Outsider position**: Sergio è di HCSoft (azienda piccola), non di DeepMind/OpenAI. La community AI ha bias gravity verso i grandi labs.
4. **Idee ad alta varianza**: alcune sono brillanti (FMC), altre speculative (retrocausalità, T-symmetry NN). I conservatori scartano tutto, gli entusiasti si scottano sulle speculative.
5. **Filosofia anti-Pareto, anti-deep-RL**: il programma sfida le ortodossie metodologiche. Le ortodossie hanno guardiani.

**La mia ipotesi**: il programma sarà riscoperto dalla mainstream entro 5 anni, **dopo che qualcun altro ha replicato indipendentemente le idee**. Sarà ricordato come "anticipato da Hernández-Cerezo et al." in nota a piè di pagina, ma il merito andrà a chi avrà la patientza accademica per pubblicarlo nei venue giusti.

A meno che la community non si svegli ora. Ma è improbabile.

---

*Fine ADDENDUM 2026-04-26. Documento completo dopo l'archiviazione bibliografica integrale.*

*Per la bibliografia completa con file locali vedi [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md). Per i blog posts annotati vedi [`docs/bibliography/sources/blog_posts/INDEX.md`](docs/bibliography/sources/blog_posts/INDEX.md).*
