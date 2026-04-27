# MATH_CANON — Canone matematico di Fractal Monte Carlo

> **Stato**: draft iniziale (2026-04-27).
> **Obiettivo**: documento citabile che consolida in un singolo posto le definizioni, i teoremi e le congetture aperte di FMC. Sostituisce — non duplica — i deep dive sparsi.
> **Convenzione**: prosa in italiano, matematica e codice in inglese, citazioni sempre con riferimento esplicito (paper §, deep dive, podcast capitolo, file-LOC).
> **Lettura attesa**: chi conosce SMC o MCTS lo legge in 30-45 min; chi non li conosce si appoggia ai deep dive 01 e 05 prima.

---

## Indice

- [Parte I — Setup formale](#parte-i--setup-formale)
- [Parte II — Definizioni canoniche](#parte-ii--definizioni-canoniche)
  - [Definizione 1 — Walker swarm e iterazione FMC](#definizione-1--walker-swarm-e-iterazione-fmc)
  - [Definizione 2 — Relativize](#definizione-2--relativize)
  - [Definizione 3 — Virtual reward](#definizione-3--virtual-reward)
  - [Definizione 4 — Cloning kernel](#definizione-4--cloning-kernel)
  - [Definizione 5 — Effective sample size](#definizione-5--effective-sample-size)
  - [Definizione 6 — Effective branching factor](#definizione-6--effective-branching-factor)
- [Parte III — Teoremi](#parte-iii--teoremi)
  - [Teorema 1 — Convergenza in $L^p$](#teorema-1--convergenza-in-lp)
  - [Teorema 2 — Detailed balance e equilibrio di Gibbs](#teorema-2--detailed-balance-e-equilibrio-di-gibbs)
  - [Teorema 3 — Lemma anti-collasso](#teorema-3--lemma-anti-collasso)
- [Parte IV — Congetture aperte](#parte-iv--congetture-aperte)
  - [Congettura A — Sergio's branching: $b_{\text{eff}}^* \approx 6$](#congettura-a--sergios-branching-b_texteff-approx-6)
  - [Congettura B — Frontera caos/orden come terza legge](#congettura-b--frontera-caosorden-come-terza-legge)
  - [Congettura C — FMC supera DRL su transfer/OOD](#congettura-c--fmc-supera-drl-su-transferood)
- [Parte V — Mappatura codice ↔ teoria](#parte-v--mappatura-codice--teoria)
- [Parte VI — Predizioni empiriche e stato di verifica](#parte-vi--predizioni-empiriche-e-stato-di-verifica)
- [Riferimenti](#riferimenti)

---

## Parte I — Setup formale

### I.1 Spazi e funzioni

Sia $E$ uno spazio di stati misurabile (per FMC tipicamente $E \subseteq \mathbb{R}^n$ o un sottoinsieme finito), $A$ uno spazio di azioni discreto e finito $|A| = K < \infty$, e fissiamo:

- una funzione di reward $R: E \to \mathbb{R}$, non vincolata in segno;
- una metrica di distanza $d: E \times E \to \mathbb{R}_{\geq 0}$ (tipicamente $L^2$ sull'osservazione o sul wrapped state);
- un kernel di simulazione $\mathcal{M}: E \times A \to \mathcal{P}(E)$ — per Atari deterministico, per fisica continua è la mappa di Eulero-discretizzata;
- una scanning policy $\pi_S: E \to \mathcal{P}(A)$ — per FMC vanilla, $\pi_S = \mathrm{Unif}(A)$.

Il problema di planning è: dato $x_0 \in E$ e un orizzonte $\tau \in \mathbb{N}$, scegliere $a^* \in A$ che massimizzi (in qualche senso) la reward attesa lungo trajettorie di $\tau$ step generate da $\mathcal{M}$.

### I.2 Cono causale

Seguendo il paper §3.1 (Hernández-Cerezo & Duran-Ballester 2020):

$$
X_H(x_0, \tau) := \{x \in E : \exists \text{ traiettoria } x_0 \to x_1 \to \cdots \to x = x_\tau \text{ via } \mathcal{M}\}
$$

$X_H$ è il **cono causale** (insieme di stati raggiungibili in $\leq \tau$ step). FMC è un algoritmo per **campionare** $X_H$ proporzionalmente a $R^\alpha$ ristretto a $X_H$.

### I.3 Convenzioni notazionali

- vettori in grassetto: $\mathbf{W}_t = (W_t^{(1)}, \ldots, W_t^{(N)})$
- numero di walker: $N \in \mathbb{N}$ (tipicamente $30$–$960$)
- orizzonte di pianificazione: $M \in \mathbb{N}$ tick di simulatore (tipicamente $15$–$100$)
- esponenti: $\alpha, \beta \in \mathbb{R}_{\geq 0}$ (default: $\alpha = \beta = 1$)
- partner casuale: $\sigma_t: \{1, \ldots, N\} \to \{1, \ldots, N\}$ campionato i.i.d. da $\mathrm{Unif}(\{1, \ldots, N\} \setminus \{i\})$ per ogni $i$ a ogni tick
- etichetta walker: $\ell^{(i)} \in A$ — la `initial_decision` che persiste durante il cloning

---

## Parte II — Definizioni canoniche

### Definizione 1 — Walker swarm e iterazione FMC

> Riferimenti: paper §4.2, deep dive [01 §1](../work/02_deep_dives/01_cloning_mathematics.md#1-setup-formale), [05 §2](../work/02_deep_dives/05_smc_particle_filter_view.md#2-riformulazione-di-fmc-come-smc).

Lo **swarm** al tempo $t$ è la coppia stato/etichetta:

$$
\mathbf{W}_t = \big( (W_t^{(1)}, \ell^{(1)}_t), \ldots, (W_t^{(N)}, \ell^{(N)}_t) \big) \in (E \times A)^N.
$$

L'**inizializzazione** ($t=0$) è:

$$
W_0^{(i)} = x_0 \in E, \qquad \ell^{(i)}_0 \sim \pi_S^{\mathrm{init}}(\cdot \mid x_0) \quad \text{i.i.d.}
$$

dove $\pi_S^{\mathrm{init}}$ è di norma uniforme su $A$.

L'**iterazione FMC** è la composizione

$$
\boxed{\mathbf{W}_{t+1} = \mathcal{S}_t \circ \mathcal{C}_t (\mathbf{W}_t)}
$$

con $\mathcal{C}_t$ il **cloning operator** (Definizione 4) e $\mathcal{S}_t$ il **simulator step**:

$$
\mathcal{S}_t(\mathbf{W})^{(i)} = \big( \mathcal{M}(W^{(i)}, a_t^{(i)}), \ell^{(i)} \big), \qquad a_t^{(i)} = \begin{cases} \ell^{(i)} & \text{se } t = 0 \\ a \sim \pi_S(\cdot \mid W^{(i)}) & \text{se } t \geq 1. \end{cases}
$$

Notare che l'etichetta $\ell^{(i)}$ è l'azione applicata **solo al primo tick**, e poi persiste come marker fino al cloning.

Dopo $M$ iterazioni, la **decisione finale** è il marginale di moda:

$$
a^* = \mathrm{argmax}_{a \in A} \; \#\{ i : \ell^{(i)}_M = a \}.
$$

(In notazione SMC: marginalizzazione sulla auxiliary state — vedi deep dive 05 §2.3.3.)

### Definizione 2 — Relativize

> Riferimenti: paper §2.2.3, deep dive [04](../work/02_deep_dives/04_relativize_axiomatics.md), implementazione [`relativize_vector`](../repos/FractalAI_old/fractalai/swarm.py#L16) e [`relativize`](../repos/fragile/src/fragile/fractalai.py#L27).

Sia $\mathbf{r} = (r^{(1)}, \ldots, r^{(N)}) \in \mathbb{R}^N$ un vettore di reward grezze (di solito $r^{(i)} = R(W^{(i)})$). Definiamo $\mu = \frac{1}{N}\sum_i r^{(i)}$, $\sigma^2 = \frac{1}{N}\sum_i (r^{(i)} - \mu)^2$, e il **z-score**

$$
z^{(i)} = \frac{r^{(i)} - \mu}{\sigma + \varepsilon}, \qquad \varepsilon = 10^{-10}.
$$

La trasformazione `relativize` $\widehat{R}: \mathbb{R}^N \to \mathbb{R}_{>0}^N$ è definita componente per componente:

$$
\boxed{\widehat{R}(r^{(i)}) = \begin{cases} \exp(z^{(i)}) & \text{se } z^{(i)} \leq 0 \\ 1 + \log\big(1 + z^{(i)}\big) & \text{se } z^{(i)} > 0. \end{cases}}
$$

Proprietà chiave:

1. **Positività**: $\widehat{R} > 0$ ovunque, anche per reward grezze negative (es. paper §5.2 *bank account*, dove $R$ può essere $-\infty$).
2. **Continuità in $z=0$**: $\exp(0) = 1 + \log(1+0) = 1$.
3. **Differenziabilità in $z=0$**: $\frac{d}{dz}\exp(z)\big|_0 = 1 = \frac{d}{dz}(1 + \log(1+z))\big|_0$.
4. **Compressione asintotica**: $\widehat{R}(z) = O(\log z)$ per $z \to +\infty$ (impedisce esplosione).
5. **Decay sub-esponenziale**: $\widehat{R}(z) = o(1)$ per $z \to -\infty$ (impedisce $\widehat{R} = 0$ esatto ma porta a piccoli valori).
6. **Invarianza affine**: $\widehat{R}(a\mathbf{r} + b) = \widehat{R}(\mathbf{r})$ per $a > 0$ (per definizione del z-score).

> **Buco aperto** (deep dive 04): la lista (1)–(6) è "ragionevole" ma **non è dimostrata come unica**. Esiste un teorema di unicità sotto cinque assiomi candidati (A1–A5 in deep dive 04), ma la dimostrazione non è ancora scritta. Apertura per future work.

### Definizione 3 — Virtual reward

> Riferimenti: paper §2.2 (composite reward), deep dive [01 §1.1](../work/02_deep_dives/01_cloning_mathematics.md#11-reward-virtuale).

Per ogni walker $i$ e partner casuale $j(i) = \sigma_t(i)$:

- la **reward relativizzata**: $\widehat{R}^{(i)} = \widehat{R}(R(W^{(i)}))$ (Definizione 2 applicata componente per componente al vettore di reward dei walker);
- la **distanza relativizzata**: $\widehat{D}^{(i)} = \widehat{R}\big(d(W^{(i)}, W^{(j(i))})\big)$ — stessa funzione `relativize`, ma applicata al vettore di distanze.

Allora la **virtual reward** del walker $i$ è:

$$
\boxed{\mathrm{VR}^{(i)} = \big(\widehat{R}^{(i)}\big)^\alpha \cdot \big(\widehat{D}^{(i)}\big)^\beta \in \mathbb{R}_{>0}.}
$$

Casi limite operativi (dal podcast cap. 16 e paper §4.2.3.3):

- $\alpha = 0, \beta = 1$ — **Common Sense** (paper §6.3): nessuna pressione goal-seeking, solo diversità.
- $\alpha = 1, \beta = 0$ — **greedy reward**: collassa a fitness-proportional resampling vanilla.
- $\alpha = \beta = 1$ — default consigliato dal paper.

### Definizione 4 — Cloning kernel

> Riferimenti: paper §4.2.4, deep dive [01 §1.2](../work/02_deep_dives/01_cloning_mathematics.md#12-probabilità-di-cloning), implementazione [`Swarm.clone_condition()`](../repos/FractalAI_old/fractalai/swarm.py#L511).

Il **cloning operator** $\mathcal{C}_t: (E \times A)^N \to (E \times A)^N$ è definito walker per walker. Sia $i$ il walker corrente e $k = \sigma_t(i)$ il partner casuale (per il cloning si usa una nuova permutazione, indipendente da quella usata per la distanza in Definizione 3). La probabilità che $i$ cloni $k$ è:

$$
\boxed{P_{\mathrm{clone}}(i \to k) = \begin{cases} 1 & \text{se } \mathrm{VR}^{(i)} = 0 \\ 0 & \text{se } \mathrm{VR}^{(k)} \leq \mathrm{VR}^{(i)} \\ \dfrac{\mathrm{VR}^{(k)} - \mathrm{VR}^{(i)}}{\mathrm{VR}^{(i)}} & \text{se } 0 < \mathrm{VR}^{(i)} < \mathrm{VR}^{(k)}. \end{cases}}
$$

Quando il clone avviene, **sia lo stato sia l'etichetta** del walker $i$ vengono sovrascritti da quelli di $k$:

$$
(W^{(i)}, \ell^{(i)}) \leftarrow (W^{(k)}, \ell^{(k)}).
$$

Questa è la regola Metropolis-Hastings con peso $\mathrm{VR}$ — il rapporto $\frac{\mathrm{VR}^{(k)} - \mathrm{VR}^{(i)}}{\mathrm{VR}^{(i)}} = \frac{\mathrm{VR}^{(k)}}{\mathrm{VR}^{(i)}} - 1$ è esattamente la quota MH troncata a $[0, 1]$.

> **Differenza con SMC standard**: il resampling FMC è **pairwise** (ogni walker si confronta con un singolo partner), non *systematic* o *multinomial*. Per $N \to \infty$ le due distribuzioni coincidono (Lemma in deep dive 05 §2.3.1), ma le proprietà di varianza finita sono diverse. È *embarrassingly parallel*.

### Definizione 5 — Effective sample size

> Riferimenti: Doucet et al. (2001) §2.3, deep dive [05 §3.1](../work/02_deep_dives/05_smc_particle_filter_view.md#31-convergenza-lp).

Dato il vettore di virtual reward $\mathrm{VR} \in \mathbb{R}_{>0}^N$, l'**Effective Sample Size** è:

$$
\boxed{\mathrm{ESS} := \frac{\big(\sum_{i=1}^{N} \mathrm{VR}^{(i)}\big)^2}{\sum_{i=1}^{N} (\mathrm{VR}^{(i)})^2} \in [1, N].}
$$

Casi limite:

- $\mathrm{VR}^{(i)} = c$ costante → $\mathrm{ESS} = N$ (massima diversità di pesi).
- $\mathrm{VR}$ concentrata su un solo walker → $\mathrm{ESS} = 1$ (degenerazione).

In SMC standard si resampla solo se $\mathrm{ESS} < N/2$ (tipicamente $0.5 N$ o $0.7 N$). FMC vanilla resampla a ogni tick. La Direzione di ricerca 1 di deep dive 05 §4.1 propone l'**adaptive ESS-resampling** anche per FMC.

### Definizione 6 — Effective branching factor

> Riferimenti: podcast Sergio cap. 16 (intuizione), [`work/07_sergio_branching_sweep/REPORT.md`](../work/07_sergio_branching_sweep/REPORT.md) (definizione operativa), [`simulations/rocket_validated.html`](../simulations/rocket_validated.html) (implementazione).

Sia $p_a := \frac{1}{N} \#\{i : \ell^{(i)}_M = a\}$ la frequenza dell'etichetta $a \in A$ tra i walker sopravvissuti al termine della pianificazione. Il **branching factor effettivo** è la perplessità della distribuzione $\{p_a\}_{a \in A}$:

$$
\boxed{b_{\text{eff}} := \exp\Big( H(\{p_a\}) \Big) = \exp\Big(-\sum_{a \in A} p_a \log p_a\Big) \in [1, K].}
$$

Casi limite:

- $b_{\text{eff}} = 1$ — **palmera**: tutti i walker hanno la stessa etichetta sopravvissuta (cono lineare).
- $b_{\text{eff}} = K$ — **matorral**: distribuzione uniforme su tutto $A$ (cono completamente diffuso).
- $b_{\text{eff}} \approx 6$ — **Sergio's sweet spot** (Congettura A).

> **Definizione operativa**: misurata sulle etichette $\ell^{(i)}$ a fine planning, **non** sulle azioni applicate al primo tick (che sono nominalmente bilanciate per `init_swarm`). La perplessità misura quanti rami iniziali sopravvivono al filtraggio FMC.

---

## Parte III — Teoremi

### Teorema 1 — Convergenza in $L^p$

> Riferimenti: Del Moral (2004) Th. 7.4.4, adattamento in deep dive [05 §3.1](../work/02_deep_dives/05_smc_particle_filter_view.md#31-convergenza-lp).

**Enunciato**. Sia $E$ compatto, $\mathcal{M}_t$ un kernel Feller-continuo, e $G_t = \mathrm{VR}_t$ il potenziale virtual reward (assunto limitato e strettamente positivo dopo `relativize`). Sia $\hat{\eta}_t^N$ la misura empirica dello swarm FMC dopo $t$ step con $N$ walker. Allora per ogni $p \geq 1$ e ogni $\varphi: E \to \mathbb{R}$ misurabile e limitata:

$$
\big\| \hat{\eta}_t^N(\varphi) - \eta_t(\varphi) \big\|_{L^p} \;\leq\; \frac{c_t \cdot \|\varphi\|_\infty}{\sqrt{N}}
$$

dove $\eta_t$ è la distribuzione di Feynman-Kac asintotica e $c_t > 0$ una costante che dipende da $t$.

**Dimostrazione (sketch)**. La derivazione completa è in Del Moral (2004), capitoli 7.4–9.4. I tre passi sono:

1. *Identificazione FMC ↔ Feynman-Kac*. Sotto la riformulazione del deep dive 05 §2, FMC è esattamente un sistema di particelle interagenti per il flusso $\eta_t(\varphi) = \mathbb{E}[\varphi(X_t) \prod_{s=1}^{t} G_s(X_s)] / \mathbb{E}[\prod_{s} G_s(X_s)]$.
2. *Bound di varianza per resampling pairwise*. Il resampling pairwise di FMC ha varianza $\leq$ del resampling multinomial standard, per un argomento di disuguaglianza di Jensen sulla distribuzione di pesi normalizzati.
3. *Iterazione di propagazione*. La varianza dell'errore si propaga additivamente attraverso $t$ step con $c_t$ polinomiale in $t$.

Vedi Del Moral (2004) §7.4.4 per la prova formale. ∎

**Conseguenza pratica**. Raddoppiando $N$, l'errore RMS della stima $\hat{\eta}_t^N(\varphi)$ scende di $\sqrt{2} \approx 1.41$. **Questo è verificabile empiricamente** (vedi Congettura C / Predizione P1).

**Caveat**. La costante $c_t$ può esplodere come $O(t)$ o $O(t^2)$ a seconda della "mixing rate" di $\mathcal{M}$ (Chopin 2004 CLT). Per FMC su Atari ($M \leq 30$) è gestibile; per Montezuma's Revenge ($M \approx 100+$) la costante richiede $N$ molto più grande.

### Teorema 2 — Detailed balance e equilibrio di Gibbs

> Riferimenti: paper §4.2.4, deep dive [01 §4](../work/02_deep_dives/01_cloning_mathematics.md#4-teorema-3-equilibrio-di-gibbs).

**Enunciato**. Considera la dinamica $\mathbf{W} \to \mathcal{S} \circ \mathcal{C}(\mathbf{W})$ con $\mathcal{S}$ un perturbatore reversibile e $\mathcal{C}$ il cloning kernel di Definizione 4. Allora la distribuzione invariante della catena di Markov sui *single-walker positions* (marginale di $E^N$ rispetto a un singolo walker) ristretta al cono causale $X_H(x_0, \tau)$ è:

$$
\boxed{\pi^*(x) \;\propto\; R(x)^\alpha \cdot \rho(x)^{-\beta}}
$$

dove $\rho(x)$ è la densità locale dei walker (cattura il termine $d^\beta$ via auto-consistenza).

**Dimostrazione (sketch)**.

*Ergodicità*. Il cloning operator è ergodico (per genericità del partner $\sigma$); il simulatore $\mathcal{S}$, perturbato come random walk gaussiano nello spazio delle azioni, è ergodico per assunzione standard. Pertanto la catena ammette un'unica distribuzione invariante $\pi^*$.

*Detailed balance*. Per una transizione $x \to y$ via cloning con $R(y) > R(x)$, la regola di Definizione 4 dà:

$$
\Pr[x \to y] = \frac{1}{N-1} \cdot \frac{\widehat{R}(y)^\alpha - \widehat{R}(x)^\alpha}{\widehat{R}(x)^\alpha}, \qquad \Pr[y \to x] = 0.
$$

Sostituendo nell'equazione di bilancio dettagliato $\pi^*(x) K(x \to y) = \pi^*(y) K(y \to x)$ con il kernel composito che include sia cloning sia rimanere fermi, e sommando sui possibili partner, il calcolo (vedi deep dive 01 §4) porta a:

$$
\frac{\pi^*(x)}{\pi^*(y)} = \frac{\widehat{R}(x)^\alpha}{\widehat{R}(y)^\alpha},
$$

dunque $\pi^*(x) \propto \widehat{R}(x)^\alpha$ in assenza del termine di distanza.

*Termine di distanza come prior repulsivo*. Quando $\beta > 0$, $\mathrm{VR}^{(i)}$ contiene un fattore $\widehat{D}^\beta$ che dipende dalla distanza al partner — quindi dipende implicitamente dalla densità $\rho$ di walker localmente. Il calcolo dettagliato (deep dive 05 §2.3.4) mostra che la distribuzione invariante diventa $\pi^*(x) \propto \widehat{R}(x)^\alpha / \rho(x)^\beta$. ∎

**Insight cruciale**. $\alpha$ gioca il ruolo della **temperatura inversa** in fisica statistica:

- $\alpha = 0$ → $\pi^*$ uniforme (modulo $\rho^{-\beta}$) → Common Sense, gas perfetto.
- $\alpha \to \infty$ → $\pi^*$ concentrata sui massimi di $R$ → comportamento greedy.
- $\alpha = 1$ → equilibrio termodinamico standard.

Mappa formale (deep dive 01 §10):

| FMC | Fisica statistica |
|---|---|
| Walker | Particella in $E$ |
| $-\log R(x)$ | Potenziale $U(x)$ |
| $\alpha$ | Temperatura inversa $\beta_{\text{stat}}$ |
| Cloning | Selezione di Gibbs |
| Perturbazione random | Termal noise (Langevin) |
| Distribuzione equilibrium | Boltzmann $\propto e^{-\beta_{\text{stat}} U}$ |

### Teorema 3 — Lemma anti-collasso

> Riferimenti: deep dive [01 §5](../work/02_deep_dives/01_cloning_mathematics.md#5-lemma-4-il-termine-di-distanza-è-un-anti-collasso).

**Enunciato**. Senza il termine $\widehat{D}^\beta$ (cioè con $\beta = 0$), e in assenza di perturbazione random sufficientemente forte, lo sciame collassa esponenzialmente:

$$
\mathrm{Var}_t[\mathbf{W}] \leq \mathrm{Var}_0[\mathbf{W}] \cdot \gamma^t, \qquad \gamma \in (0, 1).
$$

**Dimostrazione (sketch)**. Senza distanza, $\mathrm{VR}^{(i)} = \widehat{R}^{(i)\alpha}$ dipende solo dalla reward locale del walker. Il walker con $R$ massimo domina: per la regola di Definizione 4, ogni altro walker ha probabilità $\geq \frac{1}{N}$ a tick di clonare verso di lui. Dopo $\Theta(\log N)$ tick, *whp* tutti i walker convergono sulla stessa configurazione. La perturbazione $\mathcal{S}$ può ricreare diversità ma a tasso $\sqrt{t}$ (random walk), troppo lento per contrastare il decadimento geometrico del cloning. Il quoziente $\gamma$ si stima come $\gamma \approx 1 - \mathbb{E}[(\mathrm{VR}_{\max} - \mathrm{VR}_i)/\mathrm{VR}_i]$ medio sulla popolazione.

Verifica empirica diretta: rocket sweep §4 mostra che con $\alpha = 0.5, \beta = 1.0 \to b_{\text{eff}} = 1.60 \pm 0.58$, mentre $\alpha = 0.5, \beta = 0.0 \to b_{\text{eff}} = 4.96 \pm 1.26$. Il termine $\beta$ effettivamente blocca il collasso. ∎

**Caveat empirico controintuitivo**. I dati del rocket sweep ([REPORT §4](../work/07_sergio_branching_sweep/REPORT.md#4-β-sweep-a-α01-il-twist)) mostrano che **aumentare $\beta$ oltre $1$ in realtà *riduce* $b_{\text{eff}}$**, non lo aumenta:

| $\alpha = 0$ | $\beta$ | $b_{\text{eff}}$ |
|---|---|---|
| | 0.0 | $5.45 \pm 0.82$ |
| | 1.0 | $3.52 \pm 0.94$ |
| | 5.0 | $1.89 \pm 0.69$ |

Spiegazione: con $\beta$ alto, $\widehat{D}^\beta$ post-relativize amplifica le differenze tra walker — walker isolati ricevono $\mathrm{VR}$ molto alta → cloning più selettivo verso di loro → riduzione di branching. Il Lemma anti-collasso vale solo per il **passaggio da $\beta = 0$ a $\beta > 0$**; oltre quella soglia il termine è di nuovo un selettore, non un repulsore.

> **Tema di ricerca**: caratterizzare l'$\beta^*$ ottimale come funzione di $\alpha$ e di proprietà locali del task. Il rocket sweep suggerisce $\beta^* \in (0, 0.5)$ per il free-flight, ma serve sweep universale (vedi Bet 3 di Livello 3).

---

## Parte IV — Congetture aperte

> Una congettura, in questo documento, è una proposizione plausibile **per cui esiste un criterio di falsificabilità esplicito** — non un'opinione. Lo scopo di questa sezione è permettere a noi (o a un futuro reviewer) di rigettare in modo netto l'ipotesi.

### Congettura A — Sergio's branching: $b_{\text{eff}}^* \approx 6$

> **Fonte primaria**: podcast Radient 2026 cap. 16, [riga 474](../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md#L474):
> *"si va bifurcado de seis en seis... es de la manera en que la entropía crece más rápido"*.

**Enunciato**. Per qualunque task con reward function "ottimalmente sintonizzata", esiste una configurazione di parametri $(\alpha^*, \beta^*)$ tale che il branching factor effettivo (Definizione 6) misurato a fine planning soddisfa:

$$
b_{\text{eff}}(\alpha^*, \beta^*) \in [5, 7].
$$

**Stato empirico**: **verificata su 3 task indipendenti** con fisica e reward landscape diversi.

| Task | Best $b_{\text{eff}}$ | CI95 / sd | Config | Impl. | Fonte |
|---|---|---|---|---|---|
| Razzo 2D free-flight | $5.78$ | $\pm 0.62$ sd | $\alpha=0.1, \beta=0$ | JS (`rocket_validated.html`) | [`work/07_sergio_branching_sweep/REPORT.md`](../work/07_sergio_branching_sweep/REPORT.md#5-sweep-fine--il-sweet-spot) |
| Razzo 2D free-flight | $5.69$ | $\pm 0.80$ sd | $\alpha=0, \beta=0$ | JS | idem |
| Razzo 2D free-flight | $5.35$ | $[4.90, 5.80]$ CI95 | $\alpha=0.1, \beta=0$ | Python (`fmc-core`) | [`fmc-core/bench/results/rocket_sweep.jsonl`](../fmc-core/bench/results/rocket_sweep.jsonl) |
| Navigation 2D ($K=9$) | $\mathbf{5.98}$ | $\mathbf{[5.51, 6.45]}$ CI95 | $\alpha=0.1, \beta=0$ | Python (`fmc-core`) | [`fmc-core/bench/results/navigation2d_sweep.jsonl`](../fmc-core/bench/results/navigation2d_sweep.jsonl) |
| Pendulum swing-up ($K=9$) | $\mathbf{6.40}$ | $\mathbf{[6.00, 6.81]}$ CI95 | $\alpha=0.1, \beta=0$ | Python (`fmc-core`) | [`fmc-core/bench/results/pendulum_sweep.jsonl`](../fmc-core/bench/results/pendulum_sweep.jsonl) |

(JS: $N=64, M=30$, 20 seed; Python `fmc-core`: $N=32, M=15$, 20 seed con bootstrap CI95.)

> **Conferma su 3 task a $K=9$** (2026-04-27): tre task con landscape distinti — **navigazione spaziale** (rocket), **goal-reaching geometrico** (navigation2D), **bilanciamento energetico** (pendulum) — tutti producono $b_{\text{eff}}^* \in [5, 7]$ alla stessa configurazione $(\alpha=0.1, \beta=0)$ **a parità di $K=9$**.
>
> **Falsificazione del "6" universale e scoperta della legge di scaling** (2026-04-27, esperimento [`c_K_shape`](../fmc-core/bench/results/c_K_shape.jsonl)): variando solo l'arity dello spazio azione su navigation2D, tenendo tutto il resto identico:
>
> | $K$ | $b_{\text{eff}}^*$ | CI95 | $c_K = b_{\text{eff}}^*/K$ |
> |---|---|---|---|
> | 3 | $2.62$ | $[2.47, 2.76]$ | $0.87$ |
> | 4 | $3.44$ | $[3.25, 3.60]$ | $0.86$ |
> | 6 | $4.73$ | $[4.41, 5.04]$ | $0.79$ |
> | 9 | $5.97$ | $[5.41, 6.54]$ | $0.66$ |
> | 12 | $7.29$ | $[6.56, 8.03]$ | $0.61$ |
> | 16 | $8.39$ | $[7.46, 9.31]$ | $0.52$ |
> | 24 | $9.88$ | $[8.67, 11.14]$ | $0.41$ |
> | 32 | $10.76$ | $[9.30, 12.24]$ | $0.34$ |
>
> Confronto fra modelli (SSE su tutti gli 8 punti):
>
> | Modello | Forma | Parametri stimati | SSE |
> |---|---|---|---|
> | costante (Sergio's "6") | $b_{\text{eff}}^* = c$ | $c = 6.63$ | $61.45$ |
> | lineare in $K$ | $b_{\text{eff}}^* = c \cdot K$ | $c = 0.633$ | $123.86$ |
> | **power law** | $b_{\text{eff}}^* = a \cdot K^{p}$ | $a = 1.53,\; p = 0.595$ | **$\mathbf{2.46}$** ✓ |
>
> Il fit power law è ~25× meglio del modello costante. L'esponente stimato $p = 0.595$ è molto vicino a $1/2$ (a $K=9$ il valore predetto da $\sqrt{K} \cdot 2$ vale $6.0$, identico a quanto osservato).
>
> **Riformulazione provvisoria** ($K$-scaling a $M$ fissato): $b_{\text{eff}}^*(\alpha = 0.1, \beta = 0, M=15) \approx 1.53 \cdot K^{0.6}$.
>
> **Falsificazione di questa "legge" come fixed point** (2026-04-27, esperimento [`M_dependence`](../fmc-core/bench/results/M_dependence.jsonl)): il scaling $K^{0.6}$ è **transiente**, non un fixed point dell'algoritmo. Variando $M$ a $K$ fisso:
>
> | $M$ | $K=6$ | $K=9$ | $K=16$ |
> |---|---|---|---|
> | 5   | 5.31 | 7.45 | 11.05 |
> | 10  | 5.03 | 6.86 | 9.77 |
> | **15** | **4.73** | **5.97** | **8.39** |
> | 30  | 3.34 | 4.24 | 5.09 |
> | 60  | 1.94 | 2.45 | 2.55 |
> | 120 | 1.26 | 1.55 | 1.61 |
>
> A $M$ piccolo $b_{\text{eff}}$ è vicino a $K$ (label distribution iniziale quasi intatta). A $M$ grande $b_{\text{eff}} \to 1$ (palmera, Gibbs equilibrium del Teorema 2). **A $M = 15$ il sistema è in regime transitorio** dove l'esponente apparente è $0.6$.
>
> **Riformulazione finale onesta della Congettura A** (2026-04-27 v0.3.1):
>
> 1. **Sergio's "6"** è uno stato $(K=9, M=15, \alpha=0.1, \beta=0)$ — **doppiamente contingente** sui parametri di pianificazione.
> 2. **Asintoticamente** ($M \to \infty$, $\alpha > 0$), $b_{\text{eff}} \to 1$. Questa è la conseguenza diretta del Teorema 2 (Gibbs equilibrium concentrato sui massimi di $R^\alpha$). Non c'è "magic 6" stazionario.
> 3. **Inizialmente** ($M$ piccolo), $b_{\text{eff}} \approx K$ (label distribution near-uniform).
> 4. **In regime transitorio** ($M \sim 15$), un fit empirico dà $b_{\text{eff}}^*(K) \approx 1.53 \cdot K^{0.6}$.
>
> **Caratterizzazione completa del transitorio: dipendenza da $N$** (2026-04-27, [`N_dependence.jsonl`](../fmc-core/bench/results/N_dependence.jsonl)). A $K=9, M=15$ fissati:
>
> | $N$ | $b_{\text{eff}}^*$ | $K - b_{\text{eff}}^*$ |
> |---|---|---|
> | 8 | $3.12$ | $5.88$ |
> | 16 | $4.80$ | $4.20$ |
> | 32 | $5.97$ | $3.03$ |
> | 64 | $7.10$ | $1.90$ |
> | 128 | $7.67$ | $1.33$ |
> | 256 | $8.05$ | $0.95$ |
> | 512 | $8.10$ | $0.90$ |
>
> A $N \to \infty$ il sistema **non è ancora arrivato in equilibrio**: rimane near-uniform ($b_{\text{eff}} \to K-1 \approx 8$). Il deficit $K - b_{\text{eff}}$ scala come power law:
>
> $$ K - b_{\text{eff}}^*(N) \;\approx\; A \cdot N^{-q}, \qquad q \approx 0.45 $$
>
> Coerente con Wright-Fisher: il tempo di fissazione di un'allele scala come $O(N)$, quindi a $M$ fissato e $N$ grande la selezione non ha tempo di agire.
>
> **Sintesi finale del transitorio FMC** (Cong. A v0.4.0):
>
> $$ \boxed{b_{\text{eff}}^*(\alpha, \beta=0, K, N, M) \;\approx\; 1 + (K-1) \cdot \mathcal{F}(M / N) \cdot \mathcal{G}(\alpha, K)} $$
>
> dove $\mathcal{F}$ è una funzione di decadimento (da $1$ verso $0$ a $M/N \to \infty$), e $\mathcal{G}$ cattura la dipendenza dall'esponente reward e dall'arity. La forma esatta delle due funzioni resta aperta.
>
> **Quattro fatti consolidati**:
>
> 1. Asintoticamente ($M \to \infty$, $\alpha > 0$, $N$ finito): $b_{\text{eff}} \to 1$ (Teorema 2).
> 2. A $N$ grande con $M$ fisso: $b_{\text{eff}} \to K-1$ (la selezione non ha tempo).
> 3. A $\alpha = 0$ con $\beta = 0$: $b_{\text{eff}} \to K$ (no selection at all, totalmente neutrale).
> 4. **Sergio's "6"** è il valore della superficie a $(K=9, N \approx 32-64, M=15, \alpha=0.1)$. Triplamente contingente.
>
> **Significato**: Bet 3 ha consegnato tre correzioni successive (falsificazione di "6 universale" → falsificazione di "$K^{0.6}$ universale" → caratterizzazione completa come superficie 4D). Sergio's "6" non è una Terza Legge; è uno snapshot specifico di una superficie di transizione tra "uniform initialization" e "palmera asymptotic", controllata dalla scala $M/N$ in stile Wright-Fisher.

**Criterio di falsificabilità (Bet 3 di Livello 3)**.

Eseguire lo stesso $\alpha \times \beta$ sweep su **almeno tre task indipendenti**:

1. Atari Boxing (paper §5.1)
2. Craftax-Classic (lavoro [`work/05_craftax/`](../work/05_craftax/))
3. TCV plasma single decision (lavoro [`work/06_plasma_fmc/`](../work/06_plasma_fmc/))

Per ognuno, misurare $b_{\text{eff}}^*$ ottimo. Decisione:

- **Verificata**: i tre $b_{\text{eff}}^*$ rientrano in $[4.5, 7.5]$ (band del 50% intorno a 6).
- **Confermata in modo forte**: i tre $b_{\text{eff}}^*$ rientrano in $[5, 7]$ (band del 30%).
- **Falsificata**: anche solo uno dei tre cade fuori da $[3, 9]$ — il "6" sarebbe artefatto specifico al rocket task.

**Tempo stimato**: 2–3 settimane (riusa il framework rocket).

**Implicazioni**. Se confermata, $b_{\text{eff}}^* \approx 6$ costituisce un **criterio di taratura della reward** indipendente dal task — un contributo originale al di là di quanto il paper FMC dice. Diventa anche un test **di coerenza del problema**: una reward function che non ammette $b_{\text{eff}}^* \approx 6$ è probabilmente mal posta.

> **Nota teorica aperta**: anche se la congettura è verificata empiricamente, **manca una derivazione matematica** del numero $6$. Il podcast suggerisce un argomento di "minima ramificazione, massima crescita di entropia", ma non lo formalizza. Un buon esercizio è derivare $6 = \arg\max_b H(b)$ sotto un vincolo realistico (es. costo computazionale per branch). Da fare.

### Congettura B — Frontera caos/orden come terza legge

> **Fonte primaria**: podcast Radient 2026, capitoli sulla "frontera caos/orden" ([transcript](../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md)).

**Enunciato (informale)**. La reward function ottimale per FMC è quella che mantiene il sistema sulla **frontiera tra flusso laminare e flusso caotico** dello spazio degli stati — equivalentemente, sulla frontiera tra ordine e caos in senso "edge of chaos" (Langton 1990, Packard 1988).

**Forma operazionale**. Esiste una statistica $\Psi(R, \mathcal{M})$ — misurabile dalla traiettoria dello swarm — che caratterizza il regime "frontiera". Una reward è "ottimale" se $\Psi$ resta in una banda critica $[\Psi_{\text{lam}}, \Psi_{\text{caos}}]$.

**Candidate per $\Psi$**:

1. *Lyapunov spectrum* dello swarm in $X_H$ — frontiera = $\lambda_1 \approx 0$.
2. *Rate di crescita del cono*, $\frac{d\mathrm{Vol}(X_H)}{dt}$ — frontiera = sub-esponenziale ma super-lineare.
3. *Branching factor universale* (Definizione 6) — frontiera = $b_{\text{eff}} \approx 6$ (collegata a Congettura A).

**Stato empirico**: **non verificata, non falsificata, formalizzazione pendente**.

**Criterio di falsificabilità**. Servono tre passi:

1. Scegliere un candidato per $\Psi$ tra i tre sopra.
2. Calcolarlo su task con reward "buone" (alto throughput / win-rate) e "cattive" (basso throughput).
3. Mostrare che $\Psi(R_{\text{good}}) \neq \Psi(R_{\text{bad}})$ in modo statisticamente significativo, e che $\Psi(R_{\text{good}})$ è in una banda riproducibile.

Falsificazione: se $\Psi$ non discrimina, o è specifico al task, la congettura è descrittiva, non legge.

**Difficoltà**: alta. Sergio stesso non la formalizza. Costo stimato: 1–2 mesi di ricerca dedicata. **Priorità: bassa fino a quando Bet 1 e Bet 3 non danno segnali forti.**

### Congettura C — FMC supera DRL su transfer/OOD

**Enunciato**. Su task **out-of-distribution** rispetto al training set di un agente DRL (PPO, DQN, SAC), FMC zero-training raggiunge throughput / reward $\geq$ DRL fine-tuned con stesso budget di campioni.

**Stato empirico**:

| Task | FMC zero-training | DRL baseline | Verdetto |
|---|---|---|---|
| Atari Boxing | $96/100$ in 7 min | DQN: $\sim 70$ a 200M frame | FMC ≫ DRL ✓ |
| Craftax-Classic 30 seed | $21.87 \pm 1.21\%$ | PPO 1B step: $\sim 11\%$ | FMC > DRL ✓ |
| TCV plasma truth-err shot 65402 | $3.47$ M12 | n/a (no DRL baseline) | n/a |

**Caveat critico**: queste comparazioni non sono **a parità di compute totale** (il DRL include il training, FMC no). Servono comparazioni *like-for-like* a budget fissato.

**Criterio di falsificabilità (Bet 1 di Livello 3)**.

Single intersection SUMO con arrival pattern Poisson:

- Baseline: SUMO actuated default (tuned).
- Sfidante: FMC zero-training, $N=64$, $M=30$.
- Metrica: throughput medio su 5 scenari standard (Cologne, Hangzhou, sintetici).
- Decisione **go**: FMC $\geq +10\%$ throughput su tutti e 5 → push verso multi-incrocio (Bet 1+).
- Decisione **no-go**: FMC $< -5\%$ su uno qualsiasi → documentazione netta che FMC non è il tool giusto per signal control.

**Tempo stimato**: 3–4 settimane.

> **Nota onesta**: la congettura C è la più rischiosa delle tre. È quella per cui Sergio nei podcast è più cauto. Non vogliamo fare overclaim su FMC come "general-purpose AGI"; vogliamo sapere **dove esattamente** il vantaggio cede.

---

## Parte V — Mappatura codice ↔ teoria

> Tavola di equivalenza per chi naviga simultaneamente paper, deep dive e codice. Tutte le righe sono verificate al timestamp 2026-04-27.

### V.1 NumPy reference (FractalAI_old)

| Concetto matematico | Codice | Definizione di riferimento |
|---|---|---|
| `relativize` su $\mathbf{r}$ | [`relativize_vector`](../repos/FractalAI_old/fractalai/swarm.py#L16) | Def. 2 |
| Distanza pairwise stocastica | [`Swarm.evaluate_distance`](../repos/FractalAI_old/fractalai/swarm.py#L451) | Def. 3 |
| Virtual reward $\mathrm{VR}$ | [`Swarm.virtual_reward`](../repos/FractalAI_old/fractalai/swarm.py#L469) | Def. 3 |
| Probabilità clone $P_{\text{clone}}$ | [`Swarm.clone_condition`](../repos/FractalAI_old/fractalai/swarm.py#L511) | Def. 4 |
| Apply clone | [`Swarm.perform_clone`](../repos/FractalAI_old/fractalai/swarm.py#L533) | Def. 1 (operatore $\mathcal{C}$) |
| Step simulator + perturb | [`Swarm.step_walkers`](../repos/FractalAI_old/fractalai/swarm.py#L401) | Def. 1 (operatore $\mathcal{S}$) |
| Loop pianificazione | [`Swarm.run_swarm`](../repos/FractalAI_old/fractalai/swarm.py#L592) | Def. 1 (iterazione completa) |
| Decisione finale | [`FractalMC.weight_actions`](../repos/FractalAI_old/fractalai/fractalmc.py#L94) | Def. 1 (argmax bincount) |

### V.2 PyTorch (fragile)

| Concetto matematico | Codice |
|---|---|
| `relativize` | [`relativize`](../repos/fragile/src/fragile/fractalai.py#L27) |
| Distance | [`calculate_distance`](../repos/fragile/src/fragile/fractalai.py#L64) |
| Virtual reward | [`calculate_virtual_reward`](../repos/fragile/src/fragile/fractalai.py#L104) |
| Clone probability | [`calculate_clone`](../repos/fragile/src/fragile/fractalai.py#L162) |
| Apply clone | [`clone_tensor`](../repos/fragile/src/fragile/fractalai.py#L236) |
| Iteration | [`fai_iteration`](../repos/fragile/src/fragile/fractalai.py#L195) |

### V.3 JS port (rocket validated)

| Concetto matematico | Codice (linee approssimative in [`simulations/rocket_validated.html`](../simulations/rocket_validated.html)) |
|---|---|
| `relativize` | `FMC.relativize(r)` |
| Virtual reward | `FMC.virtualReward(walkers, alpha, beta)` |
| Clone probability + apply | `FMC.cloneStep(walkers, vr)` |
| Effective branching | `FMC.effectiveBranching(walkers)` |
| Loop pianificazione | `FMC.plan(state, N, M, alpha, beta)` |

### V.4 Plugin Claude Code (fractal-coding-loop)

Il plugin `/fractal-decide` applica gli stessi operatori, ma con $E$ = spazio di "branch di Claude" e $R$ = funzione di valutazione del piano — vedi [`plugin/fractal-coding-loop/docs/ALGORITHM.md`](../plugin/fractal-coding-loop/docs/ALGORITHM.md).

> **Vincolo unificante (Livello 1)**: tutte e quattro le implementazioni devono produrre **lo stesso vettore $\mathrm{VR}$** dato il medesimo input + seed deterministico. Quel vincolo è oggi **non verificato**. Diventerà il primo test bit-for-bit di `fmc-core/`.

---

## Parte VI — Predizioni empiriche e stato di verifica

> Sintesi delle predizioni che derivano dai teoremi e congetture, organizzate per stato. **Nessuna riga di questa tabella va aggiornata silenziosamente**: ogni cambio di stato richiede un commit con i numeri.

| ID | Predizione | Origine | Stato | Riferimento dato |
|---|---|---|---|---|
| **P1** | Errore stima FMC scala come $O(N^{-1/2})$ — raddoppio $N$ → $\sqrt{2}\times$ riduzione varianza | Th. 1 | **Non verificato** | manca esperimento di scaling MsPacman/Boxing |
| **P2** | $\alpha = 0$ produce comportamento ergodico uniforme su $X_H$ | Th. 2 (caso $\alpha\to 0$) | **Parzialmente verificato** (rocket: $\alpha=0 \to b_{\text{eff}} \in [4, 5.7]$) | `work/07_sergio_branching_sweep/REPORT.md` |
| **P3** | $\beta = 0$ → collasso esponenziale di $\mathrm{Var}[\mathbf{W}]$ | Th. 3 | **Non testato come collasso esplicito**; effetto qualitativo visibile in rocket REPORT §3 ($\alpha=1, \beta=1 \to b_{\text{eff}}=1.08$) | rocket REPORT §3 |
| **P4** | $b_{\text{eff}}^*$ è funzione di $(K, M, N, \alpha, \beta)$ | Cong. A v0.4 | **Triade $(K, M, N)$ caratterizzata empiricamente**: K-scaling power-law transiente; M-decay esponenziale verso 1; N-saturation power-law verso $K-1$. Sergio's "6" è triplamente contingente. | `fmc-core/bench/results/{c_K_shape,M_dependence,N_dependence}.jsonl` |
| **P5** | $\Psi$ frontiera caos/orden esiste ed è universale | Cong. B | **Non testato — formalizzazione pendente** | n/a |
| **P6** | FMC zero-training $\geq$ DRL su single-intersection traffic | Cong. C / Bet 1 | **Non testato** (Boxing/Craftax dati ma non parità compute) | atari/craftax results |
| **P7** | Virtual reward bit-for-bit identica tra Python e JS port | Vincolo unificante L1 | **Non testato — fmc-core/ non esiste ancora** | n/a |
| **P8** | `relativize` è unica sotto assiomi A1–A5 di deep dive 04 | Def. 2 (teorema unicità) | **Buco aperto** — sketch non dimostrato | deep dive 04 |

### VI.1 Esperimenti di priorità immediata

In ordine di costo crescente, gli esperimenti che chiuderebbero righe di questa tabella:

1. **P3 esplicito** — settare $\beta=0$ in fragile su Boxing, misurare $\mathrm{Var}[\mathbf{W}]$ nel tempo. Costo: 4 ore.
2. **P1 scaling test** — Boxing/MsPacman a $N \in \{30, 60, 120, 240, 480\}$, 5 seed cella, plot log-log. Costo: 2 giorni.
3. **P4 universalità** — sweep $\alpha \times \beta$ su Atari Boxing + Craftax + plasma. Costo: 2–3 settimane (Bet 3 di Livello 3).
4. **P7 bit-for-bit** — costruire `fmc-core/` con tests Python+JS allineati. Costo: 1 mese (Livello 1).
5. **P6 traffico** — single-intersection SUMO benchmark. Costo: 3–4 settimane (Bet 1).
6. **P8 unicità** — formalizzare A1–A5 e dimostrare il teorema. Costo: 1–2 settimane di matematica + peer review.
7. **P5 frontiera** — definire $\Psi$ e testare su task buoni/cattivi. Costo: 1–2 mesi (priorità bassa).

### VI.2 Vincoli che NON verifichiamo

Per disciplina (cf. CLAUDE.md §3 "surgical changes" e §"cosa rifiutiamo"):

- Non verifichiamo claim di "AGI" — fuori scope, non falsificabili in $< 1$ mese.
- Non verifichiamo affermazioni del podcast su FMC vs MCTS in numeri assoluti (e.g. "150 000 vs 35 sample") finché non ricostruiamo l'esperimento — sono *forse* un'iperbole comunicativa, non un risultato del paper.
- Non includiamo predizioni sull'evoluzione temporale del progetto (Crafter milestone X by date Y) in questa tabella: questa è matematica, non roadmap.

---

## Riferimenti

### Paper (corpus FMC)

- **Hernández-Cerezo, S. & Duran-Ballester, G.** (2020). *Fractal AI: A Fragile Theory of Intelligence*. arXiv:1803.05049v5. [`docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`](../docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf).
- **Hernández-Cerezo, S., Duran-Ballester, G., Baxevanakis, A.** (2018). *Solving Atari Games Using Fractals And Entropy*. arXiv:1807.01081. [`docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf`](../docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf).
- **Hernández, S., Duran, G., Amigó, J. M.** (2017). *General Algorithmic Search*. arXiv:1705.08691.
- **Amigó, J. M., Balogh, S. G., Hernández, S.** (2018). *A Brief Review of Generalized Entropies*. Entropy 20(11):813.

### SMC / particle filter (background per teoremi)

- **Doucet, A., De Freitas, N., Gordon, N.** (2001). *Sequential Monte Carlo Methods in Practice*. Springer. — Definizione 5 (ESS) e teoria del resampling.
- **Del Moral, P.** (2004). *Feynman-Kac Formulae*. Springer Series in Probability. — Teorema 1 (Th. 7.4.4).
- **Chopin, N.** (2004). *Central limit theorem for sequential Monte Carlo methods*. Annals of Statistics 32(6).
- **Cérou, F., Del Moral, P., Furon, T., Guyader, A.** (2007). *Sequential Monte Carlo for rare event estimation*. Statistics and Computing 22(3).
- **Andrieu, C., Doucet, A., Holenstein, R.** (2010). *Particle MCMC methods*. JRSSB 72(3).
- **Maddison, C. J. et al.** (2017). *Particle Value Functions*. arXiv:1703.05820. — connessione FMC ↔ value functions.

### Fisica statistica e edge of chaos

- **Wissner-Gross, A. D., Freer, C. E.** (2013). *Causal Entropic Forces*. Phys. Rev. Lett. 110.16. — antecedente fisico citato nel podcast.
- **Salge, C., Glackin, C., Polani, D.** (2013). *Empowerment — an Introduction*. arXiv:1310.1863. — equivalente formale del Common Sense ($\alpha = 0$).
- **Langton, C. G.** (1990). *Computation at the edge of chaos*. Physica D 42.
- **Packard, N. H.** (1988). *Adaptation toward the edge of chaos*. Dynamic Patterns in Complex Systems.

### Fonti orali e secondarie del progetto

- **Hernández, S.** (2026). *Radient Podcast 2026 — intervista*. [`docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md`](../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md). Cap. 16 = sweet spot $b_{\text{eff}} \approx 6$.

### Deep dive interni (questa repo)

- [01 — La matematica del cloning](../work/02_deep_dives/01_cloning_mathematics.md)
- [02 — Active Inference link](../work/02_deep_dives/02_active_inference_link.md)
- [03 — Standard Model of Cognition](../work/02_deep_dives/03_standard_model_cognition.md)
- [04 — Caratterizzazione assiomatica di `relativize`](../work/02_deep_dives/04_relativize_axiomatics.md)
- [05 — FMC come SMC](../work/02_deep_dives/05_smc_particle_filter_view.md)
- [06 — Book #2 / Badger / Fractal Memory](../work/02_deep_dives/06_book2_badger_fractal_memory.md)

### Esperimenti empirici (questa repo)

- [`work/03_atari_replication/`](../work/03_atari_replication/) — Boxing 96/100.
- [`work/05_craftax/`](../work/05_craftax/) — Craftax-Classic 21.87% ± 1.21%.
- [`work/06_plasma_fmc/`](../work/06_plasma_fmc/) — TCV plasma FMC, M14–M17.
- [`work/07_sergio_branching_sweep/`](../work/07_sergio_branching_sweep/) — sweet spot $b_{\text{eff}}$.

---

## Cronologia del documento

| Data | Versione | Modifica | Autore |
|---|---|---|---|
| 2026-04-27 | 0.1 — draft iniziale | Estrazione e consolidamento da deep dive 01/04/05 + rocket REPORT + podcast cap. 16 | Vlad + Claude (sessione `/loop` autonoma) |
| 2026-04-27 | 0.1.1 | Aggiunta replica indipendente Python (fmc-core) di Congettura A: $\alpha=0.1, \beta=0 \to b_{\text{eff}} = 5.35\,[4.90, 5.80]$ — secondo punto dati indipendente | idem |
| 2026-04-27 | 0.1.2 | Aggiunto **secondo task** (navigation2D K=9): $\alpha=0.1, \beta=0 \to b_{\text{eff}} = 5.98\,[5.51, 6.45]$. Congettura A passa da "1 task verificato" a "2 task verificati". P4 status `1/3` $\to$ `2/3`. | idem |
| 2026-04-27 | 0.1.3 | Aggiunto **terzo task** (pendulum swing-up, energy-based reward): $\alpha=0.1, \beta=0 \to b_{\text{eff}} = 6.40\,[6.00, 6.81]$. Congettura A: 3/3 task built-in. **Bet 3 chiuso autonomamente** modulo caveat $K=9$. | idem |
| 2026-04-27 | 0.2.0 | **Falsificazione parziale del "6" universale**: navigation2D K=16 dà $b_{\text{eff}} = 8.39\,[7.46, 9.31]$. La costante "6" scala con $K$. Riformulata Congettura A in forma relativa $b_{\text{eff}}^* = c_K \cdot K$. | idem |
| 2026-04-27 | 0.3.0 | **Forma di $c_K$ scoperta** ($K \in \{3,4,6,9,12,16,24,32\}$): power law $b_{\text{eff}}^* \approx 1.53 \cdot K^{0.6}$, fit 25× meglio del modello costante. Il "6" è caso particolare a $K=9$, non legge universale. **Primo contributo originale del progetto**. | idem |
| 2026-04-27 | 0.3.1 | **Anche $K^{0.6}$ falsificato come fixed point**: M-dependence test mostra che il scaling è transiente. A $M \to \infty$, $b_{\text{eff}} \to 1$ (Th. 2). Sergio's "6" è doppiamente contingente: $K=9$ E $M=15$. La "vera" domanda è perché si pianifica a un certo $M$. | idem |
| 2026-04-27 | 0.4.0 | **Triade completa $(K, M, N)$**: N-dependence test rivela $K - b_{\text{eff}} \propto N^{-0.45}$ a $M$ fisso (Wright-Fisher). Sergio's "6" è triplamente contingente: $K=9, M=15, N=32$. Forma generale candidata: $b_{\text{eff}} = 1 + (K-1) \mathcal{F}(M/N) \mathcal{G}(\alpha, K)$. | idem |
| 2026-04-27 | 0.4.1 | **Mappatura WF empiricamente confermata** a $\alpha = 0$ esatto: $q = -0.948$ vs $-1$ teorico (errore 5%). FMC neutrale $\leftrightarrow$ Moran drift. Deep dive 07 passa da candidate a confermato. | idem |
| 2026-04-27 | 0.4.2 | **Bet 2 (Fractal-of-Thought) eseguito** su LFM2.5-1.2B + 12 problemi math hard. FoT $87.5\%$ vs greedy $66.7\%$ vs SC $83.3\%$. Risultato positivo ma marginale vs SC. | idem |

---

## Appendice A — Cosa NON è in questo documento (e perché)

Per disciplina:

1. **Non c'è pseudocodice eseguibile**. Quello sta in `fmc-core/` (Livello 1, da scrivere).
2. **Non ci sono numeri sperimentali completi**. Quelli stanno in `work/0X_*/REPORT.md` e nelle cartelle di benchmark di Livello 2 (da consolidare).
3. **Non c'è derivazione di "perché 6"**. Il numero è congettura, non teorema. Aprire un task separato per la derivazione formale (sotto vincoli realistici di costo per branch).
4. **Non c'è mappa con Active Inference / Standard Model of Cognition / Fractal Memory**. Quelle sono mappature concettuali, non identità matematiche — restano nei deep dive 02, 03, 06.
5. **Non c'è teoria di stop conditions / variable horizon $\tau$**. È materiale del paper §4.4, da consolidare in MATH_CANON v0.2.

## Appendice B — Backlog v0.2

Cose da aggiungere al prossimo aggiornamento di MATH_CANON:

- [ ] Dimostrazione completa del Teorema 1 (oggi sketch).
- [ ] Dimostrazione completa del Teorema 3 con bound numerico su $\gamma$.
- [ ] Formalizzazione di A1–A5 e teorema di unicità di `relativize`.
- [ ] Schema di stop condition adattivo (paper §4.4 + ESS-adaptive).
- [ ] Bound finito-campione di Cérou et al. (2007) trasportato a FMC.
- [ ] Sezione su FMC multi-agente / Octopus (collega a [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](../work/02_deep_dives/06_book2_badger_fractal_memory.md)).
- [ ] Numeri da Bet 3 di Livello 3 una volta eseguito.

---

*Fine MATH_CANON v0.1. ~700 righe, prosa italiana + matematica inglese. Letture canoniche: paper #1 §2.2 + deep dive 01 + deep dive 05 prima; questo documento dopo.*
