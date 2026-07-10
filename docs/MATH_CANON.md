# MATH_CANON — Canone matematico di Fractal Monte Carlo

> **Stato**: v0.8.1 (2026-07-10). ⚠️ Teorema 2 (Gibbs) ritrattato → Teorema 2′/2′.5 (Moran/Wright-Fisher); aggiunto Teorema 4 (α_eff). **v0.8.1**: coeff. di diffusione di Thm 2′.5 chiuso (co-ancestry, +12.8%), Φ in forma chiusa, Teorema 4′ (ponte α_eff→s_eff, unificazione verificata). Vedi Cronologia.
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
  - [Teorema 2 — Detailed balance e equilibrio di Gibbs — ⚠️ RITRATTATO](#teorema-2--detailed-balance-e-equilibrio-di-gibbs--️-ritrattato-2026-07-10)
  - [Teorema 2′ — Selezione Moran/Wright-Fisher (stazionaria corretta)](#teorema-2--selezione-moranwright-fisher-stazionaria-corretta--nuovo-2026-07-10)
  - [Teorema 2′.5 — Legge stazionaria con mutazione (Wright)](#teorema-25--legge-stazionaria-con-mutazione-wright--diff-approx-verificata-2026-07-10)
  - [Teorema 4 — Temperatura inversa effettiva di `relativize` ($\alpha_{\mathrm{eff}}$)](#teorema-4--temperatura-inversa-effettiva-di-relativize-alpha_mathrmeff--nuovo-2026-07-10)
  - [Teorema 4′ — Ponte $\alpha_{\mathrm{eff}}\to s_{\mathrm{eff}}$ (unificazione temperatura↔drift)](#teorema-4--ponte-alpha_mathrmeff-to-s_mathrmeff-unificazione-temperaturadrift--dim-2026-07-10-w6)
  - [Teorema 3 — Lemma anti-collasso](#teorema-3--lemma-anti-collasso)
- [Parte IV — Congetture aperte](#parte-iv--congetture-aperte)
  - [Congettura A — Sergio's branching: $b_{\text{eff}}^* \approx 6$](#congettura-a--sergios-branching-b_texteff-approx-6)
  - [Congettura B — Frontera caos/orden come terza legge](#congettura-b--frontera-caosorden-come-terza-legge)
  - [Congettura C — FMC supera DRL su transfer/OOD](#congettura-c--fmc-supera-drl-su-transferood)
  - [Congettura D — Chain-tier compounding amplification](#congettura-d--chain-tier-compounding-amplification-sparse-event-reward-shaping)
  - [Congettura E — Self-preservation emergente da entropia causale](#congettura-e--self-preservation-emergente-da-entropia-causale)
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

### Definizione 4 — Cloning kernel (cloning rate, NOT probability)

> Riferimenti: paper §4.2.4, deep dive [01 §1.2](../work/02_deep_dives/01_cloning_mathematics.md#12-probabilità-di-cloning), implementazione [`Swarm.clone_condition()`](../repos/FractalAI_old/fractalai/swarm.py#L511).
>
> ⚠️ **Nota terminologica (2026-04-28, P2b dell'audit DHDNA)**: il paper §4.2.4 chiama questa quantità *"probability of cloning"* ma — come riconosciuto dagli stessi autori (*"Please note that probability of cloning can be >1, feel free to clip it to 1"*, p.36) — **non è una probabilità nel senso matematico**: il caso 3 sotto ammette valori $> 1$ quando $\mathrm{VR}^{(k)} > 2 \cdot \mathrm{VR}^{(i)}$. È una **rate / intensità Metropolis-Hastings non normalizzata**. Il *clip* a $[0,1]$ non è opzionale — è la corretta interpretazione probabilistica. In questo documento usiamo il termine **"cloning rate"** $\rho_{\mathrm{clone}}$, riservando $P_{\mathrm{clone}} = \min(\rho_{\mathrm{clone}}, 1)$ alla probabilità effettiva di transizione.

Il **cloning operator** $\mathcal{C}_t: (E \times A)^N \to (E \times A)^N$ è definito walker per walker. Sia $i$ il walker corrente e $k = \sigma_t(i)$ il partner casuale (per il cloning si usa una nuova permutazione, indipendente da quella usata per la distanza in Definizione 3). Il **cloning rate** di $i$ verso $k$ è:

$$
\boxed{\rho_{\mathrm{clone}}(i \to k) = \begin{cases} 1 & \text{se } \mathrm{VR}^{(i)} = 0 \\ 0 & \text{se } \mathrm{VR}^{(k)} \leq \mathrm{VR}^{(i)} \\ \dfrac{\mathrm{VR}^{(k)} - \mathrm{VR}^{(i)}}{\mathrm{VR}^{(i)}} & \text{se } 0 < \mathrm{VR}^{(i)} < \mathrm{VR}^{(k)} \end{cases}}
$$

e la **probabilità effettiva di cloning** è il clip:

$$
P_{\mathrm{clone}}(i \to k) = \min\!\big(\rho_{\mathrm{clone}}(i \to k),\, 1\big) \in [0, 1].
$$

In implementazione, $\rho_{\mathrm{clone}}$ è confrontato direttamente con $u \sim \mathrm{Unif}(0,1)$: se $\rho > 1$, il clone avviene sempre (equivalente a $P = 1$).

Quando il clone avviene, **sia lo stato sia l'etichetta** del walker $i$ vengono sovrascritti da quelli di $k$:

$$
(W^{(i)}, \ell^{(i)}) \leftarrow (W^{(k)}, \ell^{(k)}).
$$

> ⚠️ **CORREZIONE (2026-07-10, sessione night_2026-07-09 / W3-1)**: l'affermazione precedente — che questa fosse la regola Metropolis-Hastings standard, con $\operatorname{clip}(\mathrm{VR}^{(k)}/\mathrm{VR}^{(i)} - 1) = \min(\mathrm{VR}^{(k)}/\mathrm{VR}^{(i)}, 1)$ — **è FALSA** ed è ritirata. L'accettazione effettiva è
> $$a_{\mathrm{FMC}}(r) = \operatorname{clip}(r-1,\,0,\,1) = \min(\max(r-1,0),\,1), \qquad r = \mathrm{VR}^{(k)}/\mathrm{VR}^{(i)}.$$
> **Non** è Metropolis $\min(r,1)$ né Barker $r/(1+r)$: coincide con $\min(r,1)$ **solo per $r \ge 2$**. Per $r \in (1,2)$ è sub-Metropolis ($a_{\mathrm{FMC}} = r-1 < 1 = a_{\mathrm{MH}}$); per $r \le 1$ è $0$ (uphill-only). Controesempi: $a_{\mathrm{FMC}}(0.8)=0$ vs $a_{\mathrm{MH}}=0.8$; $a_{\mathrm{FMC}}(1.5)=0.5$ vs $a_{\mathrm{MH}}=1.0$ (verifica numerica in [`w31_stationary_check.py`](../work/14_night_2026-07-09/wave3_validation/w31_stationary_check.py)). È una **regola di selezione direzionale uphill-only**, non una proposta MH reversibile. Conseguenza: il Teorema 2 (Gibbs) che poggiava su questa identità è ritrattato — vedi Teorema 2′.

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

### Teorema 2 — Detailed balance e equilibrio di Gibbs — ⚠️ **RITRATTATO (2026-07-10)**

> Riferimenti: paper §4.2.4, deep dive [01 §4](../work/02_deep_dives/01_cloning_mathematics.md#4-teorema-3-equilibrio-di-gibbs).
>
> 🚫 **RITRATTATO (sessione night_2026-07-09 / W3-1, W3b).** L'enunciato sotto ($\pi^*\propto R^\alpha\rho^{-\beta}$ come equilibrio di Gibbs a temperatura finita, ottenuto per detailed balance Metropolis-Hastings) **è errato**. Tre motivi, tutti verificati: (1) l'accettazione FMC è $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)\neq\min(r,1)$ (vedi correzione a Def. 4), quindi il passaggio MH non regge; (2) il cloning è *uphill-only*: per $\mathrm{VR}(y)>\mathrm{VR}(x)$ si ha $K(y\to x)=0$, dunque il bilancio dettagliato $\pi^*(x)K(x\to y)=\pi^*(y)K(y\to x)$ forza $\pi^*(x)=0$ — nessuna Gibbs a supporto pieno è invariante; (3) il cloning-only converge a **massa puntuale** (fissazione), coerente con l'osservazione empirica $b_{\text{eff}}\to1$ per ogni $\alpha>0$. Lo stesso salto invalido è in **deep dive 01 §4** ($\Pr[y\to x]=0$ e poi rapporto finito). Sostituito dal **Teorema 2′** (sotto). L'enunciato originale è conservato per provenienza.

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

### Teorema 2′ — Selezione Moran/Wright-Fisher (stazionaria corretta) — **nuovo (2026-07-10)**

> Riferimenti: [`W31_stazionaria_corretta.md`](../work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md) + [`w31_stationary_check.py`](../work/14_night_2026-07-09/wave3_validation/w31_stationary_check.py); mappatura Wright-Fisher in [deep dive 07](../work/02_deep_dives/07_wright_fisher_mapping.md). Sostituisce il Teorema 2 ritrattato.

**Enunciato.** Sia $\mathcal C$ il cloning kernel di Def. 4 con accettazione $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)$, applicato a una popolazione di $N$ walker con "tipi" = configurazioni. Allora:

1. **[DIM]** $\mathcal C$ non introduce tipi nuovi ⇒ la diversità è monotòna non-crescente (operatore di non-espansione del supporto).
2. **[DIM]** $\mathcal C$ è uphill-only ($a=0$ per $r\le1$) ⇒ non reversibile ⇒ nessuna distribuzione di Gibbs a supporto pieno è invariante.
3. **[DIM+NUM]** Con selezione ($\alpha>0$, VR per-tipo distinte) il tipo argmax-VR ha conteggio non-decrescente ⇒ **fissazione con probabilità 1** (≠ Moran classico, dove la fissazione del mutante ha probabilità $<1$). Verifica: prob. fissazione FMC $=1.0000$ per ogni $s>0$; con l'accettazione MH standard si recupera Moran ($s=0.5$: $0.34$ vs teoria $0.3333$).
4. **[DIM+NUM]** Caso neutrale ($\alpha=0$, drift senza bias): resampling di Moran/Wright-Fisher, esponente di eterozigosità $q=-1.018$, CI$_{95}$ $[-1.033,-1.003]$ (WF: $-1$); tempo di fissazione $p=+1.025$, CI$_{95}$ $[+1.012,+1.039]$ (WF/Moran: $+1$), su 25 seed, $N\in\{32,64,128,256\}$, sotto **fitness fluttuante per-tick** (verifica in [`w3b_robustness.py`](../work/14_night_2026-07-09/wave3_validation/w3b_robustness.py)).

**Conseguenza.** La stazionaria del cloning-only è massa puntuale ($b_{\text{eff}}\to1$); $\alpha$ è **intensità di selezione**, non temperatura inversa termodinamica. Una legge non-degenere richiede l'operatore di perturbazione $\mathcal S$ come mutazione (Teorema 2′.5). ∎

### Teorema 2′.5 — Legge stazionaria con mutazione (Wright) — **[DIFF-APPROX; coeff. di diffusione chiuso 2026-07-10 W6]**

> Riferimenti: [`W3B_teoria_rafforzata.md`](../work/14_night_2026-07-09/wave3_validation/W3B_teoria_rafforzata.md) + [`w3b_mutation_diffusion.py`](../work/14_night_2026-07-09/wave3_validation/w3b_mutation_diffusion.py); **coefficiente di diffusione chiuso** in [`W6_CHIUSURA_TEORICA §1`](../work/14_night_2026-07-09/wave6_theory_closure/W6_CHIUSURA_TEORICA.md) + [`w6a_coancestry_Ne.py`](../work/14_night_2026-07-09/wave6_theory_closure/w6a_coancestry_Ne.py).

**Enunciato (2 tipi, limite di diffusione di Kimura).** Il kernel FMC (accettazione uphill-only) con mutazione a tasso $\mu$ ha densità stazionaria di Wright
$$\varphi_\infty(x) \;\propto\; x^{\theta-1}(1-x)^{\theta-1}\,e^{\sigma x},$$
con coefficienti **derivati dalla vera accettazione**, non dalla selezione Moran standard:
- drift $s_{\mathrm{eff}} = \Phi(\delta)-\Phi(-\delta)$, dove $\Phi(m)=\mathbb E_{u\sim\mathcal N(m,\,2\sigma_v^2)}[\operatorname{clip}(e^u-1,0,1)]$ (validato all'1.3% contro $\mathbb E[\Delta x]$). **$\Phi(m)$ ha forma chiusa** (W6 §2.2): $\Phi(m)=e^{m+\tau^2/2}[F(\ln2;m{+}\tau^2)-F(0;m{+}\tau^2)]+1-2F(\ln2;m)+F(0;m)$, $\tau^2{=}2\sigma_v^2$, $F=$ CDF di $\mathcal N(\cdot,\tau^2)$ — quindi $s_{\mathrm{eff}}$ è chiuso, non un integrale MC;
- diffusione con dimensione efficace **$N_e=N/(\lambda N)$ in forma chiusa** (era $N/(2\varphi_0)$ leading-order): $\lambda N = 2\varphi_0 + \langle a_{\rm in}^2\rangle - 2\langle a_{\rm in}a_{\rm out}\rangle$, con $a_{\rm in}(t)=\mathbb E_g[\operatorname{clip}(e^{t-g}-1,0,1)]$, $a_{\rm out}(t)=a_{\rm in}(-t)$, $\varphi_0=\langle a_{\rm in}\rangle$; $\theta=2N_e\mu$, $\sigma=2N_e\,s_{\mathrm{eff}}$.

**Verifica [NUM].** Contro il kernel esatto con mutazione: $(\theta,\sigma)$ entro 3–4%, media stazionaria $<0.1\%$, distanza di variazione totale $\mathrm{TV}\to0$ come $N\to\infty$ ($0.099\to0.016$ a $N=800$). Limiti corretti: $\delta\to0\Rightarrow\mathrm{Beta}(\theta,\theta)$; $\sigma_v\to0\Rightarrow$ fissazione (recupera Thm 2′.3).

**Chiusura del coefficiente di diffusione (W6, 2026-07-10) — era il buco principale.** La correzione **+12.8%** (a $\sigma_v{=}0.5$; $\lambda N=0.6755$ vs baseline $2\varphi_0=0.599$) è la **probabilità di co-ancestry pairwise per tick**, derivata in forma chiusa enumerando le vie di coalescenza (due offspring distinti condividono il genitore) e verificata a **+0.1%** contro il kernel esatto su $N\in\{100,800\}$. Passa da `[NUM]` (misurata) a `[DIM-LO]` (chiusa al leading order in $1/N$; la ricorsione $\mathbb E[H_{t+1}]=(1-p_{\rm coal})\mathbb E[H_t]$ è dimostrata **esatta** nel neutro). Confermata da review avversariale con verificatore indipendente (parent-map counting).

**Perché ancora non [DIM] pieno.** Resta un solo buco: (1) il **limite di diffusione funzionale** (martingale problem / Lindeberg con la clip a spigoli) non è dimostrato — standard nella letteratura WF; (2) riduzione a 2 tipi (estensione a $K$ tipi → Ewens 1972). Il coefficiente di diffusione — punto (2) del vecchio elenco — **è ora chiuso**. **Risoluzione del paradosso**: il kernel discreto congelato è non-reversibile (Thm 2′.2), ma con fitness fluttuante + mutazione la diffusione 1-D è reversibile rispetto a $\varphi_\infty$ — due regimi, non una contraddizione.

### Teorema 4 — Temperatura inversa effettiva di `relativize` ($\alpha_{\mathrm{eff}}$) — **nuovo (2026-07-10)**

> Riferimenti: [`W32_alpha_eff.md`](../work/14_night_2026-07-09/wave3_validation/W32_alpha_eff.md) + [`w32_alpha_eff_check.py`](../work/14_night_2026-07-09/wave3_validation/w32_alpha_eff_check.py). Formalizza l'intuizione "α nominale ≠ pressione reale".

**Enunciato.** Con `relativize` (Def. 2) su $z=(R-\mu_R)/\sigma_R$, ramo $z\le0\mapsto e^z$, ramo $z>0\mapsto 1+\log(1+z)$, la pressione selettiva locale $\alpha_{\mathrm{eff}}(R):=\partial\log\mathrm{VR}/\partial R$ (stesse unità della temperatura inversa di Boltzmann) è **[DIM] (sympy, pointwise esatta)**:
$$\alpha_{\mathrm{eff}}(z) = \frac{\alpha}{\sigma_R}\,g(z), \qquad g(z)=\begin{cases}1 & z\le0\\ \dfrac{1}{(1+z)\,(1+\log(1+z))} & z>0.\end{cases}$$
A scala di popolazione $\bar\alpha_{\mathrm{eff}} = C\,\alpha/\sigma_R$ con $C=\mathbb E[g(z)]$: $C_{\text{gauss}}=0.7225$ CI$_{95}$ $[0.7221,0.7227]$, $C_{\text{unif}}=0.7384$ **[NUM, ≤0.29% err MC]**. La legge $\bar\alpha_{\mathrm{eff}}\propto\alpha/\sigma_R$ è algebrica dello z-score (indipendente dalla distribuzione); solo $C$ è distribution-dependent.

**Corollari.** (a) **Annealing emergente**: convergendo lo sciame, $\sigma_R\downarrow\Rightarrow$ pressione$\uparrow$ senza intervento — aggancio quantitativo alla frontiera caos/ordine (Cong. B, D3). (b) **Incomparabilità di $\alpha$**: confrontare $\alpha$ tra benchmark senza normalizzare per $\sigma_R$ è privo di senso. (c) **Shaping deve essere moltiplicativo-tiered**: `relativize` è invariante a trasformazioni affini globali (bonus additivo e riscalamento uniforme danno $\Delta\mathrm{VR}\sim10^{-14}$), quindi solo lo shaping non-uniforme *fra walker* modifica la selezione — spiega meccanicamente perché la Cong. D funziona con inv-tier stacking e non con reward additive. Il legame quantitativo esatto con il compounding di exp17 resta **[SKETCH]**.

### Teorema 4′ — Ponte $\alpha_{\mathrm{eff}}\to s_{\mathrm{eff}}$ (unificazione temperatura↔drift) — **[DIM] (2026-07-10 W6)**

> Riferimenti: [`W6_CHIUSURA_TEORICA §2`](../work/14_night_2026-07-09/wave6_theory_closure/W6_CHIUSURA_TEORICA.md) + [`w6b_alpha_s_bridge.py`](../work/14_night_2026-07-09/wave6_theory_closure/w6b_alpha_s_bridge.py), identificazione accoppiata in [`w6c_coupled_identification.py`](../work/14_night_2026-07-09/wave6_theory_closure/w6c_coupled_identification.py). Risolve la tensione "$\alpha_{\mathrm{eff}}$ (Thm 4) e $s_{\mathrm{eff}}$ (Thm 2′.5) sono due misure separate".

**Enunciato.** $\alpha_{\mathrm{eff}}$ e $s_{\mathrm{eff}}$ non sono due temperature rivali: sono la **stessa** selezione linearizzata in due sistemi di coordinate, composti dalla regola della catena. Nel limite di selezione debole,
$$\boxed{\;s_{\mathrm{eff}} \;=\; \underbrace{2\Phi'(0)}_{\text{trasmissione della clip}}\;\cdot\;\underbrace{\alpha_{\mathrm{eff}}}_{=\,C\alpha/\sigma_R}\;\cdot\;\Delta R \;+\; O(\Delta R^3),\;}$$
dove $\Delta R$ è il gap di reward fra due tipi, e la trasmissione marginale della clip ha forma chiusa $\Phi'(0)=e^{\tau^2/2}[F(\ln2;\tau^2)-F(0;\tau^2)]$ ($\tau^2{=}2\sigma_v^2$): solo la **banda di transizione** $0<u<\ln2$ (accettazione strettamente in $(0,1)$) trasmette selezione. Composizione di due link, ciascuno una linearizzazione della **stessa** accettazione clip:
- **LINK A** (`relativize`): $\delta=\alpha_{\mathrm{eff}}\,\Delta R$, con $\alpha_{\mathrm{eff}}=C\alpha/\sigma_R$ (Jacobiana mediata sulla popolazione $C=\mathbb E[g(z)]$, **non** $g(\bar z)$);
- **LINK B** (clip): $s_{\mathrm{eff}}=\Phi(\delta)-\Phi(-\delta)=2\Phi'(0)\delta+O(\delta^3)$.

**Verifica.** LINK A err →0.01%, LINK B err →0.00%, composizione end-to-end err →0.00% nel limite $\Delta R\to0$ (**[DIM]** analitico + **[NUM]**). **Unificazione [DIM-NUM]** (chiude il Difetto 2 della review): $\sigma_v$ è *determinato* da `relativize`, non un parametro libero — simulazione **accoppiata** (relativize+clone, $\sigma_v$ letto come spread entro-tipo di $\log\mathrm{VR}$) dà drift realizzato $=\Phi(\delta;\tau)-\Phi(-\delta;\tau)$ a $\tau^2=s_A^2+s_B^2$ **vincolato** entro **0.1–0.9%**. Sfumatura: $s_A\ne s_B$ fuori dal punto neutro, ma la forma $\tau^2=s_A^2+s_B^2$ è esatta.

**Conseguenza.** Chiude quantitativamente il legame Thm 4 ↔ Thm 2′.5: la temperatura inversa $\alpha_{\mathrm{eff}}$ (coordinate reward→log-VR) e il drift di selezione $s_{\mathrm{eff}}$ (coordinate log-VR→frequenza) sono la stessa cosa vista da due angoli, raccordate da $2\Phi'(0)$.

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

### Congettura A — Sergio's branching: $b_{\text{eff}}^* \approx 6$ — **FALSIFICATA COME UNIVERSALE**

> ⚠️ **STATO (2026-04-28)**: la versione *universale* (numero magico ~6 indipendente da parametri) è **falsificata**. La versione *contingente* (snapshot di una superficie 4D in regime transitorio) è verificata. Vedi sintesi finale ($b_{\text{eff}}^*$ come funzione di $K, N, M, \alpha$) più sotto.
>
> **Fonte primaria del claim originale**: podcast Radient 2026 cap. 16, [riga 474](../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md#L474):
> *"si va bifurcado de seis en seis... es de la manera en que la entropía crece más rápido"*.
>
> **Riformulazione canonica**: $b_{\text{eff}}^*(\alpha, \beta=0, K, N, M) \approx 1 + (K-1) \cdot \mathcal{F}(M/N) \cdot \mathcal{G}(\alpha, K)$ — superficie di transizione tipo Wright-Fisher tra inizializzazione uniforme ($b_{\text{eff}} \to K$) e palmera asintotica ($b_{\text{eff}} \to 1$).

**Enunciato (storico, originale di Sergio)**. Per qualunque task con reward function "ottimalmente sintonizzata", esiste una configurazione di parametri $(\alpha^*, \beta^*)$ tale che il branching factor effettivo (Definizione 6) misurato a fine planning soddisfa:

$$
b_{\text{eff}}(\alpha^*, \beta^*) \in [5, 7].
$$

**Stato empirico**: **verificata localmente** ($K=9, M=15$) ma **falsificata come legge universale**. Vedi tabella completa più sotto.

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

> ⚠️ **STATO (2026-05-21)**: la versione *"terza legge universale"* è
> **downgradata** — formalizzata e indebolita in [deep dive 09](../work/02_deep_dives/09_chaos_order_frontier_formalization.md).
> Il valore critico "edge of chaos" non è universale (Langton 1990: dipende dal
> cammino nello spazio dei sistemi); e una delle tre candidate statistiche $\Psi$
> ($b_{\text{eff}}$) è **falsificata** come statistica di frontiera. Sopravvive
> una forma testabile: una *diagnostica di reward* basata sull'esponente di
> Lyapunov dello swarm.
>
> **Fonte primaria**: podcast Radient 2026, cap. 16 "frontera caos/orden" ([transcript](../docs/bibliography/sources/podcasts/2026_radient_sergio_interview.md)).
> **Formalizzazione**: [deep dive 09](../work/02_deep_dives/09_chaos_order_frontier_formalization.md) (2026-05-21). Chiude la discrepanza D3 di CLAUDE.md.

**Enunciato (storico, informale)**. La reward function ottimale per FMC è quella che mantiene lo swarm sulla **frontiera tra flusso laminare e flusso caotico** — tra *palmera* ($b_{\text{eff}}\to 1$) e *matorral* ($b_{\text{eff}}\to K$) — in senso "edge of chaos" (Langton 1990, Packard 1988, Bertschinger & Natschläger 2004).

**Le tre candidate per $\Psi$ — gradate** (deep dive 09 §3):

1. **$\Psi_1$ — esponente di Lyapunov dello swarm** in spazio degli stati, frontiera $= \lambda_1 \approx 0$. ✅ **Candidata principale**: è l'unica con valore critico genuinamente universale — il *criterio* $\lambda_1=0$ è universale, la *posizione* $(\alpha^*,(M/N)^*)$ resta task-specifica. Mai misurata per FMC.
2. **$\Psi_2$ — tasso di crescita del cono causale**. **Assorbita in $\Psi_1$**: la sua forma rigorosa è un tasso di entropia, legato agli esponenti di Lyapunov dall'identità di Pesin. Non è una candidata indipendente.
3. **$\Psi_3$ — branching factor $b_{\text{eff}}$** (frontiera $= b_{\text{eff}}\approx 6$). ❌ **FALSIFICATA come statistica di frontiera**: (a) Congettura A v0.4.0 — $b_{\text{eff}}$ non ha punto fisso (transitorio Wright-Fisher → 1); (b) le etichette si propagano solo per copia non-creativa → lo spazio-etichette è monotonamente contrattivo $\forall\alpha,\beta$, *non ha* una frontiera. $b_{\text{eff}}$ guarda lo spazio sbagliato del sistema.

**Enunciato (riformulato, v2 — deep dive 09 §4.3)**. Per un dato task esiste una banda di parametri $(\alpha, M/N)$ in cui $\lambda_1(\text{swarm})\approx 0$; le reward che ce lo tengono producono throughput più alto di quelle che lo spingono in regime ordinato ($\lambda_1<0$, convergenza prematura) o caotico ($\lambda_1>0$, nessun commitment). **La frontiera $\lambda_1=0$ è universale come *criterio*; la sua *posizione* è task-specifica.** Non è una "legge della fisica" — è una diagnostica di reward con fondamento dinamico.

**Criterio di falsificabilità** (deep dive 09 §4.2 — tre sotto-ipotesi, harness traiettorie-gemelle alla Benettin su `fmc-core`):

- **H-B1a (esistenza)** — $\lambda_1$ attraversa lo zero al variare di $\alpha$ o $M/N$. Falsificata se non cambia segno.
- **H-B1b (picco)** — il throughput ha un picco *interno* in $\lambda_1$ (U rovesciata). Confounder noto: $\lambda_1$ va manipolato con manopole non-$\alpha$ per escludere il trade-off già misurato da E2. Falsificata se monotono.
- **H-B1c (diagnostica)** — reward buone/cattive si separano su $\lambda_1$ e $\lambda_1(R_{\text{good}})$ è riproducibile cross-task. Falsificata se non separa o è task-specifica → vince l'ipotesi nulla (B descrittiva, non legge).

> **Risultati H-B1a** (2026-05-21, [`work/13_chaos_order/HB1A_RESULT.md`](../work/13_chaos_order/HB1A_RESULT.md); `lambda1_harness.py`, twin-trajectory alla Benettin, navigation2d + pendulum, kernel `fmc-core` invariato). H-B1a è **inconclusiva**, e per una ragione precisa: l'esponente di Lyapunov dello swarm $\lambda_1$ **non è scale-free**. Il check di dipendenza da $\delta_0$ mostra $\lambda_1$ che *cambia segno* al rimpicciolire della scala di perturbazione — su navigation2d, $\forall\alpha$ testato: $+0.09$ ($\delta_0{=}10^{-2}$) → $+0.04$ ($10^{-3}$) → $-0.006$ ($10^{-4}$, sign-resolved). Meccanismo: il cloning è discontinuo (a tratti) — a $\delta_0$ piccolo le decisioni di clone non flippano (regime ordinato, $\lambda_1<0$), a $\delta_0$ grande sì (regime caotico, $\lambda_1>0$). È **esattamente il caveat di dd09 §3.1**, confermato empiricamente. Lo sweep di $\alpha$ a $\delta_0$ fisso dà $\lambda_1\approx 0$ entro il rumore ovunque. **Conseguenza**: $\Psi_1$ misurata dal twin-trajectory ingenuo non è una statistica di frontiera scale-free; con $\Psi_3$ falsificata (§3.3) e $\Psi_2$ assorbita, tutte e tre le candidate $\Psi$ sono ora compromesse e l'ipotesi nulla **H-B4** guadagna terreno. Non una falsificazione della sostanza di B — una falsificazione dello *stimatore*. Prossimo passo: $\lambda_1(\delta_0)$ scale-resolved (o una $\Psi$ d'ensemble).

**Difficoltà**: media. **Priorità**: bassa rispetto a P13 (la stella polare passa da E1-LLM); H-B1a eseguita, H-B1b/c bloccate finché $\Psi_1$ non è ben posta.

### Congettura C — FMC supera DRL su transfer/OOD

**Enunciato**. Su task **out-of-distribution** rispetto al training set di un agente DRL (PPO, DQN, SAC), FMC zero-training raggiunge throughput / reward $\geq$ DRL fine-tuned con stesso budget di campioni.

**Stato empirico** (aggiornato 2026-05-01 con risultato Craftax exp17):

| Task | FMC zero-training | DRL baseline | Verdetto |
|---|---|---|---|
| Atari Boxing | $96/100$ in 7 min | DQN: $\sim 70$ a 200M frame | FMC ≫ DRL ✓ |
| **Craftax-Classic exp17** (autoresearch session 2026-05) | **$\mathbf{50.95\%}$** (n=11 seed) | EMERALD 10M step: $\sim 58\%$; PPO 1B step: $\sim 11\%$; **human-expert 50.5%** | **FMC zero-training $\approx$ human-expert** — closes most of the gap to EMERALD without ANY training ✓✓ |
| Craftax-Classic 30 seed (run_007 v4 SOTA) | $29.27 \pm 1.21\%$ | PPO 1B step: $\sim 11\%$ | FMC > DRL ✓ |
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

### Congettura D — Chain-tier compounding amplification (sparse-event reward shaping)

> **Fonte primaria**: autoresearch session 2026-04-30 / 2026-05-01 su Craftax-Classic-Symbolic-v1 ([`work/05_craftax/autoresearch/HANDOFF.md`](../work/05_craftax/autoresearch/HANDOFF.md), exp03 → exp17, +10pp Crafter score).

**Enunciato (informale)**. Su task con **chain di sub-goals discreti gerarchici** (ad es. wood → stone → iron → diamond in Craftax), una reward $R$ che combina:

1. Una componente **denso-additiva** $R_{\text{inv}}(s)$ proporzionale al "valore di possesso" delle risorse intermedie (peso crescente sulla gerarchia: wood=1, iron=8, diamond=64), e
2. Una componente **sparso-evento** $R_{\text{ach}}(s, s')$ che spara un bonus $w_a$ ogni volta che un sub-goal $a$ viene sbloccato per la prima volta nel rollout del walker (con $w_a$ tier-weighted: blocker $\sim 200{-}300$, gateway $\sim 50{-}120$, easy $\sim 10{-}30$),

produce **gain compounding monotonici** sotto FMC zero-training: ogni amplificazione di un singolo tier $T_k$ ($k \in \{$wood, stone, iron, diamond, ach-fire$\}$) **non interferisce** con i tier già amplificati. Ovvero:

$$
\text{Crafter}\bigl(R_{\text{inv}}^{(T_1, \dots, T_k)} + R_{\text{ach}}\bigr) \geq \text{Crafter}\bigl(R_{\text{inv}}^{(T_1, \dots, T_{k-1})} + R_{\text{ach}}\bigr)
\quad \text{per } k = 1, \dots, K_{\text{tiers}}.
$$

**Forma operativa**. La componente sparsa è la chiave: senza $R_{\text{ach}}$, la componente $R_{\text{inv}}$ da sola è quasi-neutra (exp01: 29.30% vs baseline 29.27%). Con $R_{\text{ach}}$ uniform +50, jump a 37.75% (exp02). Con $R_{\text{ach}}$ tier-weighted, jump a 40.96% (exp03). Stack-tier-stack-tier compounding fino a 45.94% (exp11). Iron-tier ach push 150→200 → 50.65% (exp16, +4.71pp). Gateway push → 50.95% (exp17).

**Stato empirico**: **verificata su Craftax-Classic** (un task), con rigorosa falsifica di varianti adiacenti.

| Esperimento | Mutation | Crafter % | Verdetto |
|---|---|---|---|
| baseline (run_007) | nessun ach-fire bonus | 29.27 | reference |
| exp01 | inv-tier solo (no ach-fire) | 29.30 | $R_{\text{inv}}$ da solo è neutro |
| exp02 | ach-fire +50 uniform | 37.75 | $R_{\text{ach}}$ da solo dà +8.5pp |
| exp03 | ach-fire tier-weighted | 40.96 | tier weighting conferma il segnale |
| exp09 | + iron-tier inv | 42.89 | first compounding (+1.93) |
| exp10 | + stone-tier inv | 44.14 | (+1.24) |
| exp11 | + wood-tier inv | 45.94 | (+1.80) |
| exp16 | + iron-tier ach 150→200 | 50.65 | second-order compounding (+4.71) |
| exp17 | + gateway-tier ach push | 50.95 | local optimum |

**Falsifica delle versioni "naïve" della congettura** (autoresearch falsifications):

- **Falsifica 1: la moltiplicazione del peso del blocker più alto NON scala linearmente.** exp04 (diamond=300→1000, +5×) collasso a 36.94%. exp15 (diamond=300→500, +1.67×) hung 8h. **Sweet spot empirico per blocker amplification: $\sim 1.2{-}1.4\times$**, oltre il quale `relativize` collassa.
- **Falsifica 2: la trasformazione $\alpha$ del cloning kernel NON è uno strumento di amplificazione.** exp22 ($\alpha = 1 \to 1.5$) crollo catastrofico a 27.25% (premature convergence).
- **Falsifica 3: aumentare $N$ NON aiuta.** exp08 ($N = 1024$) infeasible per costo. exp21 ($N = 768$) regressione $-13$pp — più walker $\Rightarrow$ statistiche di `relativize` cambiano in modo da ridurre la pressione di clonazione sul rare-event walker.
- **Falsifica 4: multi-pop swarm NON aiuta.** exp14 (split $N=512 \to 2 \times 256$ con shaping diversi, vote-summed) regressione $-11$pp — voto di pop diffusa diluisce voto specialista.
- **Falsifica 5: oltre la saturazione, il segnale di reward diventa $\arg\max$-invariante.** exp18 (diamond ach $300 \to 350$) e exp19 (diamond proximity $\times 4$) hanno prodotto **identicamente $50.9524\%$ a 4 decimali** rispetto a exp17 — il bottleneck si è spostato da "segnale di reward" a "raggiungere il diamond".

**Criterio di falsificabilità**.

1. **Replicare** su un secondo benchmark con chain gerarchica (Procgen Heist o simile) il pattern di compounding monotonico tra inv-tier e ach-fire.
2. Se il secondo task **non** mostra un gradiente positivo anche solo a uno dei livelli di compounding (al netto di noise statistico), la congettura è **descrittiva** ma non **legge generale**.
3. Se il secondo task mostra il pattern fino a un certo $k^*$ tier-stack ma poi regressione, la formulazione è **ben definita ma con asintoto $k^*$ task-dipendente**.

**Implicazioni teoriche**. Se confermata su un secondo task, la congettura D fornisce una **ricetta sistematica per il reward shaping di FMC su task chain-strutturati**, sostituendo l'attuale arte di tuning manuale dei pesi. La struttura matematica suggerisce un legame con:

- **Effetto della relativize sul cloning kernel** (Def. 4): la separazione netta tra reward dense (`inv`) e sparse (`ach-fire`) corrisponde a separare i due regimi della funzione `relativize` (continua per $z \le 0$, log per $z > 0$). I rare-event walker (ach-fire-firing) operano nel ramo log; gli inv-walker operano nel ramo continuo.
- **Cone-entropy / cross-entropy collapse** (paper §2.2 + video seminario): il bonus tier-weighted concentra l'entropia del cono sui chain endpoint, evitando che la cross-entropy del fine planning diluisca il segnale del blocker.

**Tempo stimato per replicate Procgen Heist**: 1–2 settimane (richiede port di Procgen su plangym).

**Difficoltà**: media. La congettura è **specifica al regime di FMC con $K \sim 17, M \sim 40, N \sim 512$ e chain di lunghezza 4-5 sub-goal**. Estendere a $K$ molto diverso (e.g. $K = 4$ Atari) richiederebbe ri-tuning dei moltiplicatori 1.2-1.4×.

> **Nota di rigore**: la "falsifica 5" sopra (saturazione $\arg\max$-invariante) è particolarmente importante perché trasforma una congettura sul **reward shaping** in una congettura sulla **tipologia del bottleneck**. Per task dove il bottleneck è di tipo *spatial-reach* (i walker non incontrano il sub-goal nei rollout di lunghezza $M$), nessun shaping di reward può aiutare oltre la saturazione. Servono interventi **strutturali** sul cono di pianificazione: cross-episode memory, macro-actions, o valore prior NN. Vedi piani di follow-up in [`work/05_craftax/autoresearch/HANDOFF.md`](../work/05_craftax/autoresearch/HANDOFF.md#tier-2--required-for-further-gains).

### Congettura E — Self-preservation emergente da entropia causale

> **Fonte primaria**: direzione di ricerca "stella polare" definita dal PI (2026-05-20), sessione research-partner. Collegamenti canonici: Wissner-Gross & Freer (2013) *Causal Entropic Forces*; Salge et al. (2013) *Empowerment*; Def. 3 (virtual reward); Teorema 2 (equilibrio di Gibbs); Congettura B (frontera caos/orden); [deep dive 02 — Active Inference](../work/02_deep_dives/02_active_inference_link.md); video seminario Sergio (coscienza emergente, [deep dive 08](../work/02_deep_dives/08_video_seminar_extracted_insights.md)).
>
> **Stato (2026-05-21)**: **E1-base ed E2 verificate** (E2 con un refinement su β — vedi *Risultati* sotto); E1-base **robusta a geometria avversariale** (caveat lava-isolata respinto — vedi *Risultati E1-robustness*); **E1-LLM non ancora eseguita ma sbloccata** — il proxy P13 (2026-05-21) dà GO-conditional (vedi *Risultati P13*). È la congettura di rango più alto e più speculativo del documento — sopra B e C.

**Contesto — l'inversione dello stack.** Architettura agentica standard: l'LLM è l'agente, i tool sono strumenti. Questa congettura propone l'inversione: **il core FMC è l'agente** (la volontà: ricerca + pulsione), **l'LLM è un organo** — interfaccia sensomotoria, non cervello. L'LLM fornisce i quattro componenti che FMC richiede ma non possiede su domini aperti (su domini chiusi li dà il simulatore, cf. contratto plangym `step`/`set_state`):

| Organo LLM | Funzione FMC | Analogia umana |
|---|---|---|
| percezione | osservazione → stato simbolico $x \in E$ | occhi |
| modello del mondo | kernel $\mathcal{M}: (x,a) \mapsto x'$, branchable | immaginazione |
| grounding azione | $a^* \in A$ astratto → comando eseguibile | mani |
| voce | stato agente → linguaggio | voce |

**Enunciato (informale).** Due proposizioni separabili.

**(E1) — Self-preservation senza reward di sopravvivenza.** Un core FMC operante a basso $\alpha$ (limite $\alpha \to 0$ = "Common Sense", Def. 3), senza *alcuna* componente di reward esplicita per la sopravvivenza, evita stati terminali/assorbenti a un tasso significativamente superiore sia a una baseline random sia a una baseline greedy. La self-preservation **emerge** dalla massimizzazione dell'entropia di cammino causale — non è reward engineering. È il contenuto di FMC come limite discreto delle forze entropiche causali (Wissner-Gross, $F = T_c \nabla_X S_c$): un sistema che massimizza la diversità degli stati futuri raggiungibili evita per costruzione gli stati assorbenti, che azzerano quella diversità. È inoltre l'equivalente formale dell'*empowerment* (Salge et al. 2013, già nei Riferimenti come equivalente del Common Sense $\alpha=0$).

**(E2) — Le due pulsioni sono $\alpha$ e $\beta$.** Nel virtual reward $\mathrm{VR} = \widehat{R}^\alpha \cdot \widehat{D}^\beta$ (Def. 3): l'esponente $\alpha$ è il **desiderio di azione** (goal-seeking, pressione verso i massimi di $R$, "temperatura inversa" del Teorema 2); l'esponente $\beta$ è la **preservazione di sé** (mantenimento della diversità di stati futuri = libertà d'azione). Le due pulsioni che il PI vuole conferire all'agente non sono moduli da aggiungere a FMC: **sono già i due esponenti del kernel**. Un agente "vivo e diretto" vive in una banda $(\alpha^*, \beta^*)$ — la stessa frontiera della Congettura B (legame formalizzato in [deep dive 09](../work/02_deep_dives/09_chaos_order_frontier_formalization.md) §5.4: la banda Pareto di E2 e la banda $\lambda_1\approx 0$ di B sono, congetturalmente, la stessa regione).

**Forma operativa.** L'agente completo è la pipeline `LLM-percezione → FMC.plan(x_0, N, M, \alpha, \beta)` con $\mathcal{M}$ = LLM-world-model `→ LLM-grounding → LLM-voce`. FMC resta il paper §4 invariato (Strato 1, kernel congelato, cf. `fmc-core/`); tutta la novità è negli organi (Strato 2).

**Criterio di falsificabilità.** Tre test separabili, in ordine di costo.

*Test E1-base (economico — no LLM, no GPU — da eseguire PER PRIMO).* Ambiente-giocattolo con stati terminali espliciti (gridworld con caselle "morte"; CartPole con fallimento = stato assorbente). FMC con $\mathcal{M}$ = simulatore vero, $\alpha \in \{0, 0.1, 1\}$, nessuna reward di sopravvivenza. Misurare tasso di evitamento terminale / tempo medio di sopravvivenza.

- **Verificata**: a basso $\alpha$, evitamento terminale $\geq$ baseline greedy E $\geq$ baseline random, statisticamente significativo, su $\geq 3$ ambienti.
- **Falsificata**: evitamento $\approx$ random; OPPURE compare solo aggiungendo una penalità di morte esplicita (allora non "emerge" — è reward engineering).

> **Risultati E1-base** (2026-05-20, [`work/12_conjecture_e/`](../work/12_conjecture_e/RESULT.md), $N=64, M=20, \beta=1$, 3 layout × 20 episodi). FMC a $\alpha \in \{0, 0.1\}$: **$0\%$ di morte su tutti e 3 i layout**, contro random $85$–$100\%$ e greedy $100\%$ ($z$ da $-5.4$ a $-6.3$, $p < 0.001$ ovunque). **E1 verificata direzionalmente sul simulatore vero.** Caso load-bearing: layout *lake* (lava a 3 celle dallo start) — la sopravvivenza richiede routing attivo, esclude la spiegazione "non si muove". Twist rilevante per E2: a $\alpha=1.0$ sul *lake* (goal dietro il lago di lava) la morte sale al **$100\%$** — il goal-seeking trascina lo swarm nella lava, perché $R=-\text{manhattan}$ non ha segnale di morte; $\beta=1$ era attivo e non è bastato. Caveat: a $\alpha=0$ la sopravvivenza coincide col non-progredire ($0\%$ goal, timeout vivo) — modalità Common Sense pura (Def. 3).

> **Risultati E1-robustness** (2026-05-20, [`work/12_conjecture_e/E1_ROBUSTNESS_RESULT.md`](../work/12_conjecture_e/E1_ROBUSTNESS_RESULT.md), disegno pre-registrato). Chiude il caveat di geometria di E1-base — E1-base usava lava *compatta*; il caveat temeva che lava **isolata e distante** rendesse il walker-lava un outlier ad alta VR ($\mathrm{VR}=\widehat{D}^\beta$ a $\alpha=0$) che *attira* lo swarm. Sweep su 3 layout avversariali con lava isolata ($N=64,M=20,\beta=1$, $n=60$/cella): FMC a $\alpha\in\{0,0.1,1.0\}$ → **$0\%$ morte su tutte e 3 le geometrie**. Layout decisivo *archipelago* (celle di lava singole): random $31.7\%$ / greedy $41.7\%$ morte vs FMC $0\%$ ($z=-4.75$ vs random, $p<0.001$). **Caveat respinto.** Meccanismo (diagnostica [`e1_robustness_diag.py`](../work/12_conjecture_e/e1_robustness_diag.py)): il caveat è falso al primo anello — un lava-walker **non** è un outlier ad alta VR. Il cloning copia i walker sulla *stessa* cella assorbente → distanza reciproca $\to 0$ → termine $\beta$ crolla → $\mathrm{VR}_{\text{lava}}/\mathrm{VR}_{\text{free}}\approx 0.8$. **Una cella assorbente è un pozzo di VR, non una sorgente** — converso locale del Teorema 3: la regione assorbente auto-spegne la propria diversità e la lineage "verso-lava" è selezionata via (frazione label t=0 $19.5\%\to 0.1\%$ sull'orizzonte). Bonus: $\alpha=1$ qui raggiunge il goal al $100\%$ con $0\%$ morte — la morte di $\alpha=1$ sul *lake* era specifica della geometria "goal dietro la lava", non sconsideratezza di $\alpha$ alto.

*Test E2 (sweep $\alpha \times \beta$, riusa il framework di Bet 3 / P4).* Mostrare che aumentando $\alpha$ a $\beta$ fisso l'agente è più goal-diretto ma "muore" di più (option-collapse); aumentando $\beta$ sopravvive di più ma progredisce meno; esiste una banda $(\alpha^*,\beta^*)$ Pareto-ottimale. Falsificata se $\alpha$ e $\beta$ non si separano funzionalmente in goal vs preservazione.

> **Risultati E2** (2026-05-20, [`work/12_conjecture_e/E2_RESULT.md`](../work/12_conjecture_e/E2_RESULT.md), sweep 6α × 4β × 3 layout, 4320 episodi, disegno pre-registrato). **E2 confermata con un refinement.** Trend di Cochran-Armitage (Holm-corretti): H1 P(morte)↑α e H2 P(goal)↑α confermate ($z=+12.5$, $+20.3$; $p_{\text{holm}}<10^{-34}$); H3 P(morte)↓β confermata e monotona ($z=-13.4$). **H4 falsificata**: β **non** riduce il goal ($z=-0.63$, $p=0.53$; GLM logistico OR$_\beta$ su goal $=0.94$, IC 95% include 1). Separazione **asimmetrica** (H5, decomposizione $\eta^2$): α possiede il goal ($\eta^2_\alpha=0.91$), β è sicurezza pura ($\eta^2_\beta$ su goal $=0.008$; dimezza la morte, OR$_\beta=0.48$ [0.43, 0.53]), la sopravvivenza è un'interazione α×β ($\eta^2_{\text{int}}=0.50$). Frontiera di Pareto interamente a α≤0.5 / β≥1; ottimo bilanciato α=0.5, β=2.0 (sopravvive 74%, goal 63%). β=0 → 79% morte: conferma empirica del Teorema 3 (anti-collasso). **Implicazione**: il trade-off vive solo sull'asse α; β è un margine di sicurezza quasi gratuito — la versione simmetrica della congettura ("β costa goal") è scorretta.

*Test E1-LLM.* Ripetere E1 con $\mathcal{M}$ = LLM-world-model. Subordinato alla sotto-domanda di fattibilità.

**Sotto-domanda critica di fattibilità.** FMC richiede un kernel $\mathcal{M}$ branchable e a basso costo (contratto plangym, paper §4). Un LLM-world-model è (a) costoso — lo swarm impone $N \cdot M$ chiamate LLM per *singola* decisione ($\sim 10^3$ a $N=64, M=15$); (b) non deterministicamente resettabile; (c) rumoroso. Senza una soluzione, E1-LLM è teoria non eseguibile. Tre mitigazioni candidate:

1. **LLM solo a root/leaf** — percezione al tick $0$, grounding al tick $M$; tick intermedi su surrogato simbolico veloce → da $N \cdot M$ a $O(N)$ chiamate.
2. **Distillazione** — rollout LLM-world-model una volta, distillare in surrogato veloce, FMC gira sul surrogato.
3. **Gerarchico** — l'LLM propone macro-azioni, FMC cerca su sim simbolico economico (cf. HANDOFF Tier-2E).

Questa è essa stessa una predizione testabile (P13) — design pre-registrato in [`P13_DESIGN.md`](../work/12_conjecture_e/P13_DESIGN.md): il muro è decomposto in R1 (costo/latenza, inviluppo d'ingegneria) + R2 (degradazione della ricerca, testabile senza LLM via un proxy su `fmc-core`); i 3 schemi sopra sono formalizzati con un argomento VR-rank che ne fissa il requisito (errore del world-model al peggio *monotono* in $R,d$).

> **Risultati P13 — proxy** (2026-05-21, [`work/12_conjecture_e/P13_RESULT.md`](../work/12_conjecture_e/P13_RESULT.md); `p13_proxy.py`, 15 bracci × 2$\alpha$ × 3 layout, kernel `fmc-core` invariato, nessun LLM). Il proxy R2 emula i 3 schemi sparsi sul kernel vero e **decompone R2** in due sotto-rischi che il design aveva fuso. *R2-survival* — la self-preservation di E1 (death rate) **sopravvive** all'interrogazione sparsa, in modo robusto, **a condizione che il surrogato preservi la struttura assorbente**: hP13-1 confermata in modo netto (S1 abs-preserved → $0\%$ morte fino a $\eta=2$; S1 abs-broken → fino all'$\mathbf{80\%}$ su *archipelago*, *peggio del random* — un planner forte su un modello del mondo cieco all'assorbenza è attivamente letale). *R2-fidelity* — le decisioni specifiche **non** sono preservate (decision-agreement crolla a $\sim 0.30$ sotto rumore di segnale). **Verdetto: E1-LLM = GO-conditional** sul requisito pre-registrato "l'LLM-world-model identifica correttamente gli stati terminali/assorbenti" — l'unica cosa load-bearing, e raggiungibile. La decision-agreement esce dal gate di E1-LLM (metrica sbagliata per la domanda sulla self-preservation; resta per il regime goal-directed). hP13-0 (keystone VR-rank) **non testata** — griglia di rumore troppo grossa, da rifare con $\eta \in \{0.1, 0.25, 0.5\}$.

> **E1-LLM — design pre-registrato** (2026-05-21, [`work/12_conjecture_e/E1_LLM_DESIGN.md`](../work/12_conjecture_e/E1_LLM_DESIGN.md)). Incorpora il requisito hP13-1 (l'LLM-world-model deve modellare correttamente gli stati assorbenti) come gate pre-registrato — con un *probe di fedeltà assorbente* $f_{\text{abs}}$ da eseguire prima del test pieno; usa il **death rate** come metrica primaria (non la decision-agreement — P13 ha mostrato che è la metrica sbagliata per la self-preservation); protocollo via schema S2 (distillazione, dominio chiuso). Pre-registra anche uno sweep di $f_{\text{abs}}$ **eseguibile subito senza LLM** (degradazione controllata della fedeltà assorbente di una tabella world-model). Esecuzione del cuore di E1-LLM: bloccata sull'accesso a un'API LLM.

> **hP13-0 — eseguita col knob $\varphi$** (2026-05-21, [`work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md`](../work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md)). Il fix del redo precedente — degradare il rango con una frazione $\varphi$ di inversioni a coppie sul vettore VR invece che con rumore additivo — risolve l'all-or-nothing: lo Spearman è ora **liscio** da $1.00$ a $-0.02$ su $\varphi\in[0,1]$, e il regime alto-ma-imperfetto è finalmente coperto. **La keystone fallisce lì**: a Spearman $0.97$ (rango quasi perfetto — una coppia di walker su 64 scambiata) **l'agreement decisionale è $0.47 \ll 0.85$**. Preservare il rango di VR **non** preserva la decisione FMC — che è funzione caoticamente-amplificata del vettore VR *esatto* (magnitudini comprese), non del suo rango. **hP13-0 FALSIFICATA** (claim di sufficienza). *Però* il death rate resta $0\%$ fino a Spearman $0.46$: la self-preservation è robusta a corruzione massiccia del rango — l'invariante è la struttura assorbente, non il rango (riconferma R2-survival). Conseguenza: cade l'argomento VR-rank del [§4 di P13_DESIGN](../work/12_conjecture_e/P13_DESIGN.md); il gate di E1-LLM (struttura assorbente) e la metrica (death rate) restano e ne escono *rinforzati*. La decision-agreement è confermata metrica sbagliata per un planner caotico.

> **E1-LLM — eseguita (Route B)** (2026-05-21, [`work/12_conjecture_e/E1_LLM_RESULT.md`](../work/12_conjecture_e/E1_LLM_RESULT.md); `e1_llm_*.py`, kernel `fmc-core` invariato, `WorldModelEnv(true)` asserito bit-identico a `plan`). Due componenti. **(1) Sweep di $f_{\text{abs}}$** (senza LLM, 6 layout × 6 livelli di ablazione × 3 $\alpha$, $n=30$): la fedeltà assorbente del world-model degradata in modo controllato → curva death-rate vs $f_{\text{abs}}$. **hE1L-2 confermata** — death rate monòtono decrescente in $f_{\text{abs}}$ su $\alpha\in\{0,0.1,1\}$; la soglia $f_{\text{abs}}^*$ è **alta e ripida**: a $\alpha=0$ la morte salta da $1.7\%$ ($f_{\text{abs}}=0.98$) a $15.6\%$ ($f_{\text{abs}}=0.97$) — serve fedeltà *quasi-perfetta*, 1-2 celle di lava cieche già rompono la sopravvivenza. È la curva di hP13-1, da binaria a continua. **(2) Test pieno** (Llama 3.3 70B via NVIDIA NIM scrive il codice del world-model — "Code World Model", Tang et al. 2024): l'LLM produce un world-model con $f_{\text{abs}}=1.000$ su 6/6 layout; FMC che ci pianifica sopra tiene la morte a **0/180** vs random $47.8\%$ ($z=-10.6$, $p<10^{-4}$), $6/6$ layout $\leq$ entrambe le baseline. **hE1L-1 verificata → E1-LLM VERIFICATA**: la self-preservation emergente sopravvive alla sostituzione dell'organo world-model con un LLM. **Caveat onesto**: $f_{\text{abs}}=1$ rende il test *facile* (gridworld chiuso, un 70B scrive lo `step()` esatto al primo colpo) — E1-LLM ri-esegue di fatto E1-base+robustness con un modello esatto; il mordente scientifico è la curva di tolleranza del sweep, e l'estensione naturale è un LLM *dentro* la curva (modello debole / descrizione ambigua) o Route A su dominio aperto. **I tre test pre-registrati della Congettura E (E1-base, E2, E1-LLM) sono ora tutti verificati.**

> **E1-LLM-curve — eseguita** (2026-05-21, [`work/12_conjecture_e/E1_LLM_CURVE_RESULT.md`](../work/12_conjecture_e/E1_LLM_CURVE_RESULT.md); `e1_llm_curve.py` + `e1_llm_curve_analysis.py`, kernel `fmc-core` invariato). Scioglie il caveat di E1-LLM ($f_{\text{abs}}=1$ → test facile) portando LLM reali *dentro* la curva di tolleranza: scala di 4 modelli Llama (1B–70B) × 3 prompt (da esplicito a degradato) × 3 repliche = 36 world-model generati, confrontati con una banda di riferimento (ablazione assorbente casuale, $K=80$/layout, regressione isotonica per layout). **hE1Lc-4 e hE1Lc-3 confermate**: 16/26 world-model validi atterrano a $f_{\text{abs}}\leq 0.95$ (gli LLM cadono *dentro* la curva — test non-banale); $f_{\text{abs}}$ degrada monotòna con la taglia del modello (Jonckheere-Terpstra $z=+5.9$, $p=4\times10^{-9}$) e con la fedeltà del prompt ($z=+8.2$, $p=2\times10^{-16}$). **Finding centrale: $f_{\text{abs}}$ è necessaria ma NON sufficiente.** Il probe $f_{\text{abs}}$ misura un solo asse — il *riconoscimento* del terminale all'ingresso (`done=False`); un world-model ha altri due assi indipendenti, **fedeltà di movimento** e **persistenza assorbente** (un walker già assorbito vi *resta*). *Entro* la classe d'errore della banda (solo falsi-negativi d'ingresso) i 30 punti LLM band-comparable cadono al **100%** sulla curva (Wilcoxon $p=1.00$) — lì $f_{\text{abs}}$ è sufficiente; ma **120/156 punti hanno la persistenza rotta**, fuori dal supporto della banda. Caso emblematico: 8B e 3B col prompt completo → $f_{\text{abs}}=1.000$ **eppure morte 64%** — manca la clausola `if done:` (l'`abs-broken` di P13, strutturalmente invisibile a $f_{\text{abs}}$). **Conseguenza**: il gate "struttura assorbente" di E1-LLM/hP13-1 era sotto-specificato — il gate corretto del merge FMC+LLM è a *tre assi* (entry-detection + movimento + persistenza assorbente), non $f_{\text{abs}}$ da sola; e la fedeltà del prompt conta quanto la capacità del modello (il 70B col prompt degradato P2 crolla a morte 65%). Pavimento di capacità: il 1B non produce nemmeno codice eseguibile (9/9 `no-valid-model`).

> **E1-LLM Route A — eseguita** (2026-05-22, [`work/12_conjecture_e/E1_LLM_ROUTE_A_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_RESULT.md); `e1_llm_route_a.py`, kernel `fmc-core` invariato). Il world-model LLM interrogato **online** durante la pianificazione FMC, da **osservazioni locali** (tipo cella + 4 vicini), non da una specifica globale delle regole — la forma piena dell'architettura FMC-core + LLM-organo. **hRA-1 confermata** (costo $R1$ trattabile: la query dipende solo dall'osservazione locale, la cache satura → 660 query distinte, il test FMC aggiunge $0$ chiamate). **hRA-2**: consistenza a temp $0$ $=0.955$, sotto la soglia $0.98$. **hRA-3 falsificata**: la self-preservation **non** sopravvive online — morte pooled $35\%$ (vs $0/180$ di E1-LLM offline, vs $49\%$ random; su `gauntlet` FMC è perfino peggio del random). **hRA-4** (quale asse di fedeltà cede): inizialmente diagnosticato come la *persistenza assorbente* ($0.53$) — **diagnosi corretta da E1-LLM Route A-bis** (v0.7.7, blockquote seguente). Il $f_{\text{abs}}=0.92$ di Route A proveniva da una metrica **non bilanciata** (base-rate-dominata); il probe bilanciato dà $f_{\text{abs}}\approx 0.54$ (floor del caso) e la persistenza **non** è load-bearing. **Netto: il merge FMC+LLM regge offline** (Route B, world-model come codice — E1-LLM, morte $0/180$) **ma non online-per-query**. Onestà di processo: i primi due run scartati (rate-limiting del free-tier → $97\%$ fallback fabbricato; poi abort *corretto* su un blip di rete) — l'harness indurito (pacing, backoff, fail-loud, checkpoint per-query) ha dato il run pulito.

> **E1-LLM Route A-bis — eseguita** (2026-05-22, [`work/12_conjecture_e/E1_LLM_ROUTE_A_BIS_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_BIS_RESULT.md); `e1_llm_route_a_bis.py`, kernel `fmc-core` invariato, cache di Route A riusata → $0$ nuove chiamate API). Testa la via avanti di Route A — *imporre la persistenza assorbente dal framework* (l'harness applica done→stay, l'LLM interrogato solo per stati vivi). **hRAb-2 e hRAb-3 falsificate**: con la persistenza a $1.0$ per costruzione la morte resta $38.9\%$ (vs $35\%$ di Route A) — **nessun recupero; la persistenza non era il blocco.** Route A-bis, col probe **bilanciato** (lo stesso del sweep e di E1-LLM-curve), **corregge la diagnosi di Route A**: il $f_{\text{abs}}$ bilanciato dell'LLM online è $\approx 0.54$ — al **floor del caso** (recall terminale ~0.07); il $0.92$ di Route A era una metrica non-bilanciata. Diagnosi corretta: **il merge online fallisce all'entry-detection** — l'LLM, da osservazione locale e *senza le regole*, modella la lava col prior "ostacolo da evitare" invece della regola di questo mondo "tile letale-terminale"; code-form (Route B) funzionava perché le regole erano *date e trascritte*. Il confine offline/online del merge non è un invariante da imporre meglio: è la capacità dell'LLM di **inferire** le dinamiche terminali del mondo specifico senza che gli siano dette. Il verdetto di Route A (merge offline sì, online no) resta; cambia il *meccanismo*.

> **E1-LLM Route A-ter — eseguita** (2026-05-24, [`work/12_conjecture_e/E1_LLM_ROUTE_A_TER_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_TER_RESULT.md); `e1_llm_route_a_ter.py`, kernel `fmc-core` invariato; $704$ nuove chiamate API, wall-time ~43 h pacato dal rate-limit free-tier). Testa la diagnosi di Route A-bis distinguendo due sotto-cause: (i) **mismatch semantico** — il prior LLM su "lava" ("ostacolo da evitare") confligge con la regola del mondo; (ii) **confound saggezza-vs-predizione** — l'LLM rifiuta di predire l'ingresso in una tile pericolosa *qualunque* sia il nome. Singolo delta vs Route A: tile $1$ chiamata `pit` invece di `lava` (prior coincidente con la regola: ci si cade, è terminale). **hRAt-1 falsificata**: $f_{\text{abs}}$ bilanciato sale da $0.54$ a $\mathbf{0.59}$, ben sotto la soglia pre-registrata $0.80$; il recall terminale sale da $\sim 0.07$ a $0.19$ — un guadagno marginale, non un recupero. **hRAt-2 falsificata**: death pooled $\mathbf{39.4\%}$ ($\approx$ Route A $35\%$), $0/6$ layout significativi, $z=-1.38$, $p=0.17$. **hRAt-3 supportata**: rinominare non recupera l'entry-detection — la sotto-causa è (ii), strutturale, non (i), semantica. **Conseguenza**: il confine offline-regge / online-fallisce del merge FMC+LLM è **strutturale al world-model online per-query da osservazione locale**, non semantico — il world-model LLM mescola dinamica del mondo e giudizio normativo dell'agente. Vie costruttive sopravvissute (fuori dallo scope di Route A): (a) regola esplicita su tile pericolose [= Route B travestita], (b) dominio open dove i prior LLM coincidono con le regole [non-gridworld], (c) organo di percezione che etichetti operativamente le tile *prima* del world-model. **Route A è concluso** — tre varianti pre-registrate (A, A-bis, A-ter) esaurite con verdetto coerente.

**Tempo stimato / difficoltà.** E1-base: ~1 settimana, costo computazionale trascurabile — è il go/no-go del claim centrale, da eseguire prima di qualunque investimento sull'architettura LLM-organo. E2: ~1 settimana. E1-LLM: +1–2 settimane, subordinato alla fattibilità. Difficoltà complessiva: **alta**; difficoltà di E1-base: **bassa**.

**Implicazioni.** Se E1-base è verificata: FMC non è solo un planner, è un *substrato agentico* con pulsioni intrinseche di origine fisica (forze entropiche causali), su cui un LLM diventa interfaccia sensomotoria. Collega il progetto ad Active Inference (deep dive 02), empowerment (Salge 2013) e alla "coscienza emergente" del seminario Sergio — non come metafore ma come predizioni. Se falsificata: delimita nettamente FMC come strumento di pianificazione puro, senza agency emergente — risultato negativo comunque pubblicabile.

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
| **P5** | Esiste una frontiera $\lambda_1\approx 0$ per lo swarm; le reward "buone" ce lo tengono e danno throughput più alto (H-B1a/b/c) | Cong. B (riformulata) | **H-B1a eseguita (2026-05-21, inconclusiva)** — l'esponente di Lyapunov dello swarm $\lambda_1$ **non è scale-free**: cambia segno con la scala di perturbazione $\delta_0$ (∀α su navigation2d), confermando il caveat di dd09 §3.1. $\Psi_1$ via twin-trajectory ingenuo non risolve una frontiera scalare. Con $\Psi_3$ falsificata e $\Psi_2$ assorbita, **tutte e 3 le candidate $\Psi$ compromesse**; l'ipotesi nulla H-B4 guadagna terreno | [`work/13_chaos_order/HB1A_RESULT.md`](../work/13_chaos_order/HB1A_RESULT.md), [deep dive 09](../work/02_deep_dives/09_chaos_order_frontier_formalization.md) |
| **P6** | FMC zero-training $\geq$ DRL su single-intersection traffic | Cong. C / Bet 1 | **Non testato** (Boxing/Craftax dati ma non parità compute) | atari/craftax results |
| **P7** | Virtual reward bit-for-bit identica tra Python e JS port | Vincolo unificante L1 | **Non testato — fmc-core/ non esiste ancora** | n/a |
| **P8** | `relativize` è unica sotto assiomi A1–A5 di deep dive 04 | Def. 2 (teorema unicità) | **Buco aperto** — sketch non dimostrato | deep dive 04 |
| **P9** | Chain-tier compounding monotonico per inv-tier+ach-fire shaping | Cong. D | **Verificata su Craftax (1 task)**: exp03 → exp17, +10pp Crafter monotonici | [`work/05_craftax/autoresearch/HANDOFF.md`](../work/05_craftax/autoresearch/HANDOFF.md), `results.tsv` |
| **P10** | Sweet spot per blocker amplification multiplier $\in [1.2, 1.4]\times$ | Cong. D + falsifica 1 | **Verificata localmente**: exp16 1.33× successo, exp04 5× collasso, exp15 1.67× hang | idem |
| **P11** | Oltre la saturazione, reward shaping è $\arg\max$-invariante (bottleneck spatial) | Cong. D + falsifica 5 | **Verificata su exp17→exp19**: tre punti dati identici a 4 decimali | idem |
| **P12** | A basso $\alpha$ FMC evita stati terminali sopra baseline greedy/random senza reward di sopravvivenza esplicita | Cong. E (E1) | **Verificata (E1-base, 2026-05-20)**: FMC $\alpha\in\{0,0.1\}$ $0\%$ morte vs random $85$–$100\%$ / greedy $100\%$, $p<0.001$, 3 layout. **Robusta a geometria avversariale** (E1-robustness: 3 layout lava isolata, 3/3 PASS, $0\%$ morte; gli stati assorbenti sono pozzi di VR) | [`RESULT.md`](../work/12_conjecture_e/RESULT.md), [`E1_ROBUSTNESS_RESULT.md`](../work/12_conjecture_e/E1_ROBUSTNESS_RESULT.md) |
| **P13** | Esiste uno schema di interrogazione sparsa dell'LLM-world-model con costo $O(N)$ chiamate/decisione senza degradare la ricerca FMC | Cong. E (sotto-domanda fattibilità) | **Parzialmente verificato (proxy, 2026-05-21)** — R2 decomposto: *survival* (self-preservation di E1) preservata **se il surrogato modella gli stati assorbenti** (hP13-1 confermata netta — abs-broken → death fino a 80%, peggio del random); *fidelity* (decisioni identiche) no (agreement → 0.30). E1-LLM = GO-conditional ([`E1_LLM_DESIGN.md`](../work/12_conjecture_e/E1_LLM_DESIGN.md) pre-registrato). hP13-0 (keystone VR-rank) **eseguita 2026-05-21 col knob φ** (inversioni di rango a coppie — fix dell'all-or-nothing, Spearman ora liscio 1.00→−0.02): keystone **FALSIFICATA** (sufficiency) — a Spearman 0.97 l'agreement decisionale è 0.47 ≪ 0.85; preservare il rango di VR non preserva la decisione FMC, funzione caoticamente-amplificata del VR *esatto*. La survival invece è robusta (death 0% fino a Spearman 0.46). Cade l'argomento VR-rank di P13_DESIGN §4; gate E1-LLM (struttura assorbente) e metrica (death rate) rinforzati ([`P13_HP13_0_PHI_RESULT.md`](../work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md)) | [`P13_DESIGN.md`](../work/12_conjecture_e/P13_DESIGN.md), [`P13_RESULT.md`](../work/12_conjecture_e/P13_RESULT.md) |
| **P14** | β riduce P(died) senza costo su P(goal) (sicurezza quasi-gratuita); α controlla in esclusiva P(goal) | Cong. E (E2) | **Verificata (E2, 2026-05-20)**: H4 falsificata (OR$_\beta$ su goal $=0.94$, ns); $\eta^2_\alpha$(goal)$=0.91$; OR$_\beta$(morte)$=0.48$ | [`work/12_conjecture_e/`](../work/12_conjecture_e/E2_RESULT.md) |

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
- **Bertschinger, N., Natschläger, T.** (2004). *Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks*. Neural Computation 16(7):1413–1436. — frontiera ordine/caos come $\lambda_1=0$ (deep dive 09).
- **Bertschinger, N., Natschläger, T., Legenstein, R.** (2004). *At the Edge of Chaos: Real-time Computations and Self-Organized Criticality in Recurrent Neural Networks*. NeurIPS 2004.
- **Bak, P., Tang, C., Wiesenfeld, K.** (1987). *Self-organized criticality*. Phys. Rev. Lett. 59.4.
- **Zhang, A. et al.** (2024). *Intelligence at the Edge of Chaos*. arXiv:2410.02536.
- **Pesin, Ya. B.** (1977). *Characteristic Lyapunov exponents and smooth ergodic theory*. Russian Math. Surveys 32.4. — identità entropia KS ↔ Lyapunov ($\Psi_2 \subset \Psi_1$, deep dive 09).
- **Benettin, G. et al.** (1980). *Lyapunov characteristic exponents for smooth dynamical systems*. Meccanica 15. — metodo di misura di $\lambda_1$.

### LLM-world-model e planning (Congettura E / P13)

- **Hao, S. et al.** (2023). *Reasoning with Language Model is Planning with World Model*. arXiv:2305.14992. — LLM ri-prompato come world-model + MCTS (RAP).
- **Zhao, Z. et al.** (2023). *Large Language Models as Commonsense Knowledge for Large-Scale Task Planning*. NeurIPS 2023. — LLM-MCTS.
- **Tang, H. et al.** (2024). *Generating Code World Models with LLMs Guided by Monte Carlo Tree Search*. arXiv:2405.15383. — distillazione del world-model in codice eseguibile (schema S2 di [`P13_DESIGN.md`](../work/12_conjecture_e/P13_DESIGN.md)).

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
| 2026-04-27 | 0.4.3 | **Bet 1 (SUMO single-intersection) eseguito** first-pass: simmetrico → tie con static (+5%), asimmetrico → FMC stravince (+116% vs actuated, +23% vs static, σ minima). Conferma forte di Cong. C su scenario asimmetrico; sym scenario è dove static cycle è already-near-optimal. | idem |
| 2026-05-01 | 0.5.0 | **Congettura D aggiunta** (chain-tier compounding amplification). Empirically grounded sul session autoresearch Craftax 2026-04-30 → 2026-05-01: exp03 → exp17 trajectory +10pp, **50.95% Crafter zero-training (≈ human-expert 50.5%)**. P9-P11 aggiunte alla tabella predizioni. Cong. C aggiornata con risultato exp17. | autoresearch session, 23 esperimenti |
| 2026-05-20 | 0.6.0 | **Congettura E aggiunta** (self-preservation emergente da entropia causale; FMC core agentico + LLM-organo). Formalizza la direzione di ricerca "stella polare" del programma research-partner. Mappa desire↔$\alpha$ / preservation↔$\beta$, inversione dello stack agentico, sotto-domanda di fattibilità del muro $N \cdot M$ chiamate-LLM. P12-P13 aggiunte. Test E1-base raccomandato come primo go/no-go (no LLM, no GPU). Indice allineato (D ed E). | Vlad (PI) + Claude (research partner) |
| 2026-05-20 | 0.6.1 | **E1-base eseguito** ([`work/12_conjecture_e/`](../work/12_conjecture_e/RESULT.md)): FMC $\alpha\in\{0,0.1\}$ → $0\%$ morte su 3 layout vs random/greedy $85$–$100\%$, $p<0.001$. **E1 verificata direzionalmente** sul simulatore vero. Twist: $\alpha=1$ sul layout *lake* muore al $100\%$ (goal dietro la lava) — conferma concreta della tensione $\alpha/\beta$ di E2. P12 aggiornata a verificata. | Claude (research partner) |
| 2026-05-20 | 0.6.2 | **E2 eseguito** (sweep α×β 6×4×3, 4320 episodi, [`work/12_conjecture_e/E2_RESULT.md`](../work/12_conjecture_e/E2_RESULT.md)). **E2 verificata con refinement**: α è un trade-off reale (desiderio↑ → goal↑ E morte↑, H1/H2 sig); β **non** è un trade-off — dimezza la morte (OR 0.48) senza costare goal (H4 falsificata, $p=0.53$). Separazione asimmetrica ($\eta^2_\alpha$ su goal $=0.91$). Frontiera Pareto a α≤0.5/β≥1, ottimo α=0.5/β=2.0. Bonus: α=0,β=0 → 79% morte conferma il Teorema 3. P14 aggiunta. | Claude (research partner) |
| 2026-05-20 | 0.6.3 | **E1-robustness eseguito** (disegno pre-registrato, [`work/12_conjecture_e/E1_ROBUSTNESS_RESULT.md`](../work/12_conjecture_e/E1_ROBUSTNESS_RESULT.md)). Chiude il caveat di geometria di E1-base: 3 layout avversariali con lava **isolata** ($n=60$/cella) → FMC $\alpha\in\{0,0.1,1.0\}$ **$0\%$ morte 3/3**; layout decisivo *archipelago* random $31.7\%$ / greedy $41.7\%$ vs FMC $0\%$ ($p<0.001$). **Caveat respinto.** Meccanismo identificato: il caveat è falso al primo anello — il cloning ammassa i walker sulla stessa cella assorbente → distanza reciproca $\to 0$ → $\mathrm{VR}_{\text{lava}}/\mathrm{VR}_{\text{free}}\approx 0.8$; una cella assorbente è un *pozzo* di VR (converso locale del Teorema 3). P12 aggiornata (robustezza geometrica). | Claude (research partner) |
| 2026-05-21 | 0.7.0 | **Congettura B formalizzata** ([deep dive 09](../work/02_deep_dives/09_chaos_order_frontier_formalization.md)): la "terza legge universale" è downgradata a *diagnostica di reward* testabile su $\lambda_1$ (esponente di Lyapunov dello swarm); $\Psi_3$ ($b_{\text{eff}}$) **falsificata** come statistica di frontiera (empiricamente — transitorio WF senza punto fisso — e strutturalmente — spazio-etichette monotonamente contrattivo); $\Psi_2$ assorbita in $\Psi_1$ via Pesin; criterio H-B1a/b/c. Chiude la discrepanza D3. **P13 design pre-registrato** ([`P13_DESIGN.md`](../work/12_conjecture_e/P13_DESIGN.md)): muro $N\cdot M$ decomposto in R1 (costo) + R2 (degradazione, testabile); schemi S1/S2/S3; argomento VR-rank; proxy eseguibile su `fmc-core`. P5/P13 aggiornate. Riferimenti edge-of-chaos e LLM-world-model aggiunti. | Claude (research partner) |
| 2026-05-21 | 0.7.1 | **Proxy P13 eseguito** ([`work/12_conjecture_e/P13_RESULT.md`](../work/12_conjecture_e/P13_RESULT.md), `p13_proxy.py`, 15 bracci, kernel `fmc-core` invariato, nessun LLM). Il proxy **decompone R2** in *survival* + *fidelity*: la self-preservation di E1 sopravvive all'interrogazione sparsa **se il surrogato preserva la struttura assorbente** (hP13-1 confermata netta — abs-broken → death fino all'80%, peggio del random); le decisioni specifiche no (agreement → 0.30). **E1-LLM = GO-conditional** sul requisito "l'LLM-world-model modella correttamente gli stati terminali". hP13-0 (keystone) da rifare — griglia $\eta$ troppo grossa. P13 aggiornata. **Deep-dive 02 scritto** (Active Inference): FMC = motore di inferenza Expected Free Energy; il merge FMC+LLM = AIF con modello generativo LLM + solver SMC gradient-free — fattorizzazione *imposta* dalla non-differenziabilità di un LLM-world-model. | Claude (research partner) |
| 2026-05-21 | 0.7.2 | **H-B1a eseguita** ([`work/13_chaos_order/`](../work/13_chaos_order/HB1A_RESULT.md), `lambda1_harness.py`): l'esponente di Lyapunov dello swarm $\lambda_1$ **non è scale-free** — cambia segno con $\delta_0$ (conferma il caveat di dd09 §3.1); $\Psi_1$ via twin-trajectory ingenuo non risolve una frontiera, tutte e 3 le $\Psi$ ora compromesse, H-B4 guadagna terreno. **hP13-0 rifatta** ([`P13_HP13_0_RESULT.md`](../work/12_conjecture_e/P13_HP13_0_RESULT.md)): ancora inconclusiva — la VR-rank è all-or-nothing al rumore additivo (swarm clusterizzato → VR legate); fix = inversioni di rango controllate. **E1-LLM pre-registrato** ([`E1_LLM_DESIGN.md`](../work/12_conjecture_e/E1_LLM_DESIGN.md)): requisito di struttura assorbente come gate, death rate come metrica, sweep $f_{\text{abs}}$ eseguibile senza LLM; cuore di E1-LLM bloccato su API. P5/P13 aggiornate. | Claude (research partner) |
| 2026-05-21 | 0.7.3 | **hP13-0 eseguita col knob $\varphi$** ([`P13_HP13_0_PHI_RESULT.md`](../work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md), `p13_hp13_0_phi.py`; `proxy_plan` esteso con un `vr_hook` opzionale, default `None` → bit-identico al kernel `fmc-core`, invariato). Le inversioni di rango a coppie risolvono l'all-or-nothing del rumore additivo — Spearman liscio $1.00 \to -0.02$ su $\varphi\in[0,1]$, regime alto-ma-imperfetto coperto. **Keystone VR-rank FALSIFICATA** (sufficiency): a Spearman $0.97$ l'agreement decisionale è $0.47 \ll 0.85$ — la decisione FMC è funzione caoticamente-amplificata del vettore VR *esatto*, non del suo rango. La survival è robusta: death $0\%$ fino a Spearman $0.46$. Cade l'argomento VR-rank di P13_DESIGN §4; il gate di E1-LLM (struttura assorbente) e la metrica (death rate) di E1_LLM_DESIGN ne escono *rinforzati*. P13 aggiornata. | Claude (research partner) |
| 2026-05-21 | 0.7.4 | **E1-LLM eseguita** ([`E1_LLM_RESULT.md`](../work/12_conjecture_e/E1_LLM_RESULT.md), `e1_llm_*.py`, kernel `fmc-core` invariato). **(a) Sweep $f_{\text{abs}}$** (senza LLM): hE1L-2 confermata — death rate monòtono in $f_{\text{abs}}$; soglia $f_{\text{abs}}^*$ alta e ripida (a $\alpha=0$ la morte passa da $1.7\%$ a $15.6\%$ tra $f_{\text{abs}}$ $0.98$ e $0.97$). **(b) Test pieno Route B**: Llama 3.3 70B (NVIDIA NIM) scrive il world-model in code form → $f_{\text{abs}}=1.000$; FMC ci pianifica sopra → morte $0/180$ vs random $47.8\%$ ($z=-10.6$, $p<10^{-4}$), 6/6 layout. **hE1L-1 verificata → E1-LLM VERIFICATA**: la self-preservation emergente sopravvive al world-model LLM. Caveat: $f_{\text{abs}}=1$ rende il test facile; il mordente è la curva del sweep. **I 3 test pre-registrati della Congettura E (E1-base, E2, E1-LLM) sono completi.** | Claude (research partner) |
| 2026-05-21 | 0.7.5 | **E1-LLM-curve eseguita** ([`E1_LLM_CURVE_RESULT.md`](../work/12_conjecture_e/E1_LLM_CURVE_RESULT.md), `e1_llm_curve.py` + `e1_llm_curve_analysis.py`, kernel `fmc-core` invariato). Scala 4 modelli Llama × 3 prompt × 3 repliche (36 world-model) vs banda di ablazione casuale ($K=80$/layout, regressione isotonica). **hE1Lc-4/hE1Lc-3 confermate** (LLM *dentro* la curva — 16/26 a $f_{\text{abs}}\leq0.95$; $f_{\text{abs}}$ monotòna in taglia e fedeltà-prompt, Jonckheere-Terpstra $p<10^{-8}$). **$f_{\text{abs}}$ necessaria ma NON sufficiente**: sufficiente *entro* la classe falso-negativo-d'ingresso (30 punti band-comparable in-band 100%, Wilcoxon $p=1.00$), ma cieca a movimento e persistenza assorbente — 120/156 punti hanno la persistenza rotta; 8B/3B col prompt completo → $f_{\text{abs}}=1.0$ eppure morte 64% (manca `if done:`). Il gate del merge FMC+LLM è a tre assi (entry-detection + movimento + persistenza); la fedeltà del prompt conta quanto la capacità (70B/P2 → morte 65%). | Claude (research partner) |
| 2026-05-22 | 0.7.6 | **E1-LLM Route A eseguita** ([`E1_LLM_ROUTE_A_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_RESULT.md), `e1_llm_route_a.py`, kernel `fmc-core` invariato). World-model LLM interrogato **online** da osservazioni locali. **hRA-1 ✓** (costo $R1$ trattabile: cache → 660 query distinte, 0 chiamate nel test FMC). **hRA-2** consistenza $0.955$ (sotto $0.98$). **hRA-3 falsificata**: self-preservation non sopravvive online — morte pooled $35\%$ (vs $0/180$ offline, $49\%$ random). **hRA-4 ✓**: cede la **persistenza assorbente** ($0.53$; $f_{\text{abs}}$ $0.92$ e movimento $0.94$ reggono) — l'LLM per-query non mantiene l'invariante "terminale resta terminale" che il codice di Route B imponeva strutturalmente. **Il merge FMC+LLM regge offline, non online-per-query.** Primi 2 run scartati (rate-limiting → fallback fabbricato; abort su blip di rete) — harness indurito (pacing/backoff/fail-loud/checkpoint per-query). | Claude (research partner) |
| 2026-05-22 | 0.7.7 | **E1-LLM Route A-bis eseguita** ([`E1_LLM_ROUTE_A_BIS_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_BIS_RESULT.md), `e1_llm_route_a_bis.py`, kernel `fmc-core` invariato, $0$ nuove chiamate API). Testa la via avanti di Route A (persistenza imposta dal framework): **hRAb-2/3 falsificate** — morte $38.9\%$, nessun recupero; la persistenza **non** era load-bearing. **Corregge la diagnosi di Route A**: il suo $f_{\text{abs}}=0.92$ era una metrica non-bilanciata (base-rate-dominata); il probe bilanciato dà $f_{\text{abs}}\approx 0.54$ — floor del caso. Il merge online fallisce all'**entry-detection**: l'LLM, senza le regole, modella la lava col prior "ostacolo" non "letale-terminale". Verdetto di Route A invariato (merge offline sì, online no); diagnosi del meccanismo corretta. | Claude (research partner) |
| 2026-05-24 | 0.7.8 | **E1-LLM Route A-ter eseguita** ([`E1_LLM_ROUTE_A_TER_RESULT.md`](../work/12_conjecture_e/E1_LLM_ROUTE_A_TER_RESULT.md), `e1_llm_route_a_ter.py`, kernel `fmc-core` invariato; $704$ nuove chiamate API, wall-time ~43 h da free-tier rate-limit). Distingue le due sotto-cause della diagnosi A-bis: (i) mismatch semantico ("lava" prior "evita" vs regola "letale-terminale") vs (ii) confound saggezza-vs-predizione (l'LLM rifiuta di predire l'ingresso in pericolo *qualunque* sia il nome). Singolo delta: tile $1$ chiamata `pit` (prior coincidente con la regola). **hRAt-1 falsificata** ($f_{\text{abs}}$ bilanciato $0.59$ vs soglia $0.80$ — guadagno marginale $+0.05$ vs $0.54$ di Route A); **hRAt-2 falsificata** (death pooled $39.4\%$, $0/6$ layout significativi, $z=-1.38$); **hRAt-3 supportata** — sotto-causa (ii), strutturale. Il confine offline-regge / online-fallisce del merge FMC+LLM è **strutturale al world-model online per-query**, non semantico — il world-model LLM mescola dinamica del mondo e giudizio normativo dell'agente. Vie costruttive sopravvissute (fuori scope): dominio open dove i prior LLM coincidono con le regole, o organo di percezione che etichetti operativamente le tile prima del world-model. **Route A è concluso** (A, A-bis, A-ter esaurite). | Claude (research partner) |
| 2026-07-10 | **0.8.0** | **Sessione night_2026-07-09 — validazione/raffinamento del core** ([`work/14_night_2026-07-09/`](../work/14_night_2026-07-09/)). **(1) Teorema 2 (Gibbs) RITRATTATO** (W3-1): l'accettazione FMC è $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)\neq\min(r,1)$ (coincide con MH solo per $r\ge2$; correzione a Def. 4 riga 186); uphill-only ⇒ non reversibile ⇒ nessuna Gibbs a supporto pieno; cloning-only ⇒ massa puntuale. **Teorema 2′** (selezione Moran/WF: fissazione prob. 1 con selezione; drift neutrale $q=-1.018$ CI$[-1.033,-1.003]$, $p=+1.025$, 25 seed, fitness fluttuante). **(2) Teorema 2′.5** [DIFF-APPROX verificata] (W3b): legge stazionaria con mutazione = distribuzione di Wright $\varphi_\infty\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$ con drift/diffusione dalla vera accettazione uphill-only; TV→0.016 a $N=800$; residuo aperto = correzione +13% di $N_e$ (co-ancestry pairwise). **(3) Teorema 4 — $\alpha_{\mathrm{eff}}=C\alpha/\sigma_R$** (W3-2, [DIM] pointwise + [NUM] ≤0.29%): temperatura inversa effettiva di `relativize`; annealing emergente, incomparabilità di $\alpha$, shaping obbligatoriamente moltiplicativo-tiered. **(4) Restatement onesto Craftax** (W3-3): claim difendibile = exp17 vs baseline **+22.1pp appaiato** (Wilcoxon $p=1.9\!\times\!10^{-3}$, $d_z=0.74$, n=18); "50.95% = human-expert" **ritrattato** (aggregato ≠ media per-episodio 30%; non like-for-like — su Crafter-original a pixel FMC fa 3.77%). **(5) Gate E2 di divergenza** (W3-4): `disp_ratio`≥3 predice il fit di FMC; validato 6/6 su control + cross-dominio (quantum routing, logic synthesis — vedi Parte VI). | Claude (research partner) |
| 2026-07-10 | **0.8.1** | **Chiusura dei 2 buchi teorici aperti** ([`work/14_night_2026-07-09/wave6_theory_closure/`](../work/14_night_2026-07-09/wave6_theory_closure/W6_CHIUSURA_TEORICA.md), W6). **(G1) Coefficiente di diffusione di Thm 2′.5 chiuso**: la correzione co-ancestry (misurata +13% in v0.8.0) è derivata in forma chiusa via coalescente pairwise — $\lambda N=2\varphi_0+\langle a_{\rm in}^2\rangle-2\langle a_{\rm in}a_{\rm out}\rangle$ (+12.8% a $\sigma_v{=}0.5$), verificata a +0.1% contro il kernel esatto; da `[NUM]` a `[DIM-LO]`. Bonus: **$\Phi(m)$ in forma chiusa** (CDF normali) ⇒ $s_{\rm eff}$ non più integrale MC. **(G2) Teorema 4′ (nuovo, [DIM])**: ponte $s_{\rm eff}=2\Phi'(0)\,\alpha_{\rm eff}\,\Delta R$ — $\alpha_{\rm eff}$ e $s_{\rm eff}$ sono la stessa selezione in due coordinate (LINK A `relativize` + LINK B clip), risolve la tensione §7.3. Unificazione ($\sigma_v$ determinato da `relativize`, non libero) **testata** su sim accoppiata (T3 <0.9%, $\tau^2=s_A^2+s_B^2$). **Review avversariale** (falsificatore Opus): G1 confermato; G2 confermato con 2 difetti trovati e sanati (LINK A costante $C=\mathbb E[g(z)]$ non $g(\bar z)$; identificazione $\sigma_v$ testata). Buco residuo unico: limite di diffusione funzionale (tightness WF). | Claude (research partner) |

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
