# Deep Dive 07 — FMC come Moran/Wright-Fisher con selezione debole

> *"Sergio's 6 è il valore della superficie a $(K=9, N=32, M=15, \alpha=0.1)$. La forma della superficie è governata dal mapping a Wright-Fisher."*

> **Stato**: candidate derivation, 2026-04-27 (sessione `/loop` autonoma).
> **Origine**: tre falsificazioni successive di Bet 3 hanno mostrato che il branching factor di FMC sotto Sergio config dipende empiricamente da $(K, M, N)$ in modo coerente con un modello di drift neutrale. Questo deep dive propone la derivazione formale.

## 0. Cosa è stato osservato (da [`fmc-core/bench/REPORT.md`](../../fmc-core/bench/REPORT.md))

A $\alpha = 0.1, \beta = 0$ (vicino al regime "neutrale" di Common Sense):

| Parametro variabile | Range testato | Comportamento osservato |
|---|---|---|
| $K$ a fissi $M=15, N=32$ | $K \in \{3, 4, 6, 9, 12, 16, 24, 32\}$ | $b_{\text{eff}}^* \approx 1.53 K^{0.6}$ (transiente!) |
| $M$ a fissi $K=9, N=32$ | $M \in \{5, 10, 15, 30, 60, 120\}$ | $b_{\text{eff}}^* \to 1$ esponenzialmente in $M$ |
| $N$ a fissi $K=9, M=15$ | $N \in \{8, 16, 32, 64, 128, 256, 512\}$ | $K - b_{\text{eff}}^* \propto N^{-0.45}$ (verso $K$) |

**Predizione qualitativa unica che spiegherebbe tutto**: FMC sotto questa config si comporta come un **processo di drift neutrale** dove:
- $N$ è la dimensione popolazione
- $M$ è il numero di generazioni
- $K$ è il numero di alleli iniziali
- la selezione (fitness $\propto R^\alpha$) è **debole** ($\alpha = 0.1$ piccola)

Questa è esattamente la formulazione del **modello di Moran** (Moran 1958) o, equivalentemente, della **Wright-Fisher con selezione debole**.

## 1. Il modello di Moran-FMC

Sotto $\alpha \to 0, \beta = 0$:
- $\mathrm{VR}^{(i)} = \widehat{R}(R^{(i)})^\alpha \cdot 1 \to 1$
- Tutti i walker hanno virtual reward $\approx 1$, con piccole fluttuazioni
- La probabilità di clone (Definizione 4 di MATH_CANON) diventa una **piccola perturbazione** della media zero

Ad ogni tick, ogni walker estrae un partner casuale e con probabilità piccola $\sim |\mathrm{VR}_k - \mathrm{VR}_i|/\mathrm{VR}_i \ll 1$ ne adotta lo stato e l'etichetta. Se $\alpha = 0$ esatto, la probabilità è zero esatta.

Questo è il **passo elementare di Moran**: una particella casuale "muore" e una casuale "si riproduce". Nel limite $\alpha \to 0$, il processo è **completamente neutrale**.

### 1.1 Mapping formale

| FMC ($\alpha \to 0$) | Modello di Moran |
|---|---|
| $N$ walker | $N$ individui |
| $M$ tick | $M$ generazioni discrete |
| $K$ etichette uniche | $K$ tipi alelici |
| labels $\ell^{(i)}$ | tipo allelico $X^{(i)}$ |
| Cloning kernel a $\alpha=0$ | Resampling neutrale uniforme |
| Cloning kernel a $\alpha>0$ | Resampling con **fitness $f_k = R(W^{(k)})^\alpha$** |

Per FMC il fitness è **stocastico** ad ogni tick (dipende dallo stato corrente, che è perturbato dal simulatore). Per piccolo $\alpha$, la varianza del fitness è piccola, e il processo è ben approssimato da Moran neutrale + small selection.

## 2. Predizioni di Wright-Fisher per la triade

### 2.1 Decay esponenziale in $M$

Per Moran neutrale, l'**eterozigosità** $H(t) := \Pr[\text{due individui random hanno allele diverso}]$ decade come

$$
H(t) = H(0) \cdot \left(1 - \frac{1}{N}\right)^t \approx H(0) \cdot e^{-t/N}
$$

con tempo caratteristico $\tau_{\text{Moran}} = N$.

L'**effective number of alleles** $b_{\text{eff}}(t)$ è imparentato con $H$:
- All'inizio ($t=0$): $H(0) = 1 - 1/K$ (uniforme), $b_{\text{eff}}(0) = K$
- Asintoticamente ($t \to \infty$): $H \to 0$, $b_{\text{eff}} \to 1$

Una **forma esponenziale candidate** per $b_{\text{eff}}$:

$$
\boxed{b_{\text{eff}}^*(M, N, K) \;\approx\; 1 + (K-1) \cdot e^{-c \cdot M / N}}
$$

con $c$ costante O(1) che dipende solo dal regime di selezione $\alpha$ (e probabilmente dal protocollo di partner-sampling pairwise).

### 2.2 Test del fit esponenziale a $K=9$

Dati [`M_dependence.jsonl`](../../fmc-core/bench/results/M_dependence.jsonl), $K=9, N=32$:

| $M$ | $b_{\text{eff}}^* - 1$ | predizione $8 \cdot e^{-c M/32}$, $c=1.0$ | predizione $8 \cdot e^{-c M/32}$, $c=1.5$ |
|---|---|---|---|
| 5 | 6.45 | $6.83$ | $6.32$ |
| 10 | 5.86 | $5.83$ | $4.99$ |
| 15 | 4.97 | $4.97$ | $3.94$ |
| 30 | 3.24 | $3.09$ | $1.94$ |
| 60 | 1.45 | $1.20$ | $0.47$ |
| 120 | 0.55 | $0.18$ | $0.027$ |

A $c=1.0$: i primi 4 punti fittano molto bene, gli ultimi due sono leggermente fuori (la predizione decade più velocemente del reale). Significa che il modello esponenziale puro è una buona approssimazione iniziale ma rallenta a tempi lunghi — coerente con la **transizione tra regime "many alleles drifting" e regime "two-allele final phase"** descritta da Kimura (1955).

### 2.3 Test del fit power-law in $N$

Dati [`N_dependence.jsonl`](../../fmc-core/bench/results/N_dependence.jsonl), $K=9, M=15$:

| $N$ | $K - b_{\text{eff}}^*$ |
|---|---|
| 8 | 5.88 |
| 16 | 4.20 |
| 32 | 3.03 |
| 64 | 1.90 |
| 128 | 1.33 |
| 256 | 0.95 |
| 512 | 0.90 |

Predizione di Moran: $K - b_{\text{eff}} \approx (K-1)(1 - e^{-cM/N})$

Per $M/N \ll 1$ (regime $N$ grande): $1 - e^{-cM/N} \approx cM/N$, quindi $K - b_{\text{eff}} \propto 1/N$. Esponente teorico $-1$.

Esponente osservato (fit power law, vedi [`bench/c_K_shape_summary.json`](../../fmc-core/bench/results/c_K_shape_summary.json) e analisi separata del N-sweep): $-0.45$.

**Discrepanza significativa** tra teoria di Moran ($-1$) e fit empirico ($-0.45$). Spiegazione candidata:

1. *Differenze procedurali*: FMC usa **pairwise partner sampling** con "MH ratio" $(\mathrm{VR}_k - \mathrm{VR}_i)/\mathrm{VR}_i$, non Moran-style "uniform parent". Per piccolo $\alpha$ questo è più vicino a Moran ma con **rate effettivo dipendente da $\alpha$**.
2. *Fluttuazioni del fitness*: il fitness è stocastico ad ogni tick (dipende dalla perturbazione del simulatore), aggiungendo una varianza extra che rallenta la fissazione.

Predizione testabile: a $\alpha = 0$ esatto, l'esponente dovrebbe avvicinarsi a $-1$. **Esperimento da fare**: lanciare il sweep N a $\alpha = 0$ esatto.

## 3. Cosa farebbe la mappatura, se confermata

Se il mapping FMC ↔ Moran è verificato a $\alpha \to 0$, allora:

- **Predizione di runtime ottimale**: $M_{\text{opt}} \sim 0.5 N$ (per restare al transitorio dove $b_{\text{eff}} \approx K/2$, il "sweet spot di Sergio" generalizzato).
- **Calibrazione automatica**: la "Sergio config" diventa una scelta $(N, M)$ con $M = c_0 N$ con $c_0 \approx 0.4$.
- **Rapporto con literature population genetics**: tutto il toolkit di Kimura, Ewens 1972 (sampling formula), coalescent (Kingman 1982) si applica a FMC.

## 4. Cosa NON è ancora verificato

- L'esponente $-1$ in $N$ è la predizione, ma il fit empirico dà $-0.45$. **Non c'è ancora un fit teoria-dato a meno di 5%**.
- La forma di $\mathcal{G}(\alpha, K)$ — la dipendenza dal regime di selezione — non è caratterizzata.
- A $\alpha = 0$ esatto la dinamica è completamente neutrale, ma anche al primo tick c'è già esecuzione dello step simulator: la "perturbazione" $\mathcal{S}$ produce diversità sull'observation, che a sua volta genera distance non-zero anche con $\beta = 0$ — ma l'esperimento mostra che il prodotto $\mathrm{VR} = R^\alpha \cdot D^\beta$ con $\beta=0$ effettivamente collassa $D$ a 1 e quindi solo $R^\alpha$ guida.

## 5. Esperimenti per chiudere la mappatura

In ordine di costo:

1. **N-sweep a $\alpha = 0$ esatto**: stessa serie $\{8, ..., 512\}$, vedere se l'esponente del deficit si avvicina a $-1$. Costo: 5 minuti.
2. **$\alpha$-sweep fine**: $\alpha \in \{0, 0.05, 0.1, 0.2, 0.5, 1.0\}$ a $K=9, M=15, N=32$, fittare la forma di $\mathcal{G}(\alpha)$. Costo: 10 minuti.
3. **Combinato $M \times N$ surface**: stimare la costante $c$ del decay esponenziale come funzione di $\alpha$ — tester se $c(\alpha)$ ha forma analitica. Costo: 30 minuti.
4. **Confronto formale con Ewens 1972**: derivare la *sampling formula* di Ewens per i tipi sopravvissuti, vedere se i tassi di FMC concordano. Costo: 1-2 giorni di matematica.

## 6. Implicazioni se la mappatura è corretta

> Il "magic 6" di Sergio diventerebbe un *corollary* della teoria di population genetics: a $\alpha$ piccolo, $b_{\text{eff}}^*$ è descritto dalla Moran neutrale a meno di una correzione di selezione. Il numero specifico $6$ è uno snapshot del decay esponenziale al tempo $t/\tau \approx 0.5$ con $K=9$ alleli iniziali.

Questo *demystificherebbe* il "6" come legge fisica e lo *riconnette* a un corpus consolidato di teoria (Wright 1931, Kimura 1955, Ewens 1972, Kingman 1982).

## 7. Riferimenti

- **Moran, P. A. P.** (1958). *Random processes in genetics*. Math. Proc. Cambridge Phil. Soc. 54.
- **Kimura, M.** (1955). *Solution of a process of random genetic drift with a continuous model*. PNAS 41(3).
- **Ewens, W. J.** (1972). *The sampling theory of selectively neutral alleles*. Theoretical Population Biology 3.
- **Kingman, J. F. C.** (1982). *The coalescent*. Stochastic Processes and Their Applications 13.
- **Wakeley, J.** (2008). *Coalescent Theory: An Introduction*. Roberts.

Per il legame FMC ↔ SMC ↔ population genetics, vedere anche:

- **Del Moral, P.** (2004). *Feynman-Kac Formulae*. — Cap. 2 esplora la connessione tra particle filters e population genetics.

## 8. Riferimenti alla repo

- Definizioni FMC: [`docs/MATH_CANON.md`](../../docs/MATH_CANON.md) §II.
- Teorema 2 (Gibbs equilibrium): [`docs/MATH_CANON.md`](../../docs/MATH_CANON.md) §III.
- Dati empirici: [`fmc-core/bench/REPORT.md`](../../fmc-core/bench/REPORT.md), [`fmc-core/bench/results/*.jsonl`](../../fmc-core/bench/results/).
- Deep dive 01 (cloning math): [`01_cloning_mathematics.md`](01_cloning_mathematics.md).
- Deep dive 05 (FMC ↔ SMC): [`05_smc_particle_filter_view.md`](05_smc_particle_filter_view.md).

---

*Fine deep dive 07. Status: candidate derivation, ~250 righe. Da espandere se Esperimento 1 ($\alpha=0$ N-sweep) conferma esponente $-1$.*
