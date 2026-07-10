# W31 — La distribuzione stazionaria corretta del kernel di cloning FMC

> **Ruolo**: dimostrazione/falsificazione, non abbellimento.
> **Data**: 2026-07-09. **Autore**: wave3 validation (matematico rigoroso).
> **Oggetto**: falsificare il Teorema 2 di [`docs/MATH_CANON.md`](../../../docs/MATH_CANON.md) righe 253-301 (Gibbs a temperatura finita via Metropolis-Hastings) e sostituirlo con l'enunciato corretto (fissazione / equilibrio mutation-selection-drift).
> **Script + output reali**: [`w31_stationary_check.py`](w31_stationary_check.py). Ogni numero in questo documento è prodotto da quello script (seed `20260709`, numpy 2.2.6). Nessun numero è inventato.
> **Marcatura**: ogni passo è etichettato `[DIMOSTRATO]`, `[SKETCH]`, `[NUMERICO]`.

---

## 0. Sintesi esecutiva

Il Teorema 2 del canone afferma che il kernel di cloning FMC è una catena Metropolis-Hastings con stazionaria di Gibbs $\pi^*(x)\propto R(x)^\alpha\rho(x)^{-\beta}$ a temperatura finita $1/\alpha$. **È falso su tre punti indipendenti, tutti verificati:**

1. **La funzione di accettazione non è quella di Metropolis** (né Barker). L'affermazione di riga 186 — "clip$(VR_k/VR_i-1)=\min(VR_k/VR_i,1)$" — è un'identità algebricamente falsa. `[DIMOSTRATO + NUMERICO]`
2. **Il kernel è uphill-only** ⇒ non reversibile ⇒ nessuna misura a supporto pieno soddisfa il bilancio dettagliato. `[DIMOSTRATO]`
3. **Il processo cloning-only converge a massa puntuale** (fissazione sul tipo con VR massima), $b_{\text{eff}}\to1$. Non esiste Gibbs a $T$ finita. `[DIMOSTRATO + NUMERICO]`

L'oggetto corretto è un **modello di Moran / Wright-Fisher**: neutrale (drift, $q\approx-1$) nel caso $\alpha=0$, con selezione direzionale nel caso $\alpha>0$. Una distribuzione stazionaria non degenere esiste **solo** aggiungendo la perturbazione $\mathcal S$ (mutazione) → equilibrio **mutation-selection-drift**, che **non** è la Gibbs $\propto R^\alpha$ salvo casi limite.

---

## 1. La regola di transizione reale (Def 4) — e la correzione di riga 186

### 1.1 Enunciato preciso `[DIMOSTRATO]`

Dal codice di riferimento (entrambe le implementazioni canoniche):

- [`repos/FractalAI_old/fractalai/swarm.py:527-528`](../../../repos/FractalAI_old/fractalai/swarm.py#L527):
  ```python
  value = (vr_compas - vir_rew) / np.where(vir_rew > 0, vir_rew, 1e-8)   # = r - 1
  clone = (value >= np.random.random()).astype(bool)                     # U <= r-1
  ```
- [`repos/fragile/src/fragile/fractalai.py:168-173`](../../../repos/fragile/src/fragile/fractalai.py#L168): regola identica (`clone_probs = (vir_rew[compas] - vir_rew)/vir_rew; will_clone = clone_probs > rand()`).

Posto $r := \mathrm{VR}_k/\mathrm{VR}_i$ (partner $k$ su self $i$), la **probabilità effettiva di clone** è
$$
a_{\mathrm{FMC}}(r) \;=\; \Pr[\,U \le r-1\,] \;=\; \operatorname{clip}(r-1,\,0,\,1) \;=\; \min\!\big(\max(r-1,\,0),\,1\big),
\qquad U\sim\mathrm{Unif}(0,1).
$$

Confronto con le due funzioni di accettazione standard:

$$
a_{\mathrm{MH}}(r)=\min(r,1)\quad\text{(Metropolis-Hastings)},\qquad
a_{\mathrm{Barker}}(r)=\frac{r}{1+r}\quad\text{(Barker)}.
$$

**Tabella prodotta dallo script (Part 0):**

| $r=\mathrm{VR}_k/\mathrm{VR}_i$ | $a_{\mathrm{FMC}}=\operatorname{clip}(r-1)$ | $a_{\mathrm{MH}}=\min(r,1)$ | $a_{\mathrm{Barker}}$ |
|---:|---:|---:|---:|
| 0.50 | 0.0000 | 0.5000 | 0.3333 |
| 0.80 | 0.0000 | 0.8000 | 0.4444 |
| 1.00 | 0.0000 | 1.0000 | 0.5000 |
| 1.20 | 0.2000 | 1.0000 | 0.5455 |
| 1.50 | 0.5000 | 1.0000 | 0.6000 |
| 1.80 | 0.8000 | 1.0000 | 0.6429 |
| 2.00 | 1.0000 | 1.0000 | 0.6667 |
| 3.00 | 1.0000 | 1.0000 | 0.7500 |

Le tre funzioni coincidono solo per $r\ge2$. Su $(0,2)$ sono diverse. In particolare $a_{\mathrm{FMC}}$ ha due proprietà che $a_{\mathrm{MH}}$ **non** ha:

- **uphill-only**: $a_{\mathrm{FMC}}(r)=0$ per ogni $r\le1$ (nessun passo verso VR minore o uguale). MH accetta i passi in discesa con probabilità $r>0$.
- **sub-Metropolis su $(1,2)$**: per $r\in(1,2)$, $a_{\mathrm{FMC}}(r)=r-1<1=a_{\mathrm{MH}}(r)$.

### 1.2 Correzione di riga 186 `[DIMOSTRATO]`

Riga 186 del canone afferma:
> "il rapporto $\frac{\mathrm{VR}_k-\mathrm{VR}_i}{\mathrm{VR}_i}=\frac{\mathrm{VR}_k}{\mathrm{VR}_i}-1$ è esattamente la quota MH che, una volta clippata, dà la probabilità di accettazione standard $P_{\mathrm{MH}}=\min(\mathrm{VR}_k/\mathrm{VR}_i,1)$."

L'identità implicita è $\operatorname{clip}(r-1,0,1)=\min(r,1)$. **È falsa.** Controesempi diretti (Part 0 dello script):

- $r=0.8$: $a_{\mathrm{FMC}}=0.000$ vs $a_{\mathrm{MH}}=0.800$ (differenza $0.800$).
- $r=1.5$: $a_{\mathrm{FMC}}=0.500$ vs $a_{\mathrm{MH}}=1.000$ (differenza $0.500$).

La quantità clippata di FMC è $\operatorname{clip}(r-1,0,1)$, non $\min(r,1)$. Le due funzioni sono uguali **solo** per $r\ge2$ (e banalmente sul comportamento a saturazione). Chiamare "regola Metropolis-Hastings" la regola di cloning è quindi un errore di identificazione con conseguenze dinamiche (§3, §4).

---

## 2. Perché il framing "Gibbs a temperatura finita" non chiude

### 2.1 Il bilancio dettagliato è impossibile con transizioni uphill-only `[DIMOSTRATO]`

Sia $\pi$ una qualunque misura a **supporto pieno** sullo spazio delle configurazioni. Siano $x,y$ due configurazioni che differiscono per il tipo di un solo walker, con $\mathrm{VR}(y)>\mathrm{VR}(x)$ (cioè $y$ è "in salita" rispetto a $x$). Il kernel di cloning dà:
$$
K(x\to y) \;=\; \tfrac{1}{N-1}\,a_{\mathrm{FMC}}\!\Big(\tfrac{\mathrm{VR}(y)}{\mathrm{VR}(x)}\Big) \;>\;0,
\qquad
K(y\to x) \;=\; \tfrac{1}{N-1}\,a_{\mathrm{FMC}}\!\Big(\tfrac{\mathrm{VR}(x)}{\mathrm{VR}(y)}\Big) \;=\;0,
$$
perché $\mathrm{VR}(x)/\mathrm{VR}(y)<1\Rightarrow a_{\mathrm{FMC}}=0$. L'equazione di bilancio dettagliato
$$
\pi(x)\,K(x\to y)=\pi(y)\,K(y\to x)=0
$$
forza $\pi(x)=0$, contraddicendo il supporto pieno. **Nessuna misura a supporto pieno (in particolare nessuna Gibbs $\propto R^\alpha$ a $T$ finita) è reversibile per il kernel di cloning.** Le uniche misure compatibili col bilancio dettagliato sono concentrate sui massimi di VR. ∎

Questo falsifica direttamente la sezione "*Detailed balance*" del Teorema 2 (righe 269-283), dove si scrive $\Pr[y\to x]=0$ e poi si conclude comunque $\pi^*(x)/\pi^*(y)=\widehat R(x)^\alpha/\widehat R(y)^\alpha$. Il passo è invalido: con $\Pr[y\to x]=0$ il bilancio dettagliato dà $\pi^*(x)=0$, non un rapporto finito. Il canone sostituisce implicitamente $a_{\mathrm{FMC}}$ con $a_{\mathrm{MH}}$ a metà della derivazione — è esattamente l'errore di riga 186 che si propaga.

### 2.2 Il kernel di cloning non introduce tipi nuovi `[DIMOSTRATO]`

Il cloning **copia** lo stato/etichetta di un walker esistente. Quindi l'insieme dei tipi presenti nella popolazione è monotòno non-crescente nel tempo. Senza un operatore di mutazione (la perturbazione $\mathcal S$), la diversità può solo **diminuire**. Gli unici stati assorbenti del solo cloning sono quelli in cui **nessuna coppia presente ha $a_{\mathrm{FMC}}>0$**, cioè tutti i walker presenti hanno la stessa VR (genericamente: popolazione monomorfica). ∎

---

## 3. La distribuzione stazionaria corretta — modello di Moran/Wright-Fisher

Mappiamo lo swarm su una popolazione di $N$ individui con "tipi" = configurazioni/etichette, $M$ tick = $M$ generazioni, come in [`work/02_deep_dives/07_wright_fisher_mapping.md`](../../02_deep_dives/07_wright_fisher_mapping.md). Il cloning è il passo di **resampling**; VR gioca il ruolo della **fitness**.

### 3.1 Caso neutrale $\alpha=0$

**(a) VR esattamente costante ($\alpha=0,\beta=0$).** Allora $r\equiv1$, $a_{\mathrm{FMC}}(1)=0$: **nessun clone avviene mai**. Il kernel di cloning è l'identità; ogni configurazione è banalmente "stazionaria" (frozen). Non c'è drift e non c'è stazionaria non-degenere in senso ergodico. `[DIMOSTRATO]`

Verifica `[NUMERICO]` (Part B, controllo, $N=64$, VR$\equiv1$): $H(\text{fine})/H(0)=1.00000$ — eterozigosità invariata, popolazione congelata.

> **Nota che chiude un buco del deep dive 07.** Il deep dive registrava drift a "$\alpha=0,\beta=0$" con $q\approx-1$; ma con VR *esattamente* costante il kernel FMC è congelato. La spiegazione corretta: il drift osservato richiede **varianza di VR** (dal termine di distanza $\beta>0$ post-relativize, o dalla fitness stocastica per-tick indotta dal simulatore). Il "neutrale" che deriva è *stochastic-fitness neutral* (nessun bias sistematico ma VR fluttuante), non VR-costante.

**(b) VR fluttuante senza bias di tipo (il modello onesto di "Common Sense", $\alpha=0,\beta>0$).** Ogni tick ogni walker estrae una VR i.i.d. indipendente dal tipo. Il cloning diventa un **resampling neutrale** ≡ drift di Moran/Wright-Fisher. Predizioni WF:

- eterozigosità $H(t)=H(0)(1-1/N_e)^t$, cioè tasso di decadimento $\lambda(N)\sim c/N$ (esponente $-1$);
- tempo di fissazione $O(N)$ generazioni;
- probabilità di fissazione di un tipo = sua frequenza iniziale;
- stato terminale ($t\to\infty$): monomorfico, $b_{\text{eff}}\to1$.

Verifica `[NUMERICO]` (Part B, kernel FMC esatto, LogNormal$(0,0.5)$, tick = generazione):

| $N$ | $\lambda$ | $\lambda\cdot N$ |
|---:|---:|---:|
| 32 | 0.02098 | 0.671 |
| 64 | 0.01098 | 0.702 |
| 128 | 0.00534 | 0.683 |
| 256 | 0.00265 | 0.680 |

Fit power-law $\lambda\sim N^q$: $\boxed{q=-0.999}$ (WF predice $-1$). $\lambda\cdot N$ è quasi costante ($\approx0.68$) su una decade di $N$. Coerente con il $q=-0.948$ (errore 5.2%) del deep dive 07. `[NUMERICO]`

Tempo di fissazione (Part C, drift neutrale):

| $N$ | $T_{\text{fix}}$ (tick) |
|---:|---:|
| 32 | 57.8 |
| 64 | 129.4 |
| 128 | 267.7 |
| 256 | 520.4 |

Fit $T_{\text{fix}}\sim N^p$: $\boxed{p=1.056}$ (WF/Moran predicono $+1$, cioè $O(N)$ generazioni). `[NUMERICO]`

**Conclusione caso neutrale:** nessuna stazionaria non-degenere. Il processo drifta verso la fissazione in tempo $O(N)$; l'unica misura invariante è concentrata sugli stati monomorfici (con probabilità = frequenza iniziale del tipo). Questo è il comportamento di Moran neutrale, non una Gibbs uniforme. `[DIMOSTRATO + NUMERICO]`

### 3.2 Caso con selezione $\alpha>0$ — modello di Moran con selezione

Assegniamo VR fisse ai tipi (landscape congelato): tipo "fit" $A$ con $\mathrm{VR}_A=1+s$, tipo "wild" $B$ con $\mathrm{VR}_B=1$, $s>0$. Update asincrono a singolo walker (= passo elementare di Moran): scegli $i$ uniforme, partner $k$ uniforme $\ne i$, clona $i\to k$ con prob $a(\mathrm{VR}_k/\mathrm{VR}_i)$. Sul conteggio $j=\#A$ il processo è una catena birth-death con
$$
T^+_j=\tfrac{(N-j)}{N}\tfrac{j}{N-1}\,a(1+s),\qquad
T^-_j=\tfrac{j}{N}\tfrac{(N-j)}{N-1}\,a\!\big(\tfrac{1}{1+s}\big).
$$

**Sotto $a_{\mathrm{MH}}$** (Metropolis vero): $a_{\mathrm{MH}}(1+s)=1$, $a_{\mathrm{MH}}(1/(1+s))=1/(1+s)$, quindi $\gamma_j:=T^-_j/T^+_j=1/(1+s)$ costante. Questo è **esattamente** il processo di Moran con selezione; la probabilità di fissazione da 1 copia è la formula classica
$$
\rho_{\mathrm{MH}}=\frac{1-\gamma}{1-\gamma^{N}},\qquad \gamma=\frac{1}{1+s}.
$$
Reversibile, con bilancio di selezione a "temperatura finita". `[DIMOSTRATO]`

**Sotto $a_{\mathrm{FMC}}$** (regola reale): $a_{\mathrm{FMC}}(1/(1+s))=0$ ⇒ $T^-_j=0$. Il conteggio $\#A$ è **non-decrescente**: un walker di tipo $A$ non clona mai verso $B$ (partner con VR $\le$ la sua ⇒ $a=0$), mentre un $B$ appaiato con $A$ clona con prob $a_{\mathrm{FMC}}(1+s)=\min(s,1)>0$. Catena finita monotòna assorbente a $j=N$ ⇒ **il tipo fit si fissa con probabilità 1** da qualsiasi $j\ge1$. `[DIMOSTRATO]`

Verifica `[NUMERICO]` (Part A, 20000 run, start = 1 copia del tipo fit):

| $s$ | $N$ | FMC (esatto) | MH | Moran teoria | neutrale $1/N$ |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 32 | **1.0000** | 0.0956 | 0.0954 | 0.0312 |
| 0.10 | 64 | **1.0000** | 0.0903 | 0.0911 | 0.0156 |
| 0.10 | 128 | **1.0000** | 0.0882 | 0.0909 | 0.0078 |
| 0.50 | 32 | **1.0000** | 0.3318 | 0.3333 | 0.0312 |
| 0.50 | 64 | **1.0000** | 0.3399 | 0.3333 | 0.0156 |
| 0.50 | 128 | **1.0000** | 0.3372 | 0.3333 | 0.0078 |
| 1.00 | 32 | **1.0000** | 0.5059 | 0.5000 | 0.0312 |
| 1.00 | 64 | **1.0000** | 0.4997 | 0.5000 | 0.0156 |
| 1.00 | 128 | **1.0000** | 0.4995 | 0.5000 | 0.0078 |

Lettura:
- $a_{\mathrm{MH}}$ riproduce la formula di Moran con selezione fino a 3 cifre ⇒ è davvero MH, reversibile, con bilancio di selezione (il framing "Gibbs" del canone descrive $a_{\mathrm{MH}}$, **non** $a_{\mathrm{FMC}}$).
- $a_{\mathrm{FMC}}$ dà fissazione $=1.0000$ per ogni $s>0$ ⇒ dinamica **assorbente uphill-only**, nessun bilancio a $T$ finita. La differenza $a_{\mathrm{FMC}}$ vs $a_{\mathrm{MH}}$ è quantitativamente enorme (es. $s=0.5,N=64$: $1.000$ vs $0.340$). `[NUMERICO]`

> **Nota sul landscape stocastico realistico.** Con VR fisse per tipo il tipo argmax si fissa con prob 1 (limite $\alpha\to\infty$). Nel FMC reale la fitness fluttua per-tick (il simulatore perturba lo stato): il "fittest" istantaneo cambia, e $\alpha$ modula l'intensità della selezione via $s\approx\alpha\cdot\mathrm{Var}(\log\widehat R)$ `[SKETCH, euristica weak-selection]`. Ma il segno del risultato non cambia: cloning-only ⇒ perdita monotòna di diversità ⇒ fissazione, $b_{\text{eff}}\to1$. Questo è esattamente ciò che predice l'asintotica di Congettura A/Teorema 2 ("palmera", $b_{\text{eff}}\to1$ per ogni $\alpha>0$) e **contraddice** una Gibbs a $T$ finita (che manterrebbe più tipi a frequenza finita).

---

## 4. Il teorema corretto (forma pubblicabile)

### Teorema 2′ (Fissazione del kernel di cloning; assenza di equilibrio di Gibbs)

**Setup.** Popolazione di $N$ walker, ciascuno con un tipo (configurazione/etichetta) in un insieme finito. Il **kernel di cloning** $\mathcal C$ agisce walker per walker: ciascun $i$ estrae un partner $k$ uniforme ($k\ne i$) e adotta il tipo di $k$ con probabilità $a_{\mathrm{FMC}}(\mathrm{VR}_k/\mathrm{VR}_i)=\operatorname{clip}(\mathrm{VR}_k/\mathrm{VR}_i-1,0,1)$. Nessuna mutazione (nessuna perturbazione $\mathcal S$).

**Ipotesi.** (H1) VR è funzione deterministica positiva del tipo (landscape congelato), oppure — caso neutrale — i.i.d. senza bias di tipo. (H2) Nel caso con selezione, VR ha massimo unico.

**Enunciato.**

1. *(Non-espansività del supporto)* `[DIMOSTRATO]` $\mathcal C$ non introduce tipi assenti; l'insieme dei tipi presenti è non-crescente. Gli stati assorbenti sono quelli in cui tutti i walker hanno VR uguale (genericamente: monomorfici).

2. *(Non reversibilità)* `[DIMOSTRATO]` Per ogni misura a supporto pieno $\pi$ e ogni coppia $x,y$ con $\mathrm{VR}(y)>\mathrm{VR}(x)$ si ha $K(x\to y)>0=K(y\to x)$; il bilancio dettagliato forza $\pi(x)=0$. Dunque **nessuna Gibbs $\pi^*\propto R^\alpha\rho^{-\beta}$ a temperatura finita è invariante-reversibile**.

3. *(Fissazione con selezione)* `[DIMOSTRATO]` Sotto (H2), il conteggio del tipo argmax-VR è non-decrescente e raggiunge $N$ con probabilità 1. L'unica distribuzione stazionaria è la **massa puntuale** sulla configurazione monomorfica argmax; $b_{\text{eff}}\to1$. (È il limite $\alpha\to\infty$, non un $\alpha$ finito.)

4. *(Drift nel caso neutrale)* `[DIMOSTRATO + NUMERICO]` Con VR i.i.d. senza bias, $\mathcal C$ è un resampling neutrale di Moran/Wright-Fisher: l'eterozigosità decade con tasso $\lambda(N)\sim c/N$ (misurato $q=-0.999$), il tempo di fissazione è $O(N)$ generazioni (misurato $p=1.056$), e ogni tipo si fissa con probabilità pari alla sua frequenza iniziale. Con VR **esattamente** costante, $\mathcal C=\mathrm{Id}$ (frozen).

5. *(Ripristino di una legge non-degenere — la mutazione è necessaria)* `[SKETCH]` Una distribuzione stazionaria non-degenere del processo completo $\mathcal S\circ\mathcal C$ esiste solo se $\mathcal S$ agisce da **mutazione** (immissione di nuovi tipi). L'equilibrio è allora un **mutation-selection-drift balance** (Wright-Fisher con selezione + mutazione), la cui forma dipende dal tasso di mutazione $\mu$ e da $s\propto\alpha\,\mathrm{Var}(\log\widehat R)$. **Non** coincide con $\pi^*\propto R^\alpha$ salvo limiti degeneri; $\alpha$ è un coefficiente di **intensità di selezione**, non un'inversa-temperatura pulita.

**Prova.** Punti 1-3: §2.1-2.2, §3.2 (argomenti finiti-Markov standard: catena monotòna assorbente ⇒ assorbimento q.c.). Punto 4: mapping di Moran/WF (deep dive 07) + verifica numerica §3.1. Punto 5: asserito, non dimostrato in forma chiusa — vedi buchi §5. ∎

**Evidenza numerica** (script `w31_stationary_check.py`, seed 20260709): Part A (fissazione: FMC $=1.0$ vs MH $=$ formula di Moran), Part B ($q=-0.999$, controllo frozen), Part C ($p=1.056$).

### Interpretazione fisica corretta di $\alpha$

Il canone (righe 285-300) chiama $\alpha$ "temperatura inversa" con equilibrio di Boltzmann. La lettura corretta:

| Claim canone (Thm 2) | Correzione |
|---|---|
| $a$ = Metropolis $\min(r,1)$ | $a_{\mathrm{FMC}}=\operatorname{clip}(r-1,0,1)$, uphill-only |
| Reversibile, detailed balance | Non reversibile (transizioni solo in salita) |
| $\pi^*\propto R^\alpha\rho^{-\beta}$ (Gibbs, $T$ finita) | Cloning-only: massa puntuale (fissazione), $b_{\text{eff}}\to1$ |
| $\alpha$ = temperatura inversa | $\alpha$ = intensità di selezione ($s\propto\alpha\,\mathrm{Var}\log\widehat R$) |
| $\alpha=0\Rightarrow\pi^*$ uniforme | $\alpha=0$: drift neutrale di Moran/WF → comunque fissazione in $O(N)$ |
| Equilibrio termodinamico | Equilibrio mutation-selection-drift (serve $\mathcal S$ come mutazione) |

---

## 5. Buchi della mia derivazione (onestà)

- **Punto 5 del teorema è SKETCH, non dimostrato.** Non ho derivato la forma chiusa della stazionaria di $\mathcal S\circ\mathcal C$ con mutazione. So che *non* è la Gibbs $\propto R^\alpha$ e che è un equilibrio mutation-selection-drift; la densità stazionaria esatta (à la Wright 1931 / diffusione) resta da caratterizzare. È il vero "teorema mancante".
- **La relazione $s\approx\alpha\,\mathrm{Var}(\log\widehat R)$ è un'euristica weak-selection**, non provata. Va derivata (o falsificata) con uno sweep in $\alpha$ contro $s$ misurato.
- **Il modello di selezione con VR fisse per tipo** è un'idealizzazione: il FMC reale ha fitness stocastica per-tick. Ho argomentato che il *segno* (fissazione, $b_{\text{eff}}\to1$) è robusto, ma la probabilità di fissazione del "fittest medio" sotto fitness fluttuante non è la formula di Moran deterministica — richiede la teoria della selezione in ambiente fluttuante (non svolta qui).
- **Il termine $\rho^{-\beta}$ (densità) non è modellato esplicitamente** nel toy: l'ho assorbito nel caso "VR fluttuante" (§3.1b). Un modello spaziale esplicito con repulsione di densità darebbe la correzione anti-collasso del Teorema 3, ma non cambia la conclusione sul supporto (cloning-only ⇒ perdita di diversità).

---

## 6. Testo esatto della correzione da applicare a MATH_CANON

### 6.1 Riga 186 — sostituire l'intera frase

> **Attenzione (correzione W31, 2026-07-09):** la funzione di accettazione effettiva del cloning è $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)=\min(\max(r-1,0),1)$ con $r=\mathrm{VR}^{(k)}/\mathrm{VR}^{(i)}$. **Non** è Metropolis-Hastings $\min(r,1)$ né Barker $r/(1+r)$: coincide con $\min(r,1)$ solo per $r\ge2$. È **uphill-only** ($a=0$ per $r\le1$: nessun passo verso VR minore o uguale) e **sub-Metropolis su $(1,2)$** ($a=r-1<1$). L'identità precedentemente scritta $\operatorname{clip}(r-1)=\min(r,1)$ è algebricamente falsa (controesempi: $r=0.8\Rightarrow0$ vs $0.8$; $r=1.5\Rightarrow0.5$ vs $1$). È una regola di **selezione direzionale**, non un proposal MH reversibile. Vedi [`work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md`](../work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md).

### 6.2 Righe 253-301 — declassare il Teorema 2 e sostituirlo

Anteporre al Teorema 2 il seguente riquadro di ritrattazione, e sostituire l'enunciato/prova con il Teorema 2′ (§4 di W31):

> ### Teorema 2 — ~~Detailed balance ed equilibrio di Gibbs~~ **RITRATTATO (W31, 2026-07-09)**
>
> **STATO: FALSIFICATO.** L'enunciato $\pi^*\propto R^\alpha\rho^{-\beta}$ come equilibrio di Gibbs Metropolis-Hastings a temperatura finita è **errato**. Tre falle indipendenti (dimostrate + numeriche in [`W31_stazionaria_corretta.md`](../work/14_night_2026-07-09/wave3_validation/W31_stazionaria_corretta.md)):
> 1. L'accettazione è $a_{\mathrm{FMC}}(r)=\operatorname{clip}(r-1,0,1)$, non $\min(r,1)$ (vedi riga 186 corretta).
> 2. Il kernel è uphill-only ⇒ non reversibile: per $x,y$ con $R(y)>R(x)$ si ha $K(y\to x)=0$, quindi il bilancio dettagliato forza $\pi^*(x)=0$. Nessuna Gibbs a supporto pieno è invariante-reversibile.
> 3. Il cloning-only converge a **massa puntuale** (fissazione sul tipo argmax-VR): $b_{\text{eff}}\to1$, coerente con l'asintotica di Congettura A e **incompatibile** con una Gibbs a $T$ finita.
>
> **Sostituito da → Teorema 2′ (Fissazione; assenza di Gibbs).** Il kernel di cloning è un **resampling di Moran/Wright-Fisher**: neutrale ($\alpha=0$) → drift verso fissazione in $O(N)$ generazioni (esponente eterozigosità $q=-0.999$, tempo di fissazione $p=1.056$, verifiche numeriche); con selezione ($\alpha>0$) → fissazione del fittest con prob 1 (limite $\alpha\to\infty$, non $\alpha$ finito). $\alpha$ è **intensità di selezione**, non temperatura inversa. Una stazionaria non-degenere esiste **solo** aggiungendo $\mathcal S$ come mutazione → equilibrio **mutation-selection-drift** (non Gibbs; forma chiusa: open problem). Enunciato completo, ipotesi e prova in W31 §4.
>
> **Nota per deep dive 01 §4:** stessa ritrattazione — la derivazione $\pi^*\propto R^\alpha$ (righe 140-150) fa lo stesso salto invalido (usa $\Pr[y\to x]=0$ e poi conclude un rapporto finito).

---

*Fine W31. Script: [`w31_stationary_check.py`](w31_stationary_check.py) — 3 parti, ~26 s, seed 20260709, numpy 2.2.6.*
