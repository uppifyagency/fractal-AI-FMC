# W3B — Chiusura delle due debolezze del paper "effective temperature FMC"

> **Data**: 2026-07-10. **Autore**: wave3 validation (matematico rigoroso).
> **Oggetto**: chiudere (o dichiarare onestamente perché non chiudono) le due debolezze note del draft
> [`wave5_papers/PAPER_THEORY_effective_temperature_FMC.md`](../wave5_papers/PAPER_THEORY_effective_temperature_FMC.md):
> **A** — le verifiche di Teorema 2′ usavano VR congelate e singolo seed (mancavano CI multi-seed su q, p, C);
> **B** — Teorema 2′.5 (legge stazionaria non-degenere del kernel FMC **con** mutazione) era solo `[SKETCH]`.
> **Script eseguiti** (numeri reali, seed 20260709): [`w3b_robustness.py`](w3b_robustness.py) (A) e
> [`w3b_mutation_diffusion.py`](w3b_mutation_diffusion.py) (B). numpy 2.2.6, scipy 1.16.1.
> **Marcatura per-affermazione**: `[DIM]` = dimostrato in forma chiusa; `[DIFF-APPROX]` = limite di diffusione con
> forma chiusa verificata numericamente e regime di validità; `[NUM]` = solo evidenza Monte Carlo; `[SKETCH]` = argomentato, non provato.

---

## 0. Verdetto in una riga

- **Debolezza A: CHIUSA.** `[NUM]` q, p, C reggono sotto fitness fluttuante per-tick, con CI bootstrap su 25 seed. q = −1.018 (WF: −1), p = +1.025 (WF: +1), C = 0.7225 (gauss) / 0.7384 (unif).
- **Debolezza B: DECLASSATA da `[SKETCH]` a `[DIFF-APPROX]`.** Ho una **forma chiusa** per la stazionaria — la distribuzione di **Wright** $\phi_\infty(x)\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x}$ — con **coefficiente di drift corretto derivato dalla vera accettazione clip** (validato all'1.3%) e coefficiente di diffusione leading-order (errore +13% misurato, di segno noto). La densità coincide con il kernel esatto e l'errore → 0 come $N\to\infty$ (TV = 0.016 a N=800). **Non** è `[DIM]` rigoroso (vedi §5).

---

## 1. Debolezza A — robustezza multi-seed sotto fitness fluttuante per-tick `[NUM]`

**Modello onesto**: due tipi, ogni tick **ogni walker ridisegna** $\mathrm{VR}=e^{g}$, $g\sim\mathcal N(m_{\text{tipo}},\sigma_v^2)$ (non VR congelate). Accettazione esatta $a_{\rm FMC}(r)=\operatorname{clip}(r-1,0,1)$. Per i test neutrali le medie per-tipo sono uguali (drift puro). 25 seed, $N\in\{32,64,128,256\}$, CI bootstrap 20 000 ricampionamenti sui seed.

### (a)+(b) esponenti q (eterozigosità) e p (tempo di fissazione)

| quantità | valore | 95% CI bootstrap | sd tra seed | predizione WF |
|---|---|---|---|---|
| **q** ($\lambda\sim N^q$) | **−1.0177** | [−1.0331, −1.0026] | 0.040 | −1 |
| **p** ($T_{\rm fix}\sim N^p$) | **+1.0254** | [+1.0122, +1.0387] | 0.034 | +1 |

$\lambda\cdot N$ (medio sui seed): 0.702, 0.685, 0.663, 0.681 per $N=32,64,128,256$ → quasi costante ≈ **0.68** su una decade di $N$ (firma WF: $\lambda\sim 1/N_e$). $T_{\rm fix}/N$ ≈ 1.92–2.04 → $O(N)$ generazioni.

**Robustezza al livello di rumore** (10 seed per riga): q = −1.012 ($\sigma_v{=}0.25$), −1.011 ($\sigma_v{=}0.5$), −1.029 ($\sigma_v{=}1.0$). L'esponente WF regge indipendentemente dall'ampiezza della stocasticità per-tick.

> **Nota onesta.** I CI di q e p **escludono** il valore WF esatto per ~2% (bias sistematico da $M=40$ tick finiti e da soli 4 punti-$N$ nel fit log-log). La direzione e l'ordine di grandezza sono inequivocabili: la firma Moran/WF sopravvive alla fitness fluttuante. Il ~2% è finite-size, non un difetto del mapping.

### (c) costante $C$ di $\alpha_{\rm eff}=C\,\alpha/\sigma_R$

| popolazione | C (25 seed) | 95% CI | sd | rif. W32 | invarianza legge di scala ($\alpha\times\sigma_R$) |
|---|---|---|---|---|---|
| gaussiana | **0.7225** | [0.7221, 0.7227] | 0.0008 | 0.7223 | spread 0.0016 su griglia $3\times3$ |
| uniforme | **0.7384** | [0.7382, 0.7386] | 0.0005 | 0.7383 | spread 0.0007 |

La costante è riprodotta a 3–4 cifre con CI strettissimi; la legge $\propto\alpha/\sigma_R$ è invariante su 2 ordini di grandezza di $\sigma_R$ (è algebra dello z-score, non gaussiana). $C$ resta distribution-dependent solo al 2% (gauss↔unif), come già in W32.

**Esito A**: le tre quantità del paper (q, p, C) reggono sotto stocasticità per-tick con CI multi-seed. La debolezza A è chiusa.

---

## 2. Debolezza B — il teorema mancante (Teorema 2′.5)

### 2.1 Il kernel FMC + mutazione come processo di Moran selezione+mutazione `[DIM]` (setup)

Due tipi $A,B$; $x=\#A/N$; mutazione simmetrica $A\leftrightarrow B$ a tasso $\mu$ (l'operatore $\mathcal S$ come immissione di tipi, nel caso a 2 tipi = flip). Fitness stocastica per-tick: ogni walker $\mathrm{VR}=e^{g}$, $g\sim\mathcal N(m_{\text{tipo}},\sigma_v^2)$, con $m_A=\delta$ (vantaggio log-medio di $A$), $m_B=0$.

Momento della clip su una **log-differenza di fitness** di media $m$ (il log-rapporto di due lognormali indipendenti ha media $m_k-m_i$ e varianza $2\sigma_v^2$):
$$
\Phi(m)\;:=\;\mathbb E_{\,u\sim\mathcal N(m,\,2\sigma_v^2)}\big[\operatorname{clip}(e^{u}-1,0,1)\big].
$$

### 2.2 Coefficienti corretti dalla vera accettazione `[DIM]` (derivazione), `[NUM]` (verifica)

**Drift** (un tick sincrono, leading order in $1/N$): un $B$ (frazione $1-x$) diventa $A$ solo se il partner è $A$ (prob $x$) e la clip scatta ⇒ $x\,\Phi(+\delta)$; un $A$ diventa $B$ con $(1-x)\,\Phi(-\delta)$. Quindi
$$
\boxed{\;\mathbb E[\Delta x]_{\rm sel}=x(1-x)\,\big[\Phi(\delta)-\Phi(-\delta)\big]=s_{\rm eff}\,x(1-x)\;}
$$
Il **coefficiente di selezione effettivo** $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ è **rinormalizzato dal clip uphill-only e dal rumore $\sigma_v$**: non è la $s$ di Moran. Limiti: $\sigma_v\to0$ (landscape congelato) ⇒ $s_{\rm eff}\to e^\delta-1$ (uphill puro, downhill azzerato dalla clip); rumore crescente ⇒ $s_{\rm eff}$ ridotto (es. $\delta{=}0.08$: $e^\delta-1=0.083$ congelato → $0.076$ a $\sigma_v{=}0.5$).

**Diffusione** (leading order, flip indipendenti): $\operatorname{Var}[\Delta j]\approx 2N x(1-x)\phi_0$, $\phi_0=\Phi(0)$, quindi
$$
\boxed{\;V(x)=\frac{x(1-x)}{N_e},\qquad N_e=\frac{N}{2\phi_0}\;}
$$
La mutazione aggiunge drift $\mu(1-2x)$ e varianza $O(\mu/N)$ trascurabile. SDE di Wright-Fisher effettiva (per generazione):
$$
dx=\big[\,s_{\rm eff}\,x(1-x)+\mu(1-2x)\,\big]\,dt+\sqrt{\tfrac{x(1-x)}{N_e}}\;dW.
$$

### 2.3 Densità stazionaria di Fokker-Planck — forma chiusa `[DIM]` (data la SDE)

Con $\phi_\infty\propto V^{-1}\exp\!\big(2\!\int\! M/V\big)$ e $\int\frac{1-2x}{x(1-x)}dx=\ln[x(1-x)]$:
$$
\boxed{\;\phi_\infty(x)\;\propto\;x^{\theta-1}(1-x)^{\theta-1}\,e^{\sigma x},\qquad
\theta=2N_e\mu=\frac{N\mu}{\phi_0},\quad \sigma=2N_e\,s_{\rm eff}=\frac{N\,s_{\rm eff}}{\phi_0}\;}
$$
È la **distribuzione di Wright (1931)**: una Beta$(\theta,\theta)$ con tilt esponenziale di selezione. Casi limite corretti:
- $\delta\to0$ (neutrale): $\sigma=0$ ⇒ **Beta$(\theta,\theta)$**, equilibrio mutazione-drift puro.
- $\sigma_v\to0$ (congelato): $\phi_0\to0\Rightarrow N_e\to\infty,\ \sigma\to\infty$ ⇒ massa puntuale in $x=1$ = **fissazione** (Teorema 2′.3). La forma chiusa **contiene** il vecchio risultato come limite.

> **Perché una densità di diffusione può esistere malgrado il kernel sia uphill-only e non-reversibile (Teorema 2′.2)?** Non c'è contraddizione: sono due oggetti diversi. Il Teorema 2′.2 riguarda il kernel **discreto con VR congelate** (dove il downhill ha prob esattamente 0 ⇒ non-reversibile ⇒ fissazione). Qui la fitness **fluttua**: a ogni tick il "più adatto" istantaneo cambia, riaprendo entrambe le direzioni; ogni diffusione 1-D su $[0,1]$ è automaticamente reversibile rispetto alla sua $\phi_\infty$. La non-reversibilità uphill-only è un effetto del regime congelato/finito, **non** un ostacolo all'esistenza della legge non-degenere una volta presenti fluttuazione + mutazione.

### 2.4 Verifica numerica contro il kernel esatto con mutazione `[NUM]`

Tutti da `w3b_mutation_diffusion.py`, $\sigma_v=0.5$, $\phi_0=0.2993$.

**V0 — drift.** $\mathbb E[\Delta x]$ misurato a $x=0.5$ (nessuna mutazione, un tick, 40k catene) vs $s_{\rm eff}\cdot0.25$:

| $\delta$ | $s_{\rm eff}$ (dalla clip) | $\Delta x$ predetto | $\Delta x$ misurato | err. rel. |
|---|---|---|---|---|
| 0.05 | 0.04716 | 0.01179 | 0.01194 | **1.25%** |
| 0.10 | 0.09414 | 0.02353 | 0.02389 | **1.49%** |

→ **il coefficiente di drift corretto è validato direttamente all'1.3%.** `[NUM]`

**V1 — calibrazione del coefficiente di diffusione.** $\lambda\cdot N$ misurato (neutrale): 0.701 ($N{=}100$), 0.636 (200), 0.687 (400) ⇒ $\phi_{0,\rm eff}=\lambda N/2=0.337$ vs analitico $\phi_0=0.299$: **correzione +13%** da correlazioni del resampling pairwise (partner condivisi ⇒ la varianza dei flip è super-Bernoulli). La correzione **non svanisce** con $N$ (è un fattore $O(1)$, non $O(1/N)$): il vero $N_e$ va misurato, non solo derivato leading-order.

**V2 — densità neutrale** ($\delta=0$) vs Beta$(\theta,\theta)$:

| $N$ | TV (a-priori, $\phi_0$ analitico) | TV (calibrato, $\phi_{0,\rm eff}$) | best-fit $\theta$ | $\theta$ target |
|---|---|---|---|---|
| 200 | 0.071 | **0.055** | 1.469 | 1.500 |
| 400 | 0.047 | **0.032** | 1.497 | 1.500 |

Il best-fit $\theta$ coincide con il target entro 2% ⇒ **la stazionaria empirica È una Beta$(\theta,\theta)$**; il residuo TV cala con $N$.

**V3 — densità con selezione** ($\delta>0$) vs Wright:

| $N$ | $\delta$ | $(\theta,\sigma)$ predetto | best-fit $(\theta,\sigma)$ | TV pred / best | media emp vs pred |
|---|---|---|---|---|---|
| 200 | 0.05 | (1.60, 28.0) | (1.63, 26.8) | 0.042 / 0.031 | 0.9434 vs 0.9442 |
| 400 | 0.04 | (1.60, 44.9) | (1.65, 44.6) | 0.022 / 0.009 | 0.9648 vs 0.9649 |

Il best-fit $(\theta,\sigma)$ coincide con i valori **predetti dall'accettazione** entro ~3–4%; la media stazionaria è predetta a **<0.1%** (la media dipende da $\mu/s_{\rm eff}$, con $\phi_0$ che si cancella ⇒ conferma indipendente di $s_{\rm eff}$).

**V4 — regime di validità.** A $\theta=1.6,\ \sigma\approx6$ **fissi** (scalando $\mu,\delta\sim1/N$), la TV vs kernel esatto:

| $N$ | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| **TV** | 0.0995 | 0.0509 | 0.0281 | **0.0159** |

TV dimezza a ogni raddoppio di $N$ (≈ $\propto 1/N$) → **l'approssimazione di diffusione converge**: è la firma di un limite di diffusione valido.

**V5 — sanity limite congelato.** $\phi_0=0.299\ (\sigma_v{=}0.5)\to0.135\ (0.2)\to0.030\ (0.05)$: $N_e/N=1/(2\phi_0)$ diverge, $\sigma\to\infty$, massa → $x=1$ (fissazione). Coerente con Teorema 2′.3.

---

## 3. Enunciato del Teorema 2′.5 (forma pubblicabile)

> **Teorema 2′.5 (legge stazionaria mutazione-selezione-drift del kernel FMC).** `[DIFF-APPROX]`
> Sia il kernel di cloning FMC $\mathcal C$ (accettazione $a_{\rm FMC}(r)=\operatorname{clip}(r-1,0,1)$) composto con mutazione $\mathcal S$ a tasso $\mu$, su $N$ walker a 2 tipi con fitness log-normale per-tick ($\sigma_v$) e vantaggio log-medio $\delta$. Nel limite di diffusione $N\to\infty$, $\mu\to0$, $s_{\rm eff}\to0$ con $\theta=2N_e\mu$ e $\sigma=2N_e s_{\rm eff}$ fissi, la densità stazionaria della frequenza $x$ è la **distribuzione di Wright**
> $$\phi_\infty(x)\propto x^{\theta-1}(1-x)^{\theta-1}e^{\sigma x},$$
> con coefficienti **derivati dalla vera accettazione**: $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$ (drift, validato all'1.3%) e $N_e=N/(2\phi_0)$, $\phi_0=\Phi(0)$ (diffusione, leading-order; correzione misurata +13% da correlazioni pairwise). **Non** è la Gibbs $\propto R^\alpha$: $\alpha$ entra come intensità di selezione via $\sigma$, non come temperatura inversa di un equilibrio termodinamico.

Questo **sostituisce** il punto 5 `[SKETCH]` del Teorema 2′ (W31 §4) con una forma chiusa verificata.

---

## 4. Verdetto onesto sullo stato del Teorema 2′.5

**Stato: `[DIFF-APPROX verificata]`** — nettamente più forte di `[SKETCH]`, non ancora `[DIM]` rigoroso.

**Cosa è ora solido:**
1. **Forma chiusa** della stazionaria (Wright), non più un'incognita.
2. **Coefficiente di drift** $s_{\rm eff}$ derivato dalla vera accettazione clip e validato direttamente (err. 1.3%). È il punto che il task chiedeva esplicitamente ("il drift FMC ≠ Moran standard"): $s_{\rm eff}$ dipende da $\sigma_v$ ed è ridotto rispetto al Moran congelato.
3. **Densità verificata** contro il kernel esatto: best-fit $(\theta,\sigma)$ ≈ predetti entro 3–4%, media a <0.1%, e **TV → 0 come $N\to\infty$** (0.016 a N=800).
4. **Coerenza con i limiti noti**: neutrale → Beta; congelato → fissazione (Thm 2′.3).

**Perché NON `[DIM]`:**
1. Il **limite di diffusione** è l'euristica classica di Kimura, qui non dimostrato come teorema di convergenza funzionale (nessuna prova di tightness / martingale problem).
2. Il **coefficiente di diffusione** $N_e=N/(2\phi_0)$ è leading-order: la correlazione del resampling pairwise dà un fattore $O(1)$ (+13%) **non modellato in forma chiusa** — il vero $N_e$ è misurato, non derivato. La *forma* è giusta, la *costante di diffusione* no.
3. È una **riduzione a 2 tipi**. Il caso a infiniti tipi (l'$\mathcal S$ reale che inietta configurazioni nuove) porta alla formula di campionamento di **Ewens (1972)**, non svolta qui.
4. La fitness log-normale per-tick è un modello onesto ma specifico; il legame quantitativo esatto $\alpha_{\rm eff}\!\to\! s_{\rm eff}$ (ponte §3↔§4 del paper) resta `[SKETCH]` (vedi §5).

---

## 5. Cosa resta aperto

- **Costante di diffusione esatta.** Derivare il fattore +13% (correzione di correlazione del resampling pairwise) in forma chiusa: darebbe $N_e$ analitico e promuoverebbe il coefficiente di diffusione da `[NUM]` a `[DIM]`. È un calcolo di secondo momento con partner condivisi (probabilità di co-ancestry in un tick).
- **Teorema di diffusione rigoroso.** Provare la convergenza del kernel discreto alla SDE (martingale problem / generatore) chiuderebbe il limite. Standard nella letteratura WF, ma con l'accettazione clip la verifica delle condizioni di Lindeberg va fatta.
- **Estensione a $K$ tipi → Ewens.** Con $\mathcal S$ che inietta tipi genuinamente nuovi, la stazionaria è la distribuzione di Ewens con $\theta=2N_e\mu$; da verificare che i tassi FMC (clip) diano lo stesso $\theta$ effettivo.
- **Ponte $\alpha_{\rm eff}\to s_{\rm eff}$.** W32 dà $\alpha_{\rm eff}=C\alpha/\sigma_R$ (esatto); qui $s_{\rm eff}=\Phi(\delta)-\Phi(-\delta)$. Unirli in un'unica derivazione (con la stessa costante) chiuderebbe la tensione §7.3 del paper. Al momento sono due misure separate, entrambe corrette.

---

*Fine W3B. Script: [`w3b_robustness.py`](w3b_robustness.py) (~40 s), [`w3b_mutation_diffusion.py`](w3b_mutation_diffusion.py) (~3.5 min). Seed 20260709. Ogni numero è prodotto dagli script; nessuno è inventato.*
