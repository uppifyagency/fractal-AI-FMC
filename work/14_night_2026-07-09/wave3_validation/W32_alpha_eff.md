# W3.2 — La "temperatura inversa effettiva" $\alpha_{\rm eff}$ di `relativize`

**Data:** 2026-07-09 · **Stato:** DIMOSTRATO (forma chiusa pointwise) + NUMERICO (costante di popolazione e legge di scala) · **Verdetto:** l'intuizione **regge**.

Script di verifica: [`w32_alpha_eff_check.py`](w32_alpha_eff_check.py) — eseguito, numeri sotto sono reali.

---

## 1. Formula esatta di `relativize` (verifica canone ↔ codice)

**Canone** — [`docs/MATH_CANON.md:116-126`](../../../docs/MATH_CANON.md) (Def. 2):

$$
z^{(i)} = \frac{r^{(i)} - \mu}{\sigma + \varepsilon},\ \varepsilon=10^{-10}
\qquad
\widehat{R}(z) = \begin{cases} e^{z} & z \le 0 \\ 1 + \log(1+z) & z > 0. \end{cases}
$$

**Codice di riferimento** — [`fmc-core/src/fmc/core.py:33-58`](../../../fmc-core/src/fmc/core.py) e [`repos/FractalAI_old/fractalai/swarm.py:16-23`](../../../repos/FractalAI_old/fractalai/swarm.py) sono **bit-identici** fra loro e implementano esattamente i due rami del boxed sopra.

**Divergenza canone ↔ codice (unica, minore, ma con significato al bordo):** il canone regolarizza *sempre* con $\sigma+\varepsilon$; il codice usa invece un ramo netto `if std == 0: return ones` e altrimenti divide per `std` **senza** $\varepsilon$ ([`core.py:50-53`](../../../fmc-core/src/fmc/core.py)). Conseguenza qualitativa nel limite $\sigma_R\to 0$: il codice **azzera** la pressione (VR tutti = 1 → nessuna selezione), mentre canone e forma chiusa danno pressione $\to\infty$. Irrilevante per $\sigma_R\gg\varepsilon$ (il regime operativo), ma va registrato: al collasso totale della popolazione il codice e il continuo non coincidono.

Per la derivazione uso la forma senza $\varepsilon$ (regime $\sigma_R\gg 10^{-10}$).

---

## 2. Definizione operativa di $\alpha_{\rm eff}$

Isolo il canale reward: tengo fisso il termine di distanza (equivale a $\beta=0$ o a $\widehat D$ costante). Allora

$$
\log \mathrm{VR} = \alpha\,\log \widehat{R}(z) + \text{const}, \qquad z=\frac{R-\mu_R}{\sigma_R}.
$$

Un selettore **Boltzmann ingenuo** userebbe $\mathrm{VR}\propto e^{\alpha_B R}$, cioè $\log\mathrm{VR}=\alpha_B R+\text{const}$: la **pressione selettiva** è $\partial_R\log\mathrm{VR}=\alpha_B$, costante, con unità $[\text{reward}]^{-1}$ = temperatura inversa. Definisco quindi la pressione selettiva effettiva di FMC nello **stesso modo dimensionale**:

$$
\boxed{\ \alpha_{\rm eff}(R) := \frac{\partial \log \mathrm{VR}}{\partial R}\ }
\qquad [\alpha_{\rm eff}] = [\text{reward}]^{-1}.
$$

Nota dimensionale cruciale: $\alpha$ in FMC è un **esponente adimensionale**; $\alpha_{\rm eff}$ ha invece unità $1/\text{reward}$. La differenza è portata da $\sigma_R$, che ha unità di reward. È *dimensionalmente inevitabile* che $\alpha_{\rm eff}$ dipenda da $\sigma_R$.

---

## 3. Forma chiusa — **DIMOSTRATO** (sympy)

Regola della catena: $\alpha_{\rm eff}(R)=\alpha\cdot\dfrac{d\log\widehat R}{dz}\cdot\dfrac{dz}{dR}$, con $\dfrac{dz}{dR}=\dfrac{1}{\sigma_R}$.

sympy ([`w32_sympy_deriv.py`](w32_sympy_deriv.py), eseguito) dà i due rami:

| ramo | $\dfrac{d\log\widehat R}{dz}$ |
|---|---|
| $z\le 0$ | $1$ |
| $z>0$ | $\dfrac{1}{(1+z)\,[1+\log(1+z)]}$ |

Da cui la **forma chiusa pointwise esatta**:

$$
\boxed{\ \alpha_{\rm eff}(z;\alpha,\sigma_R)=\frac{\alpha}{\sigma_R}\, g(z),
\qquad
g(z)=\begin{cases}1 & z\le 0\\[4pt] \dfrac{1}{(1+z)\,[1+\log(1+z)]} & z>0\end{cases}}
$$

Fatti chiave (tutti dimostrati simbolicamente):

1. **La temperatura inversa è $\alpha/\sigma_R$, non $\alpha$.** Il fattore-forma $g(z)\le 1$ modula solo la coda destra.
2. **Sotto la media la pressione è esatta e costante:** per ogni $z\le0$, $\alpha_{\rm eff}=\alpha/\sigma_R$. Tutti i walker peggiori-della-media sentono la stessa pressione.
3. **Sopra la media la pressione decade:** $g(z)\to 0$ per $z\to+\infty$ (compressione logaritmica). FMC **non** premia linearmente gli outlier: satura. Un walker eccezionale non "scappa via" con la popolazione.
4. **Continuità $C^1$ in $z=0$:** $g(0^+)=1=g(0^-)$ (limite sympy verificato) — coerente con la proprietà 3 del canone.
5. **Al valor medio** ($z=0$): $\alpha_{\rm eff}=\alpha/\sigma_R$ esatto (la linearizzazione richiesta dal task).

### Scalare di popolazione (un singolo numero confrontabile con $\alpha_B$)

Per collassare $\alpha_{\rm eff}(z)$ in uno scalare si media sulla distribuzione dei $z$. La pendenza empirica naturale è quella della regressione OLS di $\log\mathrm{VR}$ su $R$:

$$
\hat\beta=\frac{\mathrm{Cov}(\log\mathrm{VR},R)}{\mathrm{Var}(R)}=\frac{\alpha}{\sigma_R}\,\mathrm{Cov}\big(\log\widehat R(z),z\big)
=\frac{\alpha}{\sigma_R}\,\mathbb E[z\,\log\widehat R(z)].
$$

**Identità di Stein** (per $z\sim\mathcal N(0,1)$: $\mathbb E[z f(z)]=\mathbb E[f'(z)]$) collassa i due modi di misurare in **uno**:

$$
\mathbb E[z\,\log\widehat R(z)]=\mathbb E[g(z)]=:C.
$$

Cioè: la **pendenza di regressione di popolazione** = la **elasticità pointwise media** = $C$. Quindi la forma chiusa di popolazione è

$$
\boxed{\ \bar\alpha_{\rm eff}(\alpha,\sigma_R)=C\,\frac{\alpha}{\sigma_R},\qquad
C=\mathbb E_{z\sim\mathcal N(0,1)}[g(z)] = 0.7223\ \text{(quadratura)}\ }
$$

$C$ è un puro numero (nessuna dipendenza da $\alpha,\sigma_R$): metà (i $z\le0$) contribuisce $0.5$, la coda destra compressa aggiunge $0.222$. Per popolazioni non-Gaussiane $C$ cambia leggermente (vedi §4E) ma la **legge di scala $\propto\alpha/\sigma_R$ è esatta indipendentemente dalla distribuzione** (è algebra dello z-score, non un'assunzione gaussiana).

---

## 4. Verifica Monte Carlo — **NUMERICO** (eseguito)

Tutti i numeri da `w32_alpha_eff_check.py`, seed 20260709.

**(A) Formula pointwise vs derivata a differenze finite di $\log\mathrm{VR}$** — su griglia $z\in\{-2,-1,-0.2,0.3,1,3,8\}$:

| $(\alpha,\sigma_R)$ | max err. rel. |
|---|---|
| $(1.0,\,1.0)$ | $2.4\times10^{-9}$ |
| $(2.0,\,5.0)$ | $3.9\times10^{-9}$ |
| $(0.5,\,0.3)$ | $7.4\times10^{-10}$ |

→ la forma chiusa pointwise è **esatta** (errore = precisione delle differenze finite).

**(B/D) Pendenza empirica di regolazione vs $C\alpha/\sigma_R$** — popolazioni gaussiane $N=2\times10^5$, griglia $\alpha\in\{0.5,1,2\}\times\sigma_R\in\{0.2,0.5,1,3,10\}$:

- **max err. rel. = 0.29%**, media = **0.11%**.
- La legge $\hat\beta\propto\alpha$ (lineare) e $\hat\beta\propto1/\sigma_R$ (iperbolica) è confermata su 2 ordini di grandezza di $\sigma_R$.

**(C) Identità di Stein** (campione $N=2\times10^7$):
$\mathbb E[z\log\widehat R]=0.72236$, $\mathbb E[g(z)]=0.72226$, err. rel. **0.014%**; vs quadratura $C=0.72233$, err. **0.011%**. → Stein confermata: regressione ≡ elasticità media.

**(E) Robustezza non-gaussiana (popolazione uniforme):** la costante diventa $C_{\rm unif}=0.7383$ (≠ gaussiana), ma con la costante specifica-della-distribuzione la legge $C_{\rm unif}\alpha/\sigma_R$ regge con **max err. rel. 0.15%**. → la struttura $\propto\alpha/\sigma_R$ è universale; solo il prefattore $C$ è distribution-dependent (varia ~2% tra gaussiana e uniforme).

**Errore relativo complessivo della verifica MC: ≤ 0.29%** (worst case, gaussiana), tipico ~0.1%.

---

## 5. Conseguenze pratiche

### 5.1 Tuning: $\alpha$ nominale ≠ pressione reale
La pressione fisica è $C\alpha/\sigma_R$. Due conseguenze operative:

- **$\alpha$ non è una temperatura assoluta.** Lo stesso $\alpha$ produce pressione diversa su task con reward-scale diversa, o in fasi diverse della *stessa* ricerca. Confrontare $\alpha$ tra benchmark senza normalizzare per $\sigma_R$ è privo di senso.
- **Ricottura automatica incorporata.** Man mano che lo swarm converge, $\sigma_R\downarrow$, quindi $\alpha_{\rm eff}=C\alpha/\sigma_R\uparrow$: la pressione selettiva **cresce da sola** verso la fine della ricerca. FMC ha un annealing di temperatura *emergente*, non programmato — esattamente il comportamento "frontiera caos/ordine" (D3) ma qui in forma quantitativa: la popolazione parte "calda" (alto $\sigma_R$, bassa pressione, esplora) e si "raffredda" (basso $\sigma_R$, alta pressione, sfrutta).
- **Saturazione della coda:** aumentare $\alpha$ **non** rende FMC greedy sugli outlier come farebbe Boltzmann, perché $g(z)\to0$ nella coda. Per spingere davvero su un walker eccezionale bisogna alzare $\alpha$ *e* la pressione resta capata dalla compressione log. Questo spiega perché $\alpha$ molto grandi danno rendimenti decrescenti.

### 5.2 Cong. D: perché lo shaping deve essere moltiplicativo — **DIMOSTRATO**
`relativize` è **invariante affine globale**: sotto $R\mapsto aR+b$ ($a>0$ uguale per tutti i walker), $z$ è invariato (verifica: $z'=(aR+b-a\mu-b)/(a\sigma)=z$). Verifica numerica ([sez. finale dello script]):

- bonus additivo costante $R+100$: $\max|\Delta\mathrm{VR}|=2.4\times10^{-14}$ → **invisibile**;
- riscalamento moltiplicativo globale $3R$: $\max|\Delta\mathrm{VR}|=1.8\times10^{-15}$ → **invisibile**.

Quindi **né** un bonus additivo uniforme **né** un riscalamento moltiplicativo uniforme cambiano la selezione: entrambi vengono cancellati dallo z-score. L'unico shaping che morde è quello **non-uniforme fra walker** (structured): shaping moltiplicativo per-tier applicato solo al sottoinsieme che ha raggiunto il tier. Verifica: shaping moltiplicativo strutturato $\Rightarrow\max|\Delta\mathrm{VR}|=0.54$ → **NON nullo, morde**.

Il meccanismo: `relativize` è invariante a $b$ e ad $a$ globali ma **non** è invariante a un cambio di $\sigma_R$ *relativo alla struttura del segnale*. Poiché $\alpha_{\rm eff}\propto1/\sigma_R$, ciò che conta è il rapporto (gap-fra-tier)/(dispersione-entro-tier). Lo shaping moltiplicativo per-tier di Cong. D **allarga i gap fra tier** in modo che superino la dispersione locale e producano un salto di $z$ abbastanza grande da sopravvivere alla compressione log — cosa che un bonus additivo (assorbito nel ricentraggio) o un fattore globale (cancellato) non possono fare. Questa è la ragione strutturale, invariante-teoretica, per cui Cong. D è moltiplicativa e tiered e non additiva-globale.

> Caveat: la parte §5.2 sul *perché* lo stacking per-tier compone (amplificazione chain-tier) è **argomentata/approssimata**, non pienamente dimostrata qui: dimostrato è l'invarianza affine (additivo e moltiplicativo globali = zero) e che lo shaping strutturato è l'unico che modifica la selezione. Il legame quantitativo esatto con il compounding di exp17 resta da formalizzare.

---

## 6. Verdetto

**L'intuizione REGGE, e in forma più forte del previsto.** Non solo $\alpha_{\rm eff}$ dipende da $\sigma_R$: dipende **esattamente** come $C\alpha/\sigma_R$ (DIMOSTRATO pointwise, NUMERICO per la costante di popolazione con err. ≤0.29%), la temperatura inversa reale è $\alpha/\sigma_R$ modulata da un fattore-forma $g(z)\le1$ che satura la coda destra, e l'identità di Stein rende regressione-di-popolazione ed elasticità-media lo stesso numero. La legge $\propto\alpha/\sigma_R$ è algebrica (indipendente dalla distribuzione); solo $C\in[0.72,0.74]$ è distribution-dependent. È effettivamente "un teorema nuovo" con forma chiusa pulita, non un risultato negativo.
