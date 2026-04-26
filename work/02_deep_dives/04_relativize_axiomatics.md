# Deep Dive 04 — Caratterizzazione assiomatica di `relativize`

> **Stato**: outline. Risponde al "buco assiomatico" segnalato in [`ANALISIS.md` §10.3](../../ANALISIS.md).

## Tesi

> *La trasformazione `relativize` introdotta nel paper §2.2.3 non è scelta ad hoc: è (a meno di equivalenza) l'unica funzione che soddisfa cinque vincoli ragionevoli sul reshaping di reward arbitrarie.*

## I cinque assiomi candidati (da formalizzare)

| Assioma | Formulazione informale |
|---|---|
| **A1 — Positività** | $\hat{R}(x) > 0$ per ogni $x$, indipendentemente dal segno di $R(x)$ |
| **A2 — Preservazione dell'ordine** | $R(x_1) > R(x_2) \Rightarrow \hat{R}(x_1) > \hat{R}(x_2)$ |
| **A3 — Invarianza affine** | $\hat{R}_{aR+b} = \hat{R}_R$ — il reshape non dipende da scala/shift |
| **A4 — Compressione asintotica** | Per $R \to +\infty$, $\hat{R} = O(\log R)$ (no esplosione) |
| **A5 — Espansione asintotica** | Per $R \to -\infty$, $\hat{R} = o(1)$ (decay sub-esponenziale) |

## La proposta del paper

```
R_N = (R - μ) / σ        (z-score)
R = exp(R_N)             se R_N ≤ 0
R = 1 + ln(1 + R_N)      se R_N > 0
```

## Outline delle sezioni da scrivere

1. **Motivazione**: perché serve un reshape (caso bank account, reward negativo)
2. **Formalizzazione di A1-A5**: linguaggio matematico rigoroso
3. **Teorema di unicità**: sotto A1-A5, qualsiasi $\hat{R}$ è equivalente alla forma del paper a meno di costanti
4. **Dimostrazione**: caso per caso (R_N ≤ 0, R_N > 0)
5. **Discussione**: cosa succede se rilassiamo A4 o A5?
6. **Alternative**: softmax, sigmoid, tanh — come si confrontano?
7. **Verifica numerica**: A1-A5 sono soddisfatti dall'implementazione in [`relativize_vector`](../../repos/FractalAI_old/fractalai/swarm.py#L16)?

## Possibile teorema di unicità (sketch)

**Teorema**. Sia $f: \mathbb{R} \to \mathbb{R}_{>0}$ una funzione $C^1$ che soddisfa:
- A2: $f$ strettamente crescente
- A3: $f$ commuta con z-scoring
- A4: $\lim_{x \to +\infty} f(x) / \log x = 1$ (normalizzazione)
- A5: $\lim_{x \to -\infty} f(x) / e^x = 1$ (normalizzazione)
- Continuità in $x = 0$

Allora $f(x) = \exp(x) \cdot \mathbb{1}[x \leq 0] + (1 + \log(1 + x)) \cdot \mathbb{1}[x > 0]$, a meno di errore $O(\epsilon)$ in un intorno di $x = 0$.

(da dimostrare)

## Implicazioni

Se il teorema regge:

1. La scelta di `relativize` smette di essere ad hoc
2. Diventa l'**unica** scelta naturale, modulo i 5 assiomi
3. Altre architetture (e.g. softmax) violano almeno uno degli assiomi e quindi falliranno in certi regime di reward

---

*Da espandere a 600-1000 righe. Priorità: media-alta (risolve un buco identificato).*
