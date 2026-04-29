# Paper FMC — correzioni strutturali per la prossima versione

> **Oggetto**: raccomandazioni concrete per il paper FMC successivo (v6 del 1803.05049, oppure paper accademico empirico nuovo).
>
> **Origine**: audit DHDNA del paper v5 in [`paper_fmc_dhdna_audit.md`](paper_fmc_dhdna_audit.md).
>
> **Scope di questo documento**: solo correzioni *strutturali e di framing* (P2a, P2b, P2c). Le verifiche empiriche (P0, P1a, P3) sono in [`protocols/`](protocols/).
>
> **Status del paper v5**: *bookware* di Sergio T₀-T₁, non submission accademica. Il paper successivo deve avere registro accademico-empirico con claim quantitative blindate.
>
> **Data**: 2026-04-28

---

## P2a — Riformulare il claim di complessità "MCTS exponential vs FMC linear"

### Problema nel v5

Paper §4.4.1, punto 5 (p.38) afferma:

> *"MCTS resources grow exponentially with scanning depth. In FMC the CPU resources grows linearly and memory resources doesn't grow with depth."*

Questo è **misleading per come è scritto**: conflate complessità *memoria* con complessità *CPU*, e implica un'asimmetria che non esiste se si paragonano gli algoritmi a budget di rollout fisso.

### Realtà tecnica

Per una decisione singola con budget $B$ rollout di profondità $D$:

| Algoritmo | Memoria | CPU per decisione |
|---|---|---|
| **MCTS-UCT vanilla** | $O(B \cdot D)$ — tutti i nodi visitati nell'albero (tipicamente $\ll$ albero completo grazie a UCB) | $O(B \cdot D)$ — un rollout per nodo |
| **PUCT (AlphaZero-style)** | $O(B \cdot D)$ ma con priori NN che concentrano sulla zona "interessante" | $O(B \cdot D \cdot c_{\mathrm{NN}})$ con costo NN per nodo |
| **FMC** | $O(N)$ — solo lo swarm corrente, niente albero | $O(N \cdot M) = O(B)$ con $B = N \cdot M$ rollout-equivalents |

**La complessità CPU per decisione è O(n) per entrambi**, dove $n$ è il *budget di rollout*. La differenza è in **memoria**: FMC mantiene solo lo swarm corrente ($O(N)$), MCTS mantiene l'albero ($O(B \cdot D)$).

### Riformulazione raccomandata per v6

Sostituire il punto 5 di §4.4.1 con:

> 5. **Memory complexity differs**: MCTS allocates an explicit search tree with $O(B \cdot D)$ memory ($B$ rollouts × depth $D$), whereas FMC keeps only the current swarm of $N$ walkers with $O(N)$ memory regardless of horizon. **CPU per decision is $O(\text{rollout budget})$ for both algorithms** when matched on samples; the FMC advantage in our experiments comes from achieving comparable performance with a smaller rollout budget, not from a fundamentally different complexity class.

### Justification

Il claim "MCTS cresce esponenzialmente" si applica solo alla *full enumeration* dell'albero di gioco (cosa che MCTS non fa — usa UCB proprio per evitare l'esplosione). Un reviewer accademico (especially someone from the AlphaZero/MuZero tradition) segnerebbe immediatamente la confusione.

---

## P2b — Rinominare "probability of cloning" → "cloning rate"

### Problema nel v5

Paper §4.2.4 (p.34) e §4.3 pseudocode (p.35-36):

> *"We will define the probability of walker $W_i$ with virtual reward $\mathrm{VR}_i$ cloning to $W_k$ state, with virtual reward $\mathrm{VR}_k$, as: (...) Prob = $(\mathrm{VR}_K - \mathrm{VR}_i) / \mathrm{VR}_i$ if $\mathrm{VR}_i \leq \mathrm{VR}_K$"*

Poi a p.36:

> *"Please note that probability of cloning can be >1, feel free to clip it to 1 for formal reasons if this is too uncomfortable for you."*

**Problema**: una *probability* per definizione è in $[0, 1]$. Se la quantità può essere > 1, **non è una probability** — è una *rate* o *intensity* non normalizzata. Il *clip* a $[0, 1]$ non è cosmetico, è la **corretta interpretazione probabilistica**.

### Riformulazione raccomandata per v6

Rinominare ovunque:

- "**probability of cloning** $P_{\mathrm{clone}}$" → "**cloning rate** $\rho_{\mathrm{clone}}$"
- Mantenere $P_{\mathrm{clone}} = \min(\rho_{\mathrm{clone}}, 1)$ come *transition probability* effettiva
- Riformulare a §4.2.4:

> The **cloning rate** of walker $W_i$ toward partner $W_k$ is defined as:
>
> $$\rho_{\mathrm{clone}}(i \to k) = \begin{cases} 1 & \text{if } \mathrm{VR}_i = 0 \\ 0 & \text{if } \mathrm{VR}_k \leq \mathrm{VR}_i \\ \frac{\mathrm{VR}_k - \mathrm{VR}_i}{\mathrm{VR}_i} & \text{otherwise} \end{cases}$$
>
> The transition probability is $P_{\mathrm{clone}} = \min(\rho_{\mathrm{clone}}, 1) \in [0, 1]$. In implementation, $\rho_{\mathrm{clone}}$ is compared directly with $u \sim \mathrm{Unif}(0, 1)$: when $\rho > 1$ the cloning is taken with certainty (equivalent to $P = 1$). This is the standard Metropolis-Hastings acceptance form $\min(\mathrm{VR}_k / \mathrm{VR}_i, 1)$ shifted by $-1$.

### Cross-reference

Già aggiornato in [`docs/MATH_CANON.md`](../MATH_CANON.md) Definizione 4 (2026-04-28).

### Justification

Un reviewer NeurIPS competente segna "Prob can be >1" come errore matematico in tempo $O(1)$. La riformulazione *cloning rate* + *clip* allinea la presentazione con la corretta interpretazione MH e rimuove l'imbarazzo formale.

---

## P2c — Spostare la sezione "Consciousness" dal paper al manifesto / Book #3

### Problema nel v5

Paper §6.4 (pp. 51-52) definisce:

> *"We can consider the vector $\{K_i\}$ as being the 'mental state' (...) Any mechanism that could automatically adjust those coefficients in order to make better decisions can be considered as a conscious mechanism."*

**Problema**: definire *consciousness* tramite "automatic adjustment of reward composition coefficients" è:
1. Definitional creep verso un termine pesantemente carico filosoficamente
2. Non operazionalizzabile — qualsiasi meta-RL agent qualificherebbe
3. Off-topic per un paper di planning algorithm
4. **Red flag immediato per reviewer** — segnale che gli autori stanno over-claiming

### Raccomandazione strutturale

**Per il paper v6 / submission accademica**: rimuovere completamente §6.4. La sezione non aggiunge valore tecnico al claim del planning algorithm; aggiunge solo rischio di rejection per "philosophical overreach".

**Dove va invece**: Book #3 manifesto / "Fractal Manifesto" (vedi profilo 4D-DHDNA Sergio T₂ in [`sergio_cognitive_profile_dhdna.md`](sergio_cognitive_profile_dhdna.md)). Lì il framing filosofico è on-brand. Il Book #1 di Sergio è già *bookware-manifesto*, quindi la sezione è coerente lì; un paper accademico empirico richiede registro più asciutto.

### Riformulazione per chi *vuole* tenerla nel paper

Se proprio si vuole un riferimento al meta-adaptation, rinominare e ridurre da §6.4 "Consciousness" a §6.4 "Meta-adaptation of reward composition" e formulare come **research direction**, non claim:

> ### 6.4 — Meta-adaptation of reward composition
>
> When the reward function is a composition of subgoals $\{G_i\}$ with weights $\{K_i\}$, an interesting research direction is whether the same FMC machinery can be applied at a meta-level to adjust $\{K_i\}$ based on long-term agent performance. This connects to existing work in meta-reinforcement learning [refs] and intrinsic motivation [refs]. We leave full development to future work.

Niente "consciousness". Niente "mental state". Niente over-claim.

### Justification

Sergio T₂ (proiezione 2030, 4D-DHDNA) è in modalità manifesto/philosophy-of-AI. La sezione *appartiene* a quel registro, non a NeurIPS. Spostarla preserva entrambi i progetti: paper rigoroso + libro filosofico, ognuno nel suo registro nativo.

---

## Sintesi tabellare

| ID | Sezione paper v5 | Cosa cambia in v6 | Drive |
|---|---|---|---|
| **P2a** | §4.4.1 punto 5 | Separare memory $O(B \cdot D)$ da CPU $O(B)$. Rimuovere "exponential vs linear". | Rigor tecnico — claim non-falso ma misleading |
| **P2b** | §4.2.4 + §4.3 pseudocode | "probability of cloning" → "cloning rate". Aggiungere clip esplicito. | Correctness matematica — un reviewer lo nota in 5s |
| **P2c** | §6.4 Consciousness | Rimuovere o ridurre a "meta-adaptation as research direction" | Strategia accademica — riduce rejection risk |

---

## Note di propagazione

Cross-doc updates già applicati al 2026-04-28:

- ✅ [`docs/MATH_CANON.md`](../MATH_CANON.md) Definizione 4 — terminologia "cloning rate" allineata
- ✅ [`docs/MATH_CANON.md`](../MATH_CANON.md) Congettura A — titolo + status header aggiornati per riflettere falsificazione del magic-6 universale
- ✅ [`CLAUDE.md`](../../CLAUDE.md) tabella discrepanze D1/D2 aggiornate con stato corrente

Documenti non-toccati intenzionalmente:
- Il paper PDF stesso (1803.05049v5.pdf) — read-only, autori esterni
- I deep dive 01-08 — già accurati nel loro scope
- Libri / slides di Sergio (`Fractal Book.md`, `2020 Fractal*.md`) — read-only, autori esterni