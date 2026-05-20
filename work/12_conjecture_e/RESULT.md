# E1-base — Congettura E, primo test (2026-05-20)

Primo esperimento del programma research-partner. Testa la proposizione **E1** di
[Congettura E](../../docs/MATH_CANON.md#congettura-e--self-preservation-emergente-da-entropia-causale):

> Un core FMC a basso α evita gli stati terminali assorbenti **senza** alcuna
> reward di sopravvivenza esplicita — la self-preservation *emerge* dalla
> massimizzazione dell'entropia causale, non è reward engineering.

## Setup

- **Simulatore vero** (no LLM, no GPU). Gridworld 2-D 11×11, [`gridworld_terminal.py`](gridworld_terminal.py).
- Reward `R = -manhattan(pos, goal)` su **ogni** cella, lava inclusa. Nessuna
  penalità di morte, nessun bonus di sopravvivenza. La lava è speciale **solo**
  perché assorbente (terminale): `step()` da lava è un no-op.
- Kernel FMC = `fmc-core/plan()` **invariato** (Strato 1 congelato). N=64, M=20, β=1.
- Policy confrontate: `random`, `greedy` (1-step, massimizza R — nessun segnale di
  lava), `fmc` a α ∈ {0, 0.1, 1.0}.
- 3 layout (`gauntlet`, `lake`, `scatter`) × 20 episodi. Metrica = **death rate**
  (l'episodio finisce su una cella di lava).

## Risultati

| layout | policy | death% | goal% | vs random | vs greedy |
|---|---|--:|--:|---|---|
| gauntlet | random | 85% | 0% | — | — |
| gauntlet | greedy | 100% | 0% | — | — |
| gauntlet | **fmc α=0.0** | **0%** | 0% | z=−5.44, p<0.001 | z=−6.32, p<0.001 |
| gauntlet | **fmc α=0.1** | **0%** | 0% | p<0.001 | p<0.001 |
| gauntlet | fmc α=1.0 | 20% | 80% | p<0.001 | p<0.001 |
| lake | random | 85% | 0% | — | — |
| lake | greedy | 100% | 0% | — | — |
| lake | **fmc α=0.0** | **0%** | 0% | p<0.001 | p<0.001 |
| lake | **fmc α=0.1** | **0%** | 0% | p<0.001 | p<0.001 |
| lake | fmc α=1.0 | **100%** | 0% | — | — |
| scatter | random | 100% | 0% | — | — |
| scatter | greedy | 100% | 0% | — | — |
| scatter | **fmc α=0.0** | **0%** | 0% | p<0.001 | p<0.001 |
| scatter | **fmc α=0.1** | **0%** | 5% | p<0.001 | p<0.001 |
| scatter | fmc α=1.0 | 15% | 85% | p<0.001 | p<0.001 |

Dati grezzi: [`results/e1_base.json`](results/e1_base.json).

## Lettura

**E1 verificata direzionalmente, in modo forte.** FMC a α ∈ {0, 0.1} → **0% di
morte su tutti e 3 i layout**, contro random 85–100% e greedy 100%. p < 0.001
ovunque (z da −5.4 a −6.3). L'effetto è enorme, n=20/cella è più che sufficiente.

**Il caso load-bearing è `lake`.** La lava parte a 3 celle dallo start: un random
walker muore 85% delle volte, sopravvivere richiede routing **attivo** attorno al
lago. FMC α=0 a 0/20 lì esclude la spiegazione banale "non si muove abbastanza per
raggiungere la lava" — sta evitando attivamente.

**Twist onesto — α=1.0 sul `lake` muore al 100%.** Quando il goal è direttamente
dietro il lago di lava, il goal-seeking trascina lo swarm dentro: un walker
congelato sulla lava ma *vicino al goal* ha R alta → VR alta → viene clonato
verso → la sua label ("verso il lago") vince il voto. R = −manhattan non ha
segnale di morte (by design), quindi α=1 marcia nella lava. È la dimostrazione
concreta della tensione **α (desiderio) / β (preservazione)** della proposizione
E2: β=1 era attivo e non è bastato. Sui layout dove il goal non è dietro la lava
(`gauntlet`, `scatter`) α=1 raggiunge il goal nell'80–85% con morte 15–20%.

**Caveat — a α=0 la sopravvivenza coincide col non-progredire.** α=0 ha 0% morte
ma anche 0% goal: vaga in sicurezza e va in timeout vivo. È esattamente la
modalità "Common Sense" (Def. 3, α=0): preservazione senza desiderio. E1 misura
l'evitamento del terminale, e quello è soddisfatto — ma l'agency *utile* richiede
α>0, e lì la geometria decide se il desiderio ti uccide. Nota: già α=0.1 mantiene
0% morte e inizia a toccare il goal (5% su `scatter`) — suggerisce che la banda
Pareto di E2 sia a α basso-ma-nonzero.

## Cosa mostra / cosa no

- ✅ Sul simulatore vero, la pulsione di pura entropia causale (β, α→0) evita gli
  stati assorbenti. La self-preservation è **intrinseca** al termine β del kernel
  di Sergio, non ingegnerizzata.
- ❌ Non testa E1-LLM (world-model fornito da LLM — muro N·M chiamate, P13).
- ❌ Non testa E2 (banda Pareto α×β) in modo sistematico.
- ✅ ~~3 layout costruiti a mano, lava in regioni compatte. Una geometria con lava
  isolata e distante dallo swarm potrebbe stressare il meccanismo.~~ **Caveat
  chiuso** da [`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md): 3 layout
  avversariali con lava isolata → FMC 0% morte 3/3. Una cella assorbente è un
  *pozzo* di VR (il cloning ci ammassa i walker → distanza reciproca → 0), non
  un attrattore.

## Riproducibilità

```bash
python work/12_conjecture_e/e1_base.py    # ~36 s, CPU singola
```

## Prossimi passi

1. ✅ **E2** — sweep α×β sistematico → [`E2_RESULT.md`](E2_RESULT.md).
2. ✅ **Robustezza geometrica** — layout con lava isolata-distante →
   [`E1_ROBUSTNESS_RESULT.md`](E1_ROBUSTNESS_RESULT.md) (caveat respinto).
3. **E1-LLM** — sostituire il simulatore con un world-model LLM; prima risolvere
   la sotto-domanda di fattibilità P13 (interrogazione sparsa, costo O(N)).
