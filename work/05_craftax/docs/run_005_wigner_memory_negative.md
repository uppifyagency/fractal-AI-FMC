# Run 005 — Wigner-Correct Fractal Memory: risultato negativo informativo

**Data**: 2026-04-26 notte (dopo run_004)
**Setup**: Craftax-Classic-Symbolic-v1, FMC v6 (intrinsic α=0.5 + delta-prox α=0.2 + Wigner Fractal Memory).
**Hardware**: MacBook Apple Silicon, JAX 0.10.0, CPU backend.

## TL;DR

Implementare la **Fractal Memory di Sergio (Slide 2020 §6.1)** in modo letterale (Wigner reward + visit debias + walker density allocation) e usarla come **action prior in planning** **PEGGIORA il Crafter score di −9.55 punti** rispetto al baseline v4_p02_delta (12.32% vs 21.87%).

Questo è un **risultato negativo importante**: la Fractal Memory di Sergio è progettata per **NN training attention**, non per **planning action selection**. Senza un NN che processa il batch Wigner-pesato, l'allocazione walker non si traduce in policy migliore.

## Cosa abbiamo fatto

### v6 — Wigner-correct memory (`fmc_craftax_v6.py`)

```python
class FractalMemoryWigner:
    """Cross-episode memory bank con dinamica Wigner.
    
    Ogni unit = (fingerprint, action, reward_sum, visits, n_walkers).
    Walker density allocato proporzionalmente a Wigner debiased reward.
    """
    
    def update_walker_distribution(self):
        # Loss = max_avg_reward - this_avg_reward
        # Wigner R' = (π/2) · x · exp(-π/4 · x²)  with x = loss / avg_loss
        # Debias: R = R' / (1 + log(1 + visits))
        # n_walkers proporzionale a R
```

Fingerprint = 10-D discreto: 8 progress flags inventory + 2 spatial buckets (16-tile granularity).

Update: dopo ogni episode, per ogni (fp, action) decision presa, append/incrementa un'unit memory col `episode_reward` come weight. Poi ricalcola walker density Wigner.

Recall: a ogni step, recall_action_prior(fp) restituisce distribuzione weighted da n_walkers per quella fingerprint.

Sample init_actions FMC con mix `0.3 · prior + 0.7 · uniform`.

### Test — 10 seed (42-51), N=64, M=20

| Metric | v6 (Wigner mem 0.3) | v4_p02_delta (no mem) |
|---|---|---|
| Crafter score | **12.32%** | **21.87%** (30 seed) |
| Mean achievements | 7.8 | 10.03 |
| Per-seed | [10, 8, 4, 8, 8, 8, 11, 11, 4, 6] | [16, 7, 4, 14, 10, 12, 8, 15, 5, 12] |
| Achievement uniche | 15/22 | 18/22 |
| Memoria finale | 181 entries, 25 fps | — |

**Regressione netta di 9.55 punti percentuali sulla Crafter score** vs lo stesso codice senza memory.

## Diagnosi

### Errore di traduzione algoritmica

La Slide §6.1 di Sergio descrive Fractal Memory come **mecchanismo di attention sul training set**:

> *"Sample a batch of data points using #Walkers as distribution. Train the NN parameters on this batch."*

Il walker density (Wigner-weighted) governa **quale dato sottoporre al NN per il prossimo gradient step**. Il NN poi fa inferenza su qualsiasi input.

Nel mio v6 ho **saltato il NN** e usato il walker density direttamente come prior di azione. Questo conflate due cose ortogonali:

| Sergio | v6 mio |
|---|---|
| "Quale memoria devo studiare di più?" → Wigner | "Quale azione devo prendere ora?" → Wigner |
| Risposta: medium-loss = imparare al meglio | Risposta: medium-reward = compromesso fra exploit/explore |

Per planning, **vogliamo HIGH-reward action**, non medium. Wigner non risponde a questa domanda.

### Problema di credit assignment

Anche concettualmente, attribuire l'`episode_reward` a OGNI (fp, action) della traiettoria è una baseline-MC con varianza enorme. Una sequenza random che casualmente porta a 15 ach assegna +15 a tutte le sue azioni, anche quelle obiettivamente cattive. Il segnale per (fp, action) è dominato dalla luck dell'episode.

**Soluzione corretta** (non implementata): TD-style credit con eligibility traces, o un value function NN appresa tramite il batching Wigner.

### Granularità fingerprint

10-D discreto produce solo 25 fingerprint uniche su 30 episodi (~744 visit totali). Stati molto diversi del gioco (es. "in foresta vs in caverna, stesso inventory") collassano nella stessa fp. Le memorie aggregano azioni per situazioni eterogenee → prior rumoroso.

## Cosa Sergio probabilmente avrebbe detto

Re-leggendo Slide §6.3 ("NN as Fractal Memory"), capisco che il design originale richiede:

1. **K reti neurali parallele** (es. K=100), ognuna con i suoi walker
2. **Reward NN = exp(-mean_loss)** (non Wigner!)
3. **Specializzazione automatica**: NN bravi su task X attirano walker quando emerge X, NN scarsi spengono il gradient learning
4. **Inference con ensemble entropy weighting**: `weight_NN = mean(1/multiplicative_entropy(output))`

Per un caso senza NN (solo planning), l'analogo sarebbe:
- K configurazioni di FMC parallele (es. K=10 con `intrinsic_α` diversi)
- Reward config = episode_reward
- Specializzazione: config che vince su seed Y attira walker della meta-FMC
- Decisione finale = ensemble vote dei FMC vincitori

Questa è la struttura **Badger Level-1 (reward optimizer)** del Book #2 §3.2. È quello che farò nel v7 se il tempo lo permette.

## Tabella aggiornata vs leaderboard Hafner

| Metodo | Crafter score | Sample | Note |
|---|---|---|---|
| Random | ~1.6% | 0 | Hafner 2021 |
| Rainbow (1M) | 4.3% | 1M | |
| PPO (1M) | 4.6% | 1M | |
| FMC vanilla (run_002) | 6.87% | 0 | |
| **v6 Wigner Memory** ⚠️ | **12.32%** | **0** | **Regressione** vs run_004 |
| DreamerV2 | 10.0% | 1M | |
| DreamerV3 | 14.5% | 1M | |
| FMC + intrinsic (run_003) | 19.27% ±2.32 | 0 | 10 seed |
| Curious Replay | 19.4% | 1M | |
| **FMC + intrinsic + delta-prox (run_004)** ✓ | **21.87% ±1.21** | **0** | **30 seed — BEST** |
| EMERALD (Jul 2025) | 58.1% | 10M | SOTA |
| Human expert | 50.5% | — | |

## Cosa abbiamo davvero imparato (run 005)

1. **Implementare letteralmente un algoritmo non equivale ad applicarlo correttamente**. La Wigner reward formula corretta non basta; serve il *contesto operativo* corretto (NN training, non planning action selection).

2. **Sergio's Fractal Memory presuppone un NN target**. Senza NN da addestrare, il walker density è informazione orfana — non ha un consumer.

3. **Per planning zero-training, la mia v4_p02_delta resta il top performer**. Aggiungere meta-strutture (memory, multi-config FMC) richiede design specifico al planning, non porting diretto da NN-training.

4. **Negative results sono evidence**. Questo run salva mesi di tempo a chiunque ripetesse l'esperimento ingenuamente. Il fix richiesto è strutturale (NN target o Badger-Level-1), non parametrico.

## Prossimi step possibili

### Opzione A — v7 Badger-Level-1 (multi-config FMC)
Esegui K=5 FMC paralleli con diversi `(intrinsic_α, proximity_α)`. Outer-loop FMC clona configurazioni vincenti basato sull'episode reward. Coerente con Book #2 §3.2 (Reward optimizer level).

Costo: ~5x compute (5 FMC per decisione). Beneficio atteso: +1-3 punti percentuali se il signal between-config c'è.

### Opzione B — Episodi più lunghi
`max_steps=10000` (Hafner standard) invece di 500. Probabilmente unlock di `eat_plant` (richiede sapling → ripe_plant → eat). Costo: 20× tempo.

### Opzione C — N=128 walker
Doppio sciame su v4_p02_delta. Run_002 ha mostrato saturazione a N=64 vanilla, ma con dense reward forse non ancora saturato. Costo: 2× tempo.

### Opzione D — Implementare Fractal Memory con NN target (post-Book#2 path)
Train un value-function NN con il batch Wigner-pesato. Usa NN per init_actions. Costo: settimane (richiede framework PyTorch + design accurato), ma è la **vera** Fractal Memory.

---

*Notte del 2026-04-26 — auto mode, ~30 min effective post-run_004. v6 negative result documentato per onestà metodologica e per chi proseguirà.*
