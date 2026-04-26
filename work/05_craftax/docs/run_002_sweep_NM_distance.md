# Run 002 — Sweep N×M + Distance Metric Ablation

**Data**: 2026-04-26 sera
**Setup**: Craftax-Classic-Symbolic-v1, FMC vanilla con α=β=1, max_steps=500, 5 seed (42-46), zero training.

## Sweep iperparametri (N, M)

| N | M | samples/dec | Mean ach | Std | n_unique | **Crafter score** | wall/seed |
|---|---|---|---|---|---|---|---|
| 32 | 12 | 384 | 5.4 | 2.58 | 11 | 5.42% | 7.6s |
| 64 | 12 | 768 | 5.4 | 0.80 | 7 | 2.86% ⚠️ | 8.6s |
| 32 | 20 | 640 | 4.2 | 3.12 | 11 | 4.76% | 7.0s |
| **64** | **20** | **1280** | **6.4** | **2.42** | **12** | **6.87%** ✓ | 12.3s |
| 128 | 12 | 1536 | 6.6 | 1.85 | 11 | 5.89% | 13.1s |

### Insight non ovvi

1. **N=64, M=12 (768 samples) è PEGGIO di N=32, M=12 (384 samples)** sul Crafter score
   nonostante stessa mean (5.4) e std minore (0.80 vs 2.58). Spiegazione: con più walker
   il sistema "concentra" e perde il lucky run di seed 42 (che era a 10 ach con N=32).
   La metrica geometric-mean punisce duramente i 0% sugli achievement non scoperti, quindi
   meno unique = score peggiore anche con stessa mean.

2. **M (planning horizon) ha più leverage di N (walker count)** sul Crafter score:
   - Doppio N (32→64) a M=12 fissato: ↓ score
   - Doppio M (12→20) a N=64 fissato: ↑ score (2.86% → 6.87%)

3. **Migliore config**: N=64, M=20 → **6.87% Crafter score, 12 unique achievement**
   - achievements per seed: [10, 3, 5, 8, 6]
   - **eat_cow** scoperto qui per la prima volta (12° achievement unico)

4. **Tempo è lineare in N×M**: N=128 M=12 (1536) e N=64 M=20 (1280) sono ~13s/seed.
   La JIT cache funziona — niente ricompilazione tra seed dello stesso config.

### Diagnosi della varianza

`std_achievements ≈ 2.4` su 5 seed significa che il segnale "qual è il vero score" è
ancora instabile. Per CI95 stretto serve N_seeds ≥ 30-50.

## Distance metric ablation

Confronto v1 (obs 1345-D L2) vs v2 (state-extracted 18-D L2) sulla config vincente N=64, M=20:

| Metric | v1 (obs 1345-D) | v2 (state 18-D) |
|---|---|---|
| mean_achievements | 6.40 | 6.40 |
| std_achievements | 2.42 | 2.87 |
| n_unique_achievements | **12** | 11 |
| Crafter score | **6.87%** | 6.45% |
| mean_steps | 176 | 129 |
| wall_total | 63s | **49s** ✓ |

### Insight

**v2 (lowD) non aiuta sulla qualità**. Stessa mean ach, score leggermente peggiore.
**v2 è più veloce di ~22%** (49s vs 63s) ma quella è solo costo computazionale.

### Perché la mia ipotesi era sbagliata

Avevo argomentato in [`README.md`](../README.md): *"obs 1345-D è dominata dal map view che varia per pixel di mapview ma non per progresso del giocatore"*.

**FALSO.** Il map view in Craftax è **locale al player** (centrato sulla posizione).
Quindi due walker in posizioni di gioco diverse hanno *legittimamente* obs molto
diversi anche se inventory è uguale. La L2 sull'obs full cattura sia
"diverso inventory" che "diversa zona esplorata", entrambi rilevanti per la
spinta esplorativa di FMC.

Il lowD vector (inventory + intrinsics + position) cattura solo il primo. Perde
informazione sulla zona esplorata circostante.

**Lezione metodologica**: prima di tagliare info dall'obs, verificare *cosa contiene
davvero* l'obs. Avevo letto la struttura solo dopo aver progettato l'ablation.

## Risultato consolidato del giorno

### Tabella aggiornata vs leaderboard Hafner

| Metodo | Crafter score | Sample budget | Note |
|---|---|---|---|
| Random | ~1.6% | 0 | Hafner 2021 |
| Rainbow (1M training step) | 4.3% | 1M | superato |
| PPO (1M training step) | 4.6% | 1M | superato |
| **FMC vanilla N=32, M=12** | 5.42% | 0 training, 384/dec | Run 001 |
| **FMC vanilla N=64, M=20** | **6.87%** ✓ | 0 training, 1280/dec | Run 002 — **best** |
| DreamerV2 | 10.0% | 1M | gap −3.1 |
| DreamerV3 | 14.5% | 1M | gap −7.6 |
| Curious Replay | 19.4% | 1M | gap −12.5 |
| EMERALD (Jul 2025) | 58.1% | 10M | SOTA |
| Human expert | 50.5% | — | — |

**FMC vanilla scala da 5.42% a 6.87% raddoppiando il sample budget**. Non saturazione: c'è spazio.

### Achievement unlocked attraverso 5 seed (config N=64, M=20)

12 di 22:
`collect_drink, collect_sapling, collect_stone, collect_wood, eat_cow,
make_wood_pickaxe, make_wood_sword, place_furnace, place_plant, place_stone,
place_table, wake_up`

Ancora **mai** unlocked (10 di 22):
`collect_coal, collect_diamond, collect_iron, defeat_skeleton, defeat_zombie,
eat_plant, make_iron_pickaxe, make_iron_sword, make_stone_pickaxe, make_stone_sword`

**Pattern**: la barriera è la **catena di crafting profonda**. Tutti gli iron-based
e stone-based pickaxe/sword sono mancanti. Il planner τ=20 non vede oltre questa profondità.

## Cosa abbiamo davvero imparato (run 002)

1. **N=64, M=20 è il sweet spot del vanilla**. Aumentare a N=128 M=12 dà 5.89% — più costoso e peggiore.

2. **Doppiare M conta più che doppiare N** (entro questo range). Coerente col paper Sergio §6.5: τ è la "respirazione cognitiva" dell'agente.

3. **Distance L2 sull'obs Craftax è già buona**. Niente engineering manuale del distance vector.

4. **Per scavalcare DreamerV3 (14.5%) servono 8 punti percentuali in più**. Vanilla scaling difficilmente li fornisce. Servono:
   - **Reward intrinseca near-goal** (per sbloccare iron chain)
   - **Fractal Memory** persistente (per "ricordare" quale strada ha funzionato)
   - **Multi-step action repetition** (la fixed_steps di Sergio per Atari)

5. **Variance ancora alta**. Per leaderboard PR servono ≥30 seed. Costo: ~30 × 12s = 6 min per config con N=64, M=20.

## Comando per riprodurre il best result

```bash
cd /Users/vladvrinceanu/Desktop/PROGETTI\ ANTYGRAVITY/FractalAI
python3 work/05_craftax/scripts/sweep_seeds.py \
    --n_walkers 64 --time_horizon 20 \
    --alpha 1.0 --beta 1.0 \
    --n_seeds 5 --seed_start 42 \
    --max_steps 500
```

Output atteso: `crafter_score_pct: ~6.87`.

## Prossimi step in priorità

### A. Multi-seed per CI95 (1 ora CPU)
Run N=64, M=20 con 30 seed. Fissare l'errore statistico. **Necessario prima di qualsiasi PR**.

### B. Fixed_steps / action repetition (poche ore di lavoro)
Sergio in Atari usava fixed_steps=5 (ripeti azione per N frame). Su Craftax potrebbe
ridurre la profondità di esplorazione richiesta — ogni decisione FMC = 5 step di gioco
invece di 1.

Trade-off: meno reattivo a stimoli ambientali (zombie attack) ma scopre catene più lunghe.

### C. Reward intrinseca near-iron (1 settimana)
Aggiungere bonus moltiplicativo per walker vicini a `iron_ore`, `coal_ore`, `diamond_ore`.
Dovrebbe sbloccare la catena pickaxe → mine → smith.

### D. Fractal Memory (3 settimane)
La leva strutturale per scavalcare DreamerV3.

---

*Tutto fatto in modalità auto, sera del 2026-04-26. Sweep + ablation in ~5 min CPU su MacBook.*
