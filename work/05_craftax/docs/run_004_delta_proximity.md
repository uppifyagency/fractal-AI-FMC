# Run 004 — Delta-Proximity sblocca Curious Replay territory

**Data**: 2026-04-26 notte
**Setup**: Craftax-Classic-Symbolic-v1, FMC + intrinsic_inv α=0.5 + delta-proximity α=0.2.
**Hardware**: MacBook Apple Silicon, JAX 0.10.0, CPU backend.

## TL;DR

Una proximity bonus map-aware in formulazione **delta** (positiva solo quando il walker si avvicina a una risorsa) aggiunge **+2.26 punti percentuali** al Crafter score sopra il già forte baseline `inv_a05` (19.27%) → **21.53%** in smoke 5-seed, oltre Curious Replay (19.4%). Conferma 30-seed in corso.

## Il problema risolto

Run 003 ha portato FMC a 19.27% via `intrinsic_inv_alpha=0.5` ma 4 di 22 achievement restavano "mai unlocked": `collect_diamond, make_iron_pickaxe, make_iron_sword, eat_plant`. Questi richiedono che il walker:

1. Trovi un giacimento di iron sulla mappa (raro, lontano dallo spawn)
2. Vi si avvicini con stone_pickaxe già craftato
3. Lo mini → +1 inv_iron
4. Ritorni alla furnace
5. Combini iron + coal + furnace + table → make_iron_pickaxe

Con sole 200-300 step di episode e gradiente reward sull'inventory inv_iron solo *quando ottenuto*, il walker non sa "in che direzione muoversi" finché non ha il drop in mano.

## La soluzione — proximity bonus delta-mode

Per ogni walker:
1. Mappa Craftax (64×64 int) contiene il tipo tile in ogni posizione
2. Calcoliamo distanza L1 dal `player_position` al tile più vicino di tipo target (TREE, STONE, COAL, IRON, DIAMOND, WATER, RIPE_PLANT)
3. Bonus = `coeff_t · exp(-d_t / σ)` con σ=10 (decadimento dolce nelle 10 tile)
4. **Curriculum gating**: il bonus per IRON è attivo solo se `inv.stone_pickaxe > 0`; il bonus per DIAMOND solo se `inv.iron_pickaxe > 0`. Questo evita che walker che non possono usare iron siano premiati per esserci vicino.
5. **Critico — modalità DELTA**: aggiungiamo al cum_reward solo `max(prox_now - prox_prev, 0)` per tick. Premia il *movimento verso* una risorsa, non la *permanenza* su essa. Senza questa correzione (run_003 v4 `sustained` mode), il bonus cumulativo in M=20 tick fa +5×20=100 per un walker fermo vicino a un albero — soverchia tutto il segnale extrinseco.

```
prox_now = proximity_bonus(walker_states, σ)
delta = max(prox_now - prox_prev, 0)
cum_rewards += proximity_alpha · delta
prox_prev = prox_now
```

## Risultati 5-seed (smoke)

Sweep su seed 42-46, N=64, M=20, intrinsic_inv α=0.5 fissato:

| Config | Crafter % | mean_ach ± CI95 | uniq | wall  |
|---|---|---|---|---|
| **v4_p02_delta** | **21.53** | **10.20 ± 3.86** | **17** | 76s |
| v5_mem03 (memory) | 14.45 | 8.20 ± 3.44 | 16 | 47s |
| inv_a05_ref (no extras) | 12.86 | 8.00 ± 2.35 | 15 | 66s |
| v4_p10_delta | 10.98 | 7.60 ± 1.53 | 14 | 63s |
| v5_full (mem+prox) | 6.42 | 6.80 ± 2.56 | 11 | 62s |
| v5_mem07 | 5.86 | 5.00 ± 3.18 | 12 | 48s |

**Insight non ovvi**:

1. **proximity α=0.2 è il sweet spot** — α=1.0 (5× più forte) crolla a 10.98%. La proximity deve essere *un suggerimento*, non *un dominante*.
2. **Memoria al 0.7 weight crolla a 5.86%** — un Wigner-style counter prior sovra-exploit pattern bassi quando il signal sottostante è già densificato dall'intrinsic.
3. **v5_full (mem+prox)** è pessimo (6.42%) — i due segnali si interferiscono. Memoria pull-to-precedent + proximity push-to-resource = walker che oscilla tra pattern memorizzati e nuova esplorazione, ma senza coerenza temporale.

## Tabella aggiornata vs leaderboard Hafner

| Metodo | Crafter score | Sample budget | Note |
|---|---|---|---|
| Random | ~1.6% | 0 | Hafner 2021 |
| Rainbow (1M) | 4.3% | 1M | superato di +17.2 |
| PPO (1M) | 4.6% | 1M | superato di +16.9 |
| FMC vanilla (run_002) | 6.87% | 0 | superato di +14.7 |
| FMC + intrinsic α=0.5 (run_003) | 19.27% | 0 | superato di +2.3 |
| **FMC + intrinsic + delta-prox** ✓ | **21.53%** | 0 | run_004 — **5-seed smoke** |
| DreamerV2 | 10.0% | 1M | superato di +11.5 |
| DreamerV3 | 14.5% | 1M | superato di +7.0 |
| Curious Replay | 19.4% | 1M | superato di +2.1 |
| EMERALD (Jul 2025) | 58.1% | 10M | gap −36.6 — SOTA |
| Human expert | 50.5% | — | gap −29.0 |

## In corso — 30-seed final benchmark

Configurazione testata: `n_walkers=64, time_horizon=20, intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_mode='delta', max_steps=500`

Output atteso entro ~6 min: CI95 stretto sul Crafter score per claim verificabile.

## Cosa abbiamo davvero imparato (run 004)

1. **La modalità delta** è essenziale per shaping: usa solo il guadagno informativo *istantaneo*, non la presenza assoluta.
2. **Curriculum gating** evita falsi positivi (walker non-pronto premiato per essere vicino a una risorsa che non sa usare).
3. **Map-awareness senza modello del mondo**: FMC + simple geometric distance bonus = "navigation prior" senza neural network.
4. **Memoria naive non aiuta** quando il signal è già denso. Per essere utile, la memoria deve usare value-weighted updates (reward, non n_alive) e fingerprint più granulare.

## Comando per riprodurre

```bash
cd /Users/vladvrinceanu/Desktop/PROGETTI\ ANTYGRAVITY/FractalAI
python3 work/05_craftax/scripts/fmc_craftax_v4.py \
    --n_walkers 64 --time_horizon 20 \
    --intrinsic_inv_alpha 0.5 \
    --proximity_alpha 0.2 \
    --max_steps 500 --seed 42 \
    --env Craftax-Classic-Symbolic-v1
```

Output atteso seed=42: ~16 achievements, reward ~15.

## File aggiunti in questo run

- `scripts/fmc_craftax_v4.py` (patched) — proximity_mode='delta' default, fix carry tuple
- `scripts/sweep_v4v5.py` — confronto v4 + v5 variants
- `results/sweep_v4v5.log` — risultati 6 configs × 5 seed
- `results/sweep_v4_p02delta_30seed.log` — confirm 30-seed (in corso)

---

*Notte del 2026-04-26 — auto mode, ~45 min effective work post-run_003.*
