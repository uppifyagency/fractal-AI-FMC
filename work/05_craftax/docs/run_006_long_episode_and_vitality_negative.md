# Run 006 — Long-episode + Vitality bonus: due ipotesi falsificate

**Data**: 2026-04-27 mattina (loop iter 2)
**Setup**: Craftax-Classic-Symbolic-v1, FMC v4_p02_delta + variations.

## TL;DR

Due ipotesi testate per spingere oltre il 21.87% di v4_p02_delta:

1. **Long-episode** (max_steps=500 → 10000, Hafner standard): **FALSIFICATA**. Episodi terminano per morte naturale (192-300 step), non per cap. Estendere il cap non aiuta.
2. **Vitality bonus** (premia delta health/food/drink/energy): **PEGGIORA** di −6.55 a −14.13 punti su Crafter score. Il vitality scoraggia risk-taking necessario per molti achievement (mining, combat, deep cave).

Conclusione: **v4_p02_delta a 21.87% rappresenta un local optimum** dell'FMC vanilla applicato a Craftax-Classic. Ulteriori miglioramenti zero-training richiedono algoritmica strutturalmente nuova (Badger Level-1, macro-actions, NN target).

## Test 1 — Long-episode validation

### Hypothesis

Episodi di 500 step potrebbero tagliare prematuramente le chain di achievement che richiedono molti step (es. `eat_plant` richiede ~30 in-game days di crescita sapling → ripe_plant; `collect_diamond` richiede deep cave traversal).

### Setup

- Best config v4_p02_delta su 3 seed rappresentativi (42, 47, 55)
- `max_steps=10000` (Hafner standard) invece di 500

### Risultato

| seed | ach (cap=500) | ach (cap=10000) | n_steps_dec |
|---|---|---|---|
| 42 | 16 | **16** | 298 |
| 47 | 12 | **12** | 348 |
| 55 | 15 | **15** | 192 |

**Identici**. Nessun episodio raggiunge nemmeno 500 step prima di terminare per `done=True` (player death).

### Diagnosi

In tutti i 3 seed l'agente muore prima del cap. Il blocker non è il **tempo disponibile**, è la **sopravvivenza**: con la nostra reward shaping, walkers prioritizzano collection (wood, stone, iron) sopra la sopravvivenza, l'agente sceglie azioni rischiose, accumula danni, muore in ~200-300 step.

Per unlock `eat_plant` servirebbe sopravvivere ~600-1000 step (sapling growth cycle). Per `collect_diamond` servirebbe survival ~500+ step. Entrambi fuori range con il behavior corrente.

## Test 2 — Vitality bonus

### Hypothesis (basata su Test 1)

Aggiungere bonus per il delta delle 4 vital stats (`health + food + drink + energy`) dovrebbe spingere il walker a:
- Mangiare cow (eat_cow)
- Bere water (collect_drink)
- Dormire (wake_up)
- Evitare combat senza ragione

→ episodi più lunghi → più time → più chain achievement.

### Setup

`fmc_craftax_v7.py` aggiunge `vitality_alpha` config che somma al cum_reward del walker il `Δ(h+f+d+e)` per tick (signed). 

Sweep su `vitality_alpha ∈ {0.1, 0.3, 1.0, 3.0}`, 5 seed (42-46), tutti gli altri parametri = v4_p02_delta best.

### Risultato (5-seed)

| Config | vitality_α | Crafter % | mean_ach | mean_steps | uniq | Δ vs baseline |
|---|---|---|---|---|---|---|
| v4_p02_delta (baseline) | 0.0 | **21.53%** | 10.20 | — | 17 | — |
| v7_vit03 | 0.3 | 15.32% | 10.00 | 177 | 15 | **−6.21** |
| v7_vit10 | 1.0 | 14.25% | 8.20 | 158 | 16 | −7.28 |
| v7_vit01 | 0.1 | 12.04% | 9.00 | 168 | 14 | −9.49 |
| v7_vit30 | 3.0 | 7.74% | 7.00 | **192** | 12 | **−13.79** |

**Tutti i vitality_alpha peggiorano**. La curva è non-monotonica: vit03 è il "meno peggio", vit01 e vit30 (estremi) crollano.

### Diagnosi: il survival paradox

Cosa colpisce: vit30 ha gli **episodi più lunghi** (192 vs 177 step) ma le **uniche più basse** (12 vs 15). L'agente sopravvive più a lungo ma esplora meno → meno achievement.

Spiegazione: molti achievement richiedono RISK-TAKE deliberato:
- `defeat_zombie/skeleton` richiede combat (HP loss garantita)
- `collect_stone/coal/iron` richiede mining in zone potenzialmente pericolose
- `make_iron_pickaxe` richiede traversal cave deep (mob density alta)

Vitality bonus penalizza questi a prescindere dal payoff. Walker che "stays healthy" wins virtual reward → walker che "go fight zombie for the achievement" lose. Net: minor exploration.

**Trade-off paretiano broken**: vitality migliora UN dimension (survival time) a costo di UN'ALTRA (achievement coverage). Il geometric-mean Crafter score punisce il calo di coverage in modo sproporzionato.

### Cosa avrebbe potuto funzionare

Una vitality SELETTIVA che attiva bonus solo quando HP/food/drink scende sotto soglia critica:

```python
critical_h = jnp.maximum(2 - state.player_health, 0)  # solo se HP < 2
critical_f = jnp.maximum(2 - state.player_food, 0)
# ...
penalty = critical_h + critical_f + critical_d + critical_e
cum_rewards = cum_rewards - VIT_A * penalty
```

Solo penalizzare la "near-death" preserva risk-take quando HP > 2. Da provare in v8 se si torna sulla survival.

## Tabella aggiornata

| Metodo | Crafter score | Sample |
|---|---|---|
| Random | 1.6% | 0 |
| Rainbow | 4.3% | 1M |
| PPO | 4.6% | 1M |
| FMC vanilla (run_002) | 6.87% | 0 |
| **v7 vit30 (peggior survival)** ⚠️ | **7.74%** | **0** |
| **v7 vit01** ⚠️ | **12.04%** | **0** |
| **v6 Wigner memory (run_005)** ⚠️ | **12.32%** | **0** |
| **v7 vit10** ⚠️ | **14.25%** | **0** |
| DreamerV3 | 14.5% | 1M |
| **v7 vit03 (best vitality)** ⚠️ | **15.32%** | **0** |
| FMC + intrinsic α=0.5 (run_003) | 19.27% | 0 |
| Curious Replay | 19.4% | 1M |
| **v4_p02_delta (run_004)** ✓ | **21.87% ±1.21** | **0** | ← BEST 30-seed |
| EMERALD (Jul 2025) | 58.1% | 10M |

## Cosa abbiamo davvero imparato (run 006)

1. **Episode length non era il problema**. La hypothesis era ragionevole ma falsificata empiricamente in <30 minuti grazie al rapid testing.

2. **Survival reward è una contromossa, non una soluzione**. Vita più lunga ≠ più exploration. Il tradeoff geometric-mean Crafter score è governato da diversity, non da depth.

3. **v4_p02_delta è un local optimum robusto**. Tre tentativi successivi (memoria Wigner, vitality 4 valori) hanno tutti regressato. Probabile saturazione di quello che FMC vanilla + reward shaping può estrarre da Craftax.

4. **Per spingere oltre servono leve strutturali**:
   - **Macro-actions** (skill primitives): "go to nearest tree", "mine until inventory+1" → riduce horizon necessario per chain profonde
   - **Badger Level-1** (Book #2 §3.2): meta-FMC su reward configs, clona configurazioni vincenti per seed
   - **NN value function** appresa offline su rollout → priora init_actions in v5/v6 in modo informato (non solo Wigner counter)
   - **Curriculum reward shaping**: cambia α_inv durante l'episode (esploration early, exploitation late)

5. **Negative results sono saving del tempo futuro**: chiunque proverà vitality bonus naive vedrà questo doc e non perderà 30 min.

## Status finale

**v4_p02_delta resta il best zero-training su Craftax-Classic con 30-seed CI95: 21.87% ±1.21.**

Per submission (workshop / arXiv) basta: 100-seed run, doc paper draft, repo cleanup. **Nessuna modifica algoritmica necessaria** sotto il framework FMC vanilla. Per spingere oltre serve cambio di paradigma (lavoro settimane/mesi).

---

*Mattina del 2026-04-27 — auto mode, ~1h effective. Loop iter 2 completata.*
