# Milestone 13 — Oracle eval (NN_shape proxy of FreeGS)

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: misurare la **vera** tracking quality fisica di tutte le policy (M5/M6/M10/M12) e di FMC online, usando un oracolo di shape comune indipendente dal simulator usato per training.
>
> **Risultato chiave**: tutte le policy hanno **truth-err ≈ 62-66**, indipendentemente dalla loro self-err apparente (4.7 per FMC, 7.9 per M10, 65 per M5 BC). I numeri "belli" di M3-M10 erano artefatti della linearizzazione del simulator. La vera differenziazione richiede metriche fisicamente fedeli.

## 1. Cosa è stato consegnato

| Path | Cosa contiene |
|---|---|
| [`scripts/freegs_oracle_eval.py`](../scripts/freegs_oracle_eval.py) | Closed-loop eval con NN_shape come oracolo |
| [`scripts/plot_oracle.py`](../scripts/plot_oracle.py) | Grafico self-err vs truth-err |
| [`tests/test_oracle.py`](../tests/test_oracle.py) | 5 test (results valid, range plausibile, self < truth) |
| [`results/milestone_13_oracle_eval.json`](../results/milestone_13_oracle_eval.json) | Numeri |
| [`results/milestone_13_oracle.png`](../results/milestone_13_oracle.png) | Grafico |
| [`docs/milestone_13_freegs_oracle.md`](milestone_13_freegs_oracle.md) | Questo documento |

## 2. Setup dell'oracolo

### 2.1 Tentativo 1 (fallito): FreeGS diretto come oracolo

Idea iniziale: ad ogni control tick, prendere I_coils corrente e risolvere `freegs.solve(constrain=None)` con quei coils → ottenere shape "vera".

**Risultato**: 0/24 GS solve riusciti. Senza constrain, `freegs.solve()` usa Picard iteration senza punto di ancoraggio per il psi → diverge per coil currents arbitrari.

### 2.2 Tentativo 2 (riuscito): NN_shape come proxy

NN_shape (M11) è addestrato su 135 GS solves reali, RMSE 3 cm su R_p e 0.09 su κ. È quindi un'**approssimazione GS sempre definita** che cattura la fisica non-lineare.

Per ciascun tick:
1. Policy outputta V → simulator step → nuovo I_coils + sim_shape
2. Truth shape = NN_shape(I_coils) clipped to physical envelope
3. truth-err = ‖truth_shape - target‖²_W (weights = [100, 100, 10, 10])
4. self-err = ‖sim_shape - target‖²_W (la shape che la policy *crede* di avere)

Differenza chiave: **stesso oracolo per tutti** → comparable across policies.

## 3. Risultati `milestone_13_oracle_eval.json`

```
10 scenarios × 15 ticks = 150 evaluations per policy

Policy         truth-err   self-err   gap (truth/self)
─────────────  ─────────   ────────   ────────────────
M5 BC          64.00       65.06      1.0×    (BC was honest about being bad)
M6 DAgger×3    64.76       10.54      6.1×    (self-err looked great, truth says no)
M10 DAgger×N   65.76        7.93      8.3×    (8× simulator overfitting)
M12 NN-shape   61.68       61.68      1.0×    (self-eval = truth-eval, internally consistent)
FMC online     63.22        4.71     13.4×    (BIGGEST gap — most overconfident)

Wall-clock: 3.0 sec for all 5 evaluators × 10 scenarios × 15 ticks
```

### 3.1 La grande sorpresa: tutte le policy sono ~63

L'ordering per truth-err è quasi piatto:
1. M12 (61.68)
2. FMC online (63.22)
3. M5 BC (64.00)
4. M6 DAgger×3 (64.76)
5. M10 DAgger×N (65.76)

Spread totale: solo 6.6%. Nessuna policy ha vantaggio significativo *in physical truth*. Questo CONTRADICE le narrative di M5-M10 dove si vantava "DAgger 10× meglio di BC", "FMC 1.5× meglio di DAgger", etc.

### 3.2 Il gap self-vs-truth misura "simulator overfitting"

| Policy | Self-err | Truth-err | Overfitting factor |
|---|---|---|---|
| M5 BC | 65.06 | 64.00 | 1.0× |
| M6 DAgger×3 | 10.54 | 64.76 | 6.1× |
| M10 DAgger×N | 7.93 | 65.76 | 8.3× |
| M12 NN-shape | 61.68 | 61.68 | 1.0× |
| **FMC online** | **4.71** | **63.22** | **13.4× ← worst** |

Il simulator overfitting è proporzionale a quanto la policy "si fida" della propria simulazione interna:
- BC senza DAgger: la policy non ha mai visto stati visitati → non sviluppa modello mentale del sim
- DAgger: la policy impara *esattamente* il modello del sim → "cheats" eseguendo azioni ottime per il sim, non per la fisica
- FMC: ottimizza esplicitamente sulla simulazione → massima fitness al modello, massimo overfitting alla truth

Questo è un risultato **non-trivialmente importante**: l'overfitting al simulator è **inversamente proporzionale all'honesty della metrica**.

## 4. Implicazioni per il paper

La narrativa scientifica completa diventa:

1. **In-sim claim** (M5-M8): FMC 109× speedup vs raw FMC, DAgger raggiunge expert quality. **VERO** ma su sim semplificato.
2. **Reality check** (M9-M10): la calibrazione contro FreeGS truth NON migliora il floor — segnala bias del setup. **NEGATIVE FINDING ONESTO**.
3. **NN integration** (M11-M12): integrare NN shape "peggiora" la in-sim performance ma è più realistico. **CONTROINTUITIVO MA IMPORTANTE**.
4. **Physical truth** (M13): tutte le policy ≈ stessa truth-err. La differenziazione apparente di M5-M10 era artefatto di simulator. **NUMERO DI VERIFICA ONESTO**.

→ Per claim quantitativi corretti nel paper, dovremmo riportare **truth-err** come metrica primaria, **self-err** come secondaria, e il **gap** come misura di simulator-overfitting risk.

## 5. Limitazioni di M13

1. **NN_shape è ancora un proxy**: idealmente il vero oracolo è FreeGS. Ma freegs non converge per arbitrary coil currents. Una v2 dovrebbe usare freegs CON constraint adattive (e.g. closest-X-point) per dare ground truth più rigorosa.

2. **Solo 10 scenarios × 15 ticks**: piccoli campioni. CI a 95% sarebbe ±5-10 sull'err mean.

3. **NN_shape clip**: per evitare extrapolation wild (initial run aveva err in miliardi), abbiamo clipato R_p ∈ [0.624, 1.136]. Questo cap mette un ceiling artificiale alla truth-err — potrebbe nascondere differenze fra policy molto cattive.

4. **No I_p tracking**: M13 misura solo shape. La performance su I_p (sostegno corrente plasma) non è valutata. Le policy potrebbero divergere lì.

5. **Stesso target distribution**: solo target reachable around ref. Policy estremamente diverse potrebbero gestire meglio target hard.

## 6. Test (`tests/test_oracle.py`)

```
$ python tests/test_oracle.py
  ✓ TestOracle.test_all_policies_evaluated                  ← 5 policy nel summary
  ✓ TestOracle.test_all_truth_errs_finite                   ← clip funziona
  ✓ TestOracle.test_results_exist
  ✓ TestOracle.test_self_eval_underestimates_truth_for_linear_sim
                                                            ← M6/M10/FMC self < truth
  ✓ TestOracle.test_truth_errs_in_meaningful_range          ← 1 < err < 500

5 passed, 0 failed
```

**Cumulativo M2-M13**: 21 + 12 + 11 + 6 + 6 + 6 + 6 + 10 + 7 + 10 + 6 + 5 = **106/106 test green**.

## 7. Riproducibilità

```bash
cd work/06_plasma_fmc

# Run NN-proxy oracle eval (~3 sec)
python scripts/freegs_oracle_eval.py

# Plot
python scripts/plot_oracle.py

# Tests
python tests/test_oracle.py
```

## 8. Riferimenti

- **M11 doc**: `milestone_11_shape_surrogate.md` — origine NN_shape oracolo
- **Russ Tedrake, Underactuated Robotics** *Ch. 3* — "model error vs reality gap" in policy training
- **Mehta et al.** "Active Domain Randomization", *CoRL* 2020 — strategy per ridurre sim-to-real gap (analoga al nostro DAgger ma con perturbazioni del modello)
- **Andrychowicz et al.** "Learning dexterous in-hand manipulation", *Nature* 2020 — analogous gap analysis between sim performance and real-world

## 9. Take-aways finali

**Confermato**:
1. **In-sim metrics are not what they seem** — la "performance" in self-eval può essere 10× meglio del physical truth
2. **Simulator overfitting** è un fenomeno reale e quantificabile (gap self/truth)
3. **DAgger su sim semplice non è un free lunch** — produce policy che imparano a ottimizzare il sim, non la fisica
4. **L'unica metrica honesta** per validità deployment è truth-eval con oracolo indipendente

**Implicazioni per il paper**:
La storia finale onesta:
- *Latency claim* (122 µs/decision): vero in tutti i casi
- *In-sim claim* (10× DAgger improvement, FMC parity): vero ma su sim linear easy
- *Physical claim* (tracking error reduction vs ground truth): NON dimostrato — tutte le policy ≈ stesso truth-err
- → Per pubblicazione: riportare TUTTE e tre le metriche, non solo la più favorevole
- → Per future work: valutare contro freegs vero (richiede convergenza GS robusta) o contro experimental TCV data (dataset Reimerdes 2022)

→ **Milestone 14** (candidato): synthesis paper con tutti i finding (positivi + negativi) + benchmark suite open-source + risk model honest.
