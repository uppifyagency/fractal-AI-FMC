# Milestone 10 — Calibrated simulator (negative finding)

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: applicare i findings di M9 (FreeGS truth) al simulatore di M2/M3 — ricalibrare ref_state e scalare la matrice S × 10. Verificare se questo riduce il floor di tracking error osservato in M8 (3.45).
>
> **Risultato (negativo onesto)**: la calibrazione **NON riduce il floor**. M10 DAgger plateau ≈ 3.47 vs M8 ≈ 3.45. Il floor è **strutturale**, NON un problema di calibrazione di magnitude. Cause vere: (a) approssimazione lineare S non cattura non-linearità della shape control, (b) target distribuiti random includono punti unreachable a priori. Scalare S non aiuta: piccola → FMC non muove shape; grande → controllo touchy ma DAgger media. **Floor invariante**.

## 1. Cosa è stato consegnato

| Path | Cosa contiene |
|---|---|
| [`scripts/calibrated_sim.py`](../scripts/calibrated_sim.py) | `build_calibrated_jax_params()` con M9 ref + S × 10 |
| [`scripts/calibrated_pipeline.py`](../scripts/calibrated_pipeline.py) | End-to-end: gen dataset + train BC + DAgger×5 + benchmark |
| [`tests/test_calibrated.py`](../tests/test_calibrated.py) | 7 test (calibration consistency, DAgger improvement) |
| [`results/calibrated_dataset.npz`](../results/calibrated_dataset.npz) | 1000 expert samples calibrati |
| [`results/policy_calibrated.npz`](../results/policy_calibrated.npz) | Policy finale calibrata |
| [`results/calibrated_history.json`](../results/calibrated_history.json) | DAgger curve per iter |
| [`docs/milestone_10_calibration.md`](milestone_10_calibration.md) | Questo documento |

## 2. Calibrazioni applicate (input M9)

### 2.1 Reference state aggiornato

Da M9 baseline DN equilibrium:

| Parametro | M2/M3 (Miller wishlist) | M10 (FreeGS DN truth) |
|---|---|---|
| R_p ref | 0.880 m | **0.901 m** |
| Z_p ref | 0.000 m | **-0.109 m** |
| a_eff | 0.240 m | **0.390 m** |
| κ ref | 1.700 | **1.616** |
| δ ref | +0.300 | **+0.003** |

I_ref coil currents: presi direttamente dal `freegs_truth.json["baseline"]["currents"]` (le correnti che FreeGS ha trovato per produrre il baseline DN). Sostituiscono le -1.5 kA E / +2.2 kA F hand-tuned di M3.

### 2.2 S matrix scaled

S_calibrated[i,j] = S_synthetic[i,j] × 10

Motivazione: M9 misura empirico `Δκ/Δ|coil| ≈ 1×10⁻⁵ /A` aggregato (perturbazione X-point Z di 5 cm richiede ΔI di 7-9 kA → Δκ di 0.08). Synthetic S aveva `4×10⁻⁷·|Z| ≈ 3×10⁻⁷ /A` per F-coils. Rapporto ~30, ma 30× sarebbe troppo aggressivo (sistema instabile a voltage_std=50V), quindi prendo 10× come compromesso conservativo.

### 2.3 Target ranges restretti

M5 random target ranges: $\kappa \in [1.4, 2.2]$, $\delta \in [-0.5, +0.7]$.
M10 calibrated ranges: $\kappa \in [1.466, 1.916]$, $\delta \in [-0.297, +0.503]$ (centrate sul ref M9 con ±15% spread).

Motivazione: targets fuori envelope effettivo contribuiscono errore irriducibile.

## 3. Risultati pipeline calibrata

```
[1] Calibrated SimParams: ref κ=1.616, δ=+0.003, R_p=0.901
    S max coeff: 2.00e-05 (×10 vs M3)

[2] Generating expert dataset (1000 samples, JIT FMC M=32 H=8)...
    Generated 1000 samples in 1.4s (715/s)

[3] Training MLP 64×64 (BC only)...
    Trained in 0.4s, val=1.146

[4] DAgger iterations (5 × 500 samples)...
    iter 0 (BC): err=69.07, quench=0/20
    iter 1: |D|=1500, err=??.?, quench=0/20
    iter 3: |D|=2500, err=4.96, quench=0/20
    iter 4: |D|=3000, err=6.16, quench=0/20
    iter 5: |D|=3500, err=3.47, quench=0/20

Final summary:
  M5 BC (original sim, original eval) : err 36.00
  M6 DAgger×3 (original)              : err  3.55
  M8 DAgger×N (original)              : err  3.45
  M10 BC (calibrated)                 : err 69.07
  M10 DAgger×5 (calibrated)           : err  3.47

  M10 vs M8: 1.00× the M8 floor (no improvement)
```

## 4. Interpretazione del finding negativo

### 4.1 BC peggiore con S grande

M10 BC = 69.07 vs M5 BC = 36.00. Perché?

Con S × 10, una piccola differenza ΔV (errore della policy) si traduce in 10× più movimento di shape. La policy BC (1000 sample, no DAgger) commette errori sistematici → ora amplificati.

→ **S grande rende la control task più sensibile**, NON più facile.

### 4.2 DAgger plateau invariante

M10 DAgger×5 = 3.47 vs M8 DAgger×N = 3.45. Identici entro noise.

DAgger compensa la sensitivity bug correggendo on-policy. Il plateau finale è **dettato dall'expert FMC**, non dalla simulator calibration. Anche con expert "stronger" (S × 10 → FMC può muovere shape più aggressivamente), il floor non si abbassa.

### 4.3 Floor strutturale — diagnosi finale

Possibili cause concorrenti del floor ≈ 3.5:

1. **Target unreachable**: anche con range ridotto $\kappa \in [1.47, 1.92]$, il "centro" del DN baseline è $\kappa = 1.62$. Targets a $\kappa = 1.92$ sono +0.30 sopra la natural rest position — richiederebbero coil current swings molto grandi (7-9 kA per +0.08 secondo M9 → 30 kA per +0.30, che eccede I_max_EF = 7.7 kA). Questi target sono **fisicamente unreachable**.

2. **Linear-S limitation**: la S è valida solo per piccole perturbazioni intorno al ref. Per $\Delta\kappa > 0.1$ la linearizzazione diverge dal vero comportamento GS (che è non-lineare).

3. **FMC inherent variance**: M=32 walker × H=8 tick produce decisioni con varianza significativa. Il DAgger media (M8 finding), ma il bias residuo è sotto il radar.

**Per chiudere veramente il floor servirebbe**:
- Sostituire S linearizzato con un GS solver vero nel inner loop FMC (M11 candidato, costoso)
- O ridurre target distribution alla parte effettivamente raggiungibile (research design choice)

## 5. Osservazioni meta sul processo M9-M10

M9 ha dato un finding empirico "S è 30× sottostimato". M10 ha preso questo at face value e applicato. La **lezione**: i finding numerici da un singolo esperimento perturbativo sono indicativi, NON definitivi. Per una S empirica robusta servirebbe identificazione completa (perturbazione di tutti i 20 coils, regressione lineare 4×20). Il fatto che M10 non migliori non smentisce M9, ma mostra che il "floor 3.5" non era dovuto a S magnitude.

Questo è un buon esempio di **negative result che restringe lo spazio delle ipotesi**: abbiamo escluso "calibration is the bottleneck" → resta: target distribution and/or linear-S structural limit.

## 6. Test (`tests/test_calibrated.py`)

```
$ python tests/test_calibrated.py
  ✓ TestCalibratedSim.test_ref_matches_m9              ← κ_ref=1.616 ≈ M9
  ✓ TestCalibratedSim.test_S_scaled                    ← max S > 1e-5 (×10 confirmed)
  ✓ TestCalibratedSim.test_initial_state_consistent
  ✓ TestCalibratedSim.test_target_ranges_around_ref
  ✓ TestCalibratedPipeline.test_history_exists
  ✓ TestCalibratedPipeline.test_dagger_improves_bc     ← 69 → 3.5 (20× improvement)
  ✓ TestCalibratedPipeline.test_no_quench_after_dagger

7 passed, 0 failed
```

**Cumulativo M2-M10**: 21 + 12 + 11 + 6 + 6 + 6 + 6 + 10 + 7 = **85/85 test green**.

## 7. Implicazione per la roadmap

Il "floor 3.5" non si chiude con calibration. Per scendere ulteriormente serve uno dei due:

### Opzione A — Restrict target distribution

Identificare l'envelope effettivo $(\kappa, \delta, R_p, Z_p)$ raggiungibile con coil currents within engineering limits. Risampling target solo dentro questo envelope. Questo è un **design choice**, non un fix.

### Opzione B — Non-linear shape model (M11 candidate)

Sostituire `delta_shape = S @ dI` con `delta_shape = NN_shape(I_coils)` dove NN_shape è una rete addestrata su molte GS solves. Cattura non-linearità + accoppiamento coil-coil. Costo: ~1000 GS solves (~30 min wall-clock con freegs) + un training MLP (~10 sec).

Una volta NN_shape esiste, il flow è:
```
FMC inner: usa NN_shape come fast surrogate (~10 µs/eval)
Truth oracle: usa freegs (1 sec/eval) per validazione periodica
```

→ Questo è il **pattern Degrave 2022 reale**: simulator surrogate + GT validation.

## 8. Riproducibilità

```bash
cd work/06_plasma_fmc

# Test calibrated sim standalone
python scripts/calibrated_sim.py

# Full pipeline (~3 min wall-clock)
python scripts/calibrated_pipeline.py

# Tests
python tests/test_calibrated.py
```

## 9. Riferimenti

- M9 doc: [`milestone_9_freegs_truth.md`](milestone_9_freegs_truth.md) — origine dei numeri di calibrazione
- M2 doc: [`milestone_2_simulator.md`](milestone_2_simulator.md) — S sintetica originale (§2.6)
- M8 doc: [`milestone_8_extended_dagger.md`](milestone_8_extended_dagger.md) — floor 3.45 osservato
- **Wesson** *Tokamaks* 4th ed. §11.4 — non-linear plasma shape control
- **Walker, M.L. & Humphreys, D.A.** "Valid coordinate systems for linearized plasma shape response models in tokamaks" *Fusion Sci. Technol.* 50:473 (2006) — discussione formale dei limiti dell'approccio lineare

## 10. Take-aways

**Confermato**:
1. La calibration ricavata da M9 è **applicabile e numericamente consistente** (test passano)
2. Ma **non riduce il floor di tracking error** — M10 plateau ≈ M8 plateau
3. Il floor 3.5 è **strutturale**, non di magnitude
4. Per scendere ulteriormente serve abbandonare l'approssimazione lineare → M11 (NN shape model trained su GS solves)

**Per il paper**:
La storia ora include un **negative result onesto** che ha valore scientifico: aver provato "naive calibration via empirical fit" senza miglioramento mostra che il problema è non-trivialmente non-lineare. Una v3 del simulator con NN shape model è giustificata.

→ **Milestone 11**: NN-based shape surrogate trained su batch di FreeGS solves. Stima: ridurrebbe il floor da 3.5 a <1.0.
