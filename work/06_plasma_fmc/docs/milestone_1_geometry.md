# Milestone 1 — TCV geometry & reference plasma shapes

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: caricare la geometria del tokamak TCV con accuratezza pubblicata, definire 3 forme di plasma di riferimento (Miller LCFS), verificare numericamente ogni claim contro fonti primarie.

## 1. Cosa è stato consegnato

| Artefatto | Path | Cosa contiene |
|---|---|---|
| Geometry config | [`config/tcv_geometry.yaml`](../config/tcv_geometry.yaml) | 16 shaping coils + 3 T + OH circuit (solenoid + C1/C2 + D1/D2), vessel approssimato, limiti ingegneristici |
| Loader + cross-check | [`scripts/tcv_geometry.py`](../scripts/tcv_geometry.py) | `TCVMachine` dataclass, `load_tcv()`, `cross_check_against_freegs()` |
| Reference shapes | [`scripts/reference_shapes.py`](../scripts/reference_shapes.py) | Miller LCFS per 3 scenari + quantità derivate (V, S, q₉₅, n_GW, β_max) |
| Plotting | [`scripts/plot_geometry.py`](../scripts/plot_geometry.py) | Visualizzazioni Milestone 1 |
| Visualizzazioni | [`results/milestone_1_geometry.png`](../results/milestone_1_geometry.png), [`results/milestone_1_shapes.png`](../results/milestone_1_shapes.png) | Cross-section + 3 LCFS overlay + per-shape derivati |
| Bibliografia | [`REFERENCES.md`](../REFERENCES.md) | Tag → fonte (TCV-overview, TCV-coils-LRP755, TCV-coils-freegs, …) |

## 2. Parametri macchina (verificati)

| Parametro | Valore | Tag fonte |
|---|---|---|
| Major radius R₀ | 0.88 m | TCV-overview |
| Minor radius a | 0.25 m | TCV-overview |
| Inverse aspect ratio ε = a/R₀ | 0.2841 | calcolato → match esatto 0.25/0.88 |
| Toroidal field B_T,max | 1.5 T | TCV-overview |
| Plasma current I_p,max | 1.0 MA | TCV-overview |
| Elongation κ_max | 2.8 | TCV-overview |
| Triangularity range δ | [-0.7, +1.0] | TCV-overview |
| Shaping coils | 16 (E1-E8 @ R=0.5050 m, F1-F8 @ R=1.3095 m) | TCV-coils-LRP755 + TCV-coils-freegs |
| T coils | 3 (R=1.5540, 1.7170, 1.7540 m, tutte Z=-0.78 m) | TCV-coils-freegs |
| OH circuit | solenoide (R=0.43 m, Z∈[-0.93, +0.93], 100 turns) + C1/C2 + D1/D2 | TCV-coils-freegs |
| Control channels | 20 = 16 shaping + 3 T + 1 OH circuit | match Degrave-2022 (19 control voltages — abbiamo 1 in più perché Degrave fonde T2/T3) |
| I_max E/F | 7.7 kA | TCV-current-limits |
| I_max OH | 20 kA flat-top | TCV-current-limits |

### Cross-check vs `freegs.machine.TCV()`

```
$ python scripts/tcv_geometry.py
Loading TCV geometry...
  Machine: R₀=0.88 m, a=0.25 m, ε=0.2841
  Shaping coils: 16 (E1-E8 + F1-F8)
  T coils: 3
  Total control channels: 20
  E coil R: 0.5050 m (E1-E8 all same)
  F coil R: 1.3095 m (F1-F8 all same)

Cross-checking against freegs.machine.TCV()...
  ✓ Perfect match for all coil positions

All verifications passed.
```

Test: ogni coil del nostro YAML, confrontato per (R, Z) con la corrispondente entry in `freegs.machine.TCV()`, ha differenza < 1e-6 m. Diffs trovati: **0**.

## 3. Reference shapes (Miller LCFS)

Parametrizzazione (REFERENCES §D.5):
$$ R(\theta) = R_0 + a\cos(\theta + \arcsin(\delta)\sin\theta), \qquad Z(\theta) = Z_0 + \kappa a \sin\theta $$

Tre scenari canonici:

| Nome | κ | δ | Z₀ [m] | a effettivo [m] | Note |
|---|---|---|---|---|---|
| `standard_single_null` | 1.7 | +0.3 | 0.0 | 0.240 (0.96·a) | Tipico ELMy SN, riferimento Reimerdes 2022 |
| `snowflake` | 2.0 | +0.4 | +0.05 | 0.230 (0.92·a) | Approssimazione Miller (vero snowflake ha 2 X-point, richiede GS solve — Milestone 2) |
| `negative_triangularity` | 1.5 | −0.4 | 0.0 | 0.240 (0.96·a) | Sauter 2014, TCV NT scenario |

Tutte le LCFS verificano `R_min ≥ vessel_inner_R = 0.624 m` e `R_max ≤ vessel_outer_R = 1.136 m` e `|Z|_max ≤ 0.75 m`. ✓ Geometry violations: 0.

## 4. Quantità derivate — verifica analitica indipendente

A B_T = 1.43 T, I_p = 200 kA, output del codice:

| Shape | V [m³] | S [m²] | q₉₅,cyl | n_GW [m⁻³] | β_max [%] |
|---|---|---|---|---|---|
| standard_single_null | 1.7009 | 11.6282 | 4.551 | 1.105×10²⁰ | 1.632 |
| snowflake | 1.8378 | 12.6340 | 5.373 | 1.203×10²⁰ | 1.703 |
| negative_triangularity | 1.5008 | 10.6287 | 3.802 | 1.105×10²⁰ | 1.632 |

### 4.1 Volume D-shape — Wesson §1.4
$$ V = 2\pi^2 R_0 a^2 \kappa $$
Standard SN: 2π² · 0.88 · 0.24² · 1.7 = 19.7392 · 0.0979 = **1.701 m³** ✓ (codice: 1.7009)

### 4.2 Surface area D-shape (1° ordine, ignora δ) — Wesson §1.4
$$ S = 4\pi^2 R_0 a \sqrt{(1+\kappa^2)/2} $$
Standard SN: 4π² · 0.88 · 0.24 · √((1+2.89)/2) = 8.342 · 1.394 = **11.628 m²** ✓

### 4.3 q₉₅ cilindrico — Wesson §3.6
$$ q_{95}^{\text{cyl}} = \frac{5\, a^2 B_T}{R_0 I_p[\text{MA}]} \cdot \frac{1+\kappa^2}{2} $$
Standard SN (a=0.24): 5·0.0576·1.43 / (0.88·0.2) · (1+2.89)/2 = 2.341 · 1.945 = **4.554** ✓ (codice: 4.551, differenza solo da arrotondamento)

### 4.4 Greenwald — Greenwald PPCF 44:R27 (2002)
$$ n_{\text{GW}} [10^{20}\,\text{m}^{-3}] = \frac{I_p[\text{MA}]}{\pi a^2[\text{m}^2]} $$
Standard SN: 0.2 / (π · 0.0576) = 0.2 / 0.1810 = **1.105** ✓

### 4.5 Troyon β_max — Troyon PPCF 26:209 (1984)
$$ \beta_{\max}[\%] = \beta_N^{\max} \cdot \frac{I_p[\text{MA}]}{a B_T} \quad\text{con}\ \beta_N^{\max} = 2.8 $$
Standard SN: 2.8 · 0.2 / (0.24 · 1.43) = 0.56 / 0.3432 = **1.632** ✓

**Conclusione**: tutte le quantità derivate riproducono a mano i valori del codice entro arrotondamento.

## 5. Note di scope (cose che Milestone 1 NON pretende)

1. **Le 3 forme NON sono equilibri Grad-Shafranov consistenti**. Sono target geometrici parametrici. Il solver Grad-Shafranov free-boundary che trova le correnti coil necessarie a riprodurle è Milestone 2.
2. La **vessel** è approssimata a un rettangolo per il plotting. Non è la vera D-shape della camera TCV. Sufficiente per check di contenimento.
3. La **mutua induttanza coil-plasma** non è ancora stata calcolata. Arriva in Milestone 2 (formula di Neumann tra anelli filiformi — REFERENCES §G H3).
4. I **voltage rails ±1500 V** sono assunti (H1 in REFERENCES §G); EPFL non pubblica i valori esatti dei thyristor PS. Si rivede se troviamo il dato.
5. Il **secondo X-point** della snowflake non è rappresentato — vero snowflake richiede uno snapshot di flusso poloidale 2D, di nuovo Milestone 2.

## 6. Riproducibilità

```bash
cd work/06_plasma_fmc
python scripts/tcv_geometry.py        # cross-check vs freegs (deve stampare ✓)
python scripts/reference_shapes.py    # quantità derivate
python scripts/plot_geometry.py       # genera 2 PNG in results/
```

Dipendenze: `freegs==0.8.2`, `numpy`, `matplotlib`, `pyyaml`.

## 7. Prossimo step

→ **Milestone 2**: simulatore di plasma a due livelli.
- **Truth (slow)**: Grad-Shafranov free-boundary con `freegs` per generare snapshot di equilibrio
- **Inner (fast, FMC-callable)**: ROM lineare 0D — energy balance (IPB98) + equazione di circuito coil M·dI/dt + R·I = V
- Target: < 100 µs per step su CPU M1 Pro, < 30 µs su Metal GPU via JAX
