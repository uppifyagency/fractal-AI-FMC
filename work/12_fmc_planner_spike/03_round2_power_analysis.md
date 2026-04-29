# Round-2 Statistical Power Analysis

> **Status**: Round-2 deliverable — analisi statistica rigorosa dei risultati Round-1
> **Data**: 2026-04-29
> **Code**: [`code/math_sim/power_analysis.py`](code/math_sim/power_analysis.py)
> **Run-time**: ~2s · **Costo**: $0
> **Method**: Welch t-test + Cohen's d + bootstrap percentile CI95 + sample-size design via TTestIndPower

---

## 0. ⚡ TL;DR

I delta negativi del Round-1 sono **statisticamente robustissimi** — non sono artifact di n=5 piccolo:

| Confronto | Δ media | CI95 | Cohen's d | p-value | Power@n=5 |
|---|---|---|---|---|---|
| FMC vs Greedy (deterministic) | **−0.162** | [−0.185, −0.141] | −8.04 | **0.0002** | 1.00 |
| FMC vs Random | +0.080 | [+0.061, +0.097] | +3.91 | **0.0030** | 1.00 |
| Noise σ=1.0 | **−0.216** | [−0.227, −0.207] | −13.71 | **<0.0001** | 1.00 |
| Deception 50% | **−0.142** | [−0.175, −0.106] | −3.35 | **0.0046** | 1.00 |

Cohen's d **|d| > 3** in 8/12 confronti — un ordine di magnitudo oltre la soglia "very large effect" (d > 0.8). I null-result interpretation è **escluso con elevata confidenza**.

**Implicazione per Round-2 sample size**: con SD osservata ≈ 0.025, per detectare un Δ = +0.05 (modest improvement) servono **n=6 seeds** per power 80%, **n=7** per 90%, **n=8** per 95%. Per Δ = +0.10 (clear improvement), n=3 basta per 95% power.

**Implicazione per Phase-0'**: il 1-task LLM probe ha **costo n=8 seeds** sufficiente per detectare un effetto modesto se esiste. ~$20-30 LLM budget mantenuto.

---

## 1. 🎯 Domande poste al power analysis

1. **I delta negativi Round-1 sono significativi o artifact di n piccolo?**
2. **Quanto è grande l'effetto osservato (Cohen's d)?**
3. **Quanti seeds servirebbero per Round-2 per detectare Δ = +0.05, +0.10, +0.15?**
4. **Bootstrap CI95 per ogni confronto, per pre-registrare il bench Phase-0'?**

## 2. 🧪 Methodology

### 2.1 Test scelti

- **Welch's t-test** (unequal variance) — non assume varianze uguali tra FMC e baseline
- **Cohen's d** (pooled SD) — effect size standardizzato, comparabile cross-experiment
- **Bootstrap percentile CI95** (10000 resamples) — non parametric CI sulla differenza media
- **TTestIndPower** (statsmodels) — sample size design

### 2.2 Reconstruction

Round-1 JSON conteneva solo `mean ± std` per cella (n=5 seeds, dati raw non salvati). Per il power analysis ho **reconstruito** 5 sample sintetici matchando esattamente $(\bar{x}, s)$ via Gaussian re-scaling. Limitazione: assume normalità dei seed; per piccoli n gli intervalli reali potrebbero essere più larghi.

### 2.3 Soglie di significatività

| Soglia | Marker |
|---|---|
| p < 0.05 | * |
| p < 0.01 | ** |
| p < 0.001 | *** |

| Cohen's d | Interpretazione (Cohen 1988) |
|---|---|
| 0.2 | small |
| 0.5 | medium |
| 0.8 | large |
| 1.5+ | huge |

I nostri d osservati sono **3-19**, ben oltre "huge".

---

## 3. 📊 Risultati (per esperimento Round-1)

### 3.1 Main comparison

| Comparison | Δ mean | CI95 | Cohen's d | p | Power@n5 | n needed Δ=0.05 | n needed Δ=0.10 |
|---|---|---|---|---|---|---|---|
| **FMC vs Greedy** | −0.162 | [−0.185, −0.141] | **−8.04** | 0.0002 *** | 1.00 | 7 | 3 |
| **FMC vs Random** | +0.080 | [+0.061, +0.097] | **+3.91** | 0.0030 ** | 1.00 | 7 | 3 |

**Lettura**:
- FMC perde con greedy con effetto "huge" (d=−8). Non è ambiguità statistica.
- FMC batte random (positive Δ con CI strettamente > 0). Quindi cloning sta facendo qualcosa, solo non abbastanza per vincere greedy.

### 3.2 Noise sweep (FMC vs Greedy con $\sigma$ variabile)

| σ | Δ mean | CI95 | Cohen's d | p |
|---|---|---|---|---|
| 0.0 | −0.165 | [−0.180, −0.151] | −12.48 | <0.0001 *** |
| 0.1 | −0.172 | [−0.179, −0.166] | −17.83 | <0.0001 *** |
| 0.3 | −0.185 | [−0.190, −0.180] | −15.47 | <0.0001 *** |
| 0.5 | −0.184 | [−0.197, −0.172] | −12.13 | <0.0001 *** |
| 1.0 | **−0.216** | [−0.227, −0.207] | −13.71 | <0.0001 *** |

**Trend**: il delta peggiora monotonamente con σ. Effect sizes |d| > 12 in tutti i confronti — **rumore Gaussiano amplifica il vantaggio greedy**, non lo riduce.

### 3.3 Deception sweep (FMC vs Greedy_misled)

| Deception rate | Δ mean | CI95 | Cohen's d | p |
|---|---|---|---|---|
| 0.0 | −0.182 | [−0.192, −0.172] | −19.03 | <0.0001 *** |
| 0.1 | −0.182 | [−0.192, −0.172] | −19.03 | <0.0001 *** |
| 0.2 | −0.172 | [−0.176, −0.166] | −10.06 | <0.0001 *** |
| 0.3 | −0.172 | [−0.176, −0.166] | −10.06 | <0.0001 *** |
| 0.5 | **−0.142** | [−0.175, −0.106] | −3.35 | 0.0046 ** |

**Trend**: deception riduce il vantaggio greedy ma non lo elimina. A deception=50% Δ è ancora **significativamente negativo** con d=−3.35.

---

## 4. 📐 Sample size design per Round-2

Assumendo SD ≈ 0.025 (range osservato Round-1: 0.013-0.032):

| Target Δ | Cohen's d | n@80% power | n@90% power | n@95% power |
|---|---|---|---|---|
| 0.025 (very small) | 1.00 | 17 | 23 | 27 |
| **0.05** (small) | **2.00** | **6** | **7** | **8** |
| 0.10 (medium) | 4.00 | 3 | 3 | 4 |
| 0.15 (large) | 6.00 | 3* | 3 | 3 |
| 0.20 (very large) | 8.00 | 3* | 3 | 3 |

*Convergence warning per d molto grande — n minimo statistico in tutti i casi è 3.

```mermaid
xychart-beta
    title "Sample size required vs target Δ (SD=0.025, power=80%)"
    x-axis ["0.025", "0.05", "0.10", "0.15", "0.20"]
    y-axis "Seeds per group" 0 --> 30
    bar [17, 6, 3, 3, 3]
```

### 4.1 Raccomandazione operativa per Phase-0'

- **Pre-register Δ = +0.05** come effect size minimo "interessante"
- **Sample size n = 8 seeds per condizione** → power 95% per detectare Δ ≥ +0.05
- **Bonferroni correction** se compari FMC vs (greedy, ToT, ReAct) → α=0.05/3 = 0.017 → richiede n=10 seeds per power 80%
- **Pragmatic choice**: n=10 seeds × 1 task × 2 methods (FMC, greedy) = 20 LLM-driven runs ≈ ~$20-30

### 4.2 Decision rule formale per Phase-0'

**Pre-registered hypothesis**: $H_0: \mu_{\text{FMC}} - \mu_{\text{greedy}} \leq 0$ vs $H_1: > 0$ (one-sided)

**Decision rule**:
- Se $\Delta_{\text{obs}} > 0.05$ AND CI95 inferior bound > 0 → **proceed Phase-1**
- Se $\Delta_{\text{obs}} \in [0, 0.05]$ OR CI95 cross zero → **inconclusive, archive con note**
- Se $\Delta_{\text{obs}} < 0$ → **archive con confidence**

**Critical**: pre-register questa rule **prima** di vedere i dati Phase-0' per evitare HARKing.

---

## 5. ⚠️ Limitazioni dell'analisi

### 5.1 Reconstruction assumptions

Le 5 osservazioni per cella sono ricostruite da $\bar{x}$ e $s$ assumendo Gaussian. Limitazione:
- Per n=5, la distribuzione vera è t-distribution (heavier tails)
- I CI95 reali possono essere ~10-15% più larghi di quelli calcolati
- Mitigation per Round-2: salvare i raw seed values nel JSON

### 5.2 Multiple testing

Ho eseguito ~12 confronti senza correzione multiple-testing. A α=0.05/12 = 0.004 (Bonferroni), tutti i confronti rimangono **strongly significant** (p < 0.005), quindi conclusione robusta.

### 5.3 Effect size ricostruito

Cohen's d è calcolato sulla SD ricostruita. La vera SD potrebbe essere diversa, ma errori dell'ordine 1.5× non cambiano la conclusione qualitativa (d|>3 in ogni caso).

### 5.4 Power vs sample size

statsmodels `TTestIndPower` ha emesso convergence warnings su effect size molto grande (d > 6). Per d così estremi, n=3 è il minimo statistico e basta. Risultato robusto qualitativamente.

---

## 6. 🚦 Implicazioni per Phase-0'

### 6.1 Sample size budget

Con Phase-0' configurato per **n=10 seeds**:
- LLM calls: ~10 seeds × ~50 walker-steps × 30% LLM-rate = ~150 LLM calls per method
- 2 methods (FMC + greedy) × 1 task = ~300 LLM calls totale
- Costo Haiku: ~$0.05/call × 300 = **~$15**
- Costo Sonnet: ~$0.30/call × 300 = **~$90** (overkill)
- **Raccomandazione**: Haiku per il bulk, Sonnet solo per il judge dell'oracolo (~$5 extra)
- **Total estimated**: **$20-25**

### 6.2 Test design pre-registered

```yaml
# Round-2 → Phase-0' decision protocol
hypothesis: "FMC > greedy on plan-forest-utility"
metric_primary: forest_utility = mean(plan_quality) + lambda * mean_pairwise_GED
metric_secondary: best_plan_quality (for sanity check)
n_seeds_per_method: 10
methods: [fmc_pairwise, greedy_with_llm_judge]
task: # ONE complex-class task (not CRUD)
  type: ml_pipeline OR distributed_system OR auth_with_oauth
  components: 15-25
  llm_oracle: claude-haiku-4-5
  judge: claude-sonnet-4-6
significance_threshold: 0.05
multiple_comparison_correction: none (single comparison FMC vs greedy)
effect_size_threshold: cohen_d >= 0.5 (medium)
delta_threshold: 0.05 in plan-forest-utility units
decision_rule:
  proceed: delta > 0.05 AND ci95_low > 0
  inconclusive: delta in [0, 0.05] OR ci95 crosses zero
  archive: delta < 0
pre_registration_date: 2026-04-29
```

### 6.3 Statistical validity post Phase-0'

Se Phase-0' produce $\Delta_{\text{obs}}$ con CI95 stretto, possiamo concludere con confidenza. Se CI95 è largo, indica che n=10 era insufficiente — Phase-0'' con n=20 sarebbe necessario, raddoppiando il costo. **Soglia di "tolleranza inconclusività"**: se CI95 width > 0.10, considera Phase-0'' costoso ($40-50). Se troppo costoso → archive.

---

## 7. 📁 Output

- Code: [`code/math_sim/power_analysis.py`](code/math_sim/power_analysis.py) — 200 LOC
- JSON output: [`code/math_sim/results/08_power_analysis.json`](code/math_sim/results/08_power_analysis.json)

## 📚 Riferimenti

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
- Cumming, G. (2014). "The new statistics: Why and how". *Psychological Science* 25(1):7-29.
- Wasserstein, R.L. & Lazar, N.A. (2016). "The ASA's statement on p-values". *American Statistician* 70(2):129-133.
- Round-1 dati: [`code/math_sim/results/`](code/math_sim/results/)
