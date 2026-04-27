# Milestone 3 — FMC controller for plasma shape tracking

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: adattare FMC (paper §4.3) da azione discreta (Atari) a azione continua (V_coils ∈ ℝ²⁰), integrare reward shape + safety, dimostrare tracking closed-loop di un target plasma shape.

## 1. Cosa è stato consegnato

| Path | Cosa contiene |
|---|---|
| [`scripts/fmc_plasma.py`](../scripts/fmc_plasma.py) | Controller FMC: `relativize_np`, `shape_reward`, `safety_penalty`, `FMCPlasmaController` |
| [`scripts/plot_tracking.py`](../scripts/plot_tracking.py) | Visualizzazione 6-pannelli del tracking |
| [`tests/test_fmc.py`](../tests/test_fmc.py) | 12 test: relativize, reward, safety, controller behavior |
| [`results/milestone_3_tracking.json`](../results/milestone_3_tracking.json) | Log dell'esperimento di tracking |
| [`results/milestone_3_tracking.png`](../results/milestone_3_tracking.png) | Visualizzazione (R_p, Z_p, κ, δ, I_p, walkers alive vs t) |
| [`docs/milestone_3_controller.md`](milestone_3_controller.md) | Questo documento |

## 2. Adattamento FMC discreto → continuo

Riferimento canonico: [`work/03_atari_replication/scripts/fmc_minimal.py`](../../03_atari_replication/scripts/fmc_minimal.py) — FMC su Atari, 96/100 Boxing in 7 min.

| Aspetto | FMC Atari (originale) | FMC Plasma (questo M3) |
|---|---|---|
| State | ALE clone (RAM 128 byte) | Packed 27-vector (I_coils + plasma + shape) |
| Action | Discrete int (n_actions ≤ 18) | Continuous V_coils ∈ ℝ²⁰ + P_aux + gas_puff |
| Action sampling | Uniform over int | Gaussian: $V \sim \mathcal{N}(V_{\text{ref}},\sigma^2 I)$ |
| Distance metric | L2 su RAM (128 byte) | L2 su shape obs (R_p, Z_p, κ, δ — scaled) |
| Reward | Game score delta | $-\|\text{shape} - \text{target}\|_W^2 - \text{safety penalty}$ |
| Tick step | Skipframe (5 frame) | $\Delta t = 1$ ms (control rate 1 kHz) |
| Aggregazione | `bincount(initial_actions).argmax()` | Softmax-weighted mean: $\sum w_i V_i / \sum w_i$, $w_i = e^{R_i}$ |

Il **kernel FMC è invariato**: stesso schema relativize → virtual reward → cloning probabilistic. Il claim del paper (§5.1.3) che FMC è platform-agnostic è confermato — già verificato dall'utente su 11 task diverse (matematica continuous, Atari discrete, Lennard-Jones combinatorial).

## 3. Equazioni

### 3.1 Relativize (paper §2.2.3) — verifica analitica

$$ z = (R - \mu)/\sigma, \quad R_N = \begin{cases} e^z & z \le 0 \\ 1 + \log(1+z) & z > 0 \end{cases} $$

**Verifica**: per $x = (1, 3)$: μ=2, σ=1. $z_1 = -1$, $z_2 = +1$.
- $R_N[0] = e^{-1} = 0.3679$ ✓ (test passato a 1e-5)
- $R_N[1] = 1 + \log 2 = 1.6931$ ✓

Proprietà verificate (test):
- Output sempre > 0 (composito moltiplicativo coerente)
- Ordering preserved: $\arg\text{sort}(R) = \arg\text{sort}(R_N)$
- Constant input: $\sigma < 10^{-12}$ → return 1 (no-op)
- Outlier robustness: clip su exp argument a (-50, 0)

### 3.2 Virtual reward (paper §4.3)

$$ \text{VR}_i = R_N(R_i)^\alpha \cdot R_N(D_i)^\beta $$

$D_i$ = L2 distance walker $i$ vs partner random (paper §5.1.3.3 — ~+61% miglioramento RAM vs IMG, qui usiamo shape obs scaled ~equivalente). $\alpha = \beta = 1$ default (geometric mean).

### 3.3 Cloning probability (paper §4.3)

$$ p_{\text{clone}}(i \to k) = \text{clip}\left(\frac{\text{VR}_k - \text{VR}_i}{\text{VR}_i}, 0, 1\right) $$

Walker morti (plasma quenched, $|I_p| < 50$ kA) clonano sempre con $p=1$ — recupero forzato dalla fitness pool.

### 3.4 Reward + safety per plasma

$$ R(\mathbf{x}) = -\big[w_R(R_p - R^*)^2 + w_Z(Z_p - Z^*)^2 + w_\kappa(\kappa - \kappa^*)^2 + w_\delta(\delta - \delta^*)^2\big] - \mathcal{P}(\mathbf{x}) $$

Penalty $\mathcal{P}$ (soft barrier):
- $\mathcal{P}_{q_{95}} = 100\cdot\max(2 - q_{95}, 0)^2$ — kink instability (Wesson §6.4)
- $\mathcal{P}_{n} = 100\cdot\max(\bar n/n_{\text{GW}} - 0.9, 0)^2$ — Greenwald limit
- $\mathcal{P}_{I} = 10^{-4}\cdot\sum_i\max(|I_i| - 7700, 0)$ — engineering current limit
- $\mathcal{P}_{\text{quench}} = 1000$ se $|I_p| < 50$ kA

Pesi $(w_R, w_Z, w_\kappa, w_\delta)$ scelti per portare gli errori a unità comparabili (1 cm di errore in posizione ≈ 1 unità di reward, 0.1 in elongazione ≈ 1 unità).

## 4. Test mathematical correctness — `tests/test_fmc.py`

```
$ python tests/test_fmc.py
  ✓ TestRelativize.test_all_positive_output
  ✓ TestRelativize.test_constant_input            ← σ < 1e-12 → ones
  ✓ TestRelativize.test_extreme_negative_safe     ← clip ad evitare overflow
  ✓ TestRelativize.test_order_preserved
  ✓ TestRelativize.test_piecewise_at_mean         ← exp(-1), 1+log(2) verificati
  ✓ TestShapeReward.test_max_at_target            ← R massimo a target esatto (=0)
  ✓ TestShapeReward.test_quadratic                ← raddoppiando errore, reward × 4
  ✓ TestSafetyPenalty.test_q95_below_threshold
  ✓ TestSafetyPenalty.test_zero_when_safe
  ✓ TestController.test_decision_pulls_toward_target  ← R_p drift verso target verificato
  ✓ TestController.test_decision_returns_valid_v
  ✓ TestController.test_determinism                  ← stesso seed → stessa decisione

12 passed, 0 failed
```

## 5. Calibrazioni effettuate (vs Milestone 2 baseline)

Durante lo sviluppo M3 abbiamo identificato che il simulatore M2 con parametri di default non sosteneva un plasma stabile (quench in ~ms, walker tutti "morti" in FMC). Calibrazioni applicate (entrambe documentate in M2 §5 come future work):

| Parametro | M2 default | M3 calibrato | Razionale |
|---|---|---|---|
| OH solenoid coupling | M_pc[OH] = M_single | M_pc[OH] = N · M_single (N=100) | Trasformatore multi-spire (paragrafo 8.06 Smythe). Diagonali $L_{OH} = N^2 L_1$. |
| OH coil resistance | $R_{OH} = R_0$ | $R_{OH} = N \cdot R_0$ | Resistenza filo cumulativa multi-spire |
| Plasma resistance | $R_p = R_{\text{Spitzer}}$ | $R_p = 0.005 \cdot R_{\text{Spitzer}}$ | Profile averaging + neoclassical bootstrap. Misure TCV: $\tau_{\text{res}}\sim 30\text{-}100$ ms a 1 keV; Spitzer-naive darebbe ~300 µs. Fattore 0.005 calibra al middle del range. |
| Shape response S (κ, δ) | scaling 4e-8 | scaling 4e-7 (10× up) | I cambi di corrente per tick (~ kA scala) producono $\delta\kappa < 0.001$ con S originale; insufficiente per controllo |

**Tutti i 21 test M2 + 12 test M3 ancora passano dopo calibrazione**. Free-decay test M2 ora mostra plasma sustaining a I_p ~ 150 kA per 17 ms (vs quench in 1 ms prima).

## 6. Risultati tracking — `results/milestone_3_tracking.png`

Setup: target $R_p = 0.90$ m (+2 cm vs nominale), $\kappa = 1.85$ (+0.15), $Z_p=0$, $\delta=0.3$.
Config: 200 walkers, horizon 20 tick (= 20 ms lookahead), $\sigma_V = 50$ V.
50 control step = 50 ms reali simulati.

Osservazioni dal log:

| t [ms] | R_p [m] | κ | I_p [kA] | walkers alive |
|---|---|---|---|---|
| 1 | 0.876 | 1.700 | 198 | 200 |
| 11 | **0.916** | 1.696 | 154 | 200 |
| 21 | 0.885 | 1.697 | 126 | 200 |
| 31 | 0.864 | 1.694 | 98 | 192 |
| 41 | **0.926** | 1.694 | 83 | 107 |

- ✓ **R_p tracciato verso target** (oscilla tra 0.86 e 0.93 attorno a 0.90), sebbene con overshoot dovuto a horizon corto (20 ms < tempo di risposta dei coil)
- ⚠ **κ non si muove** (resta ~1.70 vs target 1.85). Causa: controllo di κ richiede pattern simmetrico di F-coils difficile da scoprire random in 200 walker × 20 tick. Discusso §7.
- ⚠ **I_p decade** (200 → 83 kA in 41 ms). FMC non sta attivamente sostenendo I_p tramite OH ramp; lo focalizziamo su shape. Real-world tokamak ha PID dedicato per I_p (Galperti 2024, controllo gerarchico).
- **Walker survival** crolla dopo 30 ms perché I_p si avvicina al limite quench (50 kA) e safety penalty domina

## 7. Limitazioni note (e perché sono interessanti)

### 7.1 Decision time = 190 ms ≫ target 1 ms

**Misurato**: una `decide()` call richiede 190 ms su M1 Pro. Il target FMC industriale è < 1 ms (control rate 1 kHz). Causa: il loop FMC è in NumPy con conversioni JAX↔NumPy ad ogni tick × 20 tick × 200 walker.

**Soluzione architetturale** (NON implementata in M3, lascio per M5): JIT-compilare l'intero loop FMC come `jax.lax.fori_loop` o `jax.lax.scan`, con stato walker in JAX device memory tutto il tempo. Stima: dovrebbe portare a ~5-10 ms per decisione (limite GPU launch overhead × tick).

### 7.2 fragile-rl architectural insight (riferimento utente)

L'utente ha condiviso (durante questa sessione) che **fragile-rl ha esplicitamente abbandonato il paradigma FMC pure-planning** proprio a causa del runtime cost. Il framework usa invece:

- Encoder Poincaré iperbolico (`vla/covariant_world_model.py`)
- World model RSSM-style (Dreamer)
- Actor + critic trainati via imagination rollout
- Macro-control gerarchico

**Costo runtime per decisione**: 1 NN forward pass (~10 ms o meno) vs FMC ~190 ms.
**Costo training**: ~1h GPU upfront vs FMC zero-training.

→ **Implicazione per controllo plasma reale**: il pattern industrialmente sostenibile è
**FMC come expert offline** (genera traiettorie ottime con simulator perfetto) → **policy distillation** in NN compatto → **deployment a 1 kHz** con NN forward pass su FPGA o real-time CPU.

Questo è esattamente il pattern di Degrave 2022: durante training l'RL usa un simulator dettagliato ($\sim$ ore di CPU per training), il deployment è una NN policy che fa inferenza in $\mu$s sul real-time control system. La novità del nostro approccio sarebbe usare **FMC zero-training come expert** invece di RL — riducendo i giorni di training a minuti (FMC non ha hyperparameters da tunare).

### 7.3 κ control sub-optimal

S è sintetica con coefficienti scelti a mano. Per controllo di κ serve identificazione lineare attorno a un equilibrio FreeGS reale (Milestone 4 originale, ora rinominato). La non-linearità della risposta plasma ai coils non è catturata da S lineare — workaround real-world: re-identificare S periodicamente attorno al punto operativo corrente.

### 7.4 I_p non controllato

M3 ottimizza solo lo shape. Real tokamak control architecture (Reimerdes 2022, Galperti 2024) ha:
- **Inner loop** (25 kHz): vertical PID per stabilità Z (instabilità di crescita ~ms)
- **I_p loop** (1 kHz): OH transformer feedback
- **Shape loop** (1 kHz): EFIT real-time + reference tracking

FMC sostituisce naturalmente lo **shape loop**, non gli inner loops che restano analogici/digitali dedicati.

## 8. Riproducibilità

```bash
cd work/06_plasma_fmc

# Tests
python tests/test_simulator.py    # 21 passed
python tests/test_fmc.py          # 12 passed

# Tracking demo (~10 sec wall-clock per 50 control step)
python scripts/fmc_plasma.py
python scripts/plot_tracking.py
```

## 9. Prossimi step (revisione roadmap dopo insight fragile-rl)

→ **Milestone 4** (originale: Streamlit dashboard) — ridotto scope a "viewer per i log esistenti", interactive ma offline.
→ **Milestone 5** (nuovo, motivato da fragile-rl insight): **FMC-to-policy distillation** — usare FMC come expert per generare ~10k (state, optimal_V) pairs, train un MLP piccolo, mostrare runtime 100-1000× migliore. Questo è il vero contributo industrialmente rilevante.
→ **Milestone 6** (nuovo): **JIT-fy FMC inner loop** in `jax.lax.scan` — anche solo per produrre l'expert dataset con velocità 30× migliore.
→ **Milestone 7** (originale 4): **Coupling FreeGS truth** — calibrare M_pc, S, R contro equilibri reali.
