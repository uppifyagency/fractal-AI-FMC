# Milestone 2 — Fast inner simulator (FMC-callable)

> **Stato**: ✓ Chiuso (2026-04-27)
> **Scope**: implementare la transizione `(state, control) → state'` per il plasma TCV in forma sufficientemente veloce da consentire 4000 valutazioni per decisione FMC entro 1 ms.

## 1. Architettura — perché due livelli

FMC con M=200 walkers e horizon N=20 richiede **4000 step di simulatore per ogni decisione di controllo**. A 1 kHz (control rate target), il budget è 1 ms = 250 ns per step in forma vettorializzata.

Un solver Grad-Shafranov free-boundary (FreeGSNKE, ~50 ms/step) renderebbe FMC inusabile (4000 × 50 ms = 200 s per decisione). Quindi la pipeline a due livelli:

```
┌────────────────────────────┐         ┌─────────────────────────────┐
│   TRUTH (slow, off-line)   │  pre-   │   INNER SIM (fast, on-line) │
│   FreeGS GS solver          │ identifies│  Reduced-order: 0D energy +  │
│   ms-scale, full physics   │ ─params─▶│  linear PF circuit + linear │
│   Used: M3 calibration     │         │  shape response             │
│                            │         │  µs-scale, FMC-callable     │
└────────────────────────────┘         └─────────────────────────────┘
                                                       │
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │   FMC walker pool (M=200)    │
                                       │   N=20 lookahead              │
                                       │   < 1 ms/decision             │
                                       └──────────────────────────────┘
```

Milestone 2 consegna **solo la parte destra** (inner simulator). FreeGS coupling per generare matrici di risposta linearizzate da equilibri reali è Milestone 3.

## 2. Equazioni implementate

Stato continuo (vettore di lunghezza N+7, con N=20 canali di controllo):

```
x = [I_coils[N], I_p, W, n̄, R_p, Z_p, κ, δ]
```

Controllo per step:

```
u = [V_coils[N], P_aux, gas_puff]
```

### 2.1 Equazione di circuito coil — REFERENCES §D.6 (Walker-Humphreys 2006)

$$ \mathbf{M}\,\frac{d\mathbf{I}}{dt} + \mathbf{R}\,\mathbf{I} = \mathbf{V} $$

- **M** ∈ ℝ^{20×20} simmetrica positiva definita, mutua induttanza tra anelli filiformi (formula di Neumann, REFERENCES §G H3, Jackson §5.17).
- **R** = diag(1 mΩ) — assunzione H2 in REFERENCES §G; verifica: τ_LR ~ M_ii/R_ii ~ 6 µH / 1 mΩ = 6 ms, dello stesso ordine di magnitudo dei tempi di risposta TCV reali.
- **Implicit Euler** per stabilità incondizionata:
  $$ (\mathbf{M} + \Delta t\,\mathbf{R})\,\mathbf{I}_{k+1} = \mathbf{M}\,\mathbf{I}_k + \Delta t\,\mathbf{V} $$

### 2.2 Mutua induttanza (formula di Neumann)

Per due loop coassiali (R₁, Z₁) e (R₂, Z₂):

$$ M = \mu_0 \sqrt{R_1 R_2}\,\Big[\,\big(\tfrac{2}{k} - k\big) K(k) - \tfrac{2}{k}\,E(k)\,\Big] $$

con $k^2 = \frac{4 R_1 R_2}{(R_1+R_2)^2 + (Z_1-Z_2)^2}$, $K, E$ integrali ellittici completi.

Self-inductance (anello sottile, raggio sezione $a_w$):

$$ L_{\text{self}} = \mu_0 R\Big[\ln\frac{8R}{a_w} - \frac{7}{4}\Big] $$

**Verifica numerica** (REFERENCES §G):
- Due loop 1 m, separati 1 m: $k^2 = 4/5 = 0.8$ → $M = \mu_0 \cdot 0.394 \approx 4.95 \cdot 10^{-7}$ H. Codice: 0.4941 µH ✓
- Self-L di loop 1 m con $a_w = 1$ cm: $L = \mu_0 \cdot 1 \cdot (\ln 800 - 1.75) = 6.20$ µH ✓
- Matrice TCV 16×16: simmetrica (max|M-Mᵀ|=0), positiva definita (eig_min = 1.46 µH > 0), diagonalmente dominante (test passato per ogni riga).

### 2.3 Plasma current — flux conservation + Spitzer

$$ L_p\,\frac{dI_p}{dt} = V_{\text{loop}} - R_{\text{plasma}}\,I_p $$

dove $V_{\text{loop}} = -\mathbf{M}_{pc} \cdot d\mathbf{I}/dt$ è la tensione di loop indotta sui coils tramite il vettore di mutua $\mathbf{M}_{pc}$ (plasma trattato come singolo loop filiforme a $(R_p, Z_p)$).

**Spitzer parallel resistivity** (REFERENCES §D, Wesson §3.6):

$$ \eta_{\text{Sp}} = 5.2 \times 10^{-5}\,\ln\Lambda\,T_e[\text{keV}]^{-3/2}\quad [\Omega\,\text{m}] $$

Con $\ln\Lambda \approx 17$ a TCV temperature. $R_p = \eta_{\text{Sp}} \cdot 2\pi R_p / (\pi a^2 \kappa)$.

**Implicit Euler** anche qui — a basso $T_e$, $R_{\text{plasma}}$ può essere ~1 Ω, $\tau_{\text{LR}} = L_p/R \sim 4\,\mu$H/1Ω = 4 µs e l'esplicito sarebbe instabile per $\Delta t = 1$ ms.

### 2.4 Energy balance 0D — REFERENCES §D.4

$$ \frac{dW}{dt} = P_{\text{aux}} + P_{\text{ohm}} - \frac{W}{\tau_E} $$

con $P_{\text{ohm}} = R_{\text{plasma}}\,I_p^2$ e $\tau_E$ dalla legge IPB98(y,2):

$$ \tau_E = 0.0562\,H_{98}\,I_p^{0.93}\,B_T^{0.15}\,P^{-0.69}\,n_e^{0.41}\,M^{0.19}\,R^{1.97}\,\varepsilon^{0.58}\,\kappa^{0.78} $$

Verifica esponenti (test): raddoppiando $I_p$, $\tau_E$ scala di $2^{0.93}$ (verificato a 1e-9 precisione). A condizioni TCV tipiche (200 kA, 1 MW, $n=5\cdot 10^{19}$): $\tau_E \approx 30$ ms, consistente con misure pubblicate (range 1-100 ms ELMy H-mode).

Conversione $T_e \leftrightarrow W$:

$$ W = 3 n V T_e\quad (\text{con } T_e = T_i,\ Z_{\text{eff}} = 1) $$

→ per $W = 40.8$ kJ, $n = 5 \cdot 10^{19}$, $V = 1.7$ m³: $T_e = 1.0$ keV ✓ (test round-trip passato).

### 2.5 Particle balance

$$ \frac{d\bar n}{dt} = \frac{\Phi_{\text{gas}}}{V} - \frac{\bar n}{\tau_p}\quad \tau_p \approx 3\tau_E $$

### 2.6 Linearized shape response

$$ \begin{bmatrix}\delta R_p\\ \delta Z_p\\ \delta\kappa\\ \delta\delta\end{bmatrix} = \mathbf{S}\,(\mathbf{I}_{\text{coils}} - \mathbf{I}_{\text{ref}}) $$

$\mathbf{S} \in \mathbb{R}^{4 \times 20}$ costruita da first-principles physical signs (test verificati):
- F-coils antisimmetrici in Z → $\delta Z_p$ (instabilità verticale: coefficiente positivo, marginal stability per $\Delta t = 1$ ms)
- E vs F → $\delta R_p$ (E inboard pinge fuori, F outboard pinge dentro)
- F simmetrici → $\delta\kappa$ (elongazione enhancement)
- F off-midplane → $\delta\delta$ (triangolarità)

**NB**: In Milestone 3, la matrice S sarà sostituita con una identificata da equilibri FreeGS reali attorno a un punto operativo.

## 3. Tests — `tests/test_simulator.py`

```
$ python tests/test_simulator.py
  ✓ TestMutualInductance.test_diagonal_dominant_self_term
  ✓ TestMutualInductance.test_distance_decay
  ✓ TestMutualInductance.test_matrix_positive_definite
  ✓ TestMutualInductance.test_matrix_symmetric
  ✓ TestMutualInductance.test_reference_value
  ✓ TestMutualInductance.test_self_inductance_positive
  ✓ TestMutualInductance.test_symmetry
  ✓ TestEnergyTemperature.test_T_zero_for_empty_plasma
  ✓ TestEnergyTemperature.test_W_T_inversion
  ✓ TestEnergyTemperature.test_spitzer_scaling
  ✓ TestIPB98.test_basic_scaling
  ✓ TestIPB98.test_tcv_typical_value
  ✓ TestIPB98.test_zero_inputs_safe
  ✓ TestSimulator.test_circuit_steady_state          ← V=R·I_ref → I_coils stays
  ✓ TestSimulator.test_circuit_zero_voltage_decays   ← V=0 → I_coils decay monotonically
  ✓ TestSimulator.test_initial_state_consistent
  ✓ TestSimulator.test_plasma_volume_consistency
  ✓ TestSimulator.test_purity                        ← f(x,u) deterministic
  ✓ TestSimulator.test_shape_response_signs          ← physical signs of S correct
  ✓ TestSimulator.test_state_immutability            ← step() does not mutate input
  ✓ test_mutual_to_plasma_decreases_with_distance

21 passed, 0 failed
```

## 4. Benchmark — Apple M1 Pro

```
======================================================================
Plasma simulator benchmark — JAX backend: [CpuDevice(id=0)]
======================================================================

[1] NumPy single-step (1 walker, 1 step)
    median =   18.83 µs   p95 =   19.92 µs

[2] JAX jit single-step (1 walker, 1 step)
    median =   10.42 µs   p95 =   12.71 µs   (1.81× vs NumPy)

[3] JAX vmap+jit (B=32 walkers, 1 step)
    median =   14.60 µs total =  0.456 µs/walker

[3] JAX vmap+jit (B=128 walkers, 1 step)
    median =   54.96 µs total =  0.429 µs/walker

[3] JAX vmap+jit (B=512 walkers, 1 step)
    median =  130.29 µs total =  0.254 µs/walker

[4] JAX scan rollout (B=32  × H=20 = 640 evals)
    median =  128.10 µs total =  0.200 µs/eval

[4] JAX scan rollout (B=128 × H=20 = 2560 evals)
    median =  468.33 µs total =  0.183 µs/eval

[4] JAX scan rollout (B=256 × H=20 = 5120 evals)
    median =  953.33 µs total =  0.186 µs/eval

[4] JAX scan rollout (B=200 × H=30 = 6000 evals)
    median = 1117.00 µs total =  0.186 µs/eval
```

**Target FMC** (M=200 walkers × N=20 lookahead = 4000 evals < 1 ms):
- Estrapolando 0.186 µs/eval × 4000 = **744 µs** ✓
- Margine: 256 µs di budget per logica FMC stessa (resampling, reward, etc.)

## 5. Note onesto — limitazioni note

1. **Metal GPU non disponibile**. `jax-metal 0.1.1` non compatibile con `jax 0.10.0` (errore `UNIMPLEMENTED: default_memory_space is not supported` perfino su `jnp.asarray` minimo). Plugin Apple fermo a JAX 0.4.x. CPU JIT raggiunge comunque il target — per matrici 20×20 il GPU launch overhead (~50 µs) supererebbe il calcolo. Quando `jax-metal` aggiornerà supporto a JAX moderno, il modulo è già pronto per Metal (basta cambiare `JAX_PLATFORMS`).

2. **float32 precision**. Per compatibilità con Metal abbiamo forzato f32. Cross-check NumPy(f64) vs JAX(f32) mostra max relative error 1.6e-6 (limite f32 sui valori grandi tipo $\bar n = 5\cdot 10^{19}$). Per produzione sarà raccomandato rescalare le unità interne (n in unità di $10^{19}$, I in MA, etc.) per recuperare precisione.

3. **Calibrazione operativa**. Il simulatore mostra fisica corretta in struttura (test passati) ma con i parametri di default (R_coil=1mΩ uniforme, L_p analitico, S sintetica) NON riproduce uno stato stazionario stabile a 200 kA. Il "free-decay test" mostra quench in ~ms perché senza un OH transformer multi-turno con flux ramp adeguato, la resistenza Spitzer scarica I_p. **Calibrazione fine al regime operativo TCV reale è scope di Milestone 3** (matrice S identificata da FreeGS, M_pc da equilibri reali, R_coil da datasheet).

4. **Vessel passivo trascurato**. Modelli reali (e.g. Degrave 2022) includono ~192 filamenti toroidali del vessel come correnti indotte. Qui assumiamo zero accoppiamento passivo. Aggiungibile come blocco 192×192 in M senza cambi all'API.

5. **Plasma filamentary**. Trattiamo il plasma come singolo loop a (R_p, Z_p). Modelli reali distribuiscono la corrente plasma su una griglia ψ. Per l'interfaccia FMC questa approssimazione è adeguata; per accuratezza shape, Milestone 3 userà profili distribuiti.

## 6. Artefatti consegnati

| Path | Cosa contiene |
|---|---|
| [`scripts/mutual_inductance.py`](../scripts/mutual_inductance.py) | Formula di Neumann + self-L + matrici |
| [`scripts/plasma_simulator.py`](../scripts/plasma_simulator.py) | NumPy reference (231 righe netti) |
| [`scripts/plasma_simulator_jax.py`](../scripts/plasma_simulator_jax.py) | JAX jit/vmap/scan version |
| [`scripts/benchmark.py`](../scripts/benchmark.py) | Latency benchmark suite |
| [`tests/test_simulator.py`](../tests/test_simulator.py) | 21 test matematici (tutti passano) |
| [`docs/milestone_2_simulator.md`](milestone_2_simulator.md) | Questo documento |

## 7. Riproducibilità

```bash
cd work/06_plasma_fmc

# Math correctness
python tests/test_simulator.py

# Reference NumPy demo
python scripts/plasma_simulator.py

# JAX cross-check
JAX_PLATFORMS=cpu python scripts/plasma_simulator_jax.py

# Benchmark
python scripts/benchmark.py
```

Dipendenze: `jax==0.10`, `jaxlib==0.10`, `numpy`, `scipy`, `pyyaml`. (Opzionale: `freegs==0.8.2` per cross-check geometry contro `tcv_geometry.py`.)

## 8. Prossimo step

→ **Milestone 3**: integrazione FMC controller.
- Adattare lo schema `fragile.fractalai.swarm` (vedi `repos/fragile`) al stato continuo plasma
- Reward shaping: shape MSE rispetto al target + safety margins (q₉₅ > 2, $\bar n / n_{GW}$ < 1, $\beta < \beta_{\text{Troyon}}$)
- Cloning per cluster su (R_p, Z_p, κ, δ) — preserva diversità geometrica
- Test: tracking di una rampa shape standard → snowflake → NT

→ **Milestone 4**: visualizzazione web Streamlit + export per dashboard.
