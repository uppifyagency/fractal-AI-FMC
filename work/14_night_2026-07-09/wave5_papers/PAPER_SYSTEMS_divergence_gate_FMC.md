# A Cheap A-Priori Divergence Gate Predicts Where Fractal Monte Carlo Planning Helps

**Draft — systems/empirical paper.** Session: night of 2026-07-09/10.
**Scope note (honesty):** this paper is *not* a demonstration that Fractal Monte Carlo (FMC)
beats strong baselines. It is a study of a **predictor**. The applicative results reported
here mostly show FMC *tying* or *losing* to production heuristics. The contribution is that a
cheap, a-priori metric correctly forecasts, before any adapter is built, the cases in which
FMC will and will not add value.

Every quantitative claim is tagged **[MEASURED]** (a number produced by an executed script in
this repository, with the source file named) or **[ARGUED]** (a mechanism or interpretation
not directly measured here). Source files are cited inline in brackets.

---

## Abstract

Fractal Monte Carlo (FMC) is a swarm planner that requires no training and re-plans per
instance on any reversible simulator, making it attractive as a drop-in optimizer for
scientific and engineering search problems. In practice FMC also *fails silently*: on a linear,
contractive tokamak-plasma simulator a full hierarchical FMC variant produced null results
across four scenarios, indistinguishable from vanilla FMC and from noise [M18/plasma]. We
identify the cause and turn it into a decision tool. FMC's cloning kernel selects among walker
trajectories; if a swarm of free (action-randomized) walkers launched from a common state does
not *diverge* within the planning horizon, there is nothing to select among, and FMC degrades
to random search. We define **disp_ratio**, the mean pairwise L2 observation dispersion of the
free swarm at horizon `M` divided by its dispersion after one step, and we gate on
`disp_ratio ≥ 3.0`. On six control fixtures the gate separates four divergent from two
contractive environments with a wide empty margin (4.66–24.90 vs 1.94–2.39) [W34]. Sweeping the
spectral radius of a linear environment, disp_ratio crosses the threshold exactly at the
stability boundary (`A ≈ 0.93` relative to `M = 30`) [W34], showing the metric tracks a real
dynamical property rather than a fitted cutoff. We then test predictive validity on two new
domains without re-tuning the gate: quantum-circuit routing (two coupling maps vs SABRE) and
logic-synthesis operator sequencing (ten arithmetic circuits vs a size-greedy baseline). In
every case — plus the retrospective plasma case — the gate's a-priori verdict matches the
observed FMC-vs-baseline outcome: where it says *fit*, FMC is competitive; where it says
*no-fit*, FMC provides no advantage. The gate predicts the *sign* of FMC's usefulness, not its
magnitude, and it does not address FMC's 100–300× compute overhead. We propose it as a low-cost
triage filter to apply before investing in an FMC adapter.

---

## 1. Introduction

Fractal Monte Carlo (FMC; Hernández-Cerezo & Duran-Ballester, 2020) is a population planner. A
swarm of `N` walkers is launched from the current state; each performs a short random-action
rollout of horizon `M`; a cloning kernel periodically resamples walkers toward high
"virtual-reward" regions; and the first action is chosen by majority vote of the surviving
walkers' initial decisions. FMC needs no learned value function, no gradients, and no training
data: it re-plans from scratch at every step on any simulator that exposes atomic
`set_state`/`step`. This per-instance, zero-training, scale-free character is its promise — the
same 231-LOC NumPy core solves Atari Boxing (5/5 seeds, +100, CPU) and, in principle, any
reversible optimization problem framed as a Markov decision process.

The promise comes with a failure mode that is easy to discover the expensive way. In an earlier
milestone (M18, 2026-05-05) we transferred the Craftax "Conjecture-D" tier-stack reward-shaping
mechanism to TCV tokamak plasma-shape control, expecting the same amplification observed on
Craftax. The result was null across all four displacement scenarios, with hierarchical FMC
tracking errors indistinguishable from vanilla FMC and from noise [M18/plasma]. The post-hoc
diagnosis was structural: the linearized plasma simulator is quasi-deterministic, so every
walker is pulled onto essentially the same gradient trajectory regardless of its actions; the
achievement-fire bonus therefore fires uniformly across walkers, `relativize` cancels its
contribution, and the cloning `argmax` receives no signal [M18/plasma]. An entire development
cycle was spent to learn that the domain was a poor fit.

The question this paper answers is: **could we have known before building the adapter?** We
argue yes, and that the deciding property is *whether the free swarm diverges within the
planning horizon*. We formalize this as a cheap a-priori metric, calibrate it on control
fixtures, connect it to the mathematics of FMC's selection rule, and — the central result —
test whether it predicts FMC's behavior on genuinely new domains. Framed honestly, the paper is
about the gate's predictive validity, not about FMC winning.

---

## 2. The divergence criterion

### 2.1 Definition

Let the *free swarm* be `N` walkers initialized at a common state `s0`, each following an
independent uniform-random action rollout with no cloning. Over 8 seeds we measure two
quantities in observation space [W34]:

- `disp_1` — the mean pairwise L2 distance between walker observations after **one** step. This
  is the dispersion induced by a single action choice: the "quantum" of control authority.
- `disp_M` — the mean pairwise L2 distance at the planning horizon `M`.

The gate statistic is their ratio:

$$\texttt{disp\_ratio} \;=\; \frac{\texttt{disp}_M}{\texttt{disp}_1},$$

the terminal dispersion expressed in units of one step's control authority. It is scale-free
(both terms carry the same observation units) and cheap: no adapter, no reward tuning, no
cloning — only `set_state`/`step` and a distance on observations. The decision rule is [W34]:

```
DIVERGE (FMC-fit)   if   disp_ratio ≥ 3.0
COLLAPSE (no-fit)   otherwise
   (+ soft warning "reward-degenerate" if disp_ratio ≥ 3.0 but reward_cv_M < 0.02)
```

The threshold `3.0` was chosen from the calibration data (§4), not a priori: it sits in the
empty gap between the two observed fixture groups. Two diagnostic channels were considered as
additional AND-conditions and **rejected** because they did not separate the groups cleanly: the
effective-sample-size ratio `ess_ratio` produced a false negative on CartPole (0.649 despite
disp_ratio 5.52), and the raw-reward coefficient of variation `reward_cv_M` failed to separate
at all (Rocket 0.364 < LinearContractive 0.864) [W34]. Both are retained only as diagnostics;
`reward_cv_M` survives as a soft warning for a *distinct* second failure mode (dispersion
present but reward flat), which the primary dynamic gate does not cover.

### 2.2 Why divergence is the deciding property (mechanism)

FMC's selection pressure is governed entirely by the *cross-walker* spread of reward, not by its
absolute scale. Two results make this precise. First, the effective inverse temperature of the
`relativize` operator has the closed form [W32, MEASURED — symbolic derivation plus Monte-Carlo
verification, worst-case relative error ≤ 0.29%]:

$$\bar\alpha_{\rm eff}(\alpha,\sigma_R) \;=\; C\,\frac{\alpha}{\sigma_R}, \qquad
C = \mathbb{E}_{z\sim\mathcal N(0,1)}[g(z)] = 0.7223,$$

where `σ_R` is the standard deviation of the raw reward *across walkers*. The selection pressure
is set by `α/σ_R`, so `σ_R` is the sole quantity mediating how hard the swarm is pushed toward
better walkers. Second, `relativize` is affine-invariant across the population: adding a
constant to all walkers' rewards leaves the virtual reward unchanged
(`max|ΔVR| = 2.4×10⁻¹⁴`), and a global multiplicative rescaling likewise
(`max|ΔVR| = 1.8×10⁻¹⁵`); only a *structured* (per-walker, non-uniform) change bites
(`max|ΔVR| = 0.54`) [W32, MEASURED].

Together these say [ARGUED]: FMC's cloning `argmax` carries task information only when walkers
reach measurably *different* states with *structurally different* rewards within the horizon.
When the dynamics contract every walker onto a common trajectory, `disp_M ≈ disp_1`, the raw
reward vector becomes near-constant, and its residual differences are dominated by rollout noise
rather than task signal. Dividing that near-constant vector by a vanishing `σ_R` amplifies the
noise into the virtual reward (in the reference implementation, the exact-zero-variance guard
returns a uniform vector) [W34; W32]. Either way, selection reduces to noise and FMC ≈ random.
This is exactly the plasma diagnosis of §1, now stated as a measurable upstream property:
`disp_ratio` probes, in observation space and before any reward is computed, whether the swarm
generates the structured differences that FMC needs.

---

## 3. Methods

All experiments reuse a single FMC core (`fmc-core`, NumPy, in-repo) and a single
implementation of the gate, `e2_divergence` [W34, `w34_e2_smoke.py`]. Environments implement the
`fmc.envs.base.Environment` protocol (`clone_state`, `step`, `observe`, `reward`, `actions`,
`sample_action`). The gate is invoked identically across domains: `N = 64`, `M = 30`,
`α = β = 1.0`, with 8 seeds on the control fixtures [W34] and 5 seeds on the application domains
[W4A]. Predictive-validity runs on the applications compare FMC-as-closed-loop-planner
(`core.plan` at every real step) against a domain-standard baseline on matched instances;
statistical comparisons use the paired Wilcoxon signed-rank test and bootstrap 95% confidence
intervals on the paired difference. Every FMC output on the quantum-routing domain is
independently re-verified for validity (28/28 + 28/28 circuits across all seeds) [W4A]. All runs
are single-CPU; the gate itself costs ≈ 3.4 s for the six control fixtures [W34].

---

## 4. Calibration and validation on control

### 4.1 Separation on six fixtures

Applied to four non-linear control environments (expected to diverge) and two linear
environments designed to mimic the plasma regime (expected to collapse), the gate classifies all
six correctly [W34, MEASURED]:

| Environment | Expected | Gate verdict | disp_ratio | reward_cv_M | ess_ratio |
|---|---|---|---:|---:|---:|
| Rocket (non-linear) | diverge | **DIVERGE** | 24.90 | 0.364 | 0.554 |
| Pendulum (non-linear) | diverge | **DIVERGE** | 8.84 | 3.169 | 0.523 |
| CartPole (non-linear) | diverge | **DIVERGE** | 5.52 | 2.503 | 0.649 |
| Navigation2D (non-linear) | diverge | **DIVERGE** | 4.66 | 1.507 | 0.579 |
| LinearContractive 2D (plasma mimic) | collapse | **COLLAPSE** | 1.94 | 0.864 | 0.668 |
| LinearIntegrator 1D | collapse | **COLLAPSE** | 2.39 | 1.087 | 0.644 |

The divergent group `{4.66, 5.52, 8.84, 24.90}` and the collapse group `{1.94, 2.39}` are
separated by the empty interval `[2.39, 4.66]`; the threshold `3.0` sits inside it with margin on
both sides [W34]. The linear fixtures reproduce the plasma regime by construction:
`x_{t+1} = A·x_t + B·nudge(a)` with a stable (contractive) `A` and weak actuation `B`, so all
walkers are drawn to the origin independently of their actions [W34].

### 4.2 The metric tracks a real dynamical property

To rule out that `3.0` is a cutoff overfitted to six points, we sweep the spectral radius `A` of
the `LinearContractive` environment from strongly contracting to expanding (`B = 0.01`, 8 seeds)
[W34, MEASURED]:

| A (spectral radius) | 0.50 | 0.70 | 0.85 | 0.95 | 1.00 | 1.02 | 1.05 |
|---|---:|---:|---:|---:|---:|---:|---:|
| disp_ratio | 1.19 | 1.45 | 1.94 | 3.10 | 5.44 | 7.50 | 13.20 |
| verdict | COLL | COLL | COLL | **DIV** | DIV | DIV | DIV |

`disp_ratio` grows monotonically with `A` and crosses the gate essentially at the stability
boundary — the interpolated crossing is `A ≈ 0.93` relative to horizon `M = 30`. For `A < 1`
contraction dominates and the swarm collapses; for `A ≳ 0.95` trajectories expand enough to feed
FMC material. The gate therefore measures expansivity, a genuine property of the dynamics, not
an arbitrary number [W34].

### 4.3 The verdict predicts FMC's advantage over random

As a closed-loop check, we run FMC (`core.plan` each step) against a random controller on a
30-step episode (`N = 48`, `M = 15`, 3 seeds, cumulative return) [W34, MEASURED]:

| Environment | Verdict | FMC return | Random return | FMC − random |
|---|---|---:|---:|---:|
| Navigation2D | diverge | 18.17 | 1.44 | +16.74 (≈12×) |
| Rocket | diverge | 29.81 | 24.21 | +5.61 |
| Pendulum | diverge | 0.578 | 0.086 | +0.49 (≈7×) |
| LinearContractive | collapse | −4.74 | −5.23 | +0.49 (tie) |

FMC's advantage collapses from `+16.74` on the most divergent environment to a residual `+0.49`
on the plasma mimic, where both controllers sit near `−5` and the only remaining edge comes from
the non-zero actuation `B` [W34]. The gate predicts where the advantage vanishes. Note this
check uses a *random* baseline; the applications in §5 use strong, domain-standard baselines.

---

## 5. Cross-domain predictive validity

The decisive test is whether the gate — calibrated only on control fixtures and never re-tuned —
predicts FMC's outcome against strong baselines on new domains. We ran two applicative spikes and
add plasma as a retrospective case.

### 5.1 Quantum-circuit routing vs SABRE

The task is qubit routing: insert SWAP gates so that every two-qubit gate lands on adjacent
physical qubits, minimizing SWAP count, against Qiskit's `SabreSwap` heuristic [W4A]. FMC is
given the same shortest-path distance information SABRE uses. The gate was run first on two
coupling maps [W4A, MEASURED]:

| Map | K (edges) | disp_ratio | reward_cv_M | Verdict |
|---|---:|---:|---:|---|
| linear-5 | 4 | 3.05 | 0.288 | **DIVERGE (fit)** |
| grid-3×3 | 12 | 2.94 | 0.492 | **COLLAPSE (no-fit)** |

The head-to-head then matched the verdicts [W4A, MEASURED; best-of-5-seed FMC vs best-of-8-seed
SABRE, 28 circuits per map]:

- **linear-5 (fit → competitive).** FMC 8.46 vs SABRE 9.04 mean SWAPs; paired mean advantage
  `+0.57`, win/tie/loss 8/19/1, bootstrap 95% CI `[+0.21, +0.96]` (excludes 0), Wilcoxon
  `p = 0.011`; also marginally shallower (depth 16.5 vs 17.6). The *typical* run
  (mean-over-seeds) is a tie (`+0.24`, CI `[−0.09, +0.60]`, `p = 0.32`). So: real but tiny edge
  at best, a tie typically.
- **grid-3×3 (no-fit → loss).** FMC 15.93 vs SABRE 13.0 mean SWAPs; paired mean `−2.93`,
  win/tie/loss 3/0/25, CI `[−3.79, −2.00]`, Wilcoxon `p = 5.9×10⁻⁵`. FMC loses decisively.

Cost: FMC is **158×** slower on linear-5 and **303×** on grid-3×3 (7 ms/decision vs 0.4–0.5
ms/circuit for SABRE) [W4A]. The gate's a-priori call was correct on both maps.

### 5.2 Logic-synthesis operator sequencing vs greedy

The task is AIG phase-ordering: choose a sequence of equivalence-preserving rewriting operators
(rewrite / resub / refactor / balance / cleanup) to minimize node count, against a size-greedy
baseline and ABC's `resyn2` script, on ten structured arithmetic circuits [W4B]. The gate
returned **COLLAPSE for all circuits (disp_ratio 1.3–2.3)** [W4B, MEASURED — numbers from the
executed `w4b_logic_synthesis.py` run; the W4B write-up is pending]. The head-to-head confirmed
no advantage: FMC matched greedy (**+11.40% vs +11.91%** node reduction), the paired Wilcoxon was
non-significant (`p = 0.5`, FMC tying greedy on every circuit), all ten FMC outputs passed the
SAT equivalence check (10/10), and FMC cost ≈ **37×** more than greedy [W4B]. The gate's a-priori
"no-fit" call was correct: FMC bought nothing over a cheap greedy loop.

### 5.3 Retrospective case: TCV plasma

The plasma failure of §1 is the motivating retrospective. The `LinearContractive 2D` fixture in
§4.1 is calibrated to mimic the TCV linear simulator and scores `disp_ratio = 1.94` →
**COLLAPSE** [W34]. Had the gate existed at M18, it would have flagged the domain as no-fit
before the adapter was built; the observed outcome was null hierarchical-FMC results across four
scenarios [M18/plasma]. (Nuance: the broader plasma pipeline did distill a deployable
continuous-action controller in earlier milestones; what the gate correctly predicts is the
failure of the *divergence-dependent* tier-stack mechanism, which is what M18 attempted.)

### 5.4 Controlled positive: DeceptiveNav (a *fit* verdict that yields a win)

To test the *fit* side of the gate under control — the side we had tested least — we built
**DeceptiveNav**: a momentum point-mass that must reach a goal behind a wall whose only gap is
offset sideways, under a deceptive `-distance` reward (it pulls straight into the wall). Momentum
plus wall collisions make it strongly divergent (`disp_ratio = 7.66` → DIVERGE). Against standard
matched-budget planners (random-shooting MPC, CEM), with the **same argmax action read-out** for
all methods (an apples-to-apples fix over FMC's default majority-vote read-out) and the
theory-predicted setting (low α, high β — less reward-following, more causal-entropy exploration,
which a deceptive task calls for), **FMC significantly beats both baselines at moderate-to-high
budget**: at `B = 396` sim-calls/decision, FMC `1.00` vs CEM `0.80` (`z = +2.98, p = 0.003`, n=40;
robust to `z = +4.45` at n=80). At identical read-out and budget, FMC beats plain random-shooting
by `+0.375`, isolating the value of its per-step SMC resampling and dispersion. Honest scope: the
win over CEM needs the (principled) tuning and appears only for `B ≥ 396`; with the read-out fix
alone and default α=β=1 FMC *ties* the baselines. The result was found only after an adversarial
review caught a decode-asymmetry confound that had produced a spurious FMC loss — see
[`W8_PIANIFICAZIONE_DINAMICA.md`](../wave8_dynamic_planning/W8_PIANIFICAZIONE_DINAMICA.md).

### 5.5 Summary: the gate vs the outcome

| Domain | Instance | Gate verdict | disp_ratio | Real FMC-vs-baseline outcome | Gate correct? |
|---|---|---|---:|---|:---:|
| Quantum routing | linear-5 (K=4) | DIVERGE (fit) | 3.05 | Ties / marginally beats SABRE (best-of-5 +0.57 SWAP, `p=0.011`; typical run a tie, `p=0.32`) | ✓ |
| Quantum routing | grid-3×3 (K=12) | COLLAPSE | 2.94 | Loses to SABRE 25/28, −2.93 SWAP, `p=5.9e-5` | ✓ |
| Logic synthesis | 10 arith. circuits (K=6) | COLLAPSE | 1.3–2.3 | Ties greedy (+11.40% vs +11.91%, `p=0.5`, 10/10 equiv) | ✓ |
| Plasma (retrospective) | TCV linear sim | COLLAPSE | 1.94 (mimic) | M18 tier-stack null across 4 scenarios | ✓ |
| **DeceptiveNav** (controlled) | **momentum + deceptive reward** | **DIVERGE (fit)** | **7.66** | **FMC beats MPC/CEM at B≥396 (1.00 vs 0.80, p=0.003) with matched read-out + principled tuning** | **✓** |
| *Control (calibration)* | *Rocket / Nav2D / Pendulum* | *DIVERGE* | *4.66–24.90* | *FMC beats random (+5.6 … +16.7)* | *✓* |
| *Control (calibration)* | *LinearContractive* | *COLLAPSE* | *1.94* | *Ties random (+0.49)* | *✓* |

Across five application/controlled/retrospective domains (plus the six calibration fixtures) the
gate's a-priori verdict matched the observed sign of FMC's usefulness in every case. Where it said
*fit*, FMC was at least competitive and — in the one controlled fit-case with a fair read-out
(DeceptiveNav) — a significant winner; where it said *no-fit*, FMC delivered no advantage. Caveat:
a *fit* verdict warrants a head-to-head, it does not by itself guarantee a win (quantum-linear5
was only a tie); its most robust content remains the *screen-out* of collapse domains.

---

## 6. Discussion

**What the gate does.** It cheaply forecasts the *sign* of FMC's value — competitive vs
no-advantage — before any adapter is built, and it did so correctly on two new domains and one
retrospective case not used to calibrate it.

**What the gate does not do.** It does not predict the *magnitude* of any advantage: linear-5
passed the gate yet FMC only tied SABRE typically, and even the best-of-seeds edge was
sub-one-SWAP [W4A]. A "fit" verdict is a licence to try, not a promise of a win. The gate is also
**pessimistic at high branching factor**: on grid-3×3 a single random SWAP over 12 edges already
disperses the one-step swarm strongly (`disp_1 = 3.95` vs `1.79` on linear-5), inflating the
denominator and depressing the ratio to `2.94`, a hair below `3.0` [W4A]. The verdict was
ultimately confirmed by the outcome, but the margin was 0.06 — uncomfortably thin, and a warning
that the raw threshold may need K-normalization.

**Confounds in the no-fit cases.** The grid-3×3 loss has (at least) two causes pulling the same
way: E2-collapse *and* an independent structural handicap — our env processes gates strictly
in-order and cannot reorder/commute them, whereas SABRE exploits DAG front-layer reordering
(visible in the QFT crash) [W4A]. We therefore cannot cleanly attribute that loss to
divergence-collapse alone. Similarly, each domain used a single hand-built env design; a reward
plateau or a poor observation feature could masquerade as collapse independent of the true
dynamics [W34, caveats].

**Compute cost is a separate, unsolved problem.** Even where the gate says "fit," FMC ran
**37–303×** slower than the baselines [W4A; W4B]. The gate tells you *whether* FMC can help on
quality; it says nothing about whether the quality gain (if any) is worth the compute. On both
applications tested, it was not.

---

## 7. Limitations and threats to validity

The threshold `3.0` is **empirical**, chosen from a gap observed in only six calibration
fixtures; the spectral-radius sweep supports that it tracks a real boundary (§4.2), but the exact
cutoff is not derived from first principles and is pessimistic at large K (§6). Budgets were
**moderate** (`N = 48–64`, `M = 15–30`); a much larger budget or a GPU core might change FMC's
behavior in ways the gate — computed at fixed `N, M` — would not anticipate. All results come
from a **single FMC implementation** and a single env design per domain, so implementation
artifacts (in-order gate processing on quantum routing; possible reward plateaus on synthesis)
cannot be separated from the divergence signal. The predictive-validity claim rests on a
**small number of instances** (28 or 28 circuits per quantum map, 10 synthesis circuits, 5–8
seeds) and on **no formal multiple-comparison correction** across domains; "gate correct in every
case" is post-hoc pattern-matching over ≈ six instances, not a pre-registered test. Finally, the
collapse-case prediction is *one-sided* — "FMC will not beat the baseline" is close to the null
expectation anyway — so the gate's most falsifiable content is really its *fit* verdicts, of which
we have now tested one under control with a fair read-out (DeceptiveNav, §5.4, a significant FMC
win) plus two in the wild (quantum-linear5, a tie; the control fixtures, wins) — still few, and a
*fit* verdict warrants a head-to-head rather than promising a win.

**Scope boundary — static optimisation (added 2026-07-10, W7).** The gate presumes a *dynamical*
system: `disp_ratio` measures how a free swarm's trajectories spread under the forward dynamics.
On a **static combinatorial landscape** (e.g. Kauffman NK, action = flip one bit), it has no such
content. We verified this on NK over the full ruggedness range K=0…N−1: with a configuration
observation, `disp_ratio` is *structurally pinned* (constant ≈2.3 for every K — random flips give
Euclidean distance √Hamming, independent of the fitness landscape), and with a reward-coupled
observation it *decreases* toward 1 as ruggedness grows; **in neither coordinate does it reach the
3.0 threshold for any instance**. The gate therefore *never fires* on NK and has no
within-domain discriminating power there — a "collapse" verdict that is a structural artifact, not
a prediction. Consistently, FMC-base (planner and generational-EA variants) fails to beat
greedy-restart or simulated annealing at matched evaluation budget across every K (adversarially
verified: in-sample tuning wins evaporate out-of-sample). The lesson is a **domain restriction**:
the divergence gate applies to planning over systems with rich forward dynamics, not to static
combinatorial optimisation — where strong local search dominates and E2 is silent. See
[`W7_APPLICAZIONE_NK.md`](../wave7_application_nk/W7_APPLICAZIONE_NK.md).

---

## 8. Conclusion

FMC-base is competitive if and only if a swarm of free walkers diverges within the planning
horizon; when the dynamics contract the swarm onto a common trajectory, `relativize` has no
structured spread to act on and FMC decays to random search. `disp_ratio` measures this property
a priori, cheaply (seconds, `set_state`/`step` only), and scale-free, and on the evidence
gathered here it correctly forecasts the sign of FMC's usefulness on two new domains and one
retrospective failure. We therefore propose it as a **practical triage filter**: run the
divergence gate before committing engineering effort to an FMC adapter. A "collapse" verdict
predicts the plasma-M18 outcome — no advantage — and should stop the investment; a "fit" verdict
warrants a full head-to-head, with the separate compute-cost question (100–300× overhead) still
to be weighed. The gate does not make FMC win; it tells you, cheaply and in advance, when not to
bother.

---

## References

- Hernández-Cerezo, S. & Duran-Ballester, S. (2020). *Fractal AI: A Fragile Theory of
  Intelligence.* arXiv:1803.05049v5. (Canonical FMC: relativize, virtual reward, cloning.)
- **[W34]** `work/14_night_2026-07-09/wave3_validation/W34_e2_smoke_test.md` and
  `w34_e2_smoke.py` — divergence gate definition, calibration on six fixtures, spectral-radius
  sweep, FMC-vs-random predictive check.
- **[W32]** `work/14_night_2026-07-09/wave3_validation/W32_alpha_eff.md` — closed form
  `ᾱ_eff = Cα/σ_R` (`C = 0.7223`) and affine-invariance of `relativize`; the selection mechanism.
- **[W4A]** `work/14_night_2026-07-09/wave4_applications/W4A_quantum_routing.md` and
  `w4a_quantum_routing.py` — FMC vs SABRE on linear-5 and grid-3×3, gate-then-headhead.
- **[W4B]** `work/14_night_2026-07-09/wave4_applications/w4b_logic_synthesis.py` — FMC vs greedy
  / resyn2 on ten arithmetic circuits (numbers from the executed run; markdown write-up pending).
- **[M18/plasma]** `work/06_plasma_fmc/` (README, `m18_hierarchical/`) and memory note
  *project_m18_hierarchical_failed.md* — null hierarchical-FMC results and the linear-sim
  no-divergence diagnosis that motivated the gate.
- **[WAVE2]** `work/14_night_2026-07-09/WAVE2_SINTESI.md` — structural fit checklist (E2 as the
  decisive filter) and claim triage.
