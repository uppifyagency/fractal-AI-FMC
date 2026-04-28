# THEORY — why this plugin exists, why it works

> *This document explains the theoretical foundation of the plugin: which paper it implements, why that paper matters, why the algorithm transfers from Atari to coding, what is faithful to the source, and what is deliberately adapted.*

For step-by-step algorithm walkthrough see [`ALGORITHM.md`](ALGORITHM.md). For file-by-file reference see [`COMPONENTS.md`](COMPONENTS.md). For invocation and configuration see [`USAGE.md`](USAGE.md).

---

## 1. The paper — and why it matters

The plugin implements **Fractal Monte Carlo (FMC)**, an algorithm proposed in:

> Hernández-Cerezo, S. & Duran-Ballester, G. (2020). *Fractal AI: A Fragile Theory of Intelligence.* arXiv:1803.05049v5.

The paper makes a strong empirical claim and a strong theoretical claim, and both have stood up to verification:

**Empirical claim**: on the standard 50-game Atari benchmark, FMC beats every published planning algorithm (MCTS UCT, IW(1), p-IW(1)) on **50/50 games**, and beats every published deep RL algorithm (DQN, A3C, NoisyNet, Dueling DQN) on **45/50 games**, while using approximately **1/300th the samples per decision** (~400 vs ~150,000). It does this **with no neural network, no training, no gradient descent, no GPU**. See paper §5.1 and the standalone empirical paper [Hernández-Cerezo et al. 2018, arXiv:1807.01081](../../../docs/bibliography/sources/papers/2018_solving_atari_1807.01081.pdf).

**Theoretical claim**: the algorithm is provably correct in the limit. The walker pool's empirical distribution converges to the **Gibbs distribution proportional to reward** (paper §3.1, formalized in deep dive [`01_cloning_mathematics.md`](../../../work/02_deep_dives/01_cloning_mathematics.md) Theorem 3, with the SMC convergence theorem of Del Moral 2004 applying directly to FMC — see [`05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md)).

**Verification of empirical claim, in this repo**: a 231-line standalone NumPy reimplementation (no porting from the original codebase) achieved **96/100 on Atari Boxing in 7 minutes** on a MacBook with no GPU. See [`work/03_atari_replication/results/SMOKE_TEST_REPORT.md`](../../../work/03_atari_replication/results/SMOKE_TEST_REPORT.md) and the personal post-experiment [`analisisPost.md`](../../../analisisPost.md).

That last fact is what made this plugin worth building. The algorithm is real and reproducible. The question becomes: **can it transfer from Atari to coding?**

---

## 2. Why the algorithm should transfer

The paper's central claim is not "FMC plays Atari." It is:

> *An agent is intelligent if, at every instant, it distributes its attention over possible futures in proportion to their expected value.*

This is medium-independent. Atari is one realization. Coding is another.

The mechanism that produces this distribution is the same in any setting:

1. A **swarm** of N walkers, each holding a state of the system
2. Each walker is **labeled** by an "initial decision" `ℓ` ∈ A (the action that produced its branch)
3. Walkers **propagate forward** for M ticks via a stochastic kernel (the simulator + a perturbation)
4. Walkers **clone** between ticks: walker_i compares to a random partner, and with probability `(VR_partner − VR_self) / VR_self` (paper §4.4) replaces its state and label with the partner's
5. After M ticks: `decision = argmax bincount(ℓ)` over surviving walkers (paper §4.6)

The four ingredients required for this to work are **state**, **action**, **simulator** (or: a way to advance state given an action), and **reward**.

**Atari has all four perfectly**: ALE provides bitwise-deterministic state cloning + step. The paper exploits this.

**Coding has all four imperfectly**:

- *State*: `git HEAD` of a worktree. Perfect — git is more reliable than ALE.
- *Action*: a "strategy choice" plus an LLM that implements it. Discrete, fewer choices than Atari (3-5 vs 18), but well-defined.
- *Simulator*: an LLM sub-agent invocation. **Stochastic, expensive, not deterministic-reproducible**. This is the main source of friction.
- *Reward*: composite signal from tests + lint + diff size + LLM judge. Computable in seconds.

The plugin makes peace with the imperfect simulator: cost-aware design (small N and M), goal-directed continuations (not random perturbations), ESS-adaptive cloning to skip unnecessary work.

---

## 3. The Octopus structure — why two loops, not one

A naïve translation of FMC to coding would be: spawn N walkers, each implementing one approach, score them, pick the winner. M=1, single decision.

**This is not FMC.** It is a single-shot vote on initial states. The "Monte Carlo" in FMC comes from the iterated dynamics: walker projection → cloning → walker projection → cloning. Each cycle refines the empirical distribution toward the true reward density.

But there's a second, deeper issue. In Atari, FMC.decide outputs **one action**, applied to the game for **one frame**. Then we re-decide from the new state. The outer loop (the game) and the inner loop (FMC) are distinct. Sergio articulated this hierarchy in his [December 2015 blog post on Fractal AI Collaboration](../../../docs/bibliography/sources/blog_posts/2015-12_fractal_ai_collaboration.md):

> *"a fractal tree of intelligence layers where fingers coordinate as a hand, hands function as an octopus, and multiple octopuses respond to collective instructions like retrieving objects."*

This is the **Badger structure** of Book #2 (2020) in proto-form, five years before the formal write-up. For the plugin, the mapping is:

| Sergio's term | Plugin entity |
|---|---|
| **Finger** | a single walker sub-agent |
| **Hand** | one FMC decision (one `/fractal-decide` invocation) |
| **Octopus** | a goal-directed session (one `/octopus` invocation) |
| **Multiple octopuses** | (out of scope — Phase 4) |

The plugin therefore has two loops:

- **Inner loop** = `/fractal-decide` = one hand. N walkers × M ticks. Output: one commit.
- **Outer loop** = `/octopus` = one octopus. K iterations of `/fractal-decide`. Output: a goal-reached state.

This matches the Atari structure precisely: in Atari, the inner loop is FMC (M tick rollouts of N walkers), the outer loop is the game itself (steps until game_over). For the plugin, the outer "game" is goal-seeking — terminates when the judge says the goal is reached or `K_MAX` is hit.

---

## 4. The four faithful mappings + one deliberate departure

### 4.1 Faithful: state

| Atari | Plugin |
|---|---|
| `ALE.cloneState() / restoreState()` | `git worktree` + `git reset --hard <head>` |
| 128 bytes of RAM | the entire git tree at a SHA |
| Deterministic | Deterministic |

Both are perfect, content-addressable snapshots. Git is in fact stronger than ALE: reproducible across machines, hash-verifiable, version-controlled.

### 4.2 Faithful: action label and bincount marginalization

The walker's `init_action` (Atari: integer in `{0..n_actions-1}`; Plugin: a string label like `"extract-module"`) is the **branching marker** that propagates through clones (paper §4.2.4). At decision time, `argmax bincount(init_action)` over alive walkers (paper §4.6) gives the consensus first action.

This is implemented identically. See [`scripts/fractal_loop.py:cmd_decide`](../scripts/fractal_loop.py) lines 280-318: walkers' `init_action_label` is updated from the partner's label on every clone (lines 261-273 of `cmd_apply_clones`), and the final bincount is over those labels.

### 4.3 Faithful: reward composition

The paper §2.2.2 specifies multiplicative reward composition:

> R(s) = R_0(s) × R_1(s) × ... × R_n(s)

Sergio rejects Pareto frontiers explicitly in his [April 2016 blog post](../../../docs/bibliography/sources/blog_posts/2016-04_pareto_frontiers.md):

> *"Real-world problems typically have single underlying objectives. We only have one goal in life — maximizing long-term well-being."*

The plugin is faithful to this. See [`scripts/fractal_reward.py:composite_reward`](../scripts/fractal_reward.py) lines 86-108:

```
R = R_alive × R_tests × (1 + R_lint) × (1 + R_diff) × (1 + R_goal)
```

Hard constraints are direct multiplications (can go to 0). Soft contributions use `(1 + x)` to avoid zero-collapse while staying multiplicative. **Not** `R = w_0·R_0 + w_1·R_1 + ...` (additive Pareto-style).

#### 4.3.1 Caveat — `relativize` is insufficient on *flat-negative* reward landscapes

The plugin's `(1 + R_x)` pattern guarantees $R > 0$ everywhere by construction, so this is not a bug we hit in practice. But for researchers porting FMC to new domains with custom reward functions, a non-obvious failure mode deserves a warning.

**The trap**: if the reward is *uniformly negative* across the swarm (e.g., raw $R = -0.5$ everywhere except a hidden positive goal), `relativize` does **not** save FMC. After z-scoring, all walkers have $z \approx 0$ and $\hat{R} \approx 1$ uniformly; the differentiation collapses. Walkers diffuse without directional clone signal and the swarm fails to find the goal.

**Numerical evidence** ([`work/04_mathematical_tests/`](../../../work/04_mathematical_tests/) F11 Scenario A, all-negative landscape with hidden positive peak):

| Variant | Reaches goal | Mean walker x (start = 1, goal = 8) |
|---|---|---|
| FMC with `relativize` | 0% | 0.72 — *stalled* |
| FMC raw, sign-preserving | 0% | 0.60 — *Sergio's "fearful" agent* |
| FMC raw, clipped at 0 | 84% | 8.04 — *only this works* |
| Random walk | 2% | 3.97 |

Sergio's claim "[reward negativ → fearful, use `relativize`]" is **necessary but not sufficient**. `relativize` prevents the *catastrophic* freeze of raw negatives, but on a flat-negative landscape the swarm still cannot find a goal it has never visited.

**Sufficient conditions for FMC to converge** (refined from the paper):

1. **R > 0 everywhere** (paper §2.2.3 — guaranteed by the plugin's `(1 + x)` pattern)
2. **Some reward gradient must be reachable from the swarm's initial dispersion**. If the gradient is zero (or invisibly small) at all walker positions, FMC reduces to random walk and cannot bootstrap.

In Sergio's slide F11 Scenario B (smooth gradient even though all-negative), `relativize` recovers and *outperforms* both random walk and clip-at-zero variants. So the canonical plugin reward design — multiplicative with always-positive components and at least one component that varies smoothly with progress (e.g., `R_diff` measuring lines toward goal) — is the right architecture. **Don't compose rewards that produce flat regions.**

### 4.4 Faithful: virtual reward, stochastic distance, pairwise clone

The paper §4.4 specifies:

- VR_i = R_i^α × Dist(W_i, W_partner)^β
- Distance is computed against a single random partner (paper §4.5: O(N) stochastic estimator instead of O(N²))
- Clone probability: `1` if VR_self = 0, else `0` if VR_partner ≤ VR_self, else `(VR_partner - VR_self) / VR_self`

All implemented in [`scripts/fractal_reward.py`](../scripts/fractal_reward.py) (relativize: 115-134; virtual_reward: 160-194) and [`scripts/fractal_loop.py:cmd_step`](../scripts/fractal_loop.py) (clone probability formula at lines 200-210).

> **Caveat — `α_code ≠ α_Gibbs`**. The implementation applies α to the *relativized* reward $\hat{R}$, not raw $R$. Because `relativize` compresses positive outliers and expands negative outliers, the effective Gibbs exponent on raw reward is larger than the code-α: $\alpha_{\text{eff}} > \alpha$. Numerically (verified in [`work/04_mathematical_tests/`](../../../work/04_mathematical_tests/) Test A), the empirical walker distribution best matches $\pi^* \propto R^1$ at **code-α ≈ 0.5**, not at code-α = 1.0. Default α = 1 produces $\pi^* \propto \hat{R}^1$, sharper than the literal "P_walker ∝ R" reading of paper eq. (3). See [`ALGORITHM.md` §4.2.1](ALGORITHM.md) for the full derivation. This is not a bug — it is the canonical FMC behaviour — but matters when comparing FMC to other Gibbs samplers where the temperature acts on raw energy.

### 4.5 Departure: perturbation at tick t > 0

This is the one place the plugin **deviates** from the paper, and the deviation is principled.

In Atari, a walker's "perturbation" at t > 0 is `random_action()` — a uniform draw from the action space. Each `ALE.act(random_action)` costs ~10μs. The walker takes 14 random frames forward to *probe* the future state space, not to play well. The cloning step then *selects* the walkers whose probes landed in good places.

For coding, every "perturbation" is a sub-agent LLM call costing ~30 seconds and ~$0.50. With N=3, M=15 (the paper's parameters), one decision would cost ~225 LLM calls, $100, and 1+ hour. **Insostenibile.**

The plugin's solution is **goal-directed continuation perturbation**:

> At tick t > 0, each walker is prompted to make ONE small step toward the goal, preserving its initial strategy. Variance comes from LLM temperature/sampling, not from random action sampling.

This is less faithful to the algorithm but more economical, and it has a defensible interpretation: in continuous-state SMC, the "random kernel" can be any Markov kernel; uniform-action is one choice but not the only one. Goal-directed-with-temperature-noise is a valid Markov kernel. The convergence theorems of Del Moral 2004 still apply, with a different (probably tighter) target distribution.

**Cost reality** at N=3, M=3:

| Item | Atari (paper) | Plugin (Phase 0) |
|---|---|---|
| Walker count N | 30 | 3 |
| Time horizon M | 15 frames | 3 ticks |
| Sim calls per walker | 15 (1 frame each, 5x skip) | 3 (one commit per tick) |
| Total sim calls per decision | 30 × 15 × 5 = 2,250 | 3 × 3 = 9 |
| Sim cost per call | ~10μs (ALE) | ~30 s (LLM) |
| Wall time per decision | 7 min for 1342 decisions | 3-5 min for 1 decision |
| Decision count to terminate | 1342 (Boxing → 96/100) | 5-15 (typical goal) |

The plugin's regime is **3 orders of magnitude fewer simulator calls per decision** than Atari but **3 orders of magnitude more expensive per call**. Net: roughly comparable wall time, but a different sweet spot.

---

## 5. The convergence theorem — why this is mathematically certified

FMC inherits the convergence theory of **Sequential Monte Carlo** (Del Moral 2004; deep dive [`05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md)).

**Theorem** (Del Moral 2004, Th. 7.4.4, adapted): under regularity conditions, for any test function φ bounded:

```
‖ π̂_t^N (φ) − π_t (φ) ‖_L²  ≤  c_t · ‖φ‖_∞ / √N
```

where π̂_t^N is the FMC empirical distribution after t steps with N walkers, and π_t is the asymptotic Feynman-Kac distribution (= Gibbs distribution proportional to reward).

**Practical translation**: FMC's empirical action distribution converges to the optimal scanning density `P_R(x) = R(x) / R_TOT`, with error `O(1/√N)`. Doubling N reduces error by factor √2 ≈ 1.41.

**For the plugin**:

| N | Expected 95% CI on action selection |
|---|---|
| 3 | ~58% |
| 5 | ~45% |
| 10 | ~32% |
| 30 | ~18% |
| 300 | ~6% |

The plugin's default N=3 is statistically under-dimensioned. For high-stakes decisions, raise N. The "Boxing 96/100" was achieved with N=30. See [`USAGE.md`](USAGE.md) §"Tuning N and M" for guidance.

The math layer's [`tests/test_fractal_math.py`](../tests/test_fractal_math.py) verifies this theorem numerically. Test 3 confirms with R=(0.9, 0.5, 0.1) and M=10 (sufficient mixing time), the high-R walker wins 100% of 200 runs — the expected Gibbs delta.

---

## 6. Why ESS-adaptive cloning matters here, more than in Atari

The deep dive [`05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md) §4.1 notes that FMC vanilla performs cloning at **every** tick. This is a conservative choice that adds resampling variance.

Standard SMC (Doucet et al. 2001) only resamples when **Effective Sample Size** drops below a threshold:

```
ESS_t = (Σ VR_i)² / Σ VR_i²    range [1, N]
```

If `ESS_t > 0.7 · N`, the swarm is already diverse — resampling adds noise without selection benefit.

**For Atari**, skipping cloning saves nothing meaningful: clone is just a state-copy, microseconds.

**For coding**, skipping cloning saves the entire next tick of N sub-agent calls (because clones force walkers to redo work from a partner's branch; without clones, walkers continue from their own state). At N=3, M=3, that's potentially 30%-50% of the total cost.

The plugin includes ESS-adaptive cloning by default, with `--ess-threshold 0.7`. See [`scripts/fractal_loop.py:cmd_step`](../scripts/fractal_loop.py) lines 175-188:

```python
sum_vr = sum(vrs)
sum_vr_sq = sum(v * v for v in vrs)
ess = (sum_vr ** 2) / sum_vr_sq if sum_vr_sq > 1e-12 else 0.0
ess_threshold_abs = state["ess_threshold"] * n
skip_cloning = ess > ess_threshold_abs
```

In the smoke test, ESS=2.68 > 2.10 (= 0.7 × 3) → cloning skipped, zero git resets, zero forced re-perturbation. This is verified behavior, not aspirational.

---

## 7. The three things this plugin does NOT do, and why

### 7.1 Fractal Memory: implemented standalone, not yet auto-integrated

The Slide doc 2020 ([`2020 Fractal Slide.md`](../../../2020%20Fractal%20Slide.md)) extends FMC inward, into neural networks themselves. It proposes that past examples should be sampled with **Wigner-distributed weights** `R'(x) = (π/2) x exp(-π/4 x²)` for automatic curriculum learning.

The plugin **implements this subsystem** in [`scripts/fractal_memory.py`](../scripts/fractal_memory.py) (~370 lines, 4 CLI subcommands: `append`, `recall`, `show`, `prune`), exposed via two slash commands [`commands/fractal-recall.md`](../commands/fractal-recall.md) and [`commands/fractal-memory-show.md`](../commands/fractal-memory-show.md). The Wigner formula is verified by `e2e_test.sh` Test 5 (peak at x=1 within 5% tolerance) and the round-trip (append → show → recall) is verified by Tests 4.

What is **not yet** wired:
- `/fractal-decide` does not auto-call `fractal_memory.py append` after Phase 7. The user must invoke memory persistence explicitly. This is a Phase 0 conservative choice — adding auto-persistence is a one-line shell call in Phase 8b of the slash command, planned but not enabled by default.
- Memory recall is not yet injected into the Phase 1 strategy generation of `/fractal-decide`. A future enhancement (Phase 2) would: before generating strategies, call `fractal-recall` for the current task and bias strategy proposal toward what worked previously. See [`docs/vision/fractal_coding_loop.md`](../../../docs/vision/fractal_coding_loop.md) §V3.

So memory exists, works standalone, but the main FMC pipeline does not feed it (write) or read from it (read). It's a warm spare ready to be wired.

### 7.2 No multi-octopus coordination

Sergio's hierarchy goes one level higher: "multiple octopuses respond to collective instructions." For the plugin this would mean parallel goals on the same repo, with the octopuses negotiating shared state.

This is interesting but unimplemented. The `/octopus` command operates on one goal at a time. No shared state machine across octopuses. Phase 4 territory.

### 7.3 No α=0 Common Sense mode exposed yet

The paper §6.3 introduces α=0: "no reward weighting, only diversity maximization." The drone autopilot demo of paper §6.3 uses this mode and "never crashes" — formally equivalent to **Empowerment** (Salge-Polani 2013) and to the **epistemic value** of Active Inference (Friston).

For coding, α=0 mode would mean "explore variations of an approach for diversity, ignore whether tests pass." Useful for brainstorming, breaking out of local optima.

The plugin's [`scripts/fractal_loop.py`](../scripts/fractal_loop.py) accepts `--alpha 0` in init, and the math works correctly (verified by relativize properties in `test_fractal_math.py` test 1). But no slash command surfaces this explicitly. To use it now, the orchestrator must be invoked manually with custom alpha, which is awkward.

A future `/fractal-explore` command would expose this mode. Not yet.

---

## 8. The honest epistemic position

The plugin is a **Phase 0 Proof of Concept**. What it proves:

1. The algorithm of Hernández-Cerezo & Duran-Ballester transfers cleanly to coding decisions when properly mapped to the Octopus structure.
2. The math layer is verifiably correct (5/5 tests, deterministic).
3. The state machine handles ESS-adaptive cloning, init_commit_sha tracking, and decision marginalization correctly on synthetic data.
4. The slash commands and sub-agents define a clean orchestration that, when invoked, will produce the expected behavior.

What it does NOT prove:

1. That on real codebases, the plugin produces **better commits** than a naïve single-shot LLM call. This is an empirical question that requires real runs.
2. That the cost is **worth** the quality improvement. Each `/fractal-decide` is ~10× the cost of a single sub-agent call. The improvement in decision quality must be > 10× to break even.
3. That the LLM-as-simulator is **stable enough** for FMC's convergence guarantees. Del Moral's theorems assume well-behaved Markov kernels; LLM output distributions may not satisfy regularity conditions.

The next step after this documentation is the same as for the original FMC paper: **run it, measure, report numbers**. The "Boxing 96/100 of the second phase" will be a real codebase, a real goal, with a real measurement of decision quality vs cost vs naïve baseline.

That measurement is not in this repo yet. When it is, this section will be updated.

---

## 9. Where to read next

| If you want | Read |
|---|---|
| The original paper | [`1803.05049v5.pdf`](../../../1803.05049v5.pdf) (root of repo) |
| Personal first analysis (theoretical) | [`ANALISIS.md`](../../../ANALISIS.md) |
| Personal post-experiment (96/100 Boxing) | [`analisisPost.md`](../../../analisisPost.md) |
| Discovery of the broader corpus (Book #2, Fractal Memory) | [`analisisPost2.md`](../../../analisisPost2.md) |
| Cloning math (Markov chain, Gibbs equilibrium) | [`work/02_deep_dives/01_cloning_mathematics.md`](../../../work/02_deep_dives/01_cloning_mathematics.md) |
| FMC ↔ SMC equivalence (Del Moral, Doucet) | [`work/02_deep_dives/05_smc_particle_filter_view.md`](../../../work/02_deep_dives/05_smc_particle_filter_view.md) |
| Book #2 + Fractal Memory analysis | [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](../../../work/02_deep_dives/06_book2_badger_fractal_memory.md) |
| Vision document for this plugin | [`docs/vision/fractal_coding_loop.md`](../../../docs/vision/fractal_coding_loop.md) |
| Full corpus 2014-2026 | [`docs/bibliography/CORPUS.md`](../../../docs/bibliography/CORPUS.md) |
| Algorithm walkthrough | [`ALGORITHM.md`](ALGORITHM.md) |
| File-by-file reference | [`COMPONENTS.md`](COMPONENTS.md) |
| Invocation, config, debugging | [`USAGE.md`](USAGE.md) |
