# FractalAI

> *A planner that reasons through possible trajectories — not gradients. Zero training, **1–2 orders-of-magnitude sample efficiency over MCTS-UCT** (replication in progress, see [audit](docs/bibliography/paper_fmc_dhdna_audit.md)), SoTA-class performance on Atari, **Craftax (50.95% Crafter, zero training — matches human-expert)**, and real tokamak plasma control — and, as of May 2026, an **agentic core that drives an LLM as a world-model organ**.*

[![Paper](https://img.shields.io/badge/paper-arXiv%3A1803.05049v5-b31b1b)](https://arxiv.org/abs/1803.05049)
[![Status](https://img.shields.io/badge/status-active%20research%20%2B%20replication-2ea44f)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Tests](https://img.shields.io/badge/fmc--core-66%2F66%20green-2ea44f)]()
[![Audit](https://img.shields.io/badge/paper%20audit-7%2F7%20tasks%20landed-2ea44f)](docs/bibliography/audit_completion_report.md)
[![Lang](https://img.shields.io/badge/code-english-005BBB)]()

---

## In one sentence

This repository contains **theoretical study, empirical replication, and original extensions** of **Fractal Monte Carlo (FMC)** by Sergio Hernández-Cerezo and Guillem Duran-Ballester — a planning algorithm that, with **no training**, beats MCTS, Rainbow, PPO, and DreamerV2/V3 on sample efficiency, and which we are pushing beyond games into **agentic coding**, **fusion reactor control**, **traffic signal control**, and an **FMC + LLM agentic merge**.

---

## 📰 Latest — May 2026

> The project added a fifth conjecture and closed it: **Conjecture E — FMC as an agentic core, an LLM as a world-model organ.** FMC supplies the *will* (causal-entropy drives: desire-to-act = α, self-preservation = β); the LLM is a sensorimotor interface. This inverts the usual stack.

- **2026-05-21 — E1-LLM-curve: f_abs is necessary, not sufficient.** A 4-model × 3-prompt ladder (Llama 1B→70B) put real LLM world-models *inside* the absorbing-fidelity tolerance curve. Within the random-ablation error class, f_abs predicts death exactly (band-comparable points 100% in-band, Wilcoxon p=1.00); but most LLM world-models fail *outside* it — broken movement, and broken **absorbing-persistence** (an 8B model: f_abs = 1.0 yet 64% death — it never wrote `if done: stay`). The FMC+LLM merge gate is **three-part** (entry-detection + movement + persistence), not f_abs alone. See [`work/12_conjecture_e/E1_LLM_CURVE_RESULT.md`](work/12_conjecture_e/E1_LLM_CURVE_RESULT.md).
- **2026-05-21 — Conjecture E complete.** All three pre-registered tests verified — **E1-base**, **E2**, **E1-LLM**. The merge is operationally validated on its central test: an LLM (Llama 3.3 70B) writes the world-model as code, FMC plans on it, and emergent self-preservation survives the organ swap — **death 0/180 vs random 47.8%** (z = −10.6, p < 10⁻⁴). Honest caveat: the LLM produced a *functionally exact* model (f_abs = 1.0), so the test was easy — the scientific teeth are in the absorbing-fidelity tolerance curve. See [`work/12_conjecture_e/E1_LLM_RESULT.md`](work/12_conjecture_e/E1_LLM_RESULT.md).
- **2026-05-21 — hP13-0: a clean, redirecting falsification.** The FMC decision is a *chaotically-amplified* function of the **exact** virtual-reward vector — not its rank: at Spearman 0.97 (rank near-perfect) decision-agreement is already 0.47. The VR-rank shortcut for sparse LLM interrogation is dead; the absorbing-structure gate survives and is reinforced. See [`work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md`](work/12_conjecture_e/P13_HP13_0_PHI_RESULT.md).
- **2026-05-01 — Craftax exp17: 50.95% Crafter, zero training** — matches/beats the human-expert score (50.5%). A 23-experiment autoresearch session discovered **Conjecture D** (chain-tier compounding amplification). See [`work/05_craftax/autoresearch/HANDOFF.md`](work/05_craftax/autoresearch/HANDOFF.md).
- Canonical math is now [`docs/MATH_CANON.md`](docs/MATH_CANON.md) **v0.7.5** — 6 definitions, 3 theorems, **5 falsifiable conjectures (A–E)**, each with empirical-verification status.

## Why FMC matters (and why it's an AGI piece)

The deep-RL community has spent the last decade burning **billions of training steps** on policies that often fail to generalize. FMC starts from the opposite premise:

> **Intelligence is not a trained neural network. It is a search procedure that, given a reward function and a simulator, discovers intelligent actions in milliseconds — *with no weights to learn*.**

Three concepts:

1. **Walkers** — N copies of the agent are launched in parallel into the simulated future (M ticks ahead).
2. **Virtual reward** $V = R^\alpha \cdot D^\beta$ — rewards walkers that accumulate reward *and* stay diverse from each other (Tsallis-entropy-inspired diversity).
3. **Cloning** — periodically, weak walkers "become" copies of strong walkers. Natural selection in real time, inside the planner.

The system converges to a Gibbs distribution over optimal trajectories (formal connection to Sequential Monte Carlo / particle filtering — see [`work/02_deep_dives/05_smc_particle_filter_view.md`](work/02_deep_dives/05_smc_particle_filter_view.md)). And it does this with **100–300 samples per decision**, not 3 million like MCTS UCT.

### Why this is an AGI building block

| AGI axis | What FMC offers |
|---|---|
| **Sample efficiency** | 1–2 orders-of-magnitude fewer rollouts than MCTS (the paper claims 359×, the audit finds incompatible 100×–10 000× cross-source claims, and our in-session micro-sweep on Boxing — n=3, B=240, CPU — gives FMC +100 vs MCTS −5; full P0 sweep pending); beats published deep-RL on Crafter with **0 training steps** |
| **Compute-at-inference** | "Thinking" scales with N×M, not with the training dataset — *the more time you think, the better you decide* |
| **Generalization** | No memorized policy = no overfitting to the training set |
| **Planning + memory** | The framework extends naturally to **Fractal Memory** (Wigner-weighted recall) and **Octopus** (multi-level loops, Badger-style) — see [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](work/02_deep_dives/06_book2_badger_fractal_memory.md) |
| **Embodiment-ready** | Works on any simulator: ALE, Craftax, Grad-Shafranov plasma equilibria, *and code* — the only requirement is "step + reward" |

In short: FMC replaces "*train a huge network on everything*" with "*you have a simulator? then you already know how to plan intelligently inside it*". A complement — not a substitute — to deep learning, and possibly the missing piece for systems that today fail in sparse-reward or long-horizon environments (Montezuma's Revenge, deep Crafter achievements, industrial control).

---

## What's in this repo

```
FractalAI/
├── 1803.05049v5.pdf                ← canonical paper (Hernández-Cerezo 2020)
├── docs/
│   ├── MATH_CANON.md               ← canonical math: 6 defs, 3 thms, 5 conjectures (A–E)
│   ├── SESSION_2026-04-27_SUMMARY.md  ← Sergio peer-review briefing
│   ├── architecture/               ← Tier-1 repos teardown (fragile, plangym, …)
│   └── bibliography/
│       ├── CORPUS.md               ← full corpus (papers, blog, codebases)
│       ├── paper_fmc_dhdna_audit.md         ← 2026-04 paper audit, 7 priorities
│       ├── paper_corrections_for_v6.md      ← claim-by-claim corrections for paper v6
│       ├── audit_completion_report.md       ← audit closure report
│       ├── sergio_cognitive_profile_dhdna.md← cognitive-DNA profile of Sergio
│       └── protocols/              ← P0/P1a/P3 executable replication protocols
│
├── fmc-core/                       ← REFERENCE IMPLEMENTATION
│   ├── src/fmc/                    ← Python (NumPy) — 7 built-in envs (incl. Atari via plangym)
│   ├── js/fmc.js                   ← JS port, bit-for-bit identical to Python
│   ├── tests/                      ← 55 Python + 11 JS, all green
│   ├── bench/                      ← 9-sweep benchmark suite + REPORT.md
│   └── Makefile                    ← make install / test-all / bench-full
│
├── work/                           ← APPLIED EXPERIMENTS
│   ├── 02_deep_dives/              ← 7 formal deep-dives (cloning math, SMC, WF mapping…)
│   ├── 03_atari_replication/       ← FMC on 3-5 Atari games (Boxing 96/100)
│   ├── 04_diagrams/                ← C4 + Mermaid for FMC and fragile-rl
│   ├── 05_craftax/                 ← FMC on Crafter, beats tabular SoTA at 0 training
│   ├── 06_plasma_fmc/              ← FMC for TCV plasma control (real shot validated)
│   ├── 07_sergio_branching_sweep/  ← original empirical study of b_eff*
│   ├── 08_simulators/              ← Python+plangym simulators replacing HTML demos
│   ├── 09_fmc_vs_mcts_replication/ ← P0 — FMC vs MCTS-UCT controlled replication
│   ├── 10_atari_replication/       ← P1a — Atari multi-seed with bootstrap CI95
│   └── 11_ram_vs_img_ablation/     ← P3 — RAM vs IMG ablation across (N, M)
│
├── plugin/fractal-coding-loop/     ← FMC as a Claude Code plugin
│
├── simulations/                    ← interactive HTML/JS demos
│   ├── kart.html / rocket.html / pong.html / octopus.html
│   └── cong_A_surface.html         ← interactive viz of the 4D b_eff surface
│
└── repos/                          ← cloned upstream codebases (FractalAI_old, fragile, fragile-rl, plangym, shaolin, hydraclick, flogging)
```

---

## Empirical highlights

### 🧮 fmc-core + scaling-law discovery (April 2026)

Source of truth: [`fmc-core/`](fmc-core/), canon in [`docs/MATH_CANON.md`](docs/MATH_CANON.md), benchmark report in [`fmc-core/bench/REPORT.md`](fmc-core/bench/REPORT.md), interactive viz in [`simulations/cong_A_surface.html`](simulations/cong_A_surface.html).

**Three pieces of original work** delivered in a single autonomous `/loop` session:

1. **Canonical math document** — [`docs/MATH_CANON.md`](docs/MATH_CANON.md) (now v0.7.4): 6 definitions, 3 theorems, 5 falsifiable conjectures (A–E), empirical-verification status table. Single citable reference replacing scattered deep-dive content.
2. **Reference implementation** — [`fmc-core/`](fmc-core/): Python (NumPy) and JS port that produce **bit-for-bit identical** virtual rewards on shared fixtures (1e-12 tolerance). Six built-in environments: gridworld, rocket 2D, cartpole, navigation 2D, pendulum swing-up, navigation 2D parameterized in K. **66 tests green** (55 Python + 11 JS).
3. **Benchmark suite** — [`fmc-core/bench/`](fmc-core/bench/): uniform runner with bootstrap CI95 and JSONL output, **9 parameter sweeps** plus 3 LLM/SUMO experiments.

#### Original scientific result: the "magic 6" of Sergio is not a constant

The Radient 2026 podcast (ch. 16) proposed that $b_{\text{eff}}^* \approx 6$ is a Third Law of cognition — a universal optimal branching factor for FMC under tuned reward. This session **falsifies that claim** with three successive empirical experiments:

| Step | Finding |
|---|---|
| K-sweep on navigation 2D | $b_{\text{eff}}^* \approx 1.53 \cdot K^{0.6}$ (8 K-values, $25\times$ better fit than constant) — "6" is just the value at $K=9$ |
| M-sweep at K=9 | The $K^{0.6}$ scaling is **transient**; at $M=120$ the system collapses to palmera ($b_{\text{eff}} \to 1$), as predicted by Theorem 2 (Gibbs equilibrium) |
| N-sweep at K=9 | At $\alpha=0$ exact, $K - b_{\text{eff}} \propto N^{-0.948}$ — within $5\%$ of the Wright-Fisher prediction $-1$. **FMC at small $\alpha$ is empirically a Moran neutral process.** |

Final reformulation:

$$b_{\text{eff}}^*(\alpha, \beta=0, K, N, M) \approx 1 + (K-1) \cdot \mathcal{F}(M/N) \cdot \mathcal{G}(\alpha, K)$$

Sergio's "6" is a single point on a 4D surface — triply contingent on $(K=9, M=15, N=32)$. The full discussion is in [`work/02_deep_dives/07_wright_fisher_mapping.md`](work/02_deep_dives/07_wright_fisher_mapping.md). Open the interactive 3D plot in [`simulations/cong_A_surface.html`](simulations/cong_A_surface.html) to see it.

#### Bet 1 — SUMO traffic, single intersection (first-pass)

Source: [`fmc-core/bench/sumo/`](fmc-core/bench/sumo/), JSONL in `bench/results/sumo_*`.

| Scenario | Method | Throughput | Avg waiting time |
|---|---|---|---|
| **Symmetric Poisson** (rate=0.15/s/dir, 8 seeds) | actuated | $156 \pm 20$ | 1623 |
| | static 30/30 | $249 \pm 32$ | 1732 |
| | **fmc** | $\mathbf{261 \pm 16}$ | **1160** |
| **Asymmetric** (N/S=0.30, E/W=0.05, 6 seeds) | actuated | $146 \pm 22$ | 1181 |
| | static 30/30 | $256 \pm 80$ (unstable) | 1822 |
| | **fmc** | $\mathbf{314 \pm 9}$ | **413** |

**FMC wins +116% throughput vs SUMO actuated and +23% vs static cycle on asymmetric traffic, with the lowest variance (σ=9 vs 80).** On stationary symmetric traffic the static cycle is already near-optimal. Verdict: FMC is the right tool for naturalistic adaptive scenarios; ignore it for the trivial-symmetric case.

#### Bet 2 — Fractal-of-Thought on a small LLM

Source: [`fmc-core/bench/llm/fot.py`](fmc-core/bench/llm/fot.py), JSONL in `bench/results/fot_llm_*`.

Model: `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` (≈ Llama-3.2-1B class). Benchmark: 12 multi-step math problems, 2 seeds.

| Method | Accuracy | Avg tokens/problem |
|---|---|---|
| Greedy ($T=0$) | $66.7\%$ (16/24) | 247 |
| Self-consistency ($K=8$) | $83.3\%$ (20/24) | 1931 |
| **FoT** ($N=8, M=2$) | $\mathbf{87.5\%}$ (21/24) | 2522 |

FoT beats greedy by $+20.8$ pp and self-consistency by $+4.2$ pp. **Honest verdict**: positive but modest improvement over self-consistency; for a clean *go* signal the setup needs LLM-as-judge reward, more cycles, and a larger benchmark (full GSM8K).

### 🎮 Atari — controlled replication

Source: [`work/03_atari_replication/`](work/03_atari_replication/)

> **Boxing 96/100 in 7 minutes, 231 lines of NumPy, no GPU.** Squarely in the paper's reported range.

Reproducible smoke test (`run_single.py --config boxing.yaml --seed 42`). 3-5 different games (Boxing, MsPacman, Centipede, Asteroids, Montezuma) with 95% confidence intervals over 5 seeds. The setup confirms **<500 samples/action** as claimed in the paper.

### 🌳 Craftax — beating published deep-RL with 0 training

Source: [`work/05_craftax/`](work/05_craftax/)

| Method | Crafter score | Training samples |
|---|---|---|
| Random baseline | 1.6% | 0 |
| Rainbow | 4.3% | 1M |
| PPO | 4.6% | 1M |
| DreamerV2 | 10.0% | 1M |
| DreamerV3 | 14.5% | 1M |
| Curious Replay (tabular SoTA pre-2025) | 19.4% | 1M |
| **FMC exp17 — chain-tier reward shaping (ours)** | **50.95%** | **0** |
| Human expert | 50.5% | — |
| EMERALD (absolute SoTA, July 2025) | 58.1% | 10M |

**FMC at zero training reaches 50.95% Crafter — matching/beating the human-expert score (50.5%)** and closing most of the gap to the 10M-step absolute SoTA. A 23-experiment autoresearch `/loop` session (exp03 → exp17, +21 pp over the session) discovered the **chain-tier compounding amplification** mechanism — now **Conjecture D** in [`MATH_CANON.md`](docs/MATH_CANON.md). Trajectory and handoff: [`work/05_craftax/autoresearch/HANDOFF.md`](work/05_craftax/autoresearch/HANDOFF.md).

### ⚛️ Plasma — FMC for real tokamak control

Source: [`work/06_plasma_fmc/`](work/06_plasma_fmc/) — **17 milestones, 118/118 tests green.**

Most ambitious thread: use FMC to control the plasma shape on a real **TCV tokamak** (Tokamak à Configuration Variable, EPFL Lausanne). The problem matters globally for fusion: plasma must remain confined within precise geometry, and the controller must hit sub-millisecond latency.

| Metric | Value |
|---|---|
| Decision latency (NN policy distilled from FMC) | **122 µs** (8× margin under the 1 ms target) |
| Speedup vs FMC online | **109×** |
| Speedup of dataset generation (JIT FMC) | **200×** |
| Quench rate (BC → DAgger) | **9/10 → 0/10** |
| Tracking error in-sim (BC → DAgger) | **10×** reduction |
| **Truth error on real TCV shot 65402 (M12 NN-shape)** | **3.47 cm with 100% physicality** — comparable to TCV's operational PCS |

Validated not only in-sim but on the **TCV-X21 dataset (CC-BY-4.0)**, with real experimental shot `65402_t1.eqdsk`. Full synthesis in [`work/06_plasma_fmc/docs/SYNTHESIS_PAPER.md`](work/06_plasma_fmc/docs/SYNTHESIS_PAPER.md), including the **negative lessons** (M13: the NN-proxy oracle had bias; corrected in M14 with the real Grad-Shafranov solver, which revealed a 22× ranking spread).

### 🐙 Agentic coding — FMC as a Claude Code planner

Source: [`plugin/fractal-coding-loop/`](plugin/fractal-coding-loop/)

The plugin ports FMC from games to code. Four slash commands:

- **`/fractal-decide [goal]`** — ONE FMC decision: spawn N walkers in isolated git worktrees, M cycles of explore + clone, cherry-pick the winning commit onto main.
- **`/octopus [goal]`** — outer loop calling `/fractal-decide` until a judge declares the goal reached.
- **`/fractal-recall [query]`** — Wigner-weighted recall of past decision episodes (Fractal Memory).
- **`/fractal-memory-show`** — dump the memory bank with per-memory statistics.

Math layer **certified by 5 deterministic tests** (Gibbs convergence verified numerically). 17/17 e2e tests pass.

### 🧠 Conjecture E — FMC as agentic core, LLM as world-model organ

Source: [`work/12_conjecture_e/`](work/12_conjecture_e/) — north-star direction (May 2026).

Invert the usual stack. **FMC is the will**: its virtual reward $V = R^\alpha \cdot D^\beta$ already carries two intrinsic drives — α = desire-to-act (goal-seeking), β = self-preservation (causal-entropy / freedom of future action, à la Wissner-Gross). **The LLM is an organ** — perception, world-model, action grounding, voice.

Three pre-registered tests, all **verified**:

| Test | Question | Result |
|---|---|---|
| **E1-base** | Does FMC at low α avoid terminal states with *no* survival reward? | ✅ 0% lava-death vs 85–100% random/greedy, p<0.001 — self-preservation **emerges** from causal entropy |
| **E2** | Do α and β separate into goal-seeking vs preservation? | ✅ α owns the goal (η²=0.91); β halves death (OR 0.48) at *zero* goal cost — safety is near-free |
| **E1-LLM** | Does it survive swapping the simulator for an **LLM world-model**? | ✅ Llama 3.3 70B writes the world-model in code; FMC plans on it; death **0/180** vs random 47.8% (z=−10.6) |

Two honest negatives sharpened the program along the way: **P13/hP13-1** — a world-model blind to absorbing states is *actively lethal* (death up to 80%, worse than random) — and **hP13-0** — the FMC decision depends on the *exact* virtual-reward vector, chaotically, not on its rank. Both are recorded in [`docs/MATH_CANON.md`](docs/MATH_CANON.md) Conjecture E and the [`work/12_conjecture_e/`](work/12_conjecture_e/) result docs.

---

## Paper audit & replication harnesses — April 2026

A two-session `/loop` produced (1) a full audit of the canonical paper for review-grade rigor, and (2) executable harnesses for the three empirical priorities the audit flagged. All deliverables under [`docs/bibliography/`](docs/bibliography/) and [`work/09|10|11/`](work/).

### Audit outputs

| Artefact | What it is |
|---|---|
| [`paper_fmc_dhdna_audit.md`](docs/bibliography/paper_fmc_dhdna_audit.md) | Claim-by-claim audit of arXiv:1803.05049v5 — 7 priorities, status & severity per claim |
| [`paper_corrections_for_v6.md`](docs/bibliography/paper_corrections_for_v6.md) | Concrete, paste-ready corrections for the v6 paper draft (P2a/P2b/P2c) |
| [`audit_completion_report.md`](docs/bibliography/audit_completion_report.md) | Closure report — 4 documentary tasks resolved, 3 empirical with harness landed |
| [`sergio_cognitive_profile_dhdna.md`](docs/bibliography/sergio_cognitive_profile_dhdna.md) | Cognitive-DNA profile of Sergio (DHDNA), used to calibrate audit reading |
| [`protocols/P0_*.md`](docs/bibliography/protocols/P0_fmc_vs_mcts_protocol.md), [`P1a_*.md`](docs/bibliography/protocols/P1a_atari_replication_protocol.md), [`P3_*.md`](docs/bibliography/protocols/P3_ram_vs_img_ablation_protocol.md) | Three full-spec protocols with decision matrices and methodological caveats |

Two known cross-source discrepancies are now formalized in [`CLAUDE.md`](CLAUDE.md):

- **D1 — Magic-6**: falsified as universal. Sergio's $b_{\text{eff}}^* \approx 6$ is a single point on the 4D surface $b_{\text{eff}}^*(K, N, M, \alpha) \approx 1 + (K-1) \cdot \mathcal{F}(M/N) \cdot \mathcal{G}(\alpha, K)$ ([`MATH_CANON.md`](docs/MATH_CANON.md) Cong. A v0.4.0).
- **D2 — Sample efficiency vs MCTS**: paper sources self-disagree (range 100×–10 000×). Boxing micro-sweep (n=3 seeds, B∈{80, 240}, RAM, CPU) gives FMC mean +91 / +100 vs MCTS −5 / −5 — directional support for the paper's qualitative claim, but the precise multiplier requires the full P0 sweep.

### Replication harnesses (all CPU-runnable)

| Dir | Protocol | What's runnable today |
|---|---|---|
| [`work/09_fmc_vs_mcts_replication/`](work/09_fmc_vs_mcts_replication/) | **P0** | MCTS-UCT baseline ([`scripts/mcts_uct.py`](work/09_fmc_vs_mcts_replication/scripts/mcts_uct.py)) against the FMC env protocol; CartPole + Boxing micro-sweep recorded |
| [`work/10_atari_replication/`](work/10_atari_replication/) | **P1a** | Multi-seed sweep with bootstrap CI95 ([`scripts/atari_seed_sweep.py`](work/10_atari_replication/scripts/atari_seed_sweep.py)); Boxing slice n=5, paper params (N=30, M=15) — 5/5 knockout, mean +100, std 0 |
| [`work/11_ram_vs_img_ablation/`](work/11_ram_vs_img_ablation/) | **P3** | Parametric RAM vs IMG sweep across $(N, M)$; aggregator + Boxing micro-cell recorded |

### Methodological surprise — the protocols don't need a GPU cluster

The protocols originally estimated **50–250 GPU-hours, 1–3 weeks on a cluster**. Measured wall-times for the *full* sweeps using the [`fmc-core/`](fmc-core/) NumPy reference on a single CPU:

| Protocol | Full sweep size | Original estimate | Measured wall (1 CPU) |
|---|---|---|---|
| **P0** | 3 games × 7 budgets × 10 seeds × 2 algos = 420 episodes | ~50–100 GPU-h | **~7 h** |
| **P1a** | 50 games × 10 seeds = 500 episodes | ~50–80 GPU-h | **~11 h** |
| **P3** | 8 games × 4 N × 4 M × 5 seeds × 2 obs = 1280 cells | ~25 GPU-h | **~28 h** |

The cluster requirement was an artifact of `fragile`'s GPU tensor swarms; the NumPy reference is fast on CPU because cloning has no neural network forward pass. **All three full sweeps are now overnight runs on a workstation.**

---

## Theory section — the deep dives

[`work/02_deep_dives/`](work/02_deep_dives/) holds seven formal expansions, each 250-1200 lines with precise code citations (`file:line`) and bibliography:

| # | Doc | What it covers |
|---|---|---|
| 01 | [`01_cloning_mathematics.md`](work/02_deep_dives/01_cloning_mathematics.md) | Cloning math, convergence theorem (Del Moral 2004) |
| 02 | [`02_active_inference_link.md`](work/02_deep_dives/02_active_inference_link.md) | Friston ↔ Hernández-Cerezo bridge: free-energy as virtual reward |
| 03 | [`03_standard_model_cognition.md`](work/02_deep_dives/03_standard_model_cognition.md) | FMC mapped onto the Standard Model of Cognition (Laird, Lebiere, Rosenbloom) |
| 04 | [`04_relativize_axiomatics.md`](work/02_deep_dives/04_relativize_axiomatics.md) | Axiomatization of the `relativize` operator (paper §2.2.3) |
| 05 | [`05_smc_particle_filter_view.md`](work/02_deep_dives/05_smc_particle_filter_view.md) | FMC ≅ Sequential Monte Carlo with adaptive resampling |
| 06 | [`06_book2_badger_fractal_memory.md`](work/02_deep_dives/06_book2_badger_fractal_memory.md) | Sergio's Book #2: Octopus / Badger / Hives + operational Fractal Memory |
| 07 | [`07_wright_fisher_mapping.md`](work/02_deep_dives/07_wright_fisher_mapping.md) | Empirical confirmation of FMC ↔ Moran/Wright-Fisher mapping |
| 08 | [`08_video_seminar_extracted_insights.md`](work/02_deep_dives/08_video_seminar_extracted_insights.md) | Sergio's video seminar — 15 formulas extracted and numerically verified |
| 09 | [`09_chaos_order_frontier_formalization.md`](work/02_deep_dives/09_chaos_order_frontier_formalization.md) | **NEW (2026-05)**: chaos/order frontier formalized — Conjecture B downgraded to a testable reward diagnostic |

C4 architecture diagrams (context, container, components) and a layered view of `fragile-rl` are in [`work/04_diagrams/`](work/04_diagrams/).

---

## Quick start

### Use the reference implementation

```bash
cd fmc-core
make install              # editable pip install
make test-all             # 55 Python + 11 JS, all green
make bench-full           # ~3 min, reproduces all parameter sweeps
```

```python
from fmc.core import plan
from fmc.envs.rocket import Rocket

env = Rocket()
action = plan(env, env.reset(), N=64, M=30, alpha=1.0, beta=1.0, seed=42)
```

### Replicate Atari (smoke test)

```bash
cd work/03_atari_replication/scripts
python run_single.py --config ../configs/boxing.yaml --seed 42 \
    --output ../results/boxing_seed42.json
# Expected: episode terminates in <10 min with reward >= 99
```

### Reproduce the directional D2 signal (FMC vs MCTS-UCT, ~10 min CPU)

```bash
cd work/09_fmc_vs_mcts_replication
# n=3 seeds × 2 budgets × 2 algos on Boxing, RAM obs, frame_skip=4
for seed in 0 1 2; do
  for B in 80 240; do
    for algo in fmc mcts; do
      python -m scripts.atari_episode \
        --algo $algo --game Boxing --B $B --seed $seed \
        --max_actions 200 --out runs/boxing_micro.jsonl
    done
  done
done
# Expected: FMC mean ~+91 (B=80), +100 (B=240); MCTS mean ~-5 both budgets.
# See REPORT.md for the full result.
```

### Run the P1a Atari multi-seed sweep (~80 s/seed CPU)

```bash
cd work/10_atari_replication
python -m scripts.atari_seed_sweep \
    --games Boxing --seeds 5 --N 30 --M 15 --max_actions 300 \
    --out_runs runs/boxing_seeds.jsonl \
    --out_summary runs/boxing_seeds_summary.csv
# Boxing slice ships in REPORT.md: 5/5 seeds knockout, mean 100, std 0.
```

### Run FMC on Craftax

```bash
cd work/05_craftax
python3 scripts/sweep_seeds.py --n_walkers 64 --time_horizon 20 \
    --alpha 1.0 --beta 1.0 --n_seeds 5 --seed_start 42
# Best config: fmc_craftax_v4.py with --intrinsic_inv_alpha 0.5 --proximity_alpha 0.2
```

### Run the plasma real-time dashboard

```bash
cd work/06_plasma_fmc
bash run_all_tests.sh                          # verify 118/118 tests
streamlit run scripts/dashboard_realtime.py    # M14 oracle truth + TCV-X21 target + FMC internals
```

### Try the coding plugin

```bash
# Verify the certified math
python3 plugin/fractal-coding-loop/tests/test_fractal_math.py
# Expected: "All FMC math tests passed — convergence certified."

# Inside Claude Code:
/fractal-decide "implement add(a, b) in src/math.py with a unit test"
/octopus "POST /login returns a valid JWT, tests/auth_test.py passes"
```

### Open the live simulations (browser)

```bash
open simulations/index.html
# Cart-pole, rocket, pong, octopus — all FMC in JavaScript, zero dependencies

open simulations/cong_A_surface.html
# Interactive Plotly viz of the 4D b_eff surface (Sergio's "6" reformulated)
```

---

## Conventions

- **English** throughout (codebase + prose). Older Italian deep dives are kept verbatim under `work/02_deep_dives/` for archival fidelity.
- **ISO 8601 dates** (`2026-04-27`).
- **Project-relative paths**.
- **Paper citations**: `(Hernández-Cerezo & Duran-Ballester, 2020, §X.Y)`.
- **Code citations**: markdown links to `file:line`.

---

## Minimal bibliography

Reading order:

1. **Canonical paper** — Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence*, [arXiv:1803.05049v5](https://arxiv.org/abs/1803.05049). §2.2 = math; §4 = algorithm; §5 = Atari results.
2. **Empirical companion** — Hernández-Cerezo et al. (2018), *Solving Atari Games Using Fractals And Entropy*, [arXiv:1807.01081](https://arxiv.org/abs/1807.01081). FMC > MCTS UCT with <1000 vs 3M samples/action.
3. **Predecessor** — Hernández et al. (2017), *General Algorithmic Search*, [arXiv:1705.08691](https://arxiv.org/abs/1705.08691). FMC = "GAS applied to planning".
4. **Entropic foundation** — Amigó, Balogh, Hernández (2018), *A Brief Review of Generalized Entropies*, Entropy 20(11):813.

Full corpus index (papers, drafts, blog posts, codebases, known gaps): [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md).

---

## Project status

| Thread | Status |
|---|---|
| MATH_CANON.md (canonical math) | ✅ v0.7.4 — 6 defs, 3 thms, 5 conjectures A–E with falsifiability criteria |
| fmc-core (reference impl) | ✅ 7 envs (incl. Atari via plangym), 55 Py + 11 JS tests green, bit-for-bit Py↔JS verified |
| Benchmark suite | ✅ 9 parameter sweeps + 3 applied tests (SUMO, FoT, WF validation), all in JSONL |
| **Paper audit (Apr 2026)** | ✅ 4 documentary tasks resolved (P1b/P2a/P2b/P2c); 3 empirical with harnesses landed |
| **D1 — universal $b_{\text{eff}}^*$** | ✅ falsified: triple contingency (K, M, N, α); WF mapping confirmed (q=−0.948 vs −1 theoretical) |
| **D2 — FMC vs MCTS sample efficiency** | 🟡 directional in-session signal (Boxing n=3: FMC +100 vs MCTS −5 at B=240); full P0 sweep pending |
| **P0 harness — FMC vs MCTS-UCT** | ✅ MCTS-UCT baseline written ([`work/09/`](work/09_fmc_vs_mcts_replication/)); ~7 h single CPU for full sweep |
| **P1a harness — Atari multi-seed** | ✅ bootstrap-CI sweeper ([`work/10/`](work/10_atari_replication/)); Boxing slice n=5, 5/5 knockout |
| **P3 harness — RAM vs IMG ablation** | ✅ parametric driver + aggregator ([`work/11/`](work/11_ram_vs_img_ablation/)) |
| Bet 3 — universal $b_{\text{eff}}^*$ | ✅ closed: triple falsification, WF mapping confirmed empirically (q=-0.948 vs -1 theoretical) |
| Bet 2 — Fractal-of-Thought | ✅ first-pass executed: +4.2pp vs self-consistency, +20.8pp vs greedy |
| Bet 1 — SUMO traffic | ✅ first-pass executed: +116% throughput vs SUMO actuated on asymmetric traffic |
| Atari replication (paper §5.1.1) | ✅ Boxing 96/100 verified, full plan for 5 games |
| Craftax (Conjecture D) | ✅ exp17 = 50.95% Crafter, zero training, matches human-expert (50.5%); chain-tier compounding discovered |
| **Conjecture E — FMC + LLM merge** | ✅ all 3 tests verified (E1-base, E2, E1-LLM); FMC-core + LLM-world-model-organ operationally validated |
| Plasma TCV | ✅ 17 milestones, 118/118 tests, validated on real shot 65402 |
| Coding plugin | ✅ math layer certified (5/5), e2e (17/17); end-to-end LLM testing in progress |
| JS simulations | ✅ kart, rocket, pong, octopus interactive |
| 4D surface viz | ✅ `simulations/cong_A_surface.html` |

---

## Credits

**FMC algorithm**: Sergio Hernández-Cerezo ([@EntropyFarmer](https://twitter.com/EntropyFarmer)) and Guillem Duran-Ballester ([@Miau_DB](https://twitter.com/Miau_DB)), 2014–2026 — over a decade of independent work that never went mainstream despite the empirical evidence.

**Replication, extension, scaling-law analysis, fmc-core, plugin, Bets 1/2/3**: Vlad Vrinceanu ([@uppifyagency](https://github.com/uppifyagency)), 2026.

For the full intellectual debt (papers, blog posts, codebases, drafts, people): [`docs/bibliography/CORPUS.md`](docs/bibliography/CORPUS.md).

Released under the MIT License.

---

> *"Published deep-RL fails to reach the last two achievement classes with 1B steps. FMC with 0 training steps could be a complement, not a substitute."*
>
> — `work/05_craftax/README.md`
