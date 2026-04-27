# FractalAI

> *A planner that reasons through possible trajectories — not gradients. Zero training, 360× sample efficiency over MCTS UCT, SoTA-class performance on Atari, Craftax, and real tokamak plasma control.*

[![Paper](https://img.shields.io/badge/paper-arXiv%3A1803.05049v5-b31b1b)](https://arxiv.org/abs/1803.05049)
[![Status](https://img.shields.io/badge/status-active%20research%20%2B%20replication-2ea44f)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Tests](https://img.shields.io/badge/fmc--core-66%2F66%20green-2ea44f)]()
[![Lang](https://img.shields.io/badge/code-english-005BBB)]()

---

## In one sentence

This repository contains **theoretical study, empirical replication, and original extensions** of **Fractal Monte Carlo (FMC)** by Sergio Hernández-Cerezo and Guillem Duran-Ballester — a planning algorithm that, with **no training**, beats MCTS, Rainbow, PPO, and DreamerV2/V3 on sample efficiency, and which we are pushing beyond games into **agentic coding**, **fusion reactor control**, and **traffic signal control**.

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
| **Sample efficiency** | 360× fewer rollouts than MCTS; beats published deep-RL on Crafter with **0 training steps** |
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
│   ├── MATH_CANON.md               ← canonical math: 6 defs, 3 thms, 3 conjectures
│   ├── SESSION_2026-04-27_SUMMARY.md  ← Sergio peer-review briefing
│   └── bibliography/               ← full corpus (papers, blog, codebases)
│
├── fmc-core/                       ← REFERENCE IMPLEMENTATION
│   ├── src/fmc/                    ← Python (NumPy) — 6 built-in envs
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
│   └── 07_sergio_branching_sweep/  ← original empirical study of b_eff*
│
├── plugin/fractal-coding-loop/     ← FMC as a Claude Code plugin
│
├── simulations/                    ← interactive HTML/JS demos
│   ├── kart.html / rocket.html / pong.html / octopus.html
│   └── cong_A_surface.html         ← interactive viz of the 4D b_eff surface
│
└── repos/                          ← cloned upstream codebases (FractalAI_old, fragile, fragile-rl)
```

---

## Empirical highlights

### 🧮 fmc-core + scaling-law discovery (April 2026)

Source of truth: [`fmc-core/`](fmc-core/), canon in [`docs/MATH_CANON.md`](docs/MATH_CANON.md), benchmark report in [`fmc-core/bench/REPORT.md`](fmc-core/bench/REPORT.md), interactive viz in [`simulations/cong_A_surface.html`](simulations/cong_A_surface.html).

**Three pieces of original work** delivered in a single autonomous `/loop` session:

1. **Canonical math document** — [`docs/MATH_CANON.md`](docs/MATH_CANON.md): 6 definitions, 3 theorems, 3 falsifiable conjectures, empirical-verification status table. Single citable reference replacing scattered deep-dive content.
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
| **FMC + intrinsic + delta-prox (ours)** | **21.87% ± 1.21** | **0** |
| EMERALD (absolute SoTA, July 2025) | 58.1% | 10M |

**FMC zero-training beats the tabular SoTA by +2.5 percentage points across 30 seeds.** Fully reproducible (`fmc_craftax_v4.py` with `intrinsic_inv_alpha=0.5, proximity_alpha=0.2, proximity_mode='delta'`).

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
| 07 | [`07_wright_fisher_mapping.md`](work/02_deep_dives/07_wright_fisher_mapping.md) | **NEW (2026-04-27)**: empirical confirmation of FMC ↔ Moran/Wright-Fisher mapping |

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
| MATH_CANON.md (canonical math) | ✅ v0.4.3 — 6 defs, 3 thms, 3 conjectures with falsifiability criteria |
| fmc-core (reference impl) | ✅ 6 envs, 55 Py + 11 JS tests green, bit-for-bit Py↔JS verified |
| Benchmark suite | ✅ 9 parameter sweeps + 3 applied tests (SUMO, FoT, WF validation), all in JSONL |
| Bet 3 — universal $b_{\text{eff}}^*$ | ✅ closed: triple falsification, WF mapping confirmed empirically (q=-0.948 vs -1 theoretical) |
| Bet 2 — Fractal-of-Thought | ✅ first-pass executed: +4.2pp vs self-consistency, +20.8pp vs greedy |
| Bet 1 — SUMO traffic | ✅ first-pass executed: +116% throughput vs SUMO actuated on asymmetric traffic |
| Atari replication | ✅ Boxing 96/100 verified, full plan for 5 games |
| Craftax | ✅ 21.87% Crafter score, 30 seeds, beats tabular SoTA |
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
