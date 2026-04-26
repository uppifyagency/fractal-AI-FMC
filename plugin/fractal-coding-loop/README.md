# fractal-coding-loop

> *Fractal Monte Carlo planning for goal-directed coding decisions in Claude Code.*

A Claude Code plugin that translates the **Fractal Monte Carlo (FMC)** algorithm of Hernández-Cerezo & Duran-Ballester (2020, [arXiv:1803.05049](https://arxiv.org/abs/1803.05049)) from Atari game-playing to codebase evolution.

You give it a goal. It spawns N parallel walker sub-agents in isolated git worktrees. They explore distinct strategies. Between ticks, walkers in losing trajectories are probabilistically reset to better walkers' state. After M ticks, the dominant initial strategy wins, and its **first commit** is cherry-picked to your main branch. Repeat until the goal is reached.

It is the same mechanism that achieved **96/100 on Atari Boxing in 7 minutes with 231 lines of NumPy** (see [`SMOKE_TEST_REPORT.md`](../../work/03_atari_replication/results/SMOKE_TEST_REPORT.md)), adapted faithfully to coding via the Octopus structure of Sergio's [2015 blog post on Fractal AI Collaboration](../../docs/bibliography/sources/blog_posts/2015-12_fractal_ai_collaboration.md).

---

## What this plugin gives you

Four slash commands:

- **`/fractal-decide [goal]`** — ONE FMC decision (N walkers × M ticks of perturbation + cloning) producing one commit toward the goal.
- **`/octopus [goal]`** — outer loop that calls `/fractal-decide` repeatedly until the goal is judged complete or `K_MAX` iterations are exhausted.
- **`/fractal-recall [query]`** — Wigner-weighted recall of past decision episodes (Slide doc 2020 §"Dataset as Fractal Memory").
- **`/fractal-memory-show`** — dump the memory bank with per-memory stats (loss, visits, weights).

Two sub-agents:

- **`fractal-walker`** — dual-mode worker (init at tick 0, continuation at tick > 0) that lives in an isolated git worktree.
- **`fractal-judge`** — goal-alignment scorer used both inside `/fractal-decide` (per-walker R_goal) and inside `/octopus` (overall goal completion check).

Three scripts:

- **`scripts/fractal_reward.py`** — the math layer: `relativize` (paper §2.2.3), composite reward (paper §2.2.2 + post-Pareto 2016), virtual reward `R^α · D^β` (paper §4.4).
- **`scripts/fractal_loop.py`** — the state-machine that orchestrates the M-tick loop: init / record / step / apply-clones / decide. Includes ESS-adaptive cloning (Doucet et al. 2001).
- **`scripts/fractal_memory.py`** — Wigner-weighted memory bank: append / recall / show / prune. Each past decision is stored as a markdown file with frontmatter; recall samples by `R(x) = (π/2) x exp(-π/4 x²)` debiased for visit count.

Two test layers that certify correctness:

- **`tests/test_fractal_math.py`** — 5 deterministic math tests: R=(0.9, 0.5, 0.1) at M=10 → walker_0 wins 100% of 200 runs (Gibbs delta), R=(1.0, 0.95, 0.0) at M=8 → 59%/41%/0% split (close-top distribution).
- **`tests/e2e_test.sh`** — 17 integration tests across plugin manifest, markdown frontmatter, reward script, memory round-trip, and Wigner formula shape.

---

## Quick orientation

### The two-level structure (the Octopus)

```
USER:  /octopus "implement POST /login with JWT, test passes"
  │
  └─ Octopus (outer loop) — the "hand of hands" in Sergio's metaphor
       │
       ├─ Iteration 1: /fractal-decide
       │    ├─ spawn N=3 walkers (fingers) in worktree-isolated branches
       │    ├─ tick 0: each walker INIT mode (different strategy)
       │    ├─ tick 1: step (compute VR + ESS) → clone if needed → CONTINUATION
       │    ├─ tick 2: step → clone → CONTINUATION
       │    └─ decide: argmax bincount(init_action) → winner_init_commit_sha
       │  cherry-pick winner_init_commit_sha to main
       │  judge: goal_score = 0.42
       │
       ├─ Iteration 2: /fractal-decide → cherry-pick → goal_score = 0.78
       ├─ Iteration 3: /fractal-decide → cherry-pick → goal_score = 0.97 ✓
       │
       └─ GOAL REACHED in 3 iterations
```

This is structurally identical to the Atari main loop:

```
while not env.game_over():
    state = env.current_state
    action = fmc.decide(state, N_walkers, M_ticks)
    env.act(action)
```

### File map

| Path | Role |
|---|---|
| [`README.md`](README.md) | this file: entry point, overview, links |
| [`INSTALL.md`](INSTALL.md) | installation steps and troubleshooting |
| [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json) | plugin manifest (name, version, keywords) |
| [`scripts/fractal_reward.py`](scripts/fractal_reward.py) | math layer: relativize, composite reward, VR |
| [`scripts/fractal_loop.py`](scripts/fractal_loop.py) | state machine: init/record/step/apply-clones/decide |
| [`scripts/fractal_memory.py`](scripts/fractal_memory.py) | Wigner-weighted memory bank: append/recall/show/prune |
| [`tests/test_fractal_math.py`](tests/test_fractal_math.py) | math certification (5 deterministic tests) |
| [`tests/e2e_test.sh`](tests/e2e_test.sh) | integration test (17 checks across all components) |
| [`agents/fractal-walker.md`](agents/fractal-walker.md) | walker sub-agent with init/continuation protocols |
| [`agents/fractal-judge.md`](agents/fractal-judge.md) | goal-alignment scorer |
| [`commands/fractal-decide.md`](commands/fractal-decide.md) | one FMC decision orchestration |
| [`commands/octopus.md`](commands/octopus.md) | outer goal-directed loop |
| [`commands/fractal-recall.md`](commands/fractal-recall.md) | Wigner-weighted recall of past decisions |
| [`commands/fractal-memory-show.md`](commands/fractal-memory-show.md) | dump memory bank state with per-memory stats |
| [`docs/THEORY.md`](docs/THEORY.md) | paper foundations + Atari→coding mapping rationale |
| [`docs/ALGORITHM.md`](docs/ALGORITHM.md) | math walkthrough with paper section references |
| [`docs/COMPONENTS.md`](docs/COMPONENTS.md) | file-by-file reference |
| [`docs/USAGE.md`](docs/USAGE.md) | invocation, configuration, cost, debugging |
| [`docs/EVOLUTION.md`](docs/EVOLUTION.md) | architectural evolution paths vs the Ralph loop, escalation-tier framing |

---

## Quick start

### 1. Verify the math is certified

```bash
cd <repo root>
python3 plugin/fractal-coding-loop/tests/test_fractal_math.py
```

Expected output ends with:
```
All FMC math tests passed — convergence certified.
```

This proves the algorithmic kernel (relativize, virtual reward, clone probability, bincount marginalization) is faithful to paper §2.2.3, §4.4, §4.6.

### 2. Smoke test the CLI state machine

The state machine works without any LLM calls — pure Python on synthetic walker JSONs. Useful for verifying installation:

```bash
mkdir -p /tmp/fmc_smoke && cd /tmp/fmc_smoke
LOOP=<plugin path>/scripts/fractal_loop.py

# init session
python3 $LOOP init --task "smoke test" --n 3 --m 3
# record id from output, then:

# step through (replace SESS with the id)
python3 $LOOP record --session $SESS --file <walker JSON file>
python3 $LOOP step --session $SESS --seed 42
python3 $LOOP apply-clones --session $SESS
python3 $LOOP decide --session $SESS
```

See [`docs/USAGE.md`](docs/USAGE.md) §"CLI smoke test" for a copy-pasteable full example.

### 3. Run a real decision (requires Claude Code)

```
/fractal-decide "implement add(a, b) function in src/math.py with a unit test"
```

This will spawn 3 walker sub-agents, each in its own git worktree, each implementing a different strategy. After ~3-5 minutes you get a comparison table and the winning commit ready to cherry-pick.

### 4. Run a goal-seeking session

```
/octopus "endpoint POST /login accepts credentials, returns valid JWT, tests/auth_test.py passes"
```

This calls `/fractal-decide` up to `K_MAX=10` times, applying each winning commit, until the judge says the goal is complete.

---

## Theoretical foundation

The paper to read first is the analysis of the original work in [`ANALISIS.md`](../../ANALISIS.md), then the personal post-experiment in [`analisisPost.md`](../../analisisPost.md), then the post-corpus reflection in [`analisisPost2.md`](../../analisisPost2.md).

The formal mathematical foundation including the convergence theorem (paper §4 + Del Moral 2004) is in [`work/02_deep_dives/01_cloning_mathematics.md`](../../work/02_deep_dives/01_cloning_mathematics.md) and the SMC equivalence in [`work/02_deep_dives/05_smc_particle_filter_view.md`](../../work/02_deep_dives/05_smc_particle_filter_view.md).

The Octopus / Badger structure that motivates this plugin's two-level design is in [`work/02_deep_dives/06_book2_badger_fractal_memory.md`](../../work/02_deep_dives/06_book2_badger_fractal_memory.md), grounded in Sergio's [2015-12 blog post](../../docs/bibliography/sources/blog_posts/2015-12_fractal_ai_collaboration.md).

The vision document that defined this plugin's scope is in [`docs/vision/fractal_coding_loop.md`](../../docs/vision/fractal_coding_loop.md).

For deep theory, see [`docs/THEORY.md`](docs/THEORY.md).
For algorithmic detail with paper section references, see [`docs/ALGORITHM.md`](docs/ALGORITHM.md).
For file-by-file reference, see [`docs/COMPONENTS.md`](docs/COMPONENTS.md).
For invocation and tuning, see [`docs/USAGE.md`](docs/USAGE.md).

---

## Status and limitations

**What is verified**:
- Math layer: 5/5 deterministic tests pass. Convergence to Gibbs distribution confirmed numerically (R=0.9/0.5/0.1 at M=10 → 100% delta on top walker).
- Integration layer: 17/17 e2e_test.sh checks pass — manifest validity, markdown frontmatter on all agents/commands, reward script accepts JSON file input + dead-walker-zero invariant, memory append/show/recall round-trip works, Wigner reward peaks near x=1.
- CLI state machine: init / record / step / apply-clones / decide all work end-to-end on synthetic data.
- ESS-adaptive cloning: triggers correctly when ESS > 0.7×N (verified in smoke test: 2.68 > 2.10 → cloning_skipped).
- Memory subsystem: append → markdown file with frontmatter, recall → Wigner-weighted top-K, show → JSON listing of all entries. All standalone-functional.

**What is NOT yet verified**:
- End-to-end on a real codebase. The plugin has not yet been invoked from Claude Code on a real repo with real sub-agent calls. The state machine and memory bank are certified but the LLM-driven phases (walker init, walker continuation, judge scoring) have not been integration-tested.
- Cost calibration. Estimates in [`docs/USAGE.md`](docs/USAGE.md) are derived from typical Claude Code sub-agent costs but not measured against actual runs.
- The `${CLAUDE_PLUGIN_ROOT}` path resolution depends on the plugin being registered with Claude Code. Until then, scripts must be invoked with absolute paths.
- Auto-persistence of decisions into memory bank from `/fractal-decide`. The `fractal_memory.py` subsystem is fully working but is currently invoked **manually** via `/fractal-recall` and `/fractal-memory-show`. Auto-append after each `/fractal-decide` is a one-line addition planned but not yet wired (see `commands/fractal-decide.md` Phase 8b).

**What is intentionally out of scope (Phase 0)**:
- Multi-octopus coordination (parallel goals on the same repo). Planned for Phase 4.
- Vision dashboard (live visualization of the cone). Planned for Phase 5.
- α=0 "Common Sense" mode for divergent design as a dedicated slash command. Currently α defaults to 1.0; can be set via `fractal_loop.py init --alpha 0` flag but no `/fractal-explore` command exposes it yet.
- Hooks for auto-trigger fractal mode on complex prompts. The `hooks/` directory is empty (Phase 1).

**Known caveats**:
- N=3 is statistically under-dimensioned per Del Moral CLT. Expected error ~58%. Raise to N=5 or N=10 for goals where decision quality matters more than cost.
- M=3 means each walker rollout is just 3 commits — short futures. For complex goals consider M=5.
- The "perturbation" at tick > 0 is **continuation** (one small step toward the goal), not random action like Atari. This is a deliberate departure for cost reasons; see [`docs/THEORY.md`](docs/THEORY.md) §4.

---

## Credits

Algorithm: **Sergio Hernández Cerezo** ([@EntropyFarmer](https://twitter.com/EntropyFarmer)) and **Guillem Duran Ballester** ([@Miau_DB](https://twitter.com/Miau_DB)), 2018-2020.

Adaptation to Claude Code: **Vlad Vrinceanu** (filcarspa@gmail.com), 2026.

Released under MIT license.

For the full intellectual debt and history (10 years of work, 2014-2026), see [`docs/bibliography/CORPUS.md`](../../docs/bibliography/CORPUS.md).
