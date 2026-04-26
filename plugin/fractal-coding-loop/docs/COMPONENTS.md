# COMPONENTS — file-by-file reference

> *This document describes every file in the plugin: what it does, why it exists, and how it works internally. References include explicit line numbers into the source.*

For higher-level rationale see [`THEORY.md`](THEORY.md). For algorithm walkthrough see [`ALGORITHM.md`](ALGORITHM.md). For invocation see [`USAGE.md`](USAGE.md).

---

## Tree view

```
plugin/fractal-coding-loop/
├── README.md                          ← entry point
├── INSTALL.md                         ← installation steps
├── .claude-plugin/
│   └── plugin.json                    ← plugin manifest
├── scripts/
│   ├── fractal_reward.py              ← math layer (relativize, composite reward, VR)
│   ├── fractal_loop.py                ← state machine (init/record/step/apply-clones/decide)
│   └── fractal_memory.py              ← Wigner-weighted memory bank (append/recall/show/prune)
├── tests/
│   ├── test_fractal_math.py           ← 5 deterministic math certification tests
│   └── e2e_test.sh                    ← 17 integration tests across all components
├── agents/
│   ├── fractal-walker.md              ← walker sub-agent (dual mode: init / continuation)
│   └── fractal-judge.md               ← goal-alignment scorer
├── commands/
│   ├── fractal-decide.md              ← one FMC decision (slash command)
│   ├── octopus.md                     ← outer goal-directed loop (slash command)
│   ├── fractal-recall.md              ← Wigner-weighted memory recall (slash command)
│   └── fractal-memory-show.md         ← memory bank state dump (slash command)
└── docs/
    ├── THEORY.md                      ← why the plugin exists
    ├── ALGORITHM.md                   ← how the algorithm works
    ├── COMPONENTS.md                  ← this file
    └── USAGE.md                       ← how to invoke and tune
```

---

## 1. `.claude-plugin/plugin.json`

**Purpose**: tells Claude Code that this directory is a plugin, declares its name, version, and metadata.

**File**: [`.claude-plugin/plugin.json`](../.claude-plugin/plugin.json) — 19 lines.

**Schema**:
```json
{
  "name": "fractal-coding-loop",
  "version": "0.1.0",
  "description": "Fractal Monte Carlo planning for coding decisions...",
  "author": {"name": "...", "email": "..."},
  "homepage": "https://github.com/uppifyagency/fractal-coding-loop",
  "license": "MIT",
  "keywords": ["fmc", "fractal-ai", "planning", "multi-agent", "decision-making"]
}
```

**Why each field matters**:
- `name`: must match the directory name `fractal-coding-loop`. Claude Code keys plugins by name.
- `version`: semver. `0.1.0` indicates Phase 0 PoC. Bump on any user-visible change.
- `keywords`: helps users find this plugin in catalogues.

**Modification policy**: rarely changed. Bump `version` when making breaking changes to the slash command interface, the script CLI, or the agent JSON schemas.

---

## 2. `scripts/fractal_reward.py`

**Purpose**: pure-math layer. No state, no I/O dependencies. All functions are deterministic given inputs.

**File**: [`scripts/fractal_reward.py`](../scripts/fractal_reward.py) — ~280 lines.

**Public API**:

| Function | Lines | Purpose |
|---|---|---|
| `relativize(values)` | 115-134 | paper §2.2.3 — z-score + exp/log reshape to strictly positive, order-preserving values |
| `r_alive(walker)` | 46-48 | Hard constraint: 0 if walker fails to compile, 1 otherwise |
| `r_tests(walker)` | 51-59 | Test pass ratio in [0, 1]; defaults to 1 if no tests |
| `r_lint(walker)` | 62-68 | Lint cleanliness, smooth decay `1/(1+log(1+warnings))` |
| `r_diff(walker)` | 71-78 | Diff size penalty: `max(0, 1 - lines/200)` |
| `r_goal(walker)` | 81-83 | Goal alignment from `fractal-judge` sub-agent (default 0.7) |
| `composite_reward(walker)` | 86-108 | Multiplicative composition: `R_alive × R_tests × (1+R_lint) × (1+R_diff) × (1+R_goal)` (paper §2.2.2) |
| `file_overlap_distance(w1, w2)` | 137-150 | Jaccard distance on `files_changed` sets (1 - intersection/union) |
| `lines_distance(w1, w2)` | 153-157 | Absolute difference of total lines changed (used as distance tiebreaker) |
| `virtual_reward(walkers, rewards, alpha, beta)` | 160-194 | paper §4.4 — `VR_i = relativize(R)[i]^α × relativize(D)[i]^β` with stochastic O(N) distance |

**Why this file is separate from `fractal_loop.py`**: pure functions, no state, easy to unit-test. `fractal_loop.py` imports it via `import fractal_reward as fr`.

**CLI**: `python3 fractal_reward.py --walker-jsons '[...]'` runs `composite_reward` + `virtual_reward` and returns full breakdown. Useful for debugging walker JSONs without a full session.

**Critical design choice — `(1 + x)` for soft constraints**: see [`ALGORITHM.md`](ALGORITHM.md) §3.3. Hard constraints multiply directly (can zero out R). Soft contributions use `(1 + x)` to never zero out — they only modulate the magnitude.

**Critical design choice — stochastic distance**: see [`ALGORITHM.md`](ALGORITHM.md) §4.3. O(N) instead of O(N²). Each walker compares to ONE random partner. Faithful to paper §4.5.

**Test coverage**: [`tests/test_fractal_math.py`](../tests/test_fractal_math.py) tests 1, 2, 3, 4, 5. Test 1 specifically targets `relativize`. Tests 3-5 use `virtual_reward` indirectly via the simulation harness.

---

## 3. `scripts/fractal_loop.py`

**Purpose**: stateful orchestration of the M-tick FMC loop. Handles session lifecycle, walker tracking, ESS-adaptive cloning, decision marginalization.

**File**: [`scripts/fractal_loop.py`](../scripts/fractal_loop.py) — ~370 lines.

**Why a state machine and not a single function**: the M-tick loop is interleaved with sub-agent invocations and git commands that are run by the **orchestrator** (the slash command), not by Python. Each `step` Python invocation must persist its state to disk so the orchestrator can pick it up after running git/sub-agent operations.

**Persistence**: state lives in `.fractal/sessions/<session_id>/state.json` of the cwd. Each `fractal_loop.py` invocation reads and writes that file. See `_session_path`, `_load`, `_save` (lines 58-77).

**State schema** (excerpt from `cmd_init`, lines 81-118):

```json
{
  "session_id": "20260426_213340_97694",
  "task": "...",
  "goal": "...",
  "n": 3, "m": 3,
  "alpha": 1.0, "beta": 1.0,
  "ess_threshold": 0.7,
  "tick": 0,
  "walkers": [
    {
      "idx": 0,
      "init_action_label": "extract-module",
      "init_action_desc": "...",
      "init_commit_sha": "abc123...",
      "init_commit_message": "FMC walker [init]: extract-module — ...",
      "alive": true,
      "current_branch": "fmc-walker-0",
      "current_path": "/.../worktree-0",
      "current_head": "def456...",
      "history": [
        {"tick": 0, "walker_json": {...}, "R": 6.63, "breakdown": {...}}
      ]
    },
    ...
  ],
  "decisions": [
    {
      "tick": 0,
      "vrs": [0.91, 0.36, 0.66],
      "rewards": [6.63, 2.31, 0.37],
      "ess": 2.68,
      "ess_threshold_abs": 2.10,
      "cloning_skipped": true,
      "clone_plan": []
    }
  ]
}
```

### CLI subcommands

| Command | Lines | Purpose |
|---|---|---|
| `init --task --goal --n --m --alpha --beta --ess-threshold` | 83-118 | Create new session, return `session_id` |
| `record --session --file <walker JSONs>` | 121-167 | Record walker outputs for current tick. At tick 0, also captures `init_commit_sha` |
| `step --session [--seed]` | 170-260 | Compute virtual rewards, ESS, generate clone_plan. Skip if `ESS > threshold × N` |
| `apply-clones --session` | 263-296 | Mirror the clone_plan into state (orchestrator runs `git reset --hard` separately) |
| `decide --session` | 299-349 | Final argmax bincount, return `winner_init_commit_sha` |
| `status --session` | 352-353 | Dump full state JSON (debugging) |

### Orchestrator interaction protocol

The slash command calls `fractal_loop.py` between phases of its own work:

1. **`init`** → Python creates session
2. Orchestrator spawns N walker sub-agents in parallel (init mode), collects JSONs
3. **`record`** → Python ingests walker JSONs for tick 0
4. For tick 1..M-1:
   1. **`step`** → Python computes VR/ESS/clone_plan
   2. Orchestrator runs `git reset --hard <src_head>` for each entry in clone_plan
   3. **`apply-clones`** → Python updates state to reflect clones, advances tick
   4. Orchestrator spawns N walker sub-agents (continuation mode), collects JSONs
   5. **`record`** → Python ingests walker JSONs for tick t
5. Final: **`step`** + **`apply-clones`** + **`decide`** → Python returns `winner_init_commit_sha`

### Critical design choices

**ESS-adaptive cloning** (lines 175-188): if `ESS > 0.7 × N`, set `clone_plan = []` and `cloning_skipped = True`. The orchestrator skips the git reset and continuation phases for that tick. **This is what makes Phase 0 economically viable**.

**`init_commit_sha` tracking** (lines 138-143, 254-257, 285-289): the SHA of the walker's first commit is captured at tick 0 and propagates through clones. At decision time, this is what gets cherry-picked. **Without this, the algorithm can't tell the orchestrator what to apply to main.**

**OLD-snapshot in apply-clones** (lines 271-276): when computing all clones for a tick, the partner index `src_idx` refers to the **pre-clone** state. We snapshot all walkers' labels and SHAs *before* applying any clones. This is paper §4.4 fidelity: all clone decisions use the SAME pre-tick swarm reference, never a partial-update.

**Dead walker handling** (lines 195-209): walkers with `compile_ok=False` get `alive=False` and `R=0` in scoring. In the clone phase, dead walkers always clone (P=1) from a random alive partner. Verified by test 5.

---

## 4a. `scripts/fractal_memory.py`

**Purpose**: persistent Wigner-weighted memory bank for past `/fractal-decide` episodes. Implements the **Dataset as Fractal Memory** concept from the [2020 Fractal Slide doc](../../../2020%20Fractal%20Slide.md): each memory has a `loss` value, and recall samples by `R(x) = (π/2) x exp(-π/4 x²)` debiased for visit count.

**File**: [`scripts/fractal_memory.py`](../scripts/fractal_memory.py) — ~370 lines.

**Storage layout**: each memory is a markdown file in `<project>/.fractal/memory/<timestamp>_<slug>.md` with YAML frontmatter:

```yaml
---
task: "refactor auth middleware"
timestamp: "2026-04-26T21-15-00"
winner_idx: 1
winner_label: "extract-module"
confidence: 0.67
loss: 0.33                      # = 1 - confidence at write time
visits: 1                       # incremented on each recall (with --mark-visited)
n_walkers: 3
approaches: ["in-place", "extract-module", "test-first"]
---

# refactor auth middleware
**Winner**: walker 1 — extract-module
**Confidence**: 67.0%
## Walkers
### 0: in-place
- Files changed: 1
- Lines: +30/-15
- Tests: 12/12
- Lint warnings: 0
...
```

**Public API**:

| Function | Lines | Purpose |
|---|---|---|
| `slugify(text, max_len)` | 56-61 | Filesystem-safe slug from task description |
| `find_project_root(cwd)` | 64-72 | Walk up to find `.git`, fallback to cwd |
| `write_memory_file(...)` | 78-153 | Create a new `<timestamp>_<slug>.md` with frontmatter + body |
| `parse_frontmatter(path)` | 156-176 | Read YAML-ish frontmatter into a dict (uses `json.loads` per value) |
| `update_frontmatter(path, updates)` | 179-190 | In-place update of frontmatter (used to increment visits) |
| `wigner_reward(loss, avg_loss)` | 197-204 | The formula: `(π/2) x exp(-π/4 x²)` with `x = loss/avg_loss` |
| `memory_weight(loss, avg_loss, visits)` | 207-210 | Wigner debiased: `wigner_reward / (1 + log(1+visits))` |

**CLI subcommands**:

| Command | Lines | Purpose |
|---|---|---|
| `append --task --winner-idx --walkers-json --confidence [--reward-json]` | 217-229 | Save a new decision episode |
| `recall [--query] [--top-k N] [--mark-visited]` | 232-292 | Wigner-weighted top-K recall with optional keyword filter |
| `show` | 295-316 | List all memories with stats |
| `prune [--min-visits N]` | 319-334 | Delete memories with `loss < 0.05` AND `visits > N` (already-learned, retire) |

### Why Wigner reward

The Wigner semicircle distribution from random matrix theory peaks at `x = 1` (= average loss):
- Memories with **low loss** (already mastered) → low weight, deprioritize
- Memories with **medium loss** (active learning zone) → high weight, surface
- Memories with **high loss** (still too difficult) → low weight, postpone

This is **automatic curriculum learning** as the slide doc proposes. Verified by `e2e_test.sh` Test 5: `wigner_reward(1.0)` ≥ 0.95 × `wigner_reward(x)` for x ∈ {0.5, 1.5, 3.0}.

### Integration status

The subsystem is **standalone-functional**. It is invoked via the two slash commands `/fractal-recall` and `/fractal-memory-show`. It is **NOT** auto-called by `/fractal-decide` after a decision yet — auto-persistence is one shell call in Phase 8b of the orchestrator, not yet wired by default. Until then, the user can manually persist:

```bash
python3 ~/.claude/plugins/fractal-coding-loop/scripts/fractal_memory.py append \
    --task "<the task>" --winner-idx <idx> \
    --walkers-json '[<walker JSONs>]' --confidence <0..1>
```

---

## 4b. `tests/test_fractal_math.py`

**Purpose**: certify the math layer numerically. Run before any commit touching `fractal_reward.py` or `fractal_loop.py`.

**File**: [`tests/test_fractal_math.py`](../tests/test_fractal_math.py) — ~190 lines.

**Test list**:

| Test | Lines | What it certifies |
|---|---|---|
| `test_relativize_properties` | 76-92 | paper §2.2.3 — strict positivity, order preservation, constant input → ones, extreme range OK |
| `test_clone_prob_formula_edge_cases` | 96-114 | paper §4.4 — 5 cases: VR_self=0, equal VRs, worse partner, 50% better partner, 9× better (capped at 1) |
| `test_convergence_high_R_dominates` | 117-127 | paper §4.3 — R=(0.9, 0.5, 0.1) at M=10, 200 runs → walker_0 wins 100% (Gibbs delta) |
| `test_close_top_two_share_distribution` | 130-145 | R=(1.0, 0.95, 0.0) at M=8, 400 runs → 59%/41%/0% (close-top split, Gibbs distribution) |
| `test_dead_walker_rare_in_winner` | 148-162 | R=(0.9, 0.5, 0.0) at M=20, 300 runs → walker_2 (R=0) wins ~0.3% (squeezed out by clone-revive cycle) |

**Helper function**: `simulate_cloning_only(rewards, M, n_runs, seed_base, alpha, beta)` (lines 30-71). Runs M ticks of pure cloning over walkers with **frozen R values** and identical empty file sets (so distances are uniform after relativize). This isolates the test to R-driven dynamics, removing distance noise.

**Why frozen states**: in real FMC, walker states change between ticks (perturbation). For testing, we freeze states to isolate the convergence properties of cloning alone. The test still demonstrates faithful Gibbs convergence.

**Run with**:
```bash
python3 tests/test_fractal_math.py
```

Output ends with:
```
============================================================
All FMC math tests passed — convergence certified.
============================================================
```

If any test fails: revert to last green commit before using the plugin.

---

## 4c. `tests/e2e_test.sh`

**Purpose**: integration tests across all components — manifest validity, markdown frontmatter, reward script behavior, memory round-trip, Wigner formula shape. Complementary to `test_fractal_math.py` (which is pure math, no shell or filesystem).

**File**: [`tests/e2e_test.sh`](../tests/e2e_test.sh) — 161 lines bash.

**Test groups** (17 checks total):

| Group | What it tests |
|---|---|
| **TEST 1: plugin manifest** | `plugin.json` is valid JSON |
| **TEST 2: markdown frontmatter** | All 6 markdown files in `commands/` and `agents/` start with `---` (YAML frontmatter) |
| **TEST 3: fractal_reward.py** | Script accepts JSON file input, produces `winner_idx`, confidence in [0,1], 3-walker output, dead walker (`compile_ok=false`) gets `R=0` (multiplicative invariant) |
| **TEST 4: fractal_memory.py round-trip** | `append` creates a memory, `show` lists it, second `append` brings count to 2, `recall` returns top-K Wigner-weighted |
| **TEST 5: Wigner reward shape** | `wigner_reward(1.0) >= 0.95 × max(wigner_reward(x))` for x ∈ {0.5, 1.5, 3.0} — i.e., the curve peaks near x=1 |

**Run with**:
```bash
bash plugin/fractal-coding-loop/tests/e2e_test.sh
```

Output ends with:
```
=== SUMMARY ===
PASSED: 17
FAILED: 0
OK ✓ — all components functional
```

This is a smoke test for the full plugin surface. Run it after any change that touches multiple components, or after `git pull` of a fresh clone.

**Cleanup**: the test creates a temp directory `/tmp/fractal-e2e-XXXXXX` and removes it on exit (via `trap "rm -rf $TMP_DIR" EXIT`). Safe to run repeatedly.

---

## 5. `agents/fractal-walker.md`

**Purpose**: defines the walker sub-agent. This file is a **prompt** read by Claude Code when an `Agent(subagent_type="fractal-walker", ...)` call is made.

**File**: [`agents/fractal-walker.md`](../agents/fractal-walker.md) — ~220 lines.

**Frontmatter** (lines 1-12):
```yaml
name: fractal-walker
description: Walker sub-agent in a Fractal Monte Carlo decision...
tools: [Bash, Read, Edit, Write, Grep, Glob]
model: sonnet
```

**Why `model: sonnet`**: walker work involves real code editing, test running, syntax checking. Sonnet handles this well at moderate cost. **Could be lowered to Haiku for continuation mode** (tick > 0) to save cost — see future optimization.

**Why these tools**: Bash for `git`/test/lint, Read/Edit/Write for code modification, Grep/Glob for codebase exploration. Notably **no Task** — walkers don't spawn other agents.

### The dual mode protocol

The walker reads the prompt and selects ONE of two modes:

#### MODE: init (tick 0)

```
MODE: init

[approach_description: ...]

Goal: ...
Task: ...

Follow the fractal-walker INIT protocol below.
```

Steps (lines 64-114):
1. Snapshot init_branch + init_head
2. Comprende il codice (Grep, Read; no modifications)
3. Implementa la strategia (Edit, Write)
4. Run tests, lint, syntax check
5. **Commit ONCE** (this is the `init_commit_sha`)
6. Output JSON with `init_commit_sha`, `init_commit_message`, all reward inputs

**Critical**: at tick 0, the walker MUST make a single commit. Splitting into multiple commits would mean the `init_commit_sha` doesn't represent the full strategy — only its first piece.

#### MODE: continuation (tick > 0)

```
MODE: continuation

[strategy_label: ...]
[strategy_description: ...]

Goal: ...
Recent history in this worktree:
<git log --oneline -5>

Follow the fractal-walker CONTINUATION protocol below.
```

Steps (lines 117-152):
1. Read git log to understand current state
2. **One small step** toward goal, preserving strategy
3. Constraint: max 30 lines diff, single logical change
4. Run tests, lint, syntax check
5. Commit
6. Output JSON without `init_commit_sha` (already in state)

**Critical**: at continuation, the walker must NOT redesign or change strategy. It's a probe of "what does this strategy look like one step further?" — variance comes from LLM stochasticity, not from architectural choices.

### Output JSON schemas

INIT (lines 100-114):
```json
{
  "mode": "init",
  "approach_label": "<short label>",
  "approach_description": "<from input>",
  "worktree_path": "...", "worktree_branch": "...",
  "init_head": "<parent of init_commit>",
  "walker_head": "<= init_commit_sha at tick 0>",
  "init_commit_sha": "<SHA>",
  "init_commit_message": "<git log -1 --format=%s>",
  "files_changed": [...], "lines_added": int, "lines_deleted": int,
  "tests_run": bool, "tests_passed": int|null, "tests_total": int|null,
  "lint_warnings": int|-1, "compile_ok": bool,
  "summary": "...", "notes": "..."
}
```

CONTINUATION (lines 138-152): same shape but with `mode: "continuation"`, no `init_commit_sha` (it's preserved by the state machine via the cloning mechanism).

### Why this protocol is in markdown not Python

Sub-agent definitions in Claude Code are markdown prompts. They are read by the LLM at invocation time. Python state-machine code can't impose this protocol — it's the **prompt that disciplines the walker**.

Modifications to walker behavior happen by editing this file. No code change needed.

---

## 6. `agents/fractal-judge.md`

**Purpose**: scores goal-alignment. Used in two places:
1. Inside `/fractal-decide` to compute `R_goal` per walker
2. Inside `/octopus` to check if the overall goal is reached after each iteration's cherry-pick

**File**: [`agents/fractal-judge.md`](../agents/fractal-judge.md) — ~90 lines.

**Frontmatter**:
```yaml
name: fractal-judge
description: Evaluates goal-alignment of a walker's output...
tools: [Bash, Read, Grep]
model: sonnet
```

**Tools**: read-only — Bash for `git diff`, Read for files, Grep for searching. **No Edit/Write** — judges don't modify code.

### Input schema

```json
{
  "task": "<original task description>",
  "approach_label": "<walker's approach>",
  "worktree_path": "<path>",
  "files_changed": [...],
  "diff_summary": "<git diff --stat>"
}
```

### Procedure

1. Read full diff: `git diff HEAD~..HEAD`
2. Compare against task. Evaluate completeness, correctness, scope, approach fidelity, hidden side effects.
3. Return JSON with `goal_score ∈ [0, 1]`, plus rationale.

### Output schema

```json
{
  "goal_score": 0.85,
  "completeness": 0.9,
  "correctness": 0.95,
  "scope_purity": 0.7,
  "approach_fidelity": 1.0,
  "hidden_effects_count": 0,
  "rationale": "<2-3 sentences>",
  "red_flags": ["..."]
}
```

### Calibration scale (lines 70-79)

- 1.0 = fully complete and faithful
- 0.85 = good with minor issues
- 0.7 = partially implemented (50-70%)
- 0.5 = different interpretation, 30-50%
- 0.3 = largely missed but something relevant
- 0.0 = no relevance

### When called from `/octopus`

The judge is invoked after each cherry-pick to score the **overall goal completion** on the main branch. The prompt is slightly different from the per-walker case (lines mentioned in `commands/octopus.md` Phase 1e). Key differences:
- "task" is the goal G, not a walker's strategy
- The judge looks at `git log $START_HEAD..$NEW_HEAD` (cumulative progress)
- Threshold for "reached" is `goal_score ≥ 0.95`

---

## 7. `commands/fractal-decide.md`

**Purpose**: defines the `/fractal-decide` slash command. Orchestrates ONE FMC decision (N walkers × M ticks) and returns a winning init_commit_sha for cherry-pick.

**File**: [`commands/fractal-decide.md`](../commands/fractal-decide.md) — ~250 lines.

**Frontmatter**:
```yaml
description: ONE FMC decision — N walkers × M ticks of perturbation+cloning toward a goal...
argument-hint: "[goal description]"
allowed-tools: [Task, Bash, Read, Edit, Write, Grep, Glob, TodoWrite]
```

**Note**: `Task` tool is required — this command spawns sub-agents.

### Phase structure

| Phase | What | Why |
|---|---|---|
| 0 — Preconditions | Check git status, identify main branch, detect toolchain | Avoid corrupting state mid-decision |
| 1 — Generate strategies | Main agent proposes N=3 distinct strategies | Initial diversity is critical (paper §4.5) |
| 2 — Init session | `python3 fractal_loop.py init --task ...` | Create state file |
| 3 — Spawn N walkers (INIT) | Task tool × N in parallel, `isolation: worktree` | Walker isolation = paper's swarm independence |
| 4 — Record tick 0 | `python3 fractal_loop.py record --tick 0` | Capture init_commit_sha + first scoring |
| 5 — M-1 ticks of step+clone+continuation | Loop: step → git reset → apply-clones → spawn continuation walkers → record | The actual M-tick FMC dynamics |
| 6 — Final step + decide | Last step + apply-clones + decide → JSON with winner_init_commit_sha | paper §4.6 marginalization |
| 7 — Show comparison table to user | Markdown table with R, VR, votes, confidence | Transparency for user review |
| 8 — Cherry-pick OR return | If standalone: ask user; if from /octopus: emit JSON | Differentiates standalone vs octopus modes |
| 9 — Cleanup | Remove non-winner worktrees and branches | Don't leak temporary state |

### Why the M-tick loop is in the slash command, not in Python

Each tick of the loop involves:
- Calling `Task` tool (Python can't do this — only Claude Code can)
- Running `git reset --hard` (Python could but it's clearer in shell)
- Reading file system

The orchestrator (Claude Code) is the only entity that can do all of these. `fractal_loop.py` provides the math + state; the slash command provides the orchestration.

### Why parallel Task invocations within one tick

Per the Claude Code docs: multiple `Task` calls in a single message run in parallel. Critical for tick performance — walker work is independent per walker.

### Critical safety constraints

1. **Confidence < 0.50 warning**: surface to user as "Decisione rischiosa, considera review manuale" before cherry-pick.
2. **Cherry-pick only the winner's init_commit, not the full branch**: see [`ALGORITHM.md`](ALGORITHM.md) §7.3.
3. **Cleanup non-winner worktrees**: don't leak `git worktree` entries. Each leftover worktree consumes inode + branch namespace.

---

## 8. `commands/octopus.md`

**Purpose**: defines the `/octopus` slash command. Outer goal-directed loop that calls `/fractal-decide` repeatedly until goal is reached or budget exhausted.

**File**: [`commands/octopus.md`](../commands/octopus.md) — ~210 lines.

**Frontmatter**:
```yaml
description: Octopus loop — repeated FMC decisions toward a goal G until completion...
argument-hint: "[goal description]"
allowed-tools: [Task, Bash, Read, Edit, Write, Grep, Glob, TodoWrite]
```

### Phase structure

| Phase | What | Why |
|---|---|---|
| 0 — Setup | Init log file, define K_MAX, THRESHOLD | Audit trail and termination criteria |
| 1 — Loop main | For iter in 1..K_MAX: run fractal-decide internally → cherry-pick → judge | The actual outer loop |
| 1b — Confidence gate | If conf < 0.50, ASK USER (continue/stop/inspect) | Don't pollute main with noisy decisions |
| 1c — Cherry-pick | `git cherry-pick winner_init_commit_sha` | Apply the swarm's decision to trunk |
| 1d — Cleanup | Remove non-winner worktrees | Don't leak |
| 1e — Goal check (judge) | Sub-agent fractal-judge against current main HEAD | Goal completion = "is it done now?" |
| 1f — Termination check | If goal_score ≥ THRESHOLD or iter ≥ K_MAX, break | Loop terminator |
| 2 — Reporting | Print summary table, log final state | User-facing transparency |
| 3 — Rollback option | If user wants: `git reset --hard $START_HEAD` | Escape hatch if octopus made bad calls |

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `OCTOPUS_K_MAX` | 10 | Max iterations |
| `OCTOPUS_THRESHOLD` | 0.95 | Goal completion threshold |
| `FMC_N` | 3 | Walkers per /fractal-decide call (read by inner command) |
| `FMC_M` | 3 | Ticks per /fractal-decide call |

### Critical safety stops

1. **Cherry-pick conflict** → STOP and ask user. Walker base diverged from main; auto-resolve is unsafe.
2. **Confidence < 0.50** → ASK user. Three rejections in a row would also stop.
3. **K_MAX exhausted with goal_score < THRESHOLD** → STOP and report partial progress. User decides whether to continue or rollback.

### Why goal_score uses the same fractal-judge

Reusing the judge sub-agent for both per-walker scoring and goal-completion checking is intentional:

- Same calibration (the 0..1 scale is consistent)
- Same evaluator (no model mismatch)
- Same prompt template (extended at the boundary)

Operationally: in `/fractal-decide`, the judge sees a single walker's worktree diff and asks "did this walker do its task?" In `/octopus`, the judge sees `git log $START_HEAD..$NEW_HEAD` and asks "is the goal complete now?"

---

## 8a. `commands/fractal-recall.md`

**Purpose**: surface the Wigner-weighted recall of past `/fractal-decide` episodes. Wraps `fractal_memory.py recall`.

**File**: [`commands/fractal-recall.md`](../commands/fractal-recall.md) — ~24 lines.

**Frontmatter**:
```yaml
description: Recall the most relevant past Fractal Coding decisions (Wigner-weighted memory bank)
argument-hint: "[query keyword]"
allowed-tools: [Bash]
```

**Behavior**: invokes:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_memory.py recall \
    --query "$ARGUMENTS" --top-k 5 --mark-visited
```

The `--mark-visited` flag increments the `visits` counter on the recalled memories (so frequently-recalled ones get debiased down per the Wigner-weight formula).

**Output rendering**: the slash command instructs the main agent to render the JSON response as a markdown table with columns:

| weight | task | winner | confidence | loss | visits |
|---|---|---|---|---|---|

Plus a one-line interpretation: "memorie più rilevanti = quelle con loss vicino a media (zona di apprendimento attiva)."

**Use case**: before starting a new `/fractal-decide`, run `/fractal-recall <similar task>` to see what worked before. Currently this is **manual** — the recall is not yet auto-injected into Phase 1 strategy generation. That integration is Phase 2 of the vision document.

---

## 8b. `commands/fractal-memory-show.md`

**Purpose**: dump the full memory bank state for inspection.

**File**: [`commands/fractal-memory-show.md`](../commands/fractal-memory-show.md) — ~20 lines.

**Frontmatter**:
```yaml
description: Mostra tutte le memorie nella Fractal Memory bank con i loro stats
allowed-tools: [Bash, Read]
```

**Behavior**: invokes:
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/fractal_memory.py show
```

**Output rendering**: the slash command instructs the main agent to render as a table sorted by timestamp DESC, plus:
- Global stats: total count, average loss, average visits
- Distribution: how many in `loss < 0.2` ("learned"), `0.2-0.5` ("learning zone"), `> 0.5` ("still hard")
- Suggestion: if any memories satisfy `loss < 0.05 AND visits > 10`, suggest running `prune` to clean up.

**Use case**: ad-hoc inspection. Useful before pruning or to verify that `/fractal-decide` runs are accumulating correctly (once auto-persistence is wired).

---

## 9. `docs/` directory

Four reference documents:

| File | What | When to read |
|---|---|---|
| [`THEORY.md`](THEORY.md) | Paper foundations, Atari→coding mapping rationale, faithful vs adapted choices | First. Once. Builds intuition. |
| [`ALGORITHM.md`](ALGORITHM.md) | Math walkthrough with paper section refs, code line numbers, worked examples | When debugging or modifying the math |
| [`COMPONENTS.md`](COMPONENTS.md) | This file. Per-file purpose and structure | When modifying or extending the plugin |
| [`USAGE.md`](USAGE.md) | Invocation, configuration, cost estimates, debugging | When using the plugin |

---

## 10. What's missing (intentional gaps in Phase 0)

| Missing | Why | Where it's planned |
|---|---|---|
| `scripts/fractal_memory.py` | Wigner-weighted recall of past decisions | Phase 2 — vision §V3 |
| `hooks/` content | Auto-trigger fractal mode on complex prompts | Phase 1 — vision §Phase 1 |
| `tests/test_fractal_loop_cli.py` | End-to-end CLI integration test | Phase 0 polish — easy add |
| `tests/test_orchestration.sh` | Shell-driven E2E with mock Claude responses | Phase 1 |
| Vision dashboard | Live visualization of the cone | Phase 5 — vision §Phase 5 |
| α=0 slash command | `/fractal-explore` for divergent design | Phase 3 |

The vision document [`docs/vision/fractal_coding_loop.md`](../../../docs/vision/fractal_coding_loop.md) §V7 has the full roadmap.
