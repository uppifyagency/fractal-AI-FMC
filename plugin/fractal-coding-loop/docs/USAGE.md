# USAGE — invocation, configuration, cost, debugging

> *This document is the operator's manual. How to invoke the commands, how to tune parameters, what each invocation costs, and how to debug when things go wrong.*

For higher-level rationale see [`THEORY.md`](THEORY.md). For algorithmic detail see [`ALGORITHM.md`](ALGORITHM.md). For per-file reference see [`COMPONENTS.md`](COMPONENTS.md).

---

## 1. The four slash commands

### `/fractal-decide [goal]`

**One FMC decision**. Spawns N walkers, runs M ticks, returns one commit ready to cherry-pick.

Example:
```
/fractal-decide add a function `add(a, b)` to src/math.py with a unit test in tests/test_math.py
```

What you'll see:
1. Phase 1 — main agent generates 3 strategies (in-place, extract-module, test-first or similar)
2. Phase 3 — three walkers spawn in parallel (3-5 min)
3. Phase 5 — for each tick: ESS check, optional cloning, then continuation walkers (3-5 min × M-1 ticks)
4. Phase 6-7 — comparison table:
   ```
   | # | Approach           | Final R | VR (last) | Alive | Votes |
   |---|--------------------|---------|-----------|-------|-------|
   | 1 | in-place-minimal   | 4.32    | 0.94      | ✓     | 2     |
   | 2 | extract-module     | 2.81    | 0.41      | ✓     | 1     |
   | 3 | test-first         | 0.00    | 0.00      | ✗     | 0     |

   Winner: in-place-minimal (confidence 67%)
   First commit: a1b2c3d  "FMC walker [init]: in-place-minimal — ..."
   ```
5. Phase 8 — asks: "Apply winner to <main_branch>? (yes / no / inspect)"

Use this for: standalone "should I do A or B?" planning decisions, when you want to see and approve each step.

### `/octopus [goal]`

**Goal-directed session**. Calls `/fractal-decide` repeatedly until the judge says the goal is reached.

Example:
```
/octopus implement POST /login endpoint with JWT, tests/auth_test.py passes
```

What you'll see:
1. Phase 0 — setup, log file, K_MAX, THRESHOLD
2. Phase 1 (loop, K times):
   - Run /fractal-decide internally
   - Cherry-pick winner
   - Judge: "is goal complete now? (0..1)"
   - If ≥ 0.95, break
3. Phase 2 — final report:
   ```
   ╔══════════════════════════════════════════════════════════════╗
   ║  OCTOPUS LOOP — final report                                 ║
   ║  Goal:           implement POST /login...                    ║
   ║  Iterations:     3 / 10                                      ║
   ║  Final score:    0.97 (threshold 0.95)                       ║
   ║  Commits:        3 (from <START_HEAD>..<END_HEAD>)            ║
   ║  Status:         REACHED ✓                                    ║
   ╚══════════════════════════════════════════════════════════════╝
   ```

Use this for: feature implementation, multi-step refactors, anything where you have a clear acceptance criterion and want autonomous progress.

### `/fractal-recall [query]`

**Wigner-weighted recall** of past decision episodes from the memory bank.

Example:
```
/fractal-recall auth middleware
```

What you'll see:
- Markdown table of top-5 past decisions whose task contains "auth middleware"
- Columns: `weight | task | winner | confidence | loss | visits`
- One-line interpretation of which memories are in the "active learning zone"

The recall surfaces memories with `loss ≈ avg_loss` first (Wigner peak at x=1), debiased by visit count. **Side effect**: visited memories increment their `visits` counter — repeated recall gradually deprioritizes the same entries.

Use this for: before starting a new `/fractal-decide`, check if you've decided on something similar before. Currently the recall is **manual** — the result is informational, not auto-injected into Phase 1 strategy generation. (See [`COMPONENTS.md`](COMPONENTS.md) §8a for integration plans.)

### `/fractal-memory-show`

**Dump the memory bank** state with per-memory stats.

Example:
```
/fractal-memory-show
```

What you'll see:
- Table of all memories sorted by timestamp DESC
- Global stats: total count, average loss, average visits
- Distribution: how many memories are "learned" (loss < 0.2), "in learning zone" (0.2-0.5), "still hard" (> 0.5)
- Suggestion to prune if any memories satisfy `loss < 0.05 AND visits > 10`

Use this for: ad-hoc inspection of accumulated state, deciding whether to prune.

---

## 2. Configuration

### Default values (Phase 0)

| Parameter | Default | Where set |
|---|---|---|
| N (walkers) | 3 | `fractal_loop.py init --n 3` (called by slash command) |
| M (ticks) | 3 | `fractal_loop.py init --m 3` |
| α (exploitation) | 1.0 | `fractal_loop.py init --alpha 1.0` |
| β (exploration) | 1.0 | `fractal_loop.py init --beta 1.0` |
| ESS threshold | 0.7 | `fractal_loop.py init --ess-threshold 0.7` |
| K_MAX (octopus iter) | 10 | env `OCTOPUS_K_MAX=10` |
| THRESHOLD (octopus goal) | 0.95 | env `OCTOPUS_THRESHOLD=0.95` |

### Overriding per-invocation

For one-off changes, set environment variables before invoking:

```bash
OCTOPUS_K_MAX=15 OCTOPUS_THRESHOLD=0.90 # then in Claude Code:
/octopus <goal>
```

### Permanent override

Edit the slash command files directly:
- [`commands/fractal-decide.md`](../commands/fractal-decide.md) Phase 2 to change `--n` and `--m` defaults
- [`commands/octopus.md`](../commands/octopus.md) Phase 0 to change `K_MAX` and `THRESHOLD`

---

## 3. Tuning N and M

This is the most important configuration trade-off.

### N (number of walkers)

The Del Moral 2004 theorem (see [`THEORY.md`](THEORY.md) §5) gives error scaling `O(1/√N)`:

| N | Expected 95% CI on action selection | Total walker calls per `/fractal-decide` (with M=3) | Approx wall time |
|---|---|---|---|
| 3 (default) | ~58% | 9 | 3-5 min |
| 5 | ~45% | 15 | 5-8 min |
| 10 | ~32% | 30 | 10-15 min |
| 30 (Atari paper regime) | ~18% | 90 | 30-45 min |
| 300 | ~6% | 900 | infeasible for coding |

**Guidance**:
- N=3 is fine for **low-stakes, exploratory** decisions (small features, refactors). Accept ~50% noise.
- N=5 is the **sweet spot** for typical features. ~45% noise, manageable cost.
- N=10 for **architecturally important** decisions where wrong choice is costly.
- N=30+ only if you're trying to match Atari paper conditions for benchmarking.

### M (ticks per decision)

M is the depth of the rollout cone — how many commits each walker projects forward.

| M | What it means | Use for |
|---|---|---|
| 1 | Single-shot vote on initial commits | Degenerate; not really FMC |
| 2 | Walker makes init + 1 continuation | Minimal multi-tick |
| 3 (default) | init + 2 continuations | Reasonable for most goals |
| 5 | Long rollouts | Goals that need to "see" several steps ahead |
| 10+ | Very long rollouts | Mostly increases cost, marginal value |

**Trade-off**: M increases cost linearly (each tick is N sub-agent calls). It also increases the **mixing time** — how many ticks of cloning are needed before the walker distribution converges. For the Gibbs convergence to bite, M ≥ 3 is generally enough; deeper just adds cost.

### α and β (exploitation / exploration balance)

| Setting | Behavior | Use for |
|---|---|---|
| α=1, β=1 (default) | Standard FMC | Almost everything |
| α=2, β=1 | Greedy reward | When you trust your reward signal completely |
| α=1, β=2 | Exploratory | When you want diverse approaches |
| α=0 | "Common Sense" — pure diversity, no reward | Brainstorming, divergent design |

α=0 mode is the drone autopilot of paper §6.3. Currently exposed only via direct `fractal_loop.py init --alpha 0`, no slash command yet.

---

## 4. Cost analysis

### Per `/fractal-decide` invocation

Cost = (walker calls) × (cost per walker call) + (judge calls) × (cost per judge call)

For default N=3, M=3:
- Walker calls: 3 walkers × 3 ticks = **9 walker invocations** (sonnet, 30s-2min each)
- Judge calls: 3 (one per final-tick walker) = **3 judge invocations** (sonnet, 30s each)
- Total: ~12 sub-agent invocations
- Time: ~5-10 min (walkers in parallel within each tick)
- Cost (Claude Pro tier estimate): ~$1-3 in API time, ~10-15% of monthly quota for power users

With ESS-adaptive cloning kicking in, expect to save 20-40% of walker calls when the swarm is already diverse.

### Per `/octopus` invocation

Cost = K_iters × (cost per /fractal-decide) + K_iters × (judge call for goal check)

For typical 5-iteration goal:
- 5 × 12 = 60 sub-agent invocations
- ~25-50 min wall time
- Cost: ~$5-15

For full K_MAX=10 budget:
- 10 × 12 = 120 sub-agent invocations
- ~50-100 min wall time
- Cost: ~$10-30

**Budget your goals**. A goal that needs 10+ commits is a big investment. Consider scoping smaller.

### Per math test run

Negligible. `python3 tests/test_fractal_math.py` runs in <1 second on a MacBook with no API calls. It's deterministic Python.

---

## 5. CLI smoke test (no LLM, free)

This verifies the state machine end-to-end without spending tokens. Useful for verifying installation or after editing `fractal_loop.py`.

```bash
mkdir -p /tmp/fmc_smoke && cd /tmp/fmc_smoke
LOOP=~/.claude/plugins/fractal-coding-loop/scripts/fractal_loop.py
# (or the absolute path if not yet installed)

# 1. Init session
SESS=$(python3 $LOOP init --task "smoke test" --goal "smoke test" \
    --n 3 --m 3 --alpha 1.0 --beta 1.0 --ess-threshold 0.7 \
    | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['session_id'])")
echo "Session: $SESS"

# 2. Create synthetic walker JSONs (simulating tick 0 outputs)
cat > walkers_t0.json <<'EOF'
[
  {"approach_label":"high","approach_description":"high R","worktree_path":"/tmp/w0","worktree_branch":"w0","walker_head":"sha_w0_t0","init_commit_sha":"sha_w0_t0","init_commit_message":"init: high","files_changed":["a.py"],"lines_added":10,"lines_deleted":0,"tests_run":true,"tests_passed":10,"tests_total":10,"lint_warnings":0,"compile_ok":true,"summary":"w0"},
  {"approach_label":"med","approach_description":"med R","worktree_path":"/tmp/w1","worktree_branch":"w1","walker_head":"sha_w1_t0","init_commit_sha":"sha_w1_t0","init_commit_message":"init: med","files_changed":["b.py"],"lines_added":15,"lines_deleted":2,"tests_run":true,"tests_passed":5,"tests_total":10,"lint_warnings":3,"compile_ok":true,"summary":"w1"},
  {"approach_label":"low","approach_description":"low R","worktree_path":"/tmp/w2","worktree_branch":"w2","walker_head":"sha_w2_t0","init_commit_sha":"sha_w2_t0","init_commit_message":"init: low","files_changed":["c.py"],"lines_added":50,"lines_deleted":5,"tests_run":true,"tests_passed":1,"tests_total":10,"lint_warnings":15,"compile_ok":true,"summary":"w2"}
]
EOF

# 3. Record tick 0
python3 $LOOP record --session $SESS --file walkers_t0.json

# 4. Step (compute VR + ESS, generate clone plan)
python3 $LOOP step --session $SESS --seed 42

# 5. Apply clones (mirrors the orchestrator's git reset into state)
python3 $LOOP apply-clones --session $SESS

# 6. Decide (final argmax bincount)
python3 $LOOP decide --session $SESS

# 7. Inspect full state if needed
python3 $LOOP status --session $SESS | head -50
```

Expected: `winner_label: "high"`, `winner_init_commit_sha: "sha_w0_t0"`. The high-R walker dominates.

---

## 6. Debugging

### Walker fails with "ModuleNotFoundError: fractal_reward"

`fractal_loop.py` imports `fractal_reward` from the same directory (line 38: `sys.path.insert(0, os.path.dirname(...))`). Verify both files are in `scripts/`:

```bash
ls ~/.claude/plugins/fractal-coding-loop/scripts/
# Should show: fractal_loop.py  fractal_reward.py
```

### Walker output JSON malformed

The walker may have ended its response with text after the JSON. The orchestrator should extract just the JSON block. If not, check the walker's `stop_reason`. The walker prompt says "Return exactly this JSON as your final output (no extra prose)" — if the LLM ignores that, raise the model temperature down (in `agents/fractal-walker.md` frontmatter, but currently just `model: sonnet` — no temperature override).

### "no alive walkers with init_actions" error from `decide`

All walkers have either died (`compile_ok=False`) or never recorded an `init_action_label`. Check:
1. Did Phase 4 (record tick 0) actually run? `python3 $LOOP status --session $SESS | grep init_action_label`
2. Did the walkers all fail to compile? Check their JSON outputs for `compile_ok: false`.

If all walkers died with the same compile error, the strategy generation in Phase 1 was likely flawed (e.g., all 3 strategies require an undefined dependency). Restart with refined goal.

### `/octopus` stops at "Confidence below 0.50"

This is by design (see [`THEORY.md`](THEORY.md) §4 confidence gate). Options:
1. **continue**: accept the noisy decision
2. **inspect**: read `python3 $LOOP status --session $SESS` to see VR distribution
3. **stop**: keep main where it is, abandon this goal for now

To prevent this stop in advance, raise N: more walkers → tighter VR distribution → higher confidence.

### Cherry-pick conflict in `/octopus`

The walker's base differs from main. Could mean:
1. The previous iteration's cherry-pick changed something the walker assumed unchanged
2. The walker ignored its base state and rewrote a file from scratch

Recovery options:
- Resolve manually: `git status` + `git mergetool` + `git cherry-pick --continue`
- Abort: `git cherry-pick --abort` then `git reset --hard $START_HEAD` (loses all octopus progress)
- Continue manually from the resolved state, skipping `/octopus` for the rest

This is also why `/octopus` writes `$LOG` — to enable manual replay.

### "ess: 2.68 / threshold: 2.10 / cloning_skipped: True"

This is **good**. ESS-adaptive cloning detected the swarm is diverse enough; skipping the clone phase saves the next tick's sub-agent calls. Expected behavior.

If you NEVER see cloning happening (always skipped), it means your strategies are too uniformly good. Lower the threshold (`--ess-threshold 0.5`) to force cloning more often.

---

## 7. Common patterns

### Quick decision on a small change

Want a fast yes/no on a 2-3 line change?

Don't use this plugin. The cost (~$2-3 + 5 minutes) is not justified. Just edit and commit directly.

### Architectural decision

"Should I refactor the auth middleware to be a separate module or keep it inline?"

This is a perfect `/fractal-decide` use case. Three strategies emerge naturally (inline, extract-module, hybrid), each can be implemented and scored, the best wins.

```
/fractal-decide refactor src/auth/middleware.ts. Decide: keep inline OR extract to src/auth/standalone.ts OR hybrid wrapper. Tests in tests/auth must still pass.
```

### Multi-step feature implementation

"Implement POST /login with JWT, tests must pass."

Use `/octopus`. The K iterations let it incrementally build (route handler → JWT logic → password verification → tests → docs).

```
/octopus implement POST /login: accepts {email, password}, validates against users table, returns 200 with JWT on success or 401 on failure. tests/auth_login_test.ts must pass.
```

### Test-driven goal

"Write tests for module X, then make them pass."

Phrase the goal so the judge can verify completion:

```
/octopus the function calculate_tax in src/billing.py must have ≥80% line coverage measured by `coverage.py`. All existing tests still pass.
```

The judge reads `git log` + diff and decides whether the goal is met.

### Refactor with existing tests as oracle

"Refactor to be 30% smaller, all tests still pass."

```
/octopus refactor src/parser.py. Goal: reduce LOC by 30%+, all tests in tests/parser_test.py still pass, no public API changes.
```

The judge can verify LOC and test pass; the swarm explores ways to compress.

---

## 8. Anti-patterns (don't do this)

### Vague goal

```
/octopus make the code better
```

This will burn your budget and produce noise. The judge can't score "better." Phrase as: "reduce duplication in src/X (specifically these 3 helper functions repeated). All existing tests pass."

### No tests in target

```
/octopus refactor src/legacy.py
```

If `tests/` is empty, the reward signal is reduced to lint + diff size + judge — much weaker. The walkers can't tell if they broke things. Add tests first or accept the noise.

### Goal that requires user judgment

```
/octopus pick the best name for the new module
```

The judge can't know which name *you* prefer. Goals must be objectively verifiable. Put your preferences in the prompt: "name it after a constellation, lowercase, ≤8 characters, isn't already used in the repo."

### Running multiple `/octopus` simultaneously

The plugin doesn't yet coordinate across sessions. Each `/octopus` writes to `.fractal/sessions/<id>/` independently. Two octopuses on the same repo will fight over `git worktree` and create conflicts.

---

## 9. Logs and audit

Every session leaves traces:

- `.fractal/sessions/<session_id>/state.json` — full state of one `/fractal-decide` (walkers, decisions, ESS history)
- `.fractal/octopus_<timestamp>.log` — outer loop log (per-iteration winner, confidence, goal_score)
- `git reflog` — all worktree branch creations/deletions
- Standard claude-code session log

To replay or audit a decision:

```bash
# Find the session
ls .fractal/sessions/

# Inspect the state
cat .fractal/sessions/<id>/state.json | jq .

# See decision history
cat .fractal/sessions/<id>/state.json | jq '.decisions'

# See per-walker rewards over time
cat .fractal/sessions/<id>/state.json | jq '.walkers[].history'
```

For a complete forensic trail of a single decision, the state file has everything: each tick's VR, ESS, clone_plan, walker R/breakdown, init_action propagation. It's the ledger.

---

## 10. Performance tips

### Speed up walker init

Walkers do `Grep`/`Read` to understand the codebase. For large codebases, this is slow. Help them by including in your goal:

```
/fractal-decide [goal]. Relevant files: src/auth/*.ts, tests/auth_*.ts. Don't search elsewhere.
```

The walker reads only what's needed.

### Reduce ESS threshold to clone more

Default 0.7 is conservative — skips a lot of cloning. If you observe walkers diverging too much (low confidence often), try `--ess-threshold 0.5` for stronger consolidation.

### Use Haiku for continuation

Currently all walker work is Sonnet. Continuation walkers (tick > 0) do small follow-up work — perfect for Haiku. To switch: edit [`agents/fractal-walker.md`](../agents/fractal-walker.md) frontmatter, set `model: haiku-4.5`. Quality may drop slightly; cost drops ~5×.

### Batch goals when possible

If you have 5 small features to implement, don't run `/octopus` 5 times. Combine them into one goal with explicit acceptance criteria for each. Single setup cost, 5 separate progress checks.

```
/octopus three features: (a) GET /users endpoint, (b) GET /users/:id endpoint, (c) tests for both in tests/users_test.ts. All must pass.
```

The judge can score partial completion (a and b done, c missing → 0.7 → continue).

---

## 11. Where to ask questions

This plugin is at Phase 0 PoC. Questions, bug reports, and feature requests:

- Open an issue at the homepage URL in [`plugin.json`](../.claude-plugin/plugin.json)
- Read the parent project's docs first: [`docs/vision/fractal_coding_loop.md`](../../../docs/vision/fractal_coding_loop.md)
- Theory questions: [`work/02_deep_dives/`](../../../work/02_deep_dives/) has 6 deep dives covering the math
- Algorithm questions: [`ALGORITHM.md`](ALGORITHM.md) of this plugin
- Component questions: [`COMPONENTS.md`](COMPONENTS.md) of this plugin

For the original FMC paper questions, [@EntropyFarmer](https://twitter.com/EntropyFarmer) (Sergio) and [@Miau_DB](https://twitter.com/Miau_DB) (Guillem) are the canonical sources.
