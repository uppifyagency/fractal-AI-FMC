# EVOLUTION — how this plugin could evolve vs the Ralph loop

> *This document records the architectural reasoning about where the plugin sits in the design space relative to Geoffrey Huntley's Ralph loop (2024), and what evolution paths would genuinely justify the plugin's added complexity over time.*

For higher-level rationale see [`THEORY.md`](THEORY.md). For algorithm walkthrough see [`ALGORITHM.md`](ALGORITHM.md). For components see [`COMPONENTS.md`](COMPONENTS.md). For invocation see [`USAGE.md`](USAGE.md).

> **2026-05-01 update — empirical confirmation from Craftax autoresearch session**: the structural-introspection thesis below (§5) gained quantitative support from a 23-experiment loop on Craftax-Classic. Starting from FMC v4 baseline (29.27%), iterative reward shaping discovered a **+10pp trajectory** culminating at exp17 = **50.95% Crafter zero-training**, matching human-expert reference (50.5%). Critical mechanism: **achievement-fire bonus** (sparse-event reward shaping) compounding with **inv-tier reward stacking** — see [`work/05_craftax/autoresearch/HANDOFF.md`](../../../work/05_craftax/autoresearch/HANDOFF.md) and Cong. D in [`docs/MATH_CANON.md`](../../../docs/MATH_CANON.md#congettura-d--chain-tier-compounding-amplification-sparse-event-reward-shaping). Implication for the plugin: **the auto-cherry-pick reward function `R_goal` should be decomposed into `R_inv` (dense per-state value) + `R_ach` (sparse milestone unlock bonus)** rather than a single scalar — see new §3.8 below.

---

## 1. The structural difference

### Ralph loop

A single trajectory through time:

```
prompt → claude → diff → prompt → claude → diff → ...    (infinite loop)
```

One arrow. No comparison between alternatives. No introspection signal. Each iteration applies whatever Claude produces, and quality depends entirely on prompt quality plus model capability. Stop condition is manual (human watches and aborts) or external.

### Fractal Coding Loop

A tree that branches and prunes at every decision point:

```
state x_0
   ├── walker 0 (strategy A) → tick 1 → tick 2 → score
   ├── walker 1 (strategy B) → tick 1 → tick 2 → score
   └── walker 2 (strategy C) → tick 1 → tick 2 → score
                                              ↓
                              bincount + cloning + decision
                                              ↓
                                       ONE commit to main
                                              ↓
                                          state x_1
```

From this structural difference flow all subsequent advantages — and costs.

---

## 2. Properties Ralph **structurally cannot have**

| Capability | Why Ralph cannot have it | Implication |
|---|---|---|
| Quantitative confidence | No swarm to count → no bincount available | Ralph never knows when it's wrong |
| Tie detection | Same — needs comparison among alternatives | Ralph applies the first thing that comes to mind |
| Selective cloning | Same — no concept of "dead walkers" to revive | Ralph propagates errors forward, never eliminates them |
| Counterfactual reporting | A single trajectory has no alternatives to discard | Ralph cannot say "I could have done X instead" |
| ESS-adaptive cost | No swarm → no diversity to measure | Ralph always pays the full per-step cost |
| Cross-session memory with weighting | Possible in theory, but requires structure to weight memories (Wigner reward) | FCL is naturally architected for this; Ralph would need bolt-on |
| Multi-octopus parallelism | Single-threaded by definition | Ralph would need N separate instances with no coordination |

These are not optimizations — they are **emergent properties of the swarm**. Ralph cannot acquire them without becoming a different algorithm.

---

## 3. Seven evolution directions, ordered by realism

### 🟢 Realistic (1-3 months)

#### 3.1 Cost reduction without losing the swarm

The plugin's cost dominates. Several optimizations could close the gap with Ralph:

- **Cheap-model continuation walkers**: tick > 0 walkers do small follow-up work. Use Haiku (or a smaller model) instead of Sonnet → ~5× cost reduction on 2/3 of walker calls.
- **Adaptive N**: start with N=2, expand to N=5 only when initial signals show divergence (variance in R, low ESS).
- **Adaptive M**: stop early if walker rewards converge before max M.
- **Lazy spawn**: don't spawn N walkers upfront; spawn one, evaluate, decide whether to spawn alternatives.

**Target**: bring FCL from 10-30× cost-of-Ralph to **2-3×**, while preserving decision quality.

#### 3.2 Closing the Fractal Memory loop

The memory subsystem ([`fractal_memory.py`](../scripts/fractal_memory.py)) exists and is tested but is invoked manually only via `/fractal-recall` and `/fractal-memory-show`. Auto-wiring would mean:

- **Auto-append** every `/fractal-decide` decision to `.fractal/memory/` (one-line shell call in Phase 8b)
- **Wigner-weighted recall** at Phase 1 strategy generation: surface similar past decisions, bias strategy proposal toward what historically worked

**Result**: the plugin becomes more useful over time **for the specific user/codebase**. Ralph is amnesic — every session starts fresh. FCL accumulates intuition.

#### 3.3 Richer confidence diagnostics

The current `is_tie` + `tied_labels` are step 1. More valuable would be:

- **Disagreement explanation**: "Walker 0 wants to refactor X; walkers 1 and 2 leave X alone. Disagreement: existing test coverage of X."
- **Counterfactual reporting**: "Winner: A. Alternative B (rejected for Y). Alternative C (higher R but lower confidence due to Z)."
- **Per-walker reasoning trace**: a brief summary from each walker of *why* its strategy was chosen — surfaced to user pre-cherry-pick.

**Result**: the plugin doesn't just decide — it **communicates why**, allowing informed user intervention. Ralph cannot communicate uncertainty because it doesn't have any.

### 🟡 Plausible (6-12 months)

#### 3.4 Add Badger level 1 — reward learning

Currently `R_goal` is hardcoded to 0.7 default when judge is skipped. Future:

- The reward function itself becomes adaptive — learns from user accept/reject decisions
- Example: "Of the last 50 cherry-picks, the user reverted 30% of refactors that touched `tests/`" → `R_tests_changed` becomes a hard constraint, multiplying the reward by 0 if tests are touched
- This is exactly Book #2 §3.4 Level 1 (Reward optimizer) applied to coding

**Result**: the plugin calibrates on the user's values, not on universal defaults. Genuine personalization, structurally impossible in Ralph (which has no value model to update).

#### 3.5 Real benchmark vs Ralph (SWE-bench Lite or equivalent)

Currently we have:
- 5/5 math tests (kernel correctness)
- 22/22 e2e tests (integration soundness)
- 1 smoke run on FizzBuzz (it runs end-to-end)

What we **don't** have: empirical evidence that FCL produces **better outcomes** than Ralph or single-shot on real-world coding tasks.

Path to that evidence:
- SWE-bench Lite: 300 issue-fix pairs from real OSS projects, established baseline metrics
- Run Ralph + single-shot Sonnet + FCL on the same subset (≥30 issues for statistical signal)
- Compare success rate, diff size, time-to-fix

**Result**: numbers that justify (or kill) the plugin's complexity. The difference between "interesting idea" and "validated tool."

#### 3.6 Multi-octopus coordination

Currently `/octopus` operates on one goal at a time. Future:

- Multiple parallel octopuses on the same repo, each with its own goal
- Goals coordinate cherry-picks to avoid conflicts (if octopus A is touching `src/auth/`, octopus B working on docs proceeds independently)
- Shared reward learning across octopuses (unified `.fractal/reward/` artifact)

**Result**: a team-of-agents working in parallel on independent feature streams. Ralph is single-threaded by definition; this composition is impossible.

#### 3.8 Decompose `R_goal` into dense-inv + sparse-ach (Craftax-validated)

**Origin**: 2026-05-01 autoresearch session on Craftax-Classic. Started at 29.27%, ended at 50.95% via this single insight.

Currently the plugin's reward function `R_goal` is one scalar (e.g. tests-pass-rate × style-score × ...). For coding tasks with **chain structure** (e.g. "implement OAuth: spec → routes → middleware → tokens → tests"), this is too coarse — the gradient saturates once any sub-step is achieved.

The Craftax finding suggests:

```
R_goal = R_inv(state) + R_ach(state, prev_state)

R_inv(state)     = sum of value-of-possession for each completed sub-deliverable
                   (weight grows hierarchically: scaffolding=1, types=2, routes=4, tests=8)

R_ach(s, s')     = bonus for FIRST-time unlock of a sub-milestone in this rollout
                   (tier-weighted: blocker = 200-300, gateway = 50-120, easy = 10-30)
```

The sparse component is the breakthrough: it gives the swarm a **discrete signal** ("this walker just unlocked a key sub-goal") on top of the continuous inv-tier signal ("this walker holds more state").

**Application to coding**:
- "blocker" milestones = tests-pass, types-check, lint-clean, integration-pass
- "gateway" milestones = file-created, function-defined, test-defined, import-resolved
- "easy" = whitespace-fix, comment-add, rename

The sparse bonus fires once per walker per rollout per milestone. Walkers that reach blockers get a 10-30× multiplier in cum_reward, dominating the cloning kernel. This propagates the chain forward without requiring the dense reward to be perfectly tuned.

**Falsified mistakes from Craftax** (DO NOT REPEAT):
- Multiplying blocker weights >2× crashes the relativize statistics. Sweet spot is 1.2-1.4× per amplification step.
- Adding ach-fire only without inv-tier stack = peaks at ~37% (exp02). Need both for compounding.
- Multi-population swarm with vote-summing dilutes specialist signal. Single-pop with proper shaping wins.

**Implementation in the plugin** (~1 day):
- `R_goal` becomes a `RewardComposite { inv: number, ach: Set<MilestoneId>, ... }`
- Walker rollout returns the `ach` set incrementally; cloning kernel sums `inv` + `Σ tier_weight[ach_i]`
- Milestone definitions are user-configurable per repo via `.fractal/milestones.yaml`

**Expected gain on coding tasks**: not yet validated. The Craftax result is one task — needs replication on a coding chain (e.g. SWE-bench Lite issue with multi-file fix). See §3.5.

### 🔴 Aspirational (12+ months)

#### 3.7 Full Badger hierarchy + dashboard

The full vision from [`docs/vision/fractal_coding_loop.md`](../../../docs/vision/fractal_coding_loop.md):

- All 5 Badger levels implemented as nested FMC swarms (architecture / embedding / prediction / reward / expert)
- Live dashboard ("Iron Man" mode) showing the cone of walker possibilities in real-time
- α=0 "Common Sense" mode exposed as `/fractal-explore` for divergent design / brainstorming
- Generalization beyond coding — same architecture applied to plasma control, robotics, strategic planning

**Result**: the plugin becomes a general platform for *intelligence operations*, not just coding decisions. The unification Hernández-Cerezo aimed at in Book #2.

---

## 4. The framing that holds: FCL as escalation tier

FCL is not a Ralph replacement. It is an **escalation tier**:

| Stakes | Right mode | Cost relative to Ralph |
|---|---|---|
| Trivial (rename, lint fix, typo) | **Ralph** (single call, no plugin) | 1× |
| Routine (single small feature) | `/fractal-decide` Lite (N=3, M=1) | ~3× |
| Ambiguous (architectural choice) | `/fractal-decide` Full (N=5, M=3) | ~12× |
| Critical (production refactor) | `/octopus` (K iter, full hierarchy) | ~30-100× |

The interesting evolution is not "FCL gets cheap as Ralph" (it would lose its value). It is:

> **The main agent automatically classifies each request into a tier and shows the user the cost/quality estimate before launching.**

A `UserPromptSubmit` hook heuristically classifies the prompt:
- `"rename foo to bar"` → Ralph mode
- `"refactor the auth middleware"` → Lite mode
- `"implement OAuth flow with full tests"` → Full mode
- `"redesign the data access layer"` → Octopus mode

The user can always override the tier via flag.

In this framing, **Ralph is a special case of FCL** with N=1, M=1, no judge, no memory. The plugin doesn't compete with Ralph — it generalizes it.

---

## 5. The epistemic position

Ralph demonstrated something important: **pure iteration over a capable LLM works better than expected**. Geoffrey Huntley's empirical observation is robust.

But Ralph has a ceiling: it doesn't know when it's going wrong.

FCL adds **structural introspection** via the Hernández-Cerezo principle: intelligence emerges from comparison among alternatives, not from depth of a single trace.

This is a falsifiable claim. The benchmark in §3.5 is what falsifies or confirms it.

Honestly: I expect FCL to win on high-stakes ambiguous tasks and lose on trivial ones. The interesting question is not "Ralph vs FCL" but "**how do you compose the two**?" — an agent that uses Ralph for 90% of tasks and FCL for the 10% that genuinely matter.

That composition is where the plugin should head in the next 3 months: stop trying to replace Ralph, start serving as an escalation path that includes Ralph as the degenerate case (N=1, M=1, no judge, no memory).

---

## 6. Concrete next steps mapped to evolution paths

| Path | Estimated effort | Dependency |
|---|---|---|
| §3.2 — Wire memory auto-append | ~half a day | None — `fractal_memory.py` already works |
| §3.1.a — Haiku for continuation walkers | ~1 hour | Trivial: edit walker frontmatter |
| §3.1.b — Adaptive N (lazy spawn) | ~2 days | Touches orchestration logic in slash command |
| §3.3 — Counterfactual reporting | ~1 day | Need to capture per-walker rationale in JSON output |
| §3.5 — SWE-bench Lite baseline | ~1-2 weeks | Significant: needs full benchmark integration |
| §3.8 — Decompose `R_goal` (Craftax-validated) | ~1 day | Trivial: refactor reward type + tier weights |
| §3.4 — Reward learning (L1 Badger) | ~2-4 weeks | Needs persistent reward artifact + user feedback channel |
| §3.6 — Multi-octopus coordination | ~1-2 months | Needs cross-session state coordination protocol |
| §3.7 — Full Badger + dashboard | ~6+ months | Major undertaking; effectively a v2 |

The path of least resistance is §3.1.a → §3.2 → §3.3 → §3.5 in that order. Each builds on the prior; each is independently shippable and incrementally testable.

---

## 7. What this document is NOT

- A commitment to ship any specific feature on any specific timeline
- A claim that FCL is "better than Ralph" — that is the empirical question of §3.5
- A complete architectural specification — see `docs/vision/fractal_coding_loop.md` in the parent project for that

It is a **map of the possibility space**, written to make explicit what the plugin can become and what it cannot. The choice of which paths to pursue depends on actual usage feedback, which Phase 0 PoC has not yet collected.

When usage data exists (run logs, user feedback, benchmark numbers), this document should be revisited and the paths re-prioritized empirically rather than from first principles.
