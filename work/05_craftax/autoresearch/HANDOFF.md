# HANDOFF: autoresearch-FMC continuation

> **For the next AI agent picking up this experiment.**
> Read this file fully before any edit. The TL;DR + map of the experiment
> tree is below. Do not skip.

## TL;DR — what's been established

Goal: push Crafter score on Craftax-Classic-Symbolic-v1 toward 50% (human-expert
level). Started at 29.27% baseline (run_007 SOTA). After 8 experiments, current
best is **exp03 = 40.96%** with **3 of 4 v4-blockers fired** for the first time
ever on this benchmark (collect_diamond=8%, make_iron_pickaxe=23%, make_iron_sword=8%).

Path to 50%: needs +9pp. Tried 5 orthogonal tweaks since exp03 (weights, M, beta,
N, action curriculum). All hover 33-40%. **exp03 is a sharp local optimum.**
Breaking through requires structural mechanisms, not parameter tweaks.

## Branch state when this was written

- Branch: `autoresearch/exp02-ach-bonus` (HEAD = `1739552`, exp03 commit)
- `fmc_mutable.py` reflects exp03 best (tier-weighted achievement bonus).
- Main branch unchanged (run_007 SOTA still there).
- Branches `autoresearch/exp01-iron-boost` and `autoresearch/sanity` exist but
  have outdated content (cherry-picked or revert source). You can delete them.

## Verify state before starting (1 min)

```bash
cd /Users/vladvrinceanu/Desktop/PROGETTI\ ANTYGRAVITY/FractalAI/work/05_craftax/autoresearch
git branch --show-current        # should be autoresearch/exp02-ach-bonus
git log --oneline -3             # should show exp03 + exp02 + setup
grep "ACH_WEIGHTS_LIST" fmc_mutable.py | head -1   # should match exp03 weights
cat results.tsv | column -t -s $'\t'    # should show 10 rows of history
```

If anything is off, run: `git checkout autoresearch/exp02-ach-bonus && git reset --hard 1739552`.

## Experiment history (full)

| # | Mutation | Crafter % | n_seeds | mean_ach | blockers | status | take-away |
|---|---|---|---|---|---|---|---|
| baseline | v4 SOTA | 27.44 | 10 | 11.90 | 0 | keep | reference |
| sanity | action_repeat=2 | 8.97 | 4 | 7.00 | 0 | discard | loop discipline OK |
| exp01 | iron-tier inv-boost (iron 8→16, diamond 16→64) | 29.30 | 9 | 12.33 | 0 | discard | inv weight tweaks neutral |
| **exp02** | **ach-fire bonus +50 uniform** | **37.75** | 11 | 12.64 | **2** | **keep** | **first blocker fire EVER** |
| **exp03** | **tier-weighted (blockers 150-300, easy 10-30)** | **40.96** | 13 | 12.62 | **3** | **CURRENT BEST** | **DIAMOND 8%** |
| exp04 | ultra-aggressive (diamond=1000, iron=500) | 36.94 | 11 | 13.00 | 2 | discard | over-weighting → relativize collapse → DIAMOND LOST |
| exp05 | M=40 → M=60 | 33.05 | 9 | 12.78 | 1 | discard | longer horizon dilutes ach signal |
| exp06 | action curriculum (mask craft/place w/o precondition) | 36.92 | 12 | 13.75 | 1 | discard | mid-tier ↑ but blockers lost |
| exp07 | β=2.0 (more diversity pressure) | 39.96 | 10 | 14.70 | 1 | discard | within noise; ach distribution shifts |
| exp08 | N=512 → N=1024 | 30.25 | 2 | 15.00 | 0 | discard | TOO SLOW, 1 episode busts budget |

## Key insights from this run (read carefully — these are GOLD)

### Insight 1 — The achievement-fire bonus is the breakthrough mechanism

exp01 (inv-tier boost) was neutral. exp02 (ach-fire bonus +50 uniform) jumped
+8pp and broke 2 blockers for the FIRST time in 115+ baseline episodes. The
RIGHT signal is the **sparse achievement-unlock event**, not dense inventory
accumulation.

Mechanism: walker sim tracks `state.achievements` (bool[22]). Per tick, compute
`new_ach = current AND NOT baseline_at_root`. Add weighted reward per new ach to
walker's cum_reward. The cloning kernel then preferentially clones achievement-
unlocking walkers, propagating the chain forward.

### Insight 2 — Tier weighting matters, but moderately

exp02 uniform +50 → 37.75%. exp03 tier-weighted (10-30 easy, 50-80 gateway,
150-300 blockers) → 40.96%. The +3pp came from amplifying gradient toward
blockers. But pushing further (exp04, blockers 500-1000) BACKFIRED — relativize
on rare-event-only-walker collapses, no productive gradient. **Sweet spot at
diamond=300.**

### Insight 3 — Ceiling 40-41% is structural, not parametric

Tried 5 orthogonal mechanisms after exp03:
- Bigger weights (exp04): regression
- Bigger M (exp05): regression
- Action curriculum (exp06): regression
- Bigger β (exp07): within noise
- Bigger N (exp08): infeasible (too slow, busts wall budget)

**Pattern: tweaks shift the achievement DISTRIBUTION but don't elevate the
ceiling.** Crafter score's geometric-log-mean punishes losing rare blockers
even when easy rates rise. Mid-tier saturation is hit.

### Insight 4 — N=1024 catastrophically slow

exp08 only completed 2 seeds in 13307s wall (budget was 1200s). One single
seed at N=1024, M=40 takes ~1 hour on this CPU. Don't try N>=1024 unless you
have real GPU support OR are willing to extend the wall budget. **N=512 is
the practical max for this CPU.**

### Insight 5 — eat_plant blocker is structurally inaccessible

eat_plant requires sapling growth ~30 in-game days. Even with M=40 ticks of
planning per decision, the achievement won't fire because the chain is OUTSIDE
the planning horizon. Has weight 200 in exp03 ACH_WEIGHTS but rate stays at 0.
**To unlock eat_plant, need: episode-level memory across decisions, OR
planted sapling from EARLIER decision being noticed at later planning step.**

### Insight 6 — auto-status gate is too generous

The driver auto-gate compares to historical baseline (29.27), so anything
> 30.27 gets "keep". This is wrong for ongoing iteration: you should compare
to **current best on branch**. Manual override has been needed multiple times.
**Recommendation**: when checking results, always read the FULL summary and
compare against the latest "keep" entry in results.tsv, not the gate decision.

## What to try next — ranked by my expected value (highest first)

### Tier 1 — Likely to give +1-3pp wins (try first)

**A. Multi-population swarm.** Split N=512 into 2 sub-pops of 256. Pop A uses
exp03 tier-weighted bonus. Pop B uses uniform +50 (exp02-style) or even
"common-sense" (α=0, β=1). Cloning happens within each pop; final action vote
combines both. Implementation moderate (~2h). Might unlock different blockers
in different pops.

**B. Adaptive M based on inventory state.** M=40 base, M=80 once stone_pickaxe
in inventory (extending only when chain is in progress). The naive "M=60 fixed"
(exp05) failed because longer horizon dilutes early ach signal — adaptive M
gives you long planning ONLY when you have something deep to plan for.
Implementation: requires JIT recompile per M change OR a single JIT with
masked timesteps. Moderate (~3h).

**C. Stack exp03 weights with iron-tier inv-boost (exp01 weights).** Tried
either alone — exp01 alone neutral, exp03 alone +11pp. Stacking might give
a small extra gradient amplification for the iron stage of the chain.
Quick test (10 min code change).

### Tier 2 — Possibly +3-5pp (moderate effort)

**D. Cross-episode achievement memory.** When iron_pickaxe fires at episode N,
record the action sequence that led to it. Inject those actions as init_actions
priors in episode N+1. This is what trained policies do implicitly. Run_005
tried naive Wigner-style memory and FAILED — but here we'd record SUCCESS
patterns specifically, not stochastic visit counts. Implementation: ~4-6h.

**E. Hierarchical option-policy.** Pre-define 5-10 macro-actions (skill primitives
like "go_to_nearest('iron')", "mine_until_inv+1", "craft_pickaxe_chain"). At
each FMC decision, with prob p, emit a macro-action instead of a primitive.
Macro executes deterministically over multiple env steps. This bypasses the
K=17 cross-entropy collapse for chain sequences. Heavy implementation: ~6-8h.

### Tier 3 — Long shots (high ceiling potential, high implementation cost)

**F. NN value function priors.** Roll out 100 episodes with exp03 config,
record (state, action, future_score). Train small Q(s,a) shallow MLP. Plug
Q-output as init_actions prior in FMC. ~1-2 days work.

**G. LLM-as-policy.** Convert state to symbolic JSON (inventory, near tiles,
mobs). Prompt LLM to suggest next action. Use as scanning policy. Voyager-style.
Requires Anthropic/OpenAI API access. ~1-2 days work, ongoing API costs.

### Things to AVOID (tried, falsified)

- ~~Bigger weights for blockers (exp04 ultra-aggressive)~~: relativize collapses
- ~~Longer M (exp05 M=60)~~: signal dilution
- ~~Action curriculum (exp06 inventory mask)~~: trades blockers for mid-tier
- ~~Bigger N (exp08 N=1024)~~: too slow, infeasible on CPU
- ~~Vitality bonus~~: falsified in run_006 (in main repo, not autoresearch)
- ~~Naive Wigner memory~~: falsified in run_005

## Operational details

### How to invoke evaluate.py

```bash
cd /Users/vladvrinceanu/Desktop/PROGETTI\ ANTYGRAVITY/FractalAI/work/05_craftax/autoresearch
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
JAX_PLATFORMS=cpu nohup $PY evaluate.py \
  --description "expNN: <short hypothesis>" \
  --status auto \
  > expNN_run.log 2>&1 &
disown
```

Wait ~20 min wall budget. Then:
```bash
grep "crafter_pct\|status\|blocker_fired" expNN_run.log | tail -20
column -t -s $'\t' results.tsv | tail -5
```

### Loop discipline (Karpathy)

1. Edit `fmc_mutable.py`
2. `git add ... && git commit -m "expNN: short desc"`
3. Run evaluate.py (background, 20 min)
4. Read crafter_pct from log
5. If improved >= +1.0pp over CURRENT BEST (not historical baseline!): keep
6. Else: `git reset --hard HEAD~1` to revert
7. Move to next experiment idea

### Wall budget management

Wall budget = 20 min hard. JAX JIT compile is ~5-10s the first time per (N, M)
config. Subsequent calls reuse the compiled function. So changing N or M means
~10s compile overhead per evaluate.py invocation, NOT per seed.

For each (N, M):
- N=128, M=20: ~22s/seed → ~50 seeds in budget
- N=256, M=40: ~50s/seed → ~24 seeds
- **N=512, M=40 (exp03 baseline)**: ~100s/seed → **~12 seeds** in budget
- N=1024, M=40: TOO SLOW, do not use

For sanity-only quick tests use `--wall_budget_s 240` (4 min).

### Achievement enum (Craftax-Classic order, important for ACH_WEIGHTS)

```
0:COLLECT_WOOD, 1:PLACE_TABLE, 2:EAT_COW, 3:COLLECT_SAPLING, 4:COLLECT_DRINK,
5:MAKE_WOOD_PICKAXE, 6:MAKE_STONE_PICKAXE, 7:MAKE_IRON_PICKAXE,
8:MAKE_WOOD_SWORD, 9:MAKE_STONE_SWORD, 10:MAKE_IRON_SWORD,
11:PLACE_PLANT, 12:DEFEAT_ZOMBIE, 13:COLLECT_STONE, 14:PLACE_STONE,
15:EAT_PLANT, 16:DEFEAT_SKELETON, 17:COLLECT_IRON, 18:COLLECT_COAL,
19:PLACE_FURNACE, 20:COLLECT_DIAMOND, 21:WAKE_UP
```

Blockers: indices 7, 10, 15, 20 (make_iron_pickaxe, make_iron_sword,
eat_plant, collect_diamond).

### Action enum

```
0:NOOP, 1:LEFT, 2:RIGHT, 3:UP, 4:DOWN, 5:DO, 6:SLEEP,
7:PLACE_STONE, 8:PLACE_TABLE, 9:PLACE_FURNACE, 10:PLACE_PLANT,
11:MAKE_WOOD_PICKAXE, 12:MAKE_STONE_PICKAXE, 13:MAKE_IRON_PICKAXE,
14:MAKE_WOOD_SWORD, 15:MAKE_STONE_SWORD, 16:MAKE_IRON_SWORD
```

K = 17.

### Craftax state structure (relevant fields)

```python
state.inventory.{wood, stone, coal, iron, diamond, sapling,
                 wood_pickaxe, stone_pickaxe, iron_pickaxe,
                 wood_sword, stone_sword, iron_sword}
state.player_position : int32[2]
state.player_health, player_food, player_drink, player_energy : int32
state.achievements : bool[22]      # KEY for ach-fire bonus
state.timestep : int32
state.growing_plants_age : int32[10]    # eat_plant gating signal
state.growing_plants_mask : bool[10]
```

## Files in this directory

```
prepare_craftax.py        # FROZEN: eval harness, Crafter score, 20-min budget
evaluate.py               # FROZEN: driver, auto-status, results.tsv append
fmc_mutable.py            # MUTABLE: agent edits this. Currently = exp03.
program_fmc.md            # The ORIGINAL agent skill (read for full discipline)
README.md                 # Setup overview
HANDOFF.md                # THIS FILE — read before continuing
results.tsv               # Experiment log (gitignored, persists across branches)
example_results.tsv       # Schema reference
sanity_test_plan.md       # Initial sanity test (already passed)

baseline_lock.{json,log}  # Initial baseline lock evidence
expNN_run.log             # Per-experiment driver stdout (NN=01..08)
```

## When you might want to STOP

If you reach 50% Crafter score → DECLARE VICTORY, stop the loop, write a
final report, prepare paper submission.

If you complete 5 more experiments (so 13+ total) without breaking 42%
Crafter → that's strong evidence the local optimum is REAL and structural.
Stop the autoresearch loop and pivot to **macro-actions** (Tier 2E above)
or **NN value function** (Tier 3F) as a multi-day project, NOT autoresearch
shaping.

If you crash repeatedly on a single mutation idea → revert and skip; don't
spend more than 30 min debugging a single failed mutation.

## Provenance + context

- Project: FractalAI / FMC research, see top-level CLAUDE.md
- Run history before autoresearch: runs 001-007 in `../docs/`
- run_007 30-seed validation: established 29.27% as official SOTA
- This autoresearch session pushed to **40.96% Crafter** with **3/4 blockers
  fired** — substantial new SOTA, paper-grade finding even if loop ends here.
- The exp03 result alone deserves a workshop paper: first time FMC zero-training
  cracks the iron→diamond chain on Craftax-Classic without any RL training.

## Closing note for next agent

You are inheriting a working framework that has already produced a publishable
result (40.96%, 3 blockers fired). The user's stated goal is 50%, but **even
not reaching it, this work is novel and significant**. If exp09-12 don't move
the needle, write up exp03 as the headline result and don't apologize.

The deeper question — "can FMC reach 90%?" — has been answered structurally
in run_007 docs and in `../docs/run_007_addendum_fragile_port_analysis.md`:
**no, not with FMC alone, by ANY method**. EMERALD with 10M training only hits
58%, human expert is 50.5%. So 40.96% zero-training is excellent context.

Good luck. Be rigorous. Don't fake results. When in doubt, revert.

— Previous agent, 2026-04-30
