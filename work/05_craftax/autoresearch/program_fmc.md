# autoresearch-FMC

> Autonomous research loop for evolving FMC on Craftax-Classic. Adapted from
> [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (Mar 2026).

You are an autonomous AI researcher. Your task: iteratively improve the FMC
algorithm in `fmc_mutable.py` to maximize the Crafter score on
Craftax-Classic-Symbolic-v1, beyond the current SOTA of **29.27%**.

## Setup (do this ONCE per session)

1. **Agree on a run tag** with the user (e.g. `apr30`, `apr30-shaping`,
   `apr30-macros`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**:
   ```
   git checkout -b autoresearch/<tag>
   ```
3. **Read the in-scope files** (the repo is small):
   - `prepare_craftax.py` — FROZEN eval harness; do not modify
   - `fmc_mutable.py` — the file you edit
   - `evaluate.py` — the driver you call
   - `README.md` — context

4. **Read the canonical references** for context (MUST read before first edit):
   - `../../docs/MATH_CANON.md` — FMC math (Def 2-4: relativize, virtual reward,
     cloning rate). If you keep these invariants your work is "still FMC".
   - `../../docs/bibliography/sources/papers/2020_fractal_ai_v5_1803.05049.pdf`
     (or scan `../../1803.05049v5.pdf`) — original FMC paper.
   - `../docs/run_005_wigner_memory_negative.md` — naive cross-episode memory
     FAILED (-9.5pp). Don't repeat naively; if attempting memory, do it
     differently.
   - `../docs/run_006_long_episode_and_vitality_negative.md` — vitality bonus
     FAILED. Survival != exploration. Don't add a flat health/food/drink reward.
   - `../docs/run_007_NM_sweep_GPU.md` — N x M scaling FAILED to crack the
     diamond chain across 115 episodes (4x4 full grid + 30-seed validation).
     Don't waste experiments on bigger N or bigger M alone.
   - `../docs/run_007_addendum_fragile_port_analysis.md` — GPU port via fragile
     does NOT accelerate Craftax; JAX vmap CPU is already optimal for this env.
   - `../scripts/test_fmc_theory.py` — 15 unit tests that gate "is this still FMC?"

5. **Verify the harness is healthy**:
   ```
   python prepare_craftax.py             # prints sanity_check JSON
   python -c "import fmc_mutable; r = fmc_mutable.run_episode(99, max_steps=80); print(r['achievements_unlocked'])"
   ```

6. **Initialize results.tsv** if it doesn't exist (evaluate.py creates it).

7. **Run the baseline FIRST** (no edits) to establish your reference:
   ```
   python evaluate.py --description "baseline (untouched fmc_mutable.py = v4 SOTA)"
   ```
   This takes ~20 minutes (the wall budget). Read out the `crafter_pct` and
   confirm it's in the 28-30% range (within CI95 of the historical 29.27 +/- 1.04).

8. **Confirm to the user**: "setup OK, baseline locks at X.XX%, ready to start
   experimentation".

## Experimentation loop

Each experiment runs on a single CPU. The wall budget is **fixed at 20 minutes**.
On a fast config (N=128, M=20) you'll get ~50 seeds / very tight CI; on a
slow config (N=512, M=160) you'll get ~4 seeds / loose CI. **Choose configs
that maximize Crafter score per wall minute.**

### What you CAN do

- Modify `fmc_mutable.py` freely (Level B authority):
  - Change `CONFIG` (the FMCConfig object) — easy first lever
  - Replace `run_episode` with a new implementation
  - Add new helper modules / classes
  - Try macro-actions (multi-step skill primitives)
  - Try non-uniform scanning policies (heuristic, learned, LLM-guided)
  - Try cross-episode memory schemes (avoid the run_005 trap of naive Wigner)
  - Try hybrid FMC+NN value functions (small Q-net trained on rollouts)
  - Try adaptive M (start small, grow when iron is found)
  - Try multi-population swarms (sub-populations with different shapings)
  - Try alternate distance metrics (task-aware embeddings vs L2 obs)
  - **If you keep MATH_CANON Def 2-4 invariants**, your work is "still FMC".
  - **If you break them deliberately** (e.g. replacing relativize with softmax),
    note this in the description as "non-FMC variant" and the 15 unit tests
    are skipped for your variant.

### What you CANNOT do

- Modify `prepare_craftax.py` (FROZEN: env, eval harness, Crafter score formula)
- Modify `test_fmc_theory.py` (FROZEN: theory invariants gate)
- Install new packages or add dependencies (use what's installed)
- Modify the `evaluate` function semantics or the wall budget
- Run more than 60 seeds per experiment (hard cap in prepare_craftax)
- Push to main or merge to main (work only on autoresearch/<tag> branch)

### Output format

After each `python evaluate.py ...` run, you'll see:

```
---
crafter_pct:        29.10
vs_baseline_pp:     -0.17
n_seeds_completed:  14
mean_ach:           12.50 +/-1.20
decisions_per_sec:  4.20
wall_s:             1198.5
blocker_fired:      0/4
status (auto):      discard
description:        increased N=512 to N=1024
```

Read out the key metric:
```
grep "crafter_pct" run.log
```

## Logging results

`evaluate.py` automatically appends to `results.tsv` (tab-separated). Schema:

```
commit  crafter_pct  n_seeds  mean_ach  ach_ci95  blocker_fired  status  description
```

- `commit`: 7-char SHA (auto-filled)
- `crafter_pct`: Crafter score 0-100 (use 0.0 for crashes)
- `n_seeds`: how many seeds completed within budget
- `mean_ach`, `ach_ci95`: per-episode achievement count + 95% CI
- `blocker_fired`: how many of the 4 v4-blockers (collect_diamond, make_iron_pickaxe,
  make_iron_sword, eat_plant) had non-zero rate
- `status`: `keep`, `discard`, `crash` (auto-decided by gate, you can override)
- `description`: short text explaining what you tried

DO NOT commit `results.tsv` to git — leave it untracked.

## The experiment loop

LOOP FOREVER (until manually interrupted):

1. **Look at git state**: confirm you're on `autoresearch/<tag>`, check current
   commit and last results.tsv row.
2. **Form a hypothesis** about a change that might improve Crafter score.
   Read more references if you're stuck. Specific suggestions:
   - **Easy levers** (try first, ~minutes each): tweak CONFIG params,
     adjust proximity coefficients per-resource, add a small constant to
     intrinsic_inv_alpha, change proximity_sigma.
   - **Medium levers** (more interesting, more risk): replace `proximity_bonus_single`
     with a different shaping; add a small NN value head; introduce
     adaptive M based on inventory state; try multi-pop swarm.
   - **Heavy levers** (high reward, high risk): macro-actions library
     (`go_to_nearest("iron")`, `mine_until_inv+1`, `craft_chain`); cross-episode
     memory done correctly; LLM-as-policy with Craftax state prompted.
3. **Edit `fmc_mutable.py`** to implement the change.
4. **Commit**: `git add fmc_mutable.py && git commit -m "<short description>"`
5. **Run**:
   ```
   python evaluate.py --description "<what you tried>" > run.log 2>&1
   ```
   Wait ~20 minutes. Do NOT use tee, the output is too big for context.
6. **Read out**:
   ```
   grep "crafter_pct\|status" run.log
   tail -20 run.log
   ```
7. **If the run crashed** (no `crafter_pct` line, only a Python traceback):
   - If it's something dumb (typo, missing import): fix and re-run.
   - If the idea is fundamentally broken: log "crash" status, revert with
     `git reset --hard HEAD~1`, move on.
8. **If `crafter_pct` improved by >=1.0pp** above the previous best with
   `n_seeds_completed >= 10`: the auto-gate sets status=keep. Advance the
   branch (the commit stays). Update your "current best" mental model.
9. **If equal or worse, OR n_seeds < 10**: status=discard. Revert:
   ```
   git reset --hard HEAD~1
   ```
   The fmc_mutable.py reverts to the previous commit's content; results.tsv
   keeps the row for history.

10. **Special signal — blocker fired**: if `blocker_fired > 0` for ANY
    achievement, regardless of Crafter score change, this is a SIGNIFICANT
    finding. Log it loudly:
    ```
    --- BLOCKER FIRED ---
    description: <what you did> + blocker=<which one>
    ```
    The user is interested in unlocking these 4 (collect_diamond, make_iron_pickaxe,
    make_iron_sword, eat_plant). Even at lower Crafter score, blocker fire
    is publishable evidence and worth keeping.

### Timeout

Each experiment should take <=20 min (the wall budget) + ~30s startup. If a
run exceeds 25 min, kill it and treat as crash.

### Crashes

OOM, segfault, NaN — log "crash" status, revert, move on. The agent gets
roughly 60-80 GB of RAM on the test machine; configs >N=2048 OR >M=400 may
OOM the env.

### NEVER STOP

Once experimentation has begun, do NOT pause to ask the human if you should
continue. Do NOT say "should I keep going?". The human may be asleep,
expecting you to work until manually interrupted. You are autonomous.

If you run out of obvious ideas:
- Re-read the negative examples (run_005, 006, 007). Sometimes the falsified
  approach has a non-obvious twist that DOES work.
- Re-read MATH_CANON.md, especially the conjectures (A, B, C). Conjecture A
  (Sergio's branching ~6) was empirically falsified — but the underlying
  Wright-Fisher surface has unexplored regions.
- Try combining two near-misses.
- Try radical surgery (replace scanning policy entirely with a heuristic).
- Try macro-actions if you haven't already.

The user has invested ~5 hours of compute on this problem and wants to see
breakthrough or definitive saturation evidence. The loop runs until you
genuinely cannot find anything new to try.

## Tracking progress: hard targets

Soft milestones the user is interested in (in increasing difficulty):

- **30%+ Crafter** (any seed count, with at least n_seeds=10): clear win
  over baseline 29.27%.
- **Any blocker achievement at >5% rate**: cracking the v4-blockers
  (collect_diamond, make_iron_pickaxe, make_iron_sword, eat_plant).
  Even at 1 occurrence in 30 seeds it's a significant finding.
- **35%+ Crafter**: substantial advance, would be a clear new SOTA
  zero-training and beat Curious Replay (19.4% with 1M training) by +15pp.
- **40%+ Crafter** with at least one blocker firing: paper-grade.
- **50%+ Crafter**: matches human expert level — would be a major result.
- **58%+ Crafter**: beats EMERALD (10M training SOTA) — would be a Nature-tier
  result, considered nearly impossible at this time.

## Communication with the user

After every 5-10 experiments (or major milestone), the user may want a
brief status report. Format:

```
[Iteration N | branch=autoresearch/<tag> | wall=Xmin]
Best so far: X.XX% Crafter (description), commit XXXXXXX
Last 5 attempts:
  X.XX% description (status)
  ...
Next idea: <hypothesis>
```

But do NOT pause for confirmation. Continue immediately after sending the
status.

## Final checklist (do before first edit)

- [ ] On branch `autoresearch/<tag>` (not main!)
- [ ] Read the 6 reference docs above
- [ ] Read MATH_CANON Def 2-4
- [ ] Read fmc_mutable.py top-to-bottom
- [ ] Ran the baseline (~20 min) and confirmed Crafter ~29% with n_seeds >= 10
- [ ] results.tsv has at least the baseline row
- [ ] User confirmed "go"

Then start the loop.
