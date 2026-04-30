# autoresearch-FMC

> Autonomous research framework for evolving the FMC algorithm on Craftax-Classic.
> Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## What this is

An LLM agent (Claude Code, Codex, etc.) sits in this directory and iterates
on `fmc_mutable.py` to improve the Crafter score above the current SOTA of
29.27% (run 007 baseline). Each experiment runs for a fixed wall budget of
**20 minutes**, after which the result is auto-recorded to `results.tsv`.
Improvements are kept (branch advances), regressions are reverted.

The agent never asks for permission between iterations. It runs until you
manually stop it.

## Files

```
autoresearch/
├── prepare_craftax.py        # FROZEN: env + eval harness + Crafter score formula
├── fmc_mutable.py            # MUTABLE: the file the agent edits
├── evaluate.py               # driver: imports fmc_mutable, runs eval, logs to TSV
├── program_fmc.md            # the agent's "skill" — its instructions and rules
├── results.tsv               # log of all experiments (untracked by git)
└── README.md                 # this file
```

## Design choices (and why they differ from Karpathy's defaults)

| Choice | Karpathy autoresearch | autoresearch-FMC | Why |
|---|---|---|---|
| Mutable file | `train.py` (LLM training) | `fmc_mutable.py` (FMC planner) | Different problem |
| Metric | val_bpb (lower better) | Crafter score (higher better, 0-100) | Hafner 2021 |
| Wall budget | 5 min | 20 min | Crafter eval is per-seed, needs more time for stable CI |
| Auxiliary signals | none | `blocker_fired`, `n_seeds_completed`, `mean_ach +/- CI95` | Decision-gate + statistical robustness |
| Throughput | ~12 exp/h on H100 | ~3 exp/h on CPU (~24/8h overnight) | CPU vs H100 |
| Required reads | README, train.py, prepare.py | + MATH_CANON, run_005, run_006, run_007, addendum | More accumulated negative evidence |
| Theory invariants | none | 15 unit tests gate "is this still FMC?" (level B authority) | Avoid resurrecting falsified ideas |

## Quick start

### Prerequisites

```
Python 3.11.7
JAX 0.10.0 + jaxlib 0.10.0 (CPU backend)
craftax 1.5.0
```

If you've successfully run `work/05_craftax/scripts/test_fmc_theory.py` before,
the environment is correctly set up.

### Manual sanity check (~25 min)

Verify the harness works on the baseline:

```bash
cd work/05_craftax/autoresearch
PY=/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python
JAX_PLATFORMS=cpu $PY prepare_craftax.py            # ~1 sec, prints env info
JAX_PLATFORMS=cpu $PY evaluate.py --description "manual sanity"  # ~20 min
```

Expected: `crafter_pct` in 28-30% range, `n_seeds_completed` ~10, `blocker_fired=0/4`.

### Spinning up the agent

In Claude Code or equivalent, in a fresh session, invoke:

```
You are running autoresearch-FMC. Read program_fmc.md and execute it.

Branch tag: <propose one>
```

The agent will:
1. Create branch `autoresearch/<tag>`
2. Read the docs
3. Run a baseline (~20 min)
4. Loop forever, editing `fmc_mutable.py`, running experiments, keeping
   improvements, reverting regressions.

You walk away. When you come back, check:
- `results.tsv` for the experiment log
- `git log autoresearch/<tag>` for kept commits
- The current `fmc_mutable.py` for the latest best implementation

### Stopping the agent

- Ctrl-C in the agent's terminal — it's autonomous, that's how you stop it.
- Or, set a soft TIME_BUDGET in your prompt ("run for 4 hours, then stop").

### Merging a winning experiment back to main

After the session, manually:

```bash
git checkout main
# Inspect the work
git log autoresearch/<tag>
git diff main autoresearch/<tag> -- fmc_mutable.py

# If you like a specific commit, cherry-pick it
git cherry-pick <commit-sha>

# Or merge the entire branch
# git merge --no-ff autoresearch/<tag>
```

DON'T merge blindly — review the agent's edits. The agent works toward Crafter
score, but the user is responsible for code quality + theory soundness.

## Safety constraints

The agent operates with these implicit guardrails (built into program_fmc.md):

1. **Never touches `main` branch** — only `autoresearch/<tag>`
2. **Never modifies `prepare_craftax.py`** (FROZEN: changes here invalidate
   the metric)
3. **Never modifies `test_fmc_theory.py`** (FROZEN: theory invariants)
4. **No new dependencies** (use what's installed in the venv)
5. **No external network calls** beyond what JAX/Craftax already do
6. **20-min wall budget cap** — won't get stuck on a single config

The user retains:
- Manual review of all edits via `git log` / `git diff`
- Power to reset, cherry-pick, merge, or discard the entire branch

## Levels of agent freedom (from program_fmc.md)

- **Level A — Shaping only**: agent only tweaks reward shaping params
- **Level B — Algorithm + shaping** (default for this setup): agent can edit
  the entire `fmc_mutable.py`. If it keeps MATH_CANON Def 2-4 invariants the
  result is "still FMC". If not, it's labeled "non-FMC variant" in description.
- **Level C — Full freedom**: agent can edit anything. Reserved for explicit
  user authorization.

This setup ships with **Level B**.

## Reading the results

`results.tsv` schema:

```
commit  crafter_pct  n_seeds  mean_ach  ach_ci95  blocker_fired  status  description
```

- `crafter_pct`: 0-100 (higher better)
- `blocker_fired`: 0-4 (number of v4-blockers with non-zero rate)
- `status`: `keep` | `discard` | `crash`

Filter for kept improvements:
```bash
awk -F'\t' '$7 == "keep"' results.tsv | sort -k2 -n -r
```

Filter for blocker fires (rare, important):
```bash
awk -F'\t' '$6 != "0"' results.tsv
```

## Connection to FMC theory

The autoresearch loop is *itself* an instance of FMC operating on code space:

| FMC concept | Autoresearch equivalent |
|---|---|
| Walker swarm | Single git branch (N=1, greedy MH) |
| Scanning policy | LLM proposing edits to fmc_mutable.py |
| Cone causale | Reachable programs via incremental edits |
| Reward R | Crafter score |
| Cloning kernel | Git keep/revert (Metropolis-Hastings) |
| Time horizon M | Number of overnight iterations |

So when the agent improves FMC's algorithm, FMC is also improving itself
(via the recursive loop). Whether this leads anywhere remains empirical.
