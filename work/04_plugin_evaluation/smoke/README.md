# Smoke test — fractal-coding-loop plugin

Minimal end-to-end test of `/fractal-decide`. Goal: verify the plugin runs without unexpected breakage on a trivial coding task. NOT a quality benchmark.

## What this test is and is NOT

**Is**: a sanity check. Does Phase 1-9 of the slash command actually execute? Do worktrees get created and cleaned up? Does the cherry-pick succeed? Do the pre-written tests pass after?

**Is NOT**: a quality comparison vs baseline. Not a variance study. Not a publishable result. See [`work/02_deep_dives/`](../../02_deep_dives/) for the deeper experiments planned.

The reasoning for the limited scope is documented in the conversation log preceding this fixture. Summary: with the plugin never having run on a real codebase, the right Phase 0 test is "does it run at all" — not "is it better than X." Quality benchmarks come after the smoke run reveals (and we fix) the inevitable first-time bugs.

## Files

| File | Purpose |
|---|---|
| `setup_smoke_repo.sh` | Creates a fresh `/tmp/fmc-smoke-repo` from the fixture |
| `fixture/` | Source files for the test repo (FizzBuzz scaffold + 5 acceptance tests) |
| `PROMPT.txt` | Exact prompt to paste into Claude Code |
| `OBSERVATION_CHECKLIST.md` | What to monitor during the run |
| `RUN_LOG.md` | Template to fill in with observations |

## How to run

```bash
# 1. Verify the plugin's certified layers still pass
python3 ../../plugin/fractal-coding-loop/tests/test_fractal_math.py | tail -3
bash    ../../plugin/fractal-coding-loop/tests/e2e_test.sh         | tail -5

# 2. Set up the smoke repo
bash ./setup_smoke_repo.sh

# 3. In a NEW Claude Code session opened in /tmp/fmc-smoke-repo:
#    paste the contents of PROMPT.txt
#    (with the fractal-coding-loop plugin installed and recognized)

# 4. Monitor via OBSERVATION_CHECKLIST.md (open in a separate terminal)

# 5. After completion:
cd /tmp/fmc-smoke-repo
pytest -v                # should pass 5/5
git log --oneline        # should show 2 commits (initial + winner)
git worktree list        # should show only main

# 6. Fill in RUN_LOG.md with observations
```

## Expected outcome

- 70% probability: the plugin runs to completion but with at least 1-2 issues to fix (path resolution, prompt parsing, JSON output format, worktree cleanup)
- 25% probability: the plugin fails partway through (most likely Phase 5 — the M-tick continuation phase has the most moving parts)
- 5% probability: clean run, all 5 tests pass, no observable issues

The 5% case is the optimistic-but-implausible scenario. If it happens, suspect we missed something — go re-read the run log carefully.

## Cost

- Sub-agent calls: 3 walkers × 3 ticks + 3 judge calls + ~5 orchestrator main-agent calls = ~17 LLM invocations
- Wall time: 8-10 minutes
- Approximate cost: $3-5 in Claude Code billing

## What to do with the results

After running, regardless of outcome:

1. Fill in `RUN_LOG.md` honestly
2. For each issue observed, file a one-line bug entry (in `RUN_LOG.md` § "Bugs / surprises")
3. If the test fails: prioritize fixes by Severity (Sev 0 = blocks any further use, Sev 1 = produces wrong results, Sev 2 = annoying but workaroundable)
4. After fixes, re-run this smoke test from scratch (`setup_smoke_repo.sh` rebuilds clean)
5. Only when smoke runs cleanly: consider scaling to a real benchmark (SWE-bench Lite, HumanEval+, or similar)

## Next steps after smoke passes

If and only if the smoke test passes cleanly:

- **Step 2 (Phase 0+)**: same fixture + `/octopus` (instead of `/fractal-decide`) → verify outer loop works on a single-iteration task
- **Step 3 (Phase 1)**: a multi-step task — e.g., a small Express CRUD API — to genuinely exercise `/octopus` over multiple cherry-picks
- **Step 4 (Phase 1+)**: real benchmark integration (SWE-bench Lite recommended)

The conversation log preceding this fixture has detailed criticism of why bigger experiments come AFTER smoke validation, not before.
