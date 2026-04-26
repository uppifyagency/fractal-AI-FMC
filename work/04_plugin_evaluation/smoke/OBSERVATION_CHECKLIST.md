# Observation checklist — smoke test of /fractal-decide

> *Run this in a separate terminal in `/tmp/fmc-smoke-repo` while Claude Code executes the plugin.*

## Pre-flight (before launching /fractal-decide)

- [ ] Plugin installed: `ls ~/.claude/plugins/fractal-coding-loop/`
- [ ] Math test passes: `python3 ~/.claude/plugins/fractal-coding-loop/tests/test_fractal_math.py | tail -3`
- [ ] E2E test passes: `bash ~/.claude/plugins/fractal-coding-loop/tests/e2e_test.sh | tail -5`
- [ ] pytest available: `pytest --version`
- [ ] Repo state clean: `git status` shows clean tree
- [ ] On main branch: `git branch --show-current` shows `main`

## During Phase 1-3 (strategy generation + walker spawn)

After Claude Code reports "spawning 3 walkers in parallel":

- [ ] `git worktree list` shows 3 NEW worktrees (4 total including main)
- [ ] `ls -la /tmp/fmc-smoke-repo/.fractal/sessions/` shows a session directory
- [ ] Each walker worktree has `src/fizzbuzz.py` and `tests/test_fizzbuzz.py`

## During Phase 4-5 (record tick 0 → step → cloning → continuation)

For each tick t in 1..M-1:

- [ ] After `step --seed N`: `cat .fractal/sessions/*/state.json | python3 -m json.tool | grep -A5 '"decisions"' | head -30` shows ESS, vrs, clone_plan
- [ ] If `cloning_skipped: true`: walker worktrees unchanged at this tick
- [ ] If `cloning_skipped: false`: at least one walker's HEAD now matches another walker's HEAD (verify via `git -C <path> rev-parse HEAD` for each walker)

## During Phase 6 (final decide)

- [ ] `decide` output shows: `winner_label`, `winner_init_commit_sha`, `confidence`, `vote_distribution`
- [ ] Confidence ≥ 0.50 (otherwise plugin should warn user before applying)

## Phase 7-8 (cherry-pick to main)

- [ ] After cherry-pick: `git log --oneline` on main shows 2 commits (initial + winner)
- [ ] The new commit's SHA matches the `winner_init_commit_sha` reported by decide (NOTE: cherry-pick produces a NEW SHA, so check by message: `git log -1 --format=%s` should match the walker's init commit message)
- [ ] `pytest -v` ALL 5 TESTS PASS
- [ ] `git diff HEAD~..HEAD --stat` shows changes only in `src/fizzbuzz.py`

## Phase 9 (cleanup)

- [ ] `git worktree list` shows ONLY main (no fractal-walker-* leftovers)
- [ ] `git branch -a` shows no `fractal-walker-*` branches
- [ ] `.fractal/sessions/<id>/state.json` PRESERVED for audit (this is by design)

## Failure modes to log

If any of the above fails, capture:

1. Exact step that failed
2. Output / error message (full, not truncated)
3. State at the moment of failure (`git status`, `git worktree list`)
4. The session state JSON (`.fractal/sessions/<id>/state.json`)
5. Any walker outputs that look malformed

Add the failure to `RUN_LOG.md`.

## Total expected time

- Phases 0-2: ~30s
- Phase 3 (3 walkers in parallel): ~2-3 min
- Phase 5 × 2 ticks: ~2-3 min × 2 = ~5 min
- Phase 6-9: ~1 min
- **TOTAL: ~8-10 min wall time, ~$3-5 cost**
