# FMC plugin smoke test — FizzBuzz

This is a minimal repo to verify the `fractal-coding-loop` plugin runs end-to-end.

## Acceptance criteria

Implement `fizzbuzz(n)` in `src/fizzbuzz.py` such that all 5 tests in `tests/test_fizzbuzz.py` pass.

```bash
pytest -v
```

## What the plugin should do

1. Generate ~3 candidate strategies for implementing fizzbuzz
2. Spawn 3 walkers in isolated git worktrees
3. Each walker implements its strategy + runs tests
4. M=3 ticks of (perturbation + cloning) refine the swarm
5. The winning strategy's first commit is cherry-picked back here

## Verification after the run

```bash
pytest -v                                  # all 5 tests pass
git log --oneline                          # one commit applied to main
git worktree list                          # only main remains (no orphans)
ls .fractal/sessions/                      # state files exist for audit
```
