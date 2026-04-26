# Installation — fractal-coding-loop

## Requirements

- **Claude Code** (recent version with plugin support)
- **Python 3.8+** (for the math and state-machine scripts)
- **git ≥ 2.5** (`git worktree` is used to isolate walkers)
- *Optional*: the test/lint toolchain of your target project (pytest, eslint, ruff, etc.)

## Step 1 — Install the plugin

### Option A — symlink (recommended for active development)

```bash
mkdir -p ~/.claude/plugins
ln -s "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/plugin/fractal-coding-loop" \
      ~/.claude/plugins/fractal-coding-loop
```

Edits to the plugin source are picked up live (after Claude Code restart).

### Option B — copy (for stable use)

```bash
cp -r "/Users/vladvrinceanu/Desktop/PROGETTI ANTYGRAVITY/FractalAI/plugin/fractal-coding-loop" \
      ~/.claude/plugins/
```

## Step 2 — Verify installation

Restart Claude Code, then in any session:

```
/help
```

You should see `/fractal-decide` and `/octopus` in the slash command list.

If they don't appear:
1. Check that the symlink/copy exists: `ls ~/.claude/plugins/fractal-coding-loop/`
2. Verify [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json) is valid JSON: `python3 -c "import json; json.load(open('~/.claude/plugins/fractal-coding-loop/.claude-plugin/plugin.json'))"`
3. Restart Claude Code completely (not just a new tab — fully quit and relaunch).

## Step 3 — Verify the math layer

Run the certification tests:

```bash
python3 ~/.claude/plugins/fractal-coding-loop/tests/test_fractal_math.py
```

Expected output ends with:
```
All FMC math tests passed — convergence certified.
```

If any test fails, **do not use the plugin** until fixed. The math layer is the foundation; failures there mean the algorithm is broken.

## Step 4 — Verify the state machine CLI

```bash
python3 ~/.claude/plugins/fractal-coding-loop/scripts/fractal_loop.py --help
python3 ~/.claude/plugins/fractal-coding-loop/scripts/fractal_reward.py --help
```

Both should print help without errors.

For a fuller smoke test (init + record + step + decide on synthetic walker JSONs), see [`docs/USAGE.md`](docs/USAGE.md) §"CLI smoke test".

## Step 5 — First real use

In a clean git repo (no uncommitted changes):

```
> /fractal-decide "add a hello world function in src/util.py and a unit test"
```

This will:
1. Generate 3 strategies
2. Spawn 3 walkers in parallel git worktrees
3. After ~3-5 minutes, present a comparison table
4. Ask if you want to cherry-pick the winner

**Cost estimate for this first run**: ~9 sub-agent calls (3 walker × 3 ticks) plus 3 judge calls = ~12 LLM invocations. Approximately $1-3 in Claude Code billing depending on plan and model.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/fractal-decide` not in `/help` | Plugin not registered | Verify symlink, restart Claude Code completely |
| `plugin.json` parse error | JSON syntax | Validate with `python3 -c "import json; json.load(open(...))"` |
| `ModuleNotFoundError: fractal_reward` | Path issue when running fractal_loop.py | Both scripts must be in the same directory; check they coexist in `scripts/` |
| Walker fails with "not a git repo" | Target dir has no git history | `git init && git commit --allow-empty -m initial` first |
| All walkers return same approach | Strategies in Phase 1 too similar | Edit the prompt in `fractal-decide.md` Phase 1 to enforce orthogonality |
| Confidence always < 30% | High reward variance with low N | Raise N (currently default 3, try 5-10 in `fractal_loop.py init --n 5`) |
| `git worktree remove` fails | Worktree busy or modified | `git worktree remove <path> --force` (data may be lost — only do this for failed walker worktrees) |
| Cherry-pick conflict in `/octopus` | Main has diverged from walker base | The octopus loop stops here — investigate the conflict, optionally `git reset --hard <START_HEAD>` to abort the whole session |
| Math tests fail | Python version too old, or fractal_reward modified | Use Python 3.8+; revert `fractal_reward.py` to the canonical version |

## Uninstall

```bash
rm ~/.claude/plugins/fractal-coding-loop          # if symlink
rm -rf ~/.claude/plugins/fractal-coding-loop      # if copied
```

The plugin writes session state into `<your-project>/.fractal/sessions/` of the target repo. To clean a project's accumulated state:

```bash
cd <your-project>
rm -rf .fractal/

# Remove any leftover walker worktrees:
git worktree list
git worktree remove <path>      # for each fractal-walker-* path
git branch -D fractal-walker-*  # for each lingering branch
```

There is no global state outside the plugin directory and per-project `.fractal/` directories.
