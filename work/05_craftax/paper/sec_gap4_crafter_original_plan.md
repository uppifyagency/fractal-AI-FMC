# Gap 4 — Cross-benchmark replication on Crafter-original

> Scaffolding doc for the next agent. Full Gap 4 execution = 2–4 weeks
> per PAPER_HANDOFF; this file gets you to first port and first 5-seed
> smoke test in ~1 day.

## Why Crafter-original

Crafter-original \citep{hafner2021crafter} is the canonical, **non-JAX**
Python implementation of the Crafter benchmark. It uses the *same* 22
achievements, *same* score formula, *same* episode cap (10 000 env steps),
but with a slower simulator (~10× slower than Craftax-Classic per step).

Replicating Conjecture D on Crafter-original converts our claim from
"Craftax-Classic-specific recipe" to "law candidate that holds across
benchmark implementations of the same underlying world". Without this
step, reviewers will and should reject the generality claim.

## Install

```bash
pip install crafter
# crafter pulls: numpy, opensimplex, pillow, ruamel.yaml
# add to project's pyproject.toml [tool.poetry.dependencies] section
```

Verify:

```python
import crafter
env = crafter.Env()
obs = env.reset()           # returns (H,W,3) uint8 image, NOT JAX state
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(env.action_names)     # 17 actions, same enum as Craftax
print(info['achievements']) # dict of 22 booleans
```

## API differences vs Craftax-Classic-Symbolic-v1

| Aspect | Craftax-Classic | Crafter-original |
|---|---|---|
| Backend | JAX, vectorisable | Python, single-env |
| Observation | symbol tensor (49,) | image (64,64,3) |
| State exposure | direct dataclass (`state.inventory`, `state.player_position`) | hidden (only via `info`) |
| Step semantics | `reset(rng, params); step(rng, state, action, params)` | `reset(); step(action)` |
| Determinism | per-seed PRNGKey | per-seed `np.random.seed` |
| Action enum | 17 actions, same names | 17 actions, same names |
| Episode cap | 500 (we use) / 10 000 (full) | 10 000 default |
| Reward | env step reward + achievement gain | env step reward + achievement gain |
| Achievements format | `state.achievements` boolean array | `info['achievements']` dict |

The **most consequential** difference for FMC porting: Crafter-original
does NOT expose a programmable get/set state interface, so we cannot run
N=512 walkers in parallel by replicating state. We need an **MCTS-style
deterministic-replay** approach:

1. Save the per-step action sequence from root.
2. To branch a walker, replay actions from `state_0` up to current step.
3. Apply walker's chosen action.

This is correct but slow — ~$N \cdot t$ env steps per FMC tick instead of
$N$ in Craftax. Mitigations:

- **Use plangym's `crafter` adapter** if available (install plangym + check
  `repos/plangym` for crafter backend status).
- **Cache state snapshots** if the env exposes `__getstate__` / `__setstate__`
  (crafter does — it's just slow because of the Python state).
- **Reduce N** for the port: $N = 64, M = 20$ for the 5-seed smoke test;
  scale up if Conjecture D shows monotonic compounding even at small N.

## Step-by-step plan (estimated 1 day for first signal)

### Day 1 morning — port FMC harness

```bash
mkdir -p work/05_crafter_original_port
cp work/05_craftax/autoresearch/{prepare_craftax,fmc_mutable}.py \
   work/05_crafter_original_port/
```

Edits to `prepare_crafter.py` (renamed):

- Replace `from craftax.craftax_env import make_craftax_env_from_name`
  with `import crafter; env = crafter.Env()`.
- Replace JAX-style `env.reset(rng, params)` calls with `env.reset()`.
- Replace `env.action_space(params).n` with `env.action_space.n`.
- Add the **state-replay shim**: a small `CrafterStateProxy` class that
  records the action sequence from `reset()` and exposes
  `branch_from(action_seq) -> new_env` by replaying.
- Replace `state.achievements`, `state.inventory.*`, `state.player_position`
  with `info['achievements']`, dict access on the inventory inside `info`,
  and `info['player_position']` (or compute from observation).

### Day 1 afternoon — smoke test

```python
# run 5 seeds with N=64, M=20 v4-style baseline (NO shaping)
python evaluate_30seed.py --out_json results/crafter_baseline_5seed.json \
    --n_seeds 5 --wall_budget_s 1800
```

Expected: Crafter score ~5–10 % (PPO baselines achieve 4.6 % at 1 M
training steps; zero-training FMC v4 typically lands in this range
without shaping).

### Day 2 — apply exp17 shaping

Apply the *same* `INV_TIER_WEIGHTS` and `ACH_WEIGHTS_LIST` from
`fmc_mutable.py` (Craftax exp17 final). 5-seed smoke test should land
between 25 % and 45 % (lower than 50.95 % because the smaller N=64 shrinks
the population's exploration breadth, and we expect linear-in-N drop in
absolute Crafter score under FMC theory).

### Day 3 — Conjecture D ablation

Run the **stack-level test**:

| stage | weights | expected on Crafter-original |
|---|---|---|
| k=0 (ach-fire only) | env reward + ach bonus, no inv-tier | baseline |
| k=1 (+ iron-tier inv) | + iron×16, coal×8, diamond×64 | + 1–3 pp |
| k=2 (+ stone-tier inv) | + stone×4, stone tools×12 | + 0.5–2 pp |
| k=3 (+ wood-tier inv) | + wood×2, wood tools×6 | + 0.5–2 pp |

If monotonic Δ at every step → Conjecture D confirmed cross-benchmark
→ paper Section "Cross-benchmark replication" is now defensible.
If NOT monotonic → the conjecture is **falsified beyond Craftax**, and
the paper pivots to a "Craftax-specific shaping recipe" workshop note.

### Day 4 — full 30-seed run + statistical test

Same Gap 1 + Gap 2 procedure as Craftax: 30 seeds × N=64 × M=20 = ~5 hours
on M1 Pro (Crafter-original is ~10× slower per step, but smaller N=64 vs
512 = ~8× faster, net ~1.25× total slowdown).

## Templated Crafter-FMC harness (drop-in)

```python
# work/05_crafter_original_port/fmc_crafter.py
"""FMC port for Crafter-original. State-replay branching pattern."""
import crafter
import numpy as np

CRAFTER_ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword", "place_plant", "defeat_zombie", "collect_stone",
    "place_stone", "eat_plant", "defeat_skeleton", "collect_iron",
    "collect_coal", "place_furnace", "collect_diamond", "wake_up",
]


class CrafterReplay:
    """Snapshot-and-replay state proxy. Slow but compatible."""

    def __init__(self, seed: int):
        self.seed = seed
        self.action_history: list[int] = []
        self._env = crafter.Env(seed=seed)
        self._reset()

    def _reset(self):
        self._obs = self._env.reset()
        self._info = {}
        self._cum_r = 0.0
        self._done = False

    def step(self, action: int) -> tuple:
        self.action_history.append(int(action))
        self._obs, r, self._done, self._info = self._env.step(action)
        self._cum_r += float(r)
        return self._obs, r, self._done, self._info

    def branch(self) -> "CrafterReplay":
        """Return a fresh CrafterReplay reproducing the same trajectory
        up to the current step."""
        twin = CrafterReplay(self.seed)
        for a in self.action_history:
            twin.step(a)
        return twin


def fmc_crafter_decide(root_state: CrafterReplay,
                        N: int, M: int, n_actions: int = 17,
                        ach_weights: np.ndarray | None = None,
                        inv_tier_weights: dict | None = None,
                        rng: np.random.Generator | None = None) -> int:
    """Single FMC decision tick. Returns the argmax-vote action."""
    rng = rng or np.random.default_rng()
    walkers = [root_state.branch() for _ in range(N)]
    init_actions = rng.integers(0, n_actions, size=N)
    cum_rewards = np.zeros(N)

    for t in range(M):
        actions = init_actions if t == 0 else rng.integers(0, n_actions, size=N)
        for i, w in enumerate(walkers):
            if not w._done:
                _, r, _, info = w.step(int(actions[i]))
                cum_rewards[i] += r
                # apply shaping (omitted: identical to fmc_mutable.py)

        # relativize + cloning (identical structure to fmc_mutable.py)
        # ... (port from JAX -> NumPy)

    # vote
    votes = np.bincount(init_actions, weights=cum_rewards, minlength=n_actions)
    return int(votes.argmax())
```

The above sketch is enough to get the port to compile; the relativize +
cloning loop is direct NumPy translation of `fmc_mutable.py` lines 130–250.

## Risks and mitigation

| Risk | Mitigation |
|---|---|
| Crafter state-replay too slow | Reduce N to 32, M to 15 for first run; scale up if Conjecture D signal visible |
| Different action semantics | Verify `env.action_names` matches Craftax; both should be the 17-action default |
| Different reward function | Crafter-original has env-step reward = +1 per achievement; Craftax similar. Confirm with smoke test reward histogram |
| RNG mismatch causing non-reproducibility | Use `crafter.Env(seed=42)` deterministic seed; verify identical action history → identical outcome on re-run |
| `info['achievements']` format inconsistent | Check both possible APIs: dict-of-bool vs list-of-str; normalize at the harness layer |

## Deliverables for paper Section "Cross-benchmark replication"

After Day 4 of this plan, you will have:

1. `results/crafter_v4_30seed.json` — Crafter-original baseline.
2. `results/crafter_exp17_30seed.json` — exp17 weights ported.
3. `results/crafter_stack_ablation.json` — k=0,1,2,3 monotonicity check.
4. Paper Figure 6: same trajectory plot as Figure 1 but on Crafter-original;
   ideally shows the same monotonic compounding pattern.
5. Paper Section "Cross-benchmark replication" (~1 page) summarising
   numerical match with Craftax conjecture D within seed noise.

If 5 fails (no monotonicity), keep the negative result — it strengthens
the paper as a falsification rather than weakens it.
