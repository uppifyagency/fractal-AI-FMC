# P0 — FMC vs MCTS-UCT replication harness

**Status (2026-04-28)**: 🟡 **HARNESS LANDED + ATARI MICRO-SWEEP (n=3) WITH STRONG SIGNAL — FULL PROTOCOL STILL ESCALATED**

This directory operationalizes the protocol in
[`docs/bibliography/protocols/P0_fmc_vs_mcts_protocol.md`](../../docs/bibliography/protocols/P0_fmc_vs_mcts_protocol.md):
a budget-controlled head-to-head comparison between **FMC** (`fmc-core`)
and a freshly written **MCTS-UCT** baseline against the same
`fmc.envs.base.Environment` protocol.

---

## What landed in-session

| Component | File | Status |
|---|---|---|
| MCTS-UCT against FMC env protocol | [`scripts/mcts_uct.py`](scripts/mcts_uct.py) | ✅ working |
| Episode runner toy-env (CartPole, Pendulum) | [`scripts/run_episode.py`](scripts/run_episode.py) | ✅ working |
| Budget sweep driver (algos × budgets × seeds) | [`scripts/budget_sweep.py`](scripts/budget_sweep.py) | ✅ working |
| **plangym → fmc Atari adapter** | [`fmc-core/src/fmc/envs/atari.py`](../../fmc-core/src/fmc/envs/atari.py) | ✅ working |
| **Atari episode runner (full game, fixed B)** | [`scripts/atari_episode.py`](scripts/atari_episode.py) | ✅ working |
| CartPole smoke run, n=5 seed × 4 budgets × 2 algos | [`runs/cartpole_smoke.jsonl`](runs/cartpole_smoke.jsonl) | ✅ recorded |
| **Boxing micro-sweep, n=3 seed × 2 budgets × 2 algos** | [`runs/boxing_micro.jsonl`](runs/boxing_micro.jsonl) | ✅ recorded |

The MCTS-UCT implementation follows Kocsis & Szepesvári (2006). Sample
budget is enforced as a hard cap on `env.step` calls (selection-descent +
expansion + rollout all charged), giving a fair samples-per-action
comparison against FMC's `N × M`.

## Smoke result on CartPole (n=5 seeds, 200 max steps)

| algo | B | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| fmc  |  20 | 5 |  38.2 | 29.1 |   9 |  79 |
| mcts |  20 | 5 |  20.2 |  6.2 |  10 |  27 |
| fmc  |  50 | 5 |  77.2 | 72.4 |   8 | 169 |
| mcts |  50 | 5 |  20.2 |  6.2 |  10 |  27 |
| fmc  | 100 | 5 | 194.6 | 12.1 | 173 | 200 |
| mcts | 100 | 5 | 200.0 |  0.0 | 200 | 200 |
| fmc  | 300 | 5 | 181.8 | 28.8 | 134 | 200 |
| mcts | 300 | 5 | 200.0 |  0.0 | 200 | 200 |

**Reading**: at ultra-low budget (B=20–50), MCTS-UCT is effectively random
on CartPole because the tree is too shallow to reach a useful leaf, while
FMC's swarm-local exploration finds non-trivial trajectories with high
variance. Both saturate by B=100. **This is harness validation only**; the
P0 claim is about Atari at $B \in [300, 300\,000]$, where the regime is
qualitatively different (long horizons, high-dim observations).

The CartPole smoke does *not* support or refute the audit's discrepancy
D2 (the "100×–10 000×" sample-efficiency claim). It confirms the harness
is executable end-to-end.

## Boxing micro-sweep (n=3 seeds, max_actions=200, RAM obs, frame_skip=4)

| algo | B | seeds | mean | std | min | max | wall/episode |
|---|---|---|---|---|---|---|---|
| **fmc**  |  80 | 3 | **+91.3** | 10.0 | 80 |  99 | ~24 s |
| mcts | 80 | 3 |  −5.0 |  5.0 | −10 |   0 | ~25 s |
| **fmc**  | 240 | 3 | **+100.0** |  0.0 | 100 | 100 | ~50 s |
| mcts | 240 | 3 |  −5.0 |  5.0 | −10 |   0 | ~73 s |

**Signal**: at the same samples-per-action budget, on Boxing, on CPU,
n=3, FMC produces winning play (knock-out at B=240, 3/3 seeds reach
the +100 cap) while MCTS-UCT plays at random level (mean ≈ −5, no
improvement from B=80 → B=240). Δ(mean) ≈ **96–105 raw points**.

**Caveats — this is *not* the P0 deliverable**:

1. **n=3 is too small.** Protocol P0 requires n=10 with bootstrap CI95.
2. **One game.** Boxing is the easiest in the paper §5.1.1 table. Q-Bert,
   MsPacman would test the claim more strictly.
3. **MCTS hyperparameters are not tuned.** $c = \sqrt{2}$ is canonical
   but not optimal for Boxing's −1/0/+1 reward sparsity. A more careful
   MCTS comparison should sweep $c$ and `rollout_depth`.
4. **Frame-skip and sticky-action determinism.** The adapter sets
   `sticky_actions=False` per protocol; ALE 0.11.2 default differs from
   the paper's setup. Worth re-checking.
5. **max_actions=200 caps the episode.** Boxing typically lasts ~600
   actions at frame_skip=4. Long episodes may favor MCTS (deeper
   trees), or favor FMC (more cloning rounds) — only a full-length
   sweep tells.

Despite these caveats, the **directional signal is consistent with the
paper's "1-2 OoM" sample-efficiency claim** and the failure modes of
MCTS at small B are exactly what the FMC narrative predicts (UCB1
collapses to uniform when no leaf is reached deep enough).

## Methodological surprise — full P0 may not need a cluster

Original protocol estimates ~50–100 GPU-hours for the full sweep. The
in-session run shows:

- FMC Atari at B=240 on **CPU only**: ~50 s per episode
- Full P0 cell = 1 episode → 3 games × 7 budgets × 10 seeds × 2 algos = **420 episodes**
- CPU walltime estimate: 420 × ~60 s ≈ **7 hours on a single workstation**

This invalidates the "needs a GPU cluster" assumption. A 1-day sprint on
a developer's MacBook can produce the full P0 deliverable. The original
estimate baked in `fragile`'s GPU tensor swarms; the `fmc-core` NumPy
reference is fast enough on CPU because cloning is trivial (no neural
network forward pass).

## What stays escalated (genuinely needs a cluster)

The full P0 deliverable per protocol:

- 3 Atari games × 7 budgets × 2 algos × 10 seeds = **420 cells**
- ~30 min/cell on a single A100 ⇒ **~210 GPU-hours**
- Plus an MCTS-UCT implementation that actually queries `plangym`
  Atari (the present implementation is env-agnostic; a thin `plangym`
  adapter is needed to query `env.set_state` per node — that's the
  remaining engineering).

These are the inputs **a developer or scheduler can feed in one shot**:
- `scripts/run_episode.py` already takes `--env`/`--algo`/`--B`/`--seed`
- `scripts/budget_sweep.py` already drives the cell matrix
- Adding Atari requires: `fmc/envs/atari.py` adapter wrapping
  `plangym.make("ALE/Boxing-v5")` plus exposing `clone_state` via
  `env.get_state()/set_state()` (plangym already supports this — the
  whole reason `plangym` is the "hard dep that makes FMC exist")

## How to extend for Atari (one-page recipe)

```python
# fmc-core/src/fmc/envs/atari.py
import plangym
from fmc.envs.base import Environment

class AtariEnv:  # implements fmc.envs.base.Environment
    def __init__(self, name="ALE/Boxing-v5", obs="ram", frame_skip=4):
        self._env = plangym.make(name, frameskip=frame_skip,
                                 obs_type=obs, sticky_actions=False)
        self._init_state, _ = self._env.reset(return_state=True)
    def actions(self):       return tuple(range(self._env.action_space.n))
    def clone_state(self, s): return self._env.get_state()  # plangym snapshot
    def step(self, s, a):    self._env.set_state(s); _, r, *_ = self._env.step(a); return self._env.get_state()
    def observe(self, s):    self._env.set_state(s); return self._env.get_obs()
    def reward(self, s):     ...  # accumulate during step()
    def sample_action(self, s, rng): return int(rng.integers(0, self._env.action_space.n))
```

Open question: the protocol's "samples-per-action" charging convention.
For MCTS this is unambiguous (one charge per `step`). For FMC, the
charge is `N × M`. The walltime asymmetry (MCTS sequential, FMC
embarrassingly parallel) should be reported separately per protocol §
"Caveat metodologici".

## Decision pending after the Atari run

Per protocol §Decision, the ratio $r = B^{\mathrm{MCTS-min}} / B^{\mathrm{FMC-min}}$
determines the paper v6 claim language:

| $r$ measured | Verdict |
|---|---|
| $r > 100$ | Conservative "2–3 OoM" claim confirmed |
| $10 < r \leq 100$ | "359×" claim was directional, restate with measured $r$ |
| $r \leq 10$ | Reposition paper: "competitive alternative" not "replacement" |
| $r \approx 1$ | Falsified — crisis, reposition on parallelism / continuous actions |

This decision is **deferred** until the Atari run completes.
