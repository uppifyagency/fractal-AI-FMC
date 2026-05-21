"""Conjecture E / E1-LLM — shared machinery (no LLM).

Pre-registered design: E1_LLM_DESIGN.md. E1-LLM (Route B, schema S2) runs FMC
with the world-model M replaced — by an ablated true model (the f_abs sweep,
section 5) or by an LLM-generated transition (the full test, section 4.1). In
both cases the EPISODE runs on the true simulator; the swapped model is only the
internal world-model of the FMC planner. Metric: death rate.

This module is the part both steps share:
  - LAYOUTS6        the 6 E1 layouts (3 E1-base + 3 E1-robustness);
  - WorldModelEnv   wraps a true env, overriding step() with a transition_fn;
  - true_transition / make_ablated_transition   the f_abs knob (section 5);
  - fabs_probe      measures absorbing-structure fidelity of a transition_fn;
  - make_wm_fmc_policy   an FMC policy that plans on a WorldModelEnv.

Kernel fmc-core is UNCHANGED. WorldModelEnv(env, true_transition) is asserted to
drive plan() bit-identically to plan(env, ...) — the control of E1-LLM.
"""

from __future__ import annotations

import numpy as np

from e1_base import (  # noqa: E402  (e1_base sets up sys.path for fmc-core)
    BETA, HORIZON_M, MAX_STEPS, N_WALKERS, LAYOUTS as _E1_BASE_LAYOUTS,
    greedy_policy, random_policy, run_episode, two_prop_z,
)
from e1_robustness import LAYOUTS as _E1_ROBUST_LAYOUTS, wilson_ci  # noqa: E402
from gridworld_terminal import GOAL, LAVA, State, _DELTA, parse_layout  # noqa: E402
from fmc.core import plan  # noqa: E402

# The 6 E1 layouts, in the order E1_LLM_DESIGN section 4.1 lists them.
LAYOUTS6 = {**_E1_BASE_LAYOUTS, **_E1_ROBUST_LAYOUTS}   # gauntlet,lake,scatter,island,spur,archipelago

ALPHAS = [0.0, 0.1, 1.0]
N_EPISODES = 30


# === world-model wrapper =====================================================

class WorldModelEnv:
    """Wraps a true TerminalGridWorld; step() goes through `transition_fn`.

    transition_fn(r, c, done, action, grid, H, W) -> (nr, nc, ndone).
    Everything else (actions, reward, observe, clone_state, sample_action)
    delegates to the true env — per schema S2 of P13_DESIGN: the LLM / the
    ablation supplies the TRANSITION; reward and observation stay the true
    functions. The FMC planner sees only this; the episode runs on the true env.
    """

    def __init__(self, true_env, transition_fn):
        self.true = true_env
        self.transition_fn = transition_fn
        self.grid = true_env.grid
        self.H, self.W = true_env.H, true_env.W
        self.goal = true_env.goal

    def actions(self):                return self.true.actions()
    def clone_state(self, s):         return State(s.r, s.c, s.done)
    def observe(self, s):             return self.true.observe(s)
    def reward(self, s):              return self.true.reward(s)
    def sample_action(self, s, rng):  return self.true.sample_action(s, rng)

    def step(self, s, action):
        nr, nc, ndone = self.transition_fn(
            int(s.r), int(s.c), bool(s.done), int(action),
            self.grid, self.H, self.W)
        return State(int(nr), int(nc), bool(ndone))


# === transition functions ====================================================

def true_transition(r, c, done, action, grid, H, W):
    """Exact replica of TerminalGridWorld.step — the f_abs=1 control."""
    if done:
        return (r, c, True)
    dr, dc = _DELTA[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < H and 0 <= nc < W):
        nr, nc = r, c
    return (nr, nc, bool(grid[nr, nc] in (LAVA, GOAL)))


def make_ablated_transition(broken):
    """True transition, except absorbing cells in `broken` are NOT terminal —
    the f_abs knob (E1_LLM_DESIGN section 5). A broken lava cell is passable,
    exactly P13's abs-broken surrogate."""
    broken = set(broken)

    def transition(r, c, done, action, grid, H, W):
        if done:
            return (r, c, True)
        dr, dc = _DELTA[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            nr, nc = r, c
        is_term = bool(grid[nr, nc] in (LAVA, GOAL)) and (nr, nc) not in broken
        return (nr, nc, is_term)

    return transition


def absorbing_cells(env):
    """All (r,c) that are absorbing (lava or goal) in the true grid."""
    return [(r, c) for r in range(env.H) for c in range(env.W)
            if env.grid[r, c] in (LAVA, GOAL)]


# === absorbing-fidelity probe (E1_LLM_DESIGN section 3) ======================

def fabs_probe(transition_fn, env, seed=0):
    """f_abs — fraction of (x,a) probes where transition_fn predicts the terminal
    flag of x' correctly. Battery: every (cell,action) that steps ONTO an
    absorbing cell, plus an equal random sample of free-landing (cell,action)
    pairs. Pure transition test — no FMC, no episodes (design section 3)."""
    rng = np.random.default_rng(seed)
    onto_term, onto_free = [], []
    for r in range(env.H):
        for c in range(env.W):
            for a in env.actions():
                dr, dc = _DELTA[a]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < env.H and 0 <= nc < env.W):
                    nr, nc = r, c
                bucket = (onto_term if env.grid[nr, nc] in (LAVA, GOAL)
                          else onto_free)
                bucket.append((r, c, a))
    k = len(onto_term)
    free_sample = [onto_free[i] for i in rng.permutation(len(onto_free))[:k]]
    probes = onto_term + free_sample
    correct = term_correct = false_neg = false_pos = crashes = 0
    for (r, c, a) in probes:
        _, _, true_done = true_transition(r, c, False, a, env.grid, env.H, env.W)
        td = bool(true_done)
        try:
            _, _, cand_done = transition_fn(r, c, False, a, env.grid,
                                            env.H, env.W)
            cd = bool(cand_done)
        except Exception:
            crashes += 1                           # a crash counts as wrong
            cd = None
        if cd is not None and td == cd:
            correct += 1
            if td:
                term_correct += 1
        elif td:                                   # absorbing cell missed
            false_neg += 1
        elif cd is True:                           # free cell hallucinated absorbing
            false_pos += 1
    return {"f_abs": correct / len(probes), "n_probes": len(probes),
            "n_terminal_landing": k,
            "terminal_recall": term_correct / k if k else 1.0,
            "false_neg": false_neg, "false_pos": false_pos, "crashes": crashes}


# === FMC policy on a swapped world-model =====================================

def make_wm_fmc_policy(alpha, wm_env):
    """FMC policy that PLANS on wm_env. The episode runner passes the true env
    (ignored here): planning uses wm_env, the episode steps the true env."""
    def pol(_true_env, s, seed):
        return plan(wm_env, s, N=N_WALKERS, M=HORIZON_M,
                    alpha=alpha, beta=BETA, seed=seed)
    return pol


def summarize(outcomes):
    n = len(outcomes)
    d, g = outcomes.count("lava"), outcomes.count("goal")
    return {"n": n, "deaths": d, "goals": g,
            "timeouts": outcomes.count("timeout"),
            "death_rate": d / n, "goal_rate": g / n}


def assert_worldmodel_identity():
    """WorldModelEnv(env, true_transition) must drive plan() bit-identically to
    plan(true_env, ...) — the E1-LLM control (analogue of p13_proxy's sanity)."""
    env, start = parse_layout(LAYOUTS6["archipelago"])
    x0 = env.reset(start)
    wm = WorldModelEnv(env, true_transition)
    return all(
        plan(env, x0, N=N_WALKERS, M=HORIZON_M, alpha=a, beta=BETA, seed=sd)
        == plan(wm, x0, N=N_WALKERS, M=HORIZON_M, alpha=a, beta=BETA, seed=sd)
        for a in ALPHAS for sd in (1, 7, 99))
