"""MCTS-UCT baseline against the fmc.envs.base.Environment protocol.

Sample-budget-controlled implementation so that FMC (N walkers x M horizon)
and MCTS-UCT can be compared at equal "samples per action" budget B.

Design notes
------------
A "sample" is one env.step call. For FMC, B = N * M. For MCTS-UCT, B is
the number of simulator queries (selection-descent + rollout) up to the
limit, charged once per env.step. This makes the wall-clock comparison
slightly unfavorable to MCTS (it does extra bookkeeping per sample), but
the *sample-efficiency* claim in paper 1803.05049v5 is in samples, not
seconds — so this matches the protocol P0 metric.

References
----------
- Kocsis & Szepesvari (2006), Bandit Based Monte Carlo Planning, ECML.
- Browne et al. (2012), A Survey of MCTS Methods, IEEE TCIAIG 4(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, List, Optional

import numpy as np

from fmc.envs.base import Environment


@dataclass
class _Node:
    state: object
    parent: Optional["_Node"] = None
    action_in: Optional[Hashable] = None
    children: dict = field(default_factory=dict)  # action -> _Node
    untried: List[Hashable] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    terminal: bool = False

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.total_reward / self.visits


def _ucb1(child: _Node, parent_visits: int, c: float) -> float:
    if child.visits == 0:
        return float("inf")
    return child.value + c * math.sqrt(math.log(parent_visits) / child.visits)


def plan(
    env: Environment,
    x0,
    sample_budget: int,
    rollout_depth: int = 30,
    c_uct: float = math.sqrt(2.0),
    discount: float = 1.0,
    seed: Optional[int] = None,
) -> Hashable:
    """Run MCTS-UCT from x0 with a hard sample budget B.

    Parameters
    ----------
    env : fmc.envs.base.Environment
        Same protocol as fmc.core.plan.
    x0
        Initial state (must be deep-cloneable via env.clone_state).
    sample_budget : int
        Hard cap on env.step calls. Equivalent to FMC's N * M.
    rollout_depth : int
        Max simulator steps per random rollout. The protocol charges
        each rollout step against sample_budget.
    c_uct : float
        UCB1 exploration constant. sqrt(2) is canonical.
    discount : float
        Gamma. 1.0 for finite-horizon Atari-style.
    seed : int or None
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    actions = list(env.actions())

    root = _Node(state=env.clone_state(x0), untried=list(actions))
    samples = 0

    def _is_terminal(state) -> bool:
        # CartPole-style: state.alive flag if present, else fall through.
        # For envs without explicit terminal flag, sample_budget bounds the
        # depth so this is safe.
        return getattr(state, "alive", True) is False

    while samples < sample_budget:
        # ---- 1. SELECTION ----
        node = root
        while not node.untried and node.children and not node.terminal:
            best_a, best_child, best_score = None, None, -float("inf")
            for a, child in node.children.items():
                score = _ucb1(child, node.visits, c_uct)
                if score > best_score:
                    best_score, best_a, best_child = score, a, child
            node = best_child

        # ---- 2. EXPANSION ----
        if node.untried and not node.terminal and samples < sample_budget:
            a = node.untried.pop(rng.integers(0, len(node.untried)))
            new_state = env.step(env.clone_state(node.state), a)
            samples += 1
            child = _Node(
                state=new_state,
                parent=node,
                action_in=a,
                untried=list(actions),
                terminal=_is_terminal(new_state),
            )
            node.children[a] = child
            node = child

        # ---- 3. SIMULATION (rollout) ----
        rollout_state = env.clone_state(node.state)
        rollout_return = 0.0
        gamma = 1.0
        steps_done = 0
        while (
            not _is_terminal(rollout_state)
            and steps_done < rollout_depth
            and samples < sample_budget
        ):
            a = env.sample_action(rollout_state, rng)
            rollout_state = env.step(rollout_state, a)
            samples += 1
            rollout_return += gamma * env.reward(rollout_state)
            gamma *= discount
            steps_done += 1

        leaf_value = (
            env.reward(node.state) if node.visits == 0 else node.value
        ) + rollout_return

        # ---- 4. BACKPROPAGATION ----
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_reward += leaf_value
            cur = cur.parent

    # Robust child policy: most-visited action at root.
    if not root.children:
        return actions[rng.integers(0, len(actions))]
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]
