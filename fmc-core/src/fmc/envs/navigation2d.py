"""2D navigation in a continuous box, K=9 discrete directions.

State: (x, y) in [0, 1]^2.
Actions: 9 = {N, NE, E, SE, S, SW, W, NW, stay}.
Reward: -L2 distance to goal.
Terminal: at goal (within radius) or out of bounds.

Designed as a second non-trivial task with K=9 (same arity as rocket) for
Conjecture A testing across tasks (Bet 3 of Level 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# 8 compass directions + stay, all unit step in step-size space.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
ACTION_DIRS = (
    (0.0,  1.0),               # N    0
    (_INV_SQRT2, _INV_SQRT2),  # NE   1
    (1.0,  0.0),               # E    2
    (_INV_SQRT2, -_INV_SQRT2), # SE   3
    (0.0, -1.0),               # S    4
    (-_INV_SQRT2, -_INV_SQRT2),# SW   5
    (-1.0, 0.0),               # W    6
    (-_INV_SQRT2, _INV_SQRT2), # NW   7
    (0.0,  0.0),               # stay 8
)

STEP_SIZE = 0.04
GOAL_RADIUS = 0.05


@dataclass
class State:
    x: float
    y: float
    goal_x: float
    goal_y: float
    alive: bool


class Navigation2D:
    """Implements fmc.envs.base.Environment."""

    def __init__(self, goal_x: float = 0.85, goal_y: float = 0.85):
        self.goal_x = goal_x
        self.goal_y = goal_y

    def reset(self, x: float = 0.15, y: float = 0.15) -> State:
        return State(x=x, y=y, goal_x=self.goal_x, goal_y=self.goal_y, alive=True)

    def actions(self):
        return tuple(range(9))

    def clone_state(self, state: State) -> State:
        return State(
            x=state.x, y=state.y,
            goal_x=state.goal_x, goal_y=state.goal_y,
            alive=state.alive,
        )

    def step(self, state: State, action: int) -> State:
        if not state.alive:
            return self.clone_state(state)
        dx, dy = ACTION_DIRS[action]
        nx = state.x + dx * STEP_SIZE
        ny = state.y + dy * STEP_SIZE
        # Out of bounds -> die.
        alive = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        if not alive:
            return State(x=state.x, y=state.y, goal_x=state.goal_x, goal_y=state.goal_y, alive=False)
        return State(x=nx, y=ny, goal_x=state.goal_x, goal_y=state.goal_y, alive=True)

    def reward(self, state: State) -> float:
        if not state.alive:
            return 0.0
        d = math.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        # Higher reward closer to goal; bonus when reached.
        r = max(0.0, 1.0 - d)
        if d < GOAL_RADIUS:
            r += 5.0
        return r

    def observe(self, state: State) -> np.ndarray:
        return np.array([state.x, state.y], dtype=np.float64)

    def sample_action(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 9))
