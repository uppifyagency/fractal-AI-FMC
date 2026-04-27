"""Navigation2D variant with K=16 directions, used to test whether
Sergio's b_eff* ~ 6 is task-dependent or K-dependent.

If b_eff* still falls in [5, 7] when K=16 (instead of K=9), the constant
is plausibly universal. If it scales with K, it's an artifact of action
space size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# 16 evenly spaced directions on the unit circle.
ACTION_DIRS = tuple(
    (math.cos(2 * math.pi * i / 16), math.sin(2 * math.pi * i / 16))
    for i in range(16)
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


class Navigation2DK16:
    """Implements fmc.envs.base.Environment with K=16."""

    def __init__(self, goal_x: float = 0.85, goal_y: float = 0.85):
        self.goal_x = goal_x
        self.goal_y = goal_y

    def reset(self, x: float = 0.15, y: float = 0.15) -> State:
        return State(x=x, y=y, goal_x=self.goal_x, goal_y=self.goal_y, alive=True)

    def actions(self):
        return tuple(range(16))

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
        alive = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        if not alive:
            return State(x=state.x, y=state.y, goal_x=state.goal_x, goal_y=state.goal_y, alive=False)
        return State(x=nx, y=ny, goal_x=state.goal_x, goal_y=state.goal_y, alive=True)

    def reward(self, state: State) -> float:
        if not state.alive:
            return 0.0
        d = math.sqrt((state.x - state.goal_x) ** 2 + (state.y - state.goal_y) ** 2)
        r = max(0.0, 1.0 - d)
        if d < GOAL_RADIUS:
            r += 5.0
        return r

    def observe(self, state: State) -> np.ndarray:
        return np.array([state.x, state.y], dtype=np.float64)

    def sample_action(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 16))
