"""Navigation2D parameterized by K (action arity).

Same physics as navigation2d.py, but K compass directions evenly spaced on
the circle. Used to measure how b_eff* scales with K under Sergio's config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


STEP_SIZE = 0.04
GOAL_RADIUS = 0.05


@dataclass
class State:
    x: float
    y: float
    goal_x: float
    goal_y: float
    alive: bool


class Navigation2DKN:
    """Implements fmc.envs.base.Environment with arbitrary K >= 2."""

    def __init__(self, K: int, goal_x: float = 0.85, goal_y: float = 0.85):
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        self.K = K
        self.goal_x = goal_x
        self.goal_y = goal_y
        self._dirs = tuple(
            (math.cos(2 * math.pi * i / K), math.sin(2 * math.pi * i / K))
            for i in range(K)
        )

    def reset(self, x: float = 0.15, y: float = 0.15) -> State:
        return State(x=x, y=y, goal_x=self.goal_x, goal_y=self.goal_y, alive=True)

    def actions(self):
        return tuple(range(self.K))

    def clone_state(self, state: State) -> State:
        return State(
            x=state.x, y=state.y,
            goal_x=state.goal_x, goal_y=state.goal_y,
            alive=state.alive,
        )

    def step(self, state: State, action: int) -> State:
        if not state.alive:
            return self.clone_state(state)
        dx, dy = self._dirs[action]
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
        return int(rng.integers(0, self.K))
