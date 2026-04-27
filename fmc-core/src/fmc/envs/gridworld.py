"""Minimal 1D goal-reaching gridworld.

Used to validate the Environment protocol and end-to-end plan() loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTIONS = (-1, 0, +1)  # left, stay, right


@dataclass
class State:
    pos: int
    goal: int


class GridWorld:
    """Walker on Z, reward = -|pos - goal|, deterministic."""

    def __init__(self, goal: int = 10, length: int = 100):
        self.goal = goal
        self.length = length

    def reset(self, start: int = 0) -> State:
        return State(pos=start, goal=self.goal)

    def actions(self):
        return ACTIONS

    def clone_state(self, state: State) -> State:
        return State(pos=state.pos, goal=state.goal)

    def step(self, state: State, action: int) -> State:
        new_pos = max(0, min(self.length - 1, state.pos + action))
        return State(pos=new_pos, goal=state.goal)

    def observe(self, state: State) -> np.ndarray:
        return np.array([state.pos], dtype=np.float64)

    def reward(self, state: State) -> float:
        return -abs(state.pos - state.goal)

    def sample_action(self, state: State, rng: np.random.Generator) -> int:
        return ACTIONS[rng.integers(0, len(ACTIONS))]
