"""Classic CartPole environment.

Standard physics from Sutton & Barto / OpenAI Gym, but with no Gym dependency.
Action space: 2 (push left, push right).
Reward: +1 per timestep alive (balance).
Terminal: |angle| > 12 degrees or |position| > 2.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Standard cartpole constants.
GRAVITY = 9.8
M_CART = 1.0
M_POLE = 0.1
M_TOTAL = M_CART + M_POLE
LENGTH = 0.5  # half-pole
POLEMASS_LENGTH = M_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02

THETA_THRESHOLD = 12 * math.pi / 180  # 12 degrees
X_THRESHOLD = 2.4


@dataclass
class State:
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    alive: bool


class CartPole:
    """Implements fmc.envs.base.Environment."""

    def reset(self, seed: int | None = None) -> State:
        rng = np.random.default_rng(seed)
        # small uniform [-0.05, 0.05] perturbation as per gym.
        v = rng.uniform(-0.05, 0.05, size=4)
        return State(x=v[0], x_dot=v[1], theta=v[2], theta_dot=v[3], alive=True)

    def actions(self):
        return (0, 1)  # 0=left, 1=right

    def clone_state(self, state: State) -> State:
        return State(
            x=state.x, x_dot=state.x_dot,
            theta=state.theta, theta_dot=state.theta_dot,
            alive=state.alive,
        )

    def step(self, state: State, action: int) -> State:
        if not state.alive:
            return self.clone_state(state)

        force = FORCE_MAG if action == 1 else -FORCE_MAG
        cos_t = math.cos(state.theta)
        sin_t = math.sin(state.theta)

        temp = (force + POLEMASS_LENGTH * state.theta_dot ** 2 * sin_t) / M_TOTAL
        thetaacc = (GRAVITY * sin_t - cos_t * temp) / (
            LENGTH * (4.0 / 3.0 - M_POLE * cos_t * cos_t / M_TOTAL)
        )
        xacc = temp - POLEMASS_LENGTH * thetaacc * cos_t / M_TOTAL

        x_new = state.x + TAU * state.x_dot
        x_dot_new = state.x_dot + TAU * xacc
        theta_new = state.theta + TAU * state.theta_dot
        theta_dot_new = state.theta_dot + TAU * thetaacc

        alive = (
            -X_THRESHOLD <= x_new <= X_THRESHOLD and
            -THETA_THRESHOLD <= theta_new <= THETA_THRESHOLD
        )
        return State(x=x_new, x_dot=x_dot_new, theta=theta_new, theta_dot=theta_dot_new, alive=alive)

    def reward(self, state: State) -> float:
        return 1.0 if state.alive else 0.0

    def observe(self, state: State) -> np.ndarray:
        # Use the standard 4D obs vector. Distance metric is L2 in this space.
        return np.array(
            [state.x, state.x_dot, state.theta, state.theta_dot],
            dtype=np.float64,
        )

    def sample_action(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 2))
