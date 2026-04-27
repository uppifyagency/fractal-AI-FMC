"""Pendulum swing-up.

Classic continuous-control task: swing an inverted pendulum to upright and
balance. Reward landscape is fundamentally different from rocket/navigation:

  - rocket: navigate around walls in position space.
  - navigation2D: minimize distance in position space.
  - pendulum: maximize angular potential energy (cos(theta)) AND minimize
    angular velocity. The reward landscape lives in (angle, angular velocity)
    state space and has a sharp peak at theta=0, not in position.

Continuous torque [-2, +2] discretized in 9 buckets to keep the same K=9
arity as rocket. This is the third task for testing Conjecture A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


GRAVITY = 10.0
M = 1.0
L = 1.0
DT = 0.05
MAX_SPEED = 8.0
MAX_TORQUE = 2.0


# 9-bucket discretization of torque in [-MAX_TORQUE, +MAX_TORQUE].
_LEVELS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
ACTION_TORQUES = tuple(t * MAX_TORQUE for t in _LEVELS)


@dataclass
class State:
    theta: float        # 0 = upright
    theta_dot: float
    alive: bool


def _angle_normalize(x: float) -> float:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


class Pendulum:
    """Implements fmc.envs.base.Environment."""

    def reset(self, seed: int | None = None) -> State:
        rng = np.random.default_rng(seed)
        # Start hanging down with small noise (theta near pi).
        theta = math.pi + rng.uniform(-0.2, 0.2)
        theta_dot = rng.uniform(-0.5, 0.5)
        return State(theta=_angle_normalize(theta), theta_dot=theta_dot, alive=True)

    def actions(self):
        return tuple(range(9))

    def clone_state(self, state: State) -> State:
        return State(theta=state.theta, theta_dot=state.theta_dot, alive=state.alive)

    def step(self, state: State, action: int) -> State:
        if not state.alive:
            return self.clone_state(state)
        u = ACTION_TORQUES[action]

        new_theta_dot = state.theta_dot + (
            -3.0 * GRAVITY / (2.0 * L) * math.sin(state.theta + math.pi)
            + 3.0 / (M * L * L) * u
        ) * DT
        # Clip angular velocity.
        if new_theta_dot > MAX_SPEED:
            new_theta_dot = MAX_SPEED
        elif new_theta_dot < -MAX_SPEED:
            new_theta_dot = -MAX_SPEED
        new_theta = _angle_normalize(state.theta + new_theta_dot * DT)
        return State(theta=new_theta, theta_dot=new_theta_dot, alive=True)

    def reward(self, state: State) -> float:
        # Standard gym pendulum: -theta^2 - 0.1*theta_dot^2 - 0.001*u^2
        # We drop the action cost (no action available here) and shift to
        # a strictly non-negative range:
        #   r = (1 + cos(theta))/2 - 0.01 * theta_dot^2
        # Range roughly [-0.64, 1].
        return 0.5 * (1.0 + math.cos(state.theta)) - 0.01 * state.theta_dot * state.theta_dot

    def observe(self, state: State) -> np.ndarray:
        return np.array(
            [math.cos(state.theta), math.sin(state.theta), state.theta_dot * 0.125],
            dtype=np.float64,
        )

    def sample_action(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 9))
