"""Python port of the 2D rocket environment from simulations/rocket_validated.html.

Same physics constants, same reward composition (paper §2.2.2 multiplicative form).
Discrete action space: 9 buckets (3 thrust * 3 torque).

This is the third env shipped with fmc-core, after gridworld. Used as the first
non-trivial test of plan() on a multi-step planning task with terminal states
(walls, goal).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import math

import numpy as np


# ---- Physics constants (mirror JS exactly) ----
W = 900
H = 540
GRAVITY = 0.14
DRAG = 0.992
A_DRAG = 0.95
MAX_THRUST = 0.40
MAX_TORQUE = 0.08

WALLS = [
    (0, 0, W, 8),
    (0, H - 8, W, 8),
    (0, 0, 8, H),
    (W - 8, 0, 8, H),
    (200, 120, 30, 220),
    (400, 280, 200, 30),
    (680, 100, 30, 280),
    (100, 400, 200, 30),
    (550, 460, 200, 30),
]
GOAL = (W - 100, 60, 30)  # (x, y, r)
MAX_DIAG = math.sqrt(W * W + H * H)


# 9-bucket action grid: (thrust_bucket, torque_bucket) -> (thrust, torque) center.
# thrust buckets: low(0.05), mid(0.18), high(0.32)
# torque buckets: left(-0.05), zero(0.0), right(+0.05)
ACTION_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.05, -0.05),  # 0
    (0.05,  0.00),  # 1
    (0.05, +0.05),  # 2
    (0.18, -0.05),  # 3
    (0.18,  0.00),  # 4
    (0.18, +0.05),  # 5
    (0.32, -0.05),  # 6
    (0.32,  0.00),  # 7
    (0.32, +0.05),  # 8
)


@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    v_angle: float
    alive: bool
    fuel: float


def _point_in_walls(x: float, y: float, rad: float = 4.0) -> bool:
    for wx, wy, ww, wh in WALLS:
        if x + rad > wx and x - rad < wx + ww and y + rad > wy and y - rad < wy + wh:
            return True
    return False


def _dist_to_nearest_wall(x: float, y: float) -> float:
    best = float("inf")
    for wx, wy, ww, wh in WALLS:
        dx = max(wx - x, 0.0, x - (wx + ww))
        dy = max(wy - y, 0.0, y - (wy + wh))
        d = math.sqrt(dx * dx + dy * dy)
        if d < best:
            best = d
    return best


def _dist_to_goal(s: State) -> float:
    gx, gy, _ = GOAL
    return math.sqrt((s.x - gx) ** 2 + (s.y - gy) ** 2)


class Rocket:
    """Implements fmc.envs.base.Environment."""

    def reset(self, x: float = 80.0, y: float = H - 80.0) -> State:
        return State(
            x=x, y=y,
            vx=0.0, vy=0.0,
            angle=-math.pi / 2,
            v_angle=0.0,
            alive=True,
            fuel=1000.0,
        )

    def actions(self):
        return tuple(range(len(ACTION_TABLE)))

    def clone_state(self, state: State) -> State:
        return replace(state)

    def step(self, state: State, action_idx: int) -> State:
        if not state.alive:
            return self.clone_state(state)
        thrust, torque = ACTION_TABLE[action_idx]
        thrust = max(0.0, min(MAX_THRUST, thrust))
        torque = max(-MAX_TORQUE, min(MAX_TORQUE, torque))

        v_angle = (state.v_angle + torque) * A_DRAG
        angle = state.angle + state.v_angle + torque  # use updated v before damping for symmetry with JS
        # JS does: vAngle += torque; angle += vAngle; vAngle *= A_DRAG.
        # Reproduce exactly:
        v_angle_new = state.v_angle + torque
        angle_new = state.angle + v_angle_new
        v_angle_final = v_angle_new * A_DRAG

        vx_new = (state.vx + math.cos(angle_new) * thrust) * DRAG
        vy_new = (state.vy + math.sin(angle_new) * thrust + GRAVITY) * DRAG
        x_new = state.x + vx_new
        y_new = state.y + vy_new
        fuel_new = max(0.0, state.fuel - thrust * 5)

        if _point_in_walls(x_new, y_new):
            return State(
                x=x_new, y=y_new, vx=vx_new, vy=vy_new,
                angle=angle_new, v_angle=v_angle_final,
                alive=False, fuel=fuel_new,
            )
        return State(
            x=x_new, y=y_new, vx=vx_new, vy=vy_new,
            angle=angle_new, v_angle=v_angle_final,
            alive=True, fuel=fuel_new,
        )

    def reward(self, state: State) -> float:
        if not state.alive:
            return 0.0
        clearance = _dist_to_nearest_wall(state.x, state.y)
        r_clear = min(1.0, clearance / 50.0)
        d_goal = _dist_to_goal(state)
        r_progress = 1.0 - d_goal / MAX_DIAG  # [0, 1]
        reward = 1.0 * r_clear * (1.0 + r_progress)
        if d_goal < GOAL[2]:
            reward += 5.0
        return reward

    def observe(self, state: State) -> np.ndarray:
        return np.array([
            state.x / W,
            state.y / H,
            state.vx * 0.05,
            state.vy * 0.05,
            math.cos(state.angle) * 0.5,
            math.sin(state.angle) * 0.5,
            state.v_angle * 0.3,
        ], dtype=np.float64)

    def sample_action(self, state, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(ACTION_TABLE)))
