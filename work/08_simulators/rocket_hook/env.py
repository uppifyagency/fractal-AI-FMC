"""RocketHookEnv — plangym PlanEnv subclass for Sergio's F23 demo.

A 2D rocket tethered to a grappling hook by an elastic rope. Scenario
(from VideoTranscriptSergio.md and work/02_deep_dives/08, F23):

  1. The rocket starts mid-air. A stone falls under gravity.
  2. The hook is attached to the rocket by an elastic spring (rest length L0,
     stiffness k). The rocket can thrust and rotate; the hook follows
     ballistically + spring force.
  3. Reward (bipartite, F23):
       - Phase 1 (stone NOT yet attached):  R = 1 / dist(hook, stone)
       - Phase 2 (stone attached):          R = 1 / dist(hook, target_circle_center)
     The hook auto-grabs the stone when within HOOK_GRAB_RADIUS, and the
     stone auto-deposits when over the target circle.
  4. Termination: rocket crash (y < 0), arena exit, or success
     (stone deposited).

Action space: Discrete(6) — combinations of {no-thrust, thrust} × {no-rot, left, right}
State (15 floats):
  rocket: x, y, vx, vy, theta, omega
  hook:   hx, hy, vhx, vhy
  stone:  sx, sy, svx, svy
  flag:   stone_phase  (0.0 = falling, 1.0 = on hook, 2.0 = delivered)

Observation = state (coords). Image = small top-down RGB raster (64x64x3 uint8).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from plangym.core import PlanEnv


# --------------------------------------------------------------------- physics
G = 9.81                # gravity (m/s^2), pointing -y
STONE_G_FRAC = 0.18     # stone falls much slower than rocket — fits FMC horizons
DT_PHYS = 0.04          # physics step (sec) — 25 Hz
THRUST_ACC = 22.0       # rocket thrust acceleration (m/s^2)
ROT_TORQUE = 5.0        # angular acceleration from rotation actions (rad/s^2)
ANG_DAMPING = 0.85      # angular velocity damping per step
LIN_DAMPING = 0.995     # linear damping per step (air drag, very mild)
HOOK_MASS = 1.0
ROCKET_MASS = 4.0
SPRING_K = 30.0         # rope stiffness
SPRING_REST = 1.5       # rope rest length
SPRING_MAX = 4.0        # rope max length (clamped)
HOOK_GRAB_RADIUS = 0.6  # hook auto-grabs stone within this distance
TARGET_RADIUS = 1.0     # target circle radius
ARENA_X = (-15.0, 15.0)
ARENA_Y = (0.0, 18.0)

ACTION_TABLE = np.array(
    [
        # [thrust, rot]   rot > 0 ⇒ θ increases (rocket tilts CCW), thrust pushes LEFT
        # rot < 0 ⇒ θ decreases (CW), thrust pushes RIGHT.
        [0.0, 0.0],   # 0: noop (free fall + previous angular vel)
        [0.0, +1.0],  # 1: rotate left  (thrust would push LEFT)
        [0.0, -1.0],  # 2: rotate right (thrust would push RIGHT)
        [1.0, 0.0],   # 3: thrust straight up
        [1.0, +1.0],  # 4: thrust + tilt left
        [1.0, -1.0],  # 5: thrust + tilt right
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------- packing
STATE_DIM = 15

_FIELDS = (
    "x", "y", "vx", "vy", "theta", "omega",
    "hx", "hy", "vhx", "vhy",
    "sx", "sy", "svx", "svy",
    "stone_phase",
)
assert len(_FIELDS) == STATE_DIM


def _pack(d: dict) -> np.ndarray:
    return np.asarray([d[k] for k in _FIELDS], dtype=np.float64)


def _unpack(state: np.ndarray) -> dict:
    return {k: float(state[i]) for i, k in enumerate(_FIELDS)}


# ---------------------------------------------------------------------- env
class RocketHookEnv(PlanEnv):
    """Rocket + elastic hook + falling stone + delivery target.

    Implements the four PlanEnv abstract methods: apply_action, apply_reset,
    get_state, set_state. Adds a tiny rasterizer in get_image() so that
    fragile.FractalTree can render the best walker.
    """

    STATE_IS_ARRAY = True
    OBS_IS_ARRAY = True
    SINGLETON = False

    def __init__(
        self,
        name: str = "RocketHook-v0",
        frameskip: int = 1,
        autoreset: bool = True,
        delay_setup: bool = False,
        return_image: bool = False,
        target_xy: tuple[float, float] = (10.0, 1.5),
        seed: int | None = None,
        max_steps: int = 600,
    ):
        self.target_xy = np.asarray(target_xy, dtype=np.float64)
        self._np_random = np.random.default_rng(seed)
        self._max_steps = int(max_steps)
        self._step_counter = 0
        # Spaces are set in setup() so delay_setup=True still works.
        self.observation_space: gym.Space | None = None
        self.action_space: gym.Space | None = None
        self._state_vec = np.zeros(STATE_DIM, dtype=np.float64)
        super().__init__(
            name=name,
            frameskip=frameskip,
            autoreset=autoreset,
            delay_setup=delay_setup,
            return_image=return_image,
        )

    # -- plangym shape properties ------------------------------------------
    @property
    def obs_shape(self) -> tuple[int]:
        return (STATE_DIM,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return ()  # scalar discrete

    # -- plangym setup -----------------------------------------------------
    def setup(self) -> None:
        low = np.full(STATE_DIM, -np.inf, dtype=np.float32)
        high = np.full(STATE_DIM, np.inf, dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Discrete(len(ACTION_TABLE))

    # -- plangym serialization --------------------------------------------
    def get_state(self) -> np.ndarray:
        out = self._state_vec.copy()
        # Stash the step counter on a 16th slot so set_state can restore it.
        # We append it but keep STATE_DIM=15 for the obs vector — the env
        # tracks its own step_counter and the FMC swarm doesn't observe it.
        return np.concatenate([out, [float(self._step_counter)]])

    def set_state(self, state: np.ndarray) -> None:
        if state.shape[0] == STATE_DIM:
            self._state_vec = np.asarray(state, dtype=np.float64).copy()
            self._step_counter = 0
        else:
            self._state_vec = np.asarray(state[:STATE_DIM], dtype=np.float64).copy()
            self._step_counter = int(state[STATE_DIM])

    # -- plangym dynamics --------------------------------------------------
    def apply_reset(self, **kwargs):  # noqa: ARG002
        rng = self._np_random
        d = {k: 0.0 for k in _FIELDS}
        # Rocket starts to the left of the falling stone, with a small random
        # spread. Stone is positioned so the rocket has time to catch it.
        d["x"] = float(rng.uniform(-3.0, -1.0))
        d["y"] = float(rng.uniform(8.0, 10.0))
        d["theta"] = float(rng.uniform(-0.2, 0.2))
        # Hook hangs below the rocket on a slack rope.
        d["hx"] = d["x"]
        d["hy"] = d["y"] - SPRING_REST
        # Stone falling from above, slightly to the right of rocket.
        d["sx"] = float(rng.uniform(1.5, 3.5))
        d["sy"] = float(rng.uniform(13.0, 15.0))
        d["svy"] = float(rng.uniform(-0.2, 0.0))
        d["stone_phase"] = 0.0
        self._state_vec = _pack(d)
        self._step_counter = 0
        obs = self._observe()
        info = {}
        return obs, info

    def apply_action(self, action):
        a_idx = int(action)
        if not 0 <= a_idx < len(ACTION_TABLE):
            raise ValueError(f"invalid action {action}")
        thrust_on, rot_dir = ACTION_TABLE[a_idx]

        d = _unpack(self._state_vec)
        # Rotation
        d["omega"] = d["omega"] * ANG_DAMPING + ROT_TORQUE * rot_dir * DT_PHYS
        d["theta"] += d["omega"] * DT_PHYS
        # Thrust along rocket facing direction (theta=0 → +y up).
        ax = -np.sin(d["theta"]) * THRUST_ACC * thrust_on
        ay = np.cos(d["theta"]) * THRUST_ACC * thrust_on - G
        d["vx"] = (d["vx"] + ax * DT_PHYS) * LIN_DAMPING
        d["vy"] = (d["vy"] + ay * DT_PHYS) * LIN_DAMPING
        d["x"] += d["vx"] * DT_PHYS
        d["y"] += d["vy"] * DT_PHYS

        # Hook physics: gravity + elastic rope to rocket
        rope_x = d["hx"] - d["x"]
        rope_y = d["hy"] - d["y"]
        rope_len = float(np.hypot(rope_x, rope_y) + 1e-9)
        # Spring force on hook (pulled toward rocket if longer than rest)
        stretch = max(rope_len - SPRING_REST, 0.0)
        force_mag = SPRING_K * stretch
        fx_h = -force_mag * (rope_x / rope_len)
        fy_h = -force_mag * (rope_y / rope_len) - G * HOOK_MASS
        d["vhx"] = (d["vhx"] + (fx_h / HOOK_MASS) * DT_PHYS) * LIN_DAMPING
        d["vhy"] = (d["vhy"] + (fy_h / HOOK_MASS) * DT_PHYS) * LIN_DAMPING
        d["hx"] += d["vhx"] * DT_PHYS
        d["hy"] += d["vhy"] * DT_PHYS
        # Reaction force on rocket (Newton's third law) — softer due to mass ratio.
        d["vx"] -= (fx_h / ROCKET_MASS) * DT_PHYS
        d["vy"] -= (fy_h / ROCKET_MASS) * DT_PHYS
        # Hard clamp on rope length
        if rope_len > SPRING_MAX:
            scale = SPRING_MAX / rope_len
            d["hx"] = d["x"] + rope_x * scale
            d["hy"] = d["y"] + rope_y * scale

        # Stone physics
        phase = int(round(d["stone_phase"]))
        if phase == 0:
            # Free fall — reduced gravity so the stone gives FMC enough planning time.
            d["svy"] -= G * STONE_G_FRAC * DT_PHYS
            d["sx"] += d["svx"] * DT_PHYS
            d["sy"] += d["svy"] * DT_PHYS
            # Auto-grab if hook within radius
            if np.hypot(d["sx"] - d["hx"], d["sy"] - d["hy"]) < HOOK_GRAB_RADIUS:
                d["stone_phase"] = 1.0
        elif phase == 1:
            # Stone glued to hook
            d["sx"] = d["hx"]
            d["sy"] = d["hy"]
            d["svx"] = d["vhx"]
            d["svy"] = d["vhy"]
            # Auto-deposit if over target circle
            if np.hypot(d["sx"] - self.target_xy[0], d["sy"] - self.target_xy[1]) < TARGET_RADIUS:
                d["stone_phase"] = 2.0
        else:
            # Delivered: stone sits at target
            d["sx"] = self.target_xy[0]
            d["sy"] = self.target_xy[1]
            d["svx"] = 0.0
            d["svy"] = 0.0

        self._state_vec = _pack(d)
        self._step_counter += 1

        # Reward: F23 bipartite — exactly as Sergio describes in the seminar
        # (work/02_deep_dives/08 §F23): R = 1/dist before grab, 1/dist to target
        # after grab, big bonus on delivery. Amplitudes mildly scaled (×2) so
        # the relativize step has more dynamic range.
        if phase == 0:
            dist = np.hypot(d["sx"] - d["hx"], d["sy"] - d["hy"])
            reward = 2.0 / (dist + 1.0)
        elif phase == 1:
            dist = np.hypot(d["sx"] - self.target_xy[0], d["sy"] - self.target_xy[1])
            reward = 2.0 + 2.0 / (dist + 1.0)
        else:
            reward = 10.0  # delivery bonus

        # Termination
        terminal = False
        truncated = False
        if d["y"] <= 0.0:
            terminal = True  # rocket crashed
        elif not (ARENA_X[0] <= d["x"] <= ARENA_X[1]) or d["y"] > ARENA_Y[1]:
            terminal = True  # rocket out of arena
        elif phase == 0 and d["sy"] <= 0.0:
            terminal = True  # stone hit the ground — failure
        elif d["stone_phase"] >= 2.0:
            terminal = True  # success
        elif self._step_counter >= self._max_steps:
            truncated = True

        info = {"phase": phase, "step": self._step_counter}
        obs = self._observe()
        return obs, float(reward), bool(terminal), bool(truncated), info

    # -- helpers -----------------------------------------------------------
    def _observe(self) -> np.ndarray:
        return self._state_vec.astype(np.float32)

    def sample_action(self):
        return int(self._np_random.integers(0, len(ACTION_TABLE)))

    def get_image(self) -> np.ndarray:
        # Lazy import — avoids import-time cost when only stepping is used.
        from render import render_rocket_hook  # type: ignore[import-not-found]

        return render_rocket_hook(
            state=self._state_vec,
            target_xy=self.target_xy,
            arena_x=ARENA_X,
            arena_y=ARENA_Y,
        )
