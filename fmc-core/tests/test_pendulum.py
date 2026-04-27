"""Tests for Pendulum env."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmc.core import plan
from fmc.envs.base import Environment
from fmc.envs.pendulum import Pendulum, State, ACTION_TORQUES


def test_pendulum_implements_protocol():
    assert isinstance(Pendulum(), Environment)


def test_pendulum_step_is_deterministic():
    env = Pendulum()
    s = State(theta=math.pi / 2, theta_dot=0.0, alive=True)
    s1 = env.step(s, 4)  # zero torque
    s2 = env.step(env.clone_state(s), 4)
    assert s1.theta == pytest.approx(s2.theta)
    assert s1.theta_dot == pytest.approx(s2.theta_dot)


def test_pendulum_reward_max_at_upright():
    """Reward should be highest near theta = 0 (upright)."""
    env = Pendulum()
    r_up = env.reward(State(theta=0.0, theta_dot=0.0, alive=True))
    r_down = env.reward(State(theta=math.pi, theta_dot=0.0, alive=True))
    r_side = env.reward(State(theta=math.pi / 2, theta_dot=0.0, alive=True))
    assert r_up > r_side > r_down


def test_pendulum_gravity_pulls_down():
    """No torque applied to horizontal pendulum -> gains downward velocity.

    Standard gym convention: theta=0 is upright, theta=pi (or -pi) is down.
    Starting at theta=pi/2 with zero velocity, gravity should produce
    downward angular velocity (toward theta=pi).
    """
    env = Pendulum()
    s = State(theta=math.pi / 2, theta_dot=0.0, alive=True)
    for _ in range(50):
        s = env.step(s, 4)  # zero torque
    # After 50 steps with only gravity, theta_dot must be non-zero (gained energy).
    assert abs(s.theta_dot) > 0.5


def test_plan_runs_on_pendulum():
    env = Pendulum()
    x0 = env.reset(seed=0)
    a = plan(env, x0, N=32, M=15, alpha=1.0, beta=1.0, seed=0)
    assert a in env.actions()
