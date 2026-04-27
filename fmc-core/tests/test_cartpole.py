"""Tests for CartPole env."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmc.core import plan
from fmc.envs.base import Environment
from fmc.envs.cartpole import CartPole, State, THETA_THRESHOLD


def test_cartpole_implements_protocol():
    assert isinstance(CartPole(), Environment)


def test_cartpole_step_is_deterministic():
    env = CartPole()
    s = State(x=0.0, x_dot=0.0, theta=0.05, theta_dot=0.0, alive=True)
    s1 = env.step(s, 1)
    s2 = env.step(env.clone_state(s), 1)
    assert s1.x == pytest.approx(s2.x)
    assert s1.theta == pytest.approx(s2.theta)


def test_cartpole_dies_when_falling():
    """Pole tilted past threshold + no compensation -> dies."""
    env = CartPole()
    s = State(x=0.0, x_dot=0.0, theta=THETA_THRESHOLD * 0.9, theta_dot=2.0, alive=True)
    for _ in range(20):
        s = env.step(s, 1)  # pushing wrong way relative to fall
        if not s.alive:
            break
    assert not s.alive


def test_plan_runs_on_cartpole():
    env = CartPole()
    x0 = env.reset(seed=0)
    a = plan(env, x0, N=32, M=15, alpha=1.0, beta=1.0, seed=0)
    assert a in env.actions()
