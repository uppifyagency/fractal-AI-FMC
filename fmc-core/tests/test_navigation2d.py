"""Tests for Navigation2D env."""

from __future__ import annotations

import numpy as np
import pytest

from fmc.core import plan
from fmc.envs.base import Environment
from fmc.envs.navigation2d import Navigation2D, State, ACTION_DIRS


def test_navigation2d_implements_protocol():
    assert isinstance(Navigation2D(), Environment)


def test_step_moves_correctly():
    env = Navigation2D()
    s = env.reset()
    s_E = env.step(s, 2)  # east
    assert s_E.x > s.x and s_E.y == pytest.approx(s.y)


def test_stay_action_keeps_position():
    env = Navigation2D()
    s = env.reset()
    s2 = env.step(s, 8)
    assert s2.x == pytest.approx(s.x) and s2.y == pytest.approx(s.y)


def test_dies_when_out_of_bounds():
    env = Navigation2D()
    s = env.reset(x=0.01, y=0.5)
    for _ in range(10):
        s = env.step(s, 6)  # west
        if not s.alive:
            break
    assert not s.alive


def test_plan_chooses_diagonal_to_goal():
    """Goal at (0.85, 0.85), start (0.15, 0.15) -> should prefer NE-ish."""
    env = Navigation2D(goal_x=0.85, goal_y=0.85)
    x0 = env.reset()
    chosen = []
    for seed in range(15):
        a = plan(env, x0, N=32, M=20, alpha=1.0, beta=1.0, seed=seed)
        chosen.append(a)
    # Most plans should pick NE (1) or one of N/E (0, 2). Reject if mostly stay or south.
    nethe = sum(1 for a in chosen if a in (0, 1, 2))
    assert nethe >= len(chosen) // 2, f"Expected mostly NE-ish, got {chosen}"
