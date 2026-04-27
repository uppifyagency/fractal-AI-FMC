"""Sanity tests for the rocket env: physics determinism, reward bounds,
end-to-end plan() invocation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmc.core import plan, effective_branching_factor
from fmc.envs.base import Environment
from fmc.envs.rocket import Rocket, State, ACTION_TABLE


def test_rocket_implements_protocol():
    assert isinstance(Rocket(), Environment)


def test_rocket_step_is_deterministic():
    env = Rocket()
    s = env.reset()
    s1 = env.step(s, 4)  # idle
    s2 = env.step(env.reset(), 4)
    assert s1.x == pytest.approx(s2.x)
    assert s1.y == pytest.approx(s2.y)


def test_rocket_dies_when_falling_into_floor():
    """No thrust, gravity pulls down — eventually hits a wall and dies."""
    env = Rocket()
    s = env.reset(x=450, y=H_minus(50))
    for _ in range(200):
        s = env.step(s, 1)  # thrust=0.05 (very weak), no torque
        if not s.alive:
            break
    assert not s.alive, "rocket should die from gravity into wall"


def H_minus(d):
    from fmc.envs.rocket import H
    return H - d


def test_rocket_alive_with_strong_thrust():
    """High thrust + initial nose-up should keep rocket alive for many ticks."""
    env = Rocket()
    s = env.reset()
    for _ in range(40):
        s = env.step(s, 7)  # high thrust, no torque
        if not s.alive:
            break
    # Survived initial gravity? Should still be alive.
    assert s.alive


def test_rocket_reward_bounded():
    """Reward must be in [0, ~6] (paper §2.2.2 + goal bonus)."""
    env = Rocket()
    s = env.reset()
    for _ in range(20):
        s = env.step(s, np.random.default_rng(0).integers(0, 9))
        if not s.alive:
            break
        r = env.reward(s)
        assert 0.0 <= r <= 7.0


def test_plan_runs_on_rocket_without_errors():
    """End-to-end: plan() returns a valid action."""
    env = Rocket()
    x0 = env.reset()
    a = plan(env, x0, N=32, M=15, alpha=1.0, beta=1.0, seed=0)
    assert a in env.actions()


def test_plan_produces_branching_in_expected_range():
    """With (alpha=0.1, beta=0) (Sergio's sweet spot per rocket REPORT) the
    chosen actions across multiple seeds should NOT all be the same.

    This is a soft sanity check — ensures the planner isn't degenerate.
    """
    env = Rocket()
    x0 = env.reset()
    chosen = []
    for seed in range(10):
        a = plan(env, x0, N=32, M=15, alpha=0.1, beta=0.0, seed=seed)
        chosen.append(a)
    n_distinct = len(set(chosen))
    assert n_distinct >= 2, f"Expected planner diversity, got always {chosen[0]}"
