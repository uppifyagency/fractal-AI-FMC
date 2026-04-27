"""End-to-end tests of plan() with the gridworld example."""

from __future__ import annotations

import numpy as np
import pytest

from fmc.core import plan, effective_branching_factor
from fmc.envs.base import Environment
from fmc.envs.gridworld import GridWorld, State


def test_environment_protocol():
    env = GridWorld(goal=10)
    assert isinstance(env, Environment)


def test_plan_finds_correct_direction_when_goal_right():
    """If goal is to the right of start, FMC should prefer +1."""
    env = GridWorld(goal=20, length=50)
    x0 = env.reset(start=0)
    # Run multiple seeds, expect majority to choose +1.
    chosen_right = 0
    N_TRIALS = 9
    for seed in range(N_TRIALS):
        a = plan(env, x0, N=64, M=20, alpha=1.0, beta=1.0, seed=seed)
        if a == +1:
            chosen_right += 1
    assert chosen_right >= int(N_TRIALS * 0.6), (
        f"Expected most plans to choose +1, got {chosen_right}/{N_TRIALS}"
    )


def test_plan_finds_correct_direction_when_goal_left():
    """Symmetric: goal to the left -> prefer -1."""
    env = GridWorld(goal=2, length=50)
    x0 = env.reset(start=30)
    chosen_left = 0
    N_TRIALS = 9
    for seed in range(N_TRIALS):
        a = plan(env, x0, N=64, M=20, alpha=1.0, beta=1.0, seed=seed)
        if a == -1:
            chosen_left += 1
    assert chosen_left >= int(N_TRIALS * 0.6), (
        f"Expected most plans to choose -1, got {chosen_left}/{N_TRIALS}"
    )


def test_plan_is_deterministic_given_seed():
    """Same seed -> same plan output."""
    env = GridWorld(goal=15)
    x0 = env.reset(start=5)
    a1 = plan(env, x0, N=32, M=15, seed=7)
    a2 = plan(env, x0, N=32, M=15, seed=7)
    assert a1 == a2
