"""Environment protocol for fmc-core.

A user environment must implement:

  - actions()         -> Iterable[Hashable]    : the discrete action set A
  - clone_state(s)    -> S                     : deep copy / serialization of state
  - step(s, a)        -> S                     : apply action, return new state
  - observe(s)        -> np.ndarray            : observation used for distance
  - reward(s)         -> float                 : raw reward (pre-relativize)
  - sample_action(s, rng) -> a                 : scanning policy (default: uniform)

This is intentionally minimal. No batching, no JAX, no GPU. For research
performance, use fragile or fragile-rl. For correctness reference, use this.
"""

from __future__ import annotations

from typing import Iterable, Hashable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Environment(Protocol):
    def actions(self) -> Iterable[Hashable]:
        ...

    def clone_state(self, state):
        ...

    def step(self, state, action):
        ...

    def observe(self, state) -> np.ndarray:
        ...

    def reward(self, state) -> float:
        ...

    def sample_action(self, state, rng: np.random.Generator):
        ...
