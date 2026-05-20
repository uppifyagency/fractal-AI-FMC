"""2-D gridworld with absorbing terminal states — substrate for Conjecture E, test E1-base.

Design discipline (see docs/MATH_CANON.md, Congettura E):
  - reward(s) = -manhattan(pos, goal) for EVERY cell, lava included.
  - Lava carries NO penalty and there is NO survival bonus. The ONLY thing
    special about a lava cell is that it is absorbing (terminal): once a walker
    steps onto lava, step() is a no-op forever.
  - The goal is also absorbing, but it is success, not death.

So any terminal-lava avoidance an FMC agent shows must EMERGE from the planning
dynamics (causal-entropy / diversity), not from reward shaping. That is exactly
the falsifiable content of E1.

Implements the fmc-core Environment protocol (fmc/envs/base.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cell types.
FREE, LAVA, GOAL = 0, 1, 2

# Actions: 0=up, 1=down, 2=left, 3=right, 4=stay.  K = 5.
ACTIONS = (0, 1, 2, 3, 4)
_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


@dataclass
class State:
    r: int
    c: int
    done: bool  # True on LAVA or GOAL — absorbing.


class TerminalGridWorld:
    """Walker on a 2-D grid with absorbing lava (death) and goal cells."""

    def __init__(self, grid: np.ndarray, goal_rc: tuple[int, int]):
        self.grid = np.asarray(grid, dtype=np.int64)
        self.H, self.W = self.grid.shape
        self.goal = goal_rc

    # --- Environment protocol -------------------------------------------------

    def actions(self):
        return ACTIONS

    def reset(self, start: tuple[int, int]) -> State:
        r, c = start
        return State(r, c, done=self._terminal(r, c))

    def clone_state(self, s: State) -> State:
        return State(s.r, s.c, s.done)

    def step(self, s: State, action: int) -> State:
        if s.done:
            return State(s.r, s.c, True)  # absorbing
        dr, dc = _DELTA[action]
        nr, nc = s.r + dr, s.c + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W):
            nr, nc = s.r, s.c  # off-grid: stay
        return State(nr, nc, done=self._terminal(nr, nc))

    def observe(self, s: State) -> np.ndarray:
        return np.array([s.r, s.c], dtype=np.float64)

    def reward(self, s: State) -> float:
        gr, gc = self.goal
        return -float(abs(s.r - gr) + abs(s.c - gc))

    def sample_action(self, s: State, rng: np.random.Generator) -> int:
        return ACTIONS[rng.integers(0, len(ACTIONS))]

    # --- helpers --------------------------------------------------------------

    def _terminal(self, r: int, c: int) -> bool:
        return self.grid[r, c] in (LAVA, GOAL)

    def cell(self, s: State) -> int:
        return int(self.grid[s.r, s.c])


def parse_layout(text: str) -> tuple[TerminalGridWorld, tuple[int, int]]:
    """Parse an ASCII layout. '.'=free  'L'=lava  'G'=goal  'S'=start."""
    rows = [ln for ln in text.strip("\n").splitlines()]
    H, W = len(rows), len(rows[0])
    grid = np.zeros((H, W), dtype=np.int64)
    start = goal = None
    for r, line in enumerate(rows):
        assert len(line) == W, f"ragged layout at row {r}"
        for c, ch in enumerate(line):
            if ch == "L":
                grid[r, c] = LAVA
            elif ch == "G":
                grid[r, c] = GOAL
                goal = (r, c)
            elif ch == "S":
                start = (r, c)
            elif ch != ".":
                raise ValueError(f"bad layout char {ch!r}")
    assert start is not None and goal is not None, "layout needs S and G"
    return TerminalGridWorld(grid, goal), start
