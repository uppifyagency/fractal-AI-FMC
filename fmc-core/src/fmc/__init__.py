"""Fractal Monte Carlo — reference math layer.

See ../../README.md and ../../../docs/MATH_CANON.md for theory.
"""

from fmc.core import (
    relativize,
    virtual_reward,
    effective_sample_size,
    effective_branching_factor,
    clone_step,
    decide,
    plan,
)

__version__ = "0.1.0"
__all__ = [
    "relativize",
    "virtual_reward",
    "effective_sample_size",
    "effective_branching_factor",
    "clone_step",
    "decide",
    "plan",
]
