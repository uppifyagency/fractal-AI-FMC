"""Statistical tests for theorems in MATH_CANON.

These tests are necessarily probabilistic — we use enough seeds + tolerance
to make false-failure rate < 1%. They are slow (~5-10 sec each).
"""

from __future__ import annotations

import numpy as np
import pytest

from fmc.core import (
    relativize,
    virtual_reward,
    clone_step,
    effective_branching_factor,
)


# ---------------------------------------------------------------------------
# Theorem 3 — anti-collapse lemma (qualitative)
# ---------------------------------------------------------------------------

def test_high_alpha_collapses_swarm():
    """Th. 3 operational form: with strong reward pressure (large alpha) and
    weak perturbation, the swarm collapses toward b_eff ~ 1.

    This is the falsifiable consequence: raising alpha monotonically reduces
    b_eff. Verified on the rocket task in
    work/07_sergio_branching_sweep/REPORT.md §3.
    """
    N, M = 64, 30

    def run(alpha_value: float, seed: int) -> float:
        local_rng = np.random.default_rng(seed)
        positions = local_rng.standard_normal(N) * 5.0
        labels = local_rng.integers(0, 9, size=N)
        for _ in range(M):
            partners = local_rng.permutation(N)
            for i in range(N):
                if partners[i] == i:
                    partners[i] = (i + 1) % N
            rewards = -np.abs(positions - 3.0)
            states = positions.reshape(-1, 1)
            vr = virtual_reward(rewards, states, partners, alpha=alpha_value, beta=0.0)
            idx = clone_step(vr, local_rng)
            positions = positions[idx]
            labels = labels[idx]
            positions += local_rng.standard_normal(N) * 0.1
        return effective_branching_factor(labels)

    b_eff_a3 = np.mean([run(3.0, s) for s in range(20)])
    b_eff_a0 = np.mean([run(0.0, s) for s in range(20)])
    # alpha=3 should produce strong collapse compared to alpha=0.
    assert b_eff_a3 < b_eff_a0, (
        f"Expected alpha=3 to collapse swarm more than alpha=0, "
        f"got b_eff(alpha=3)={b_eff_a3:.2f} vs b_eff(alpha=0)={b_eff_a0:.2f}"
    )
    # Strong-alpha should be quite low (palmera-ish).
    assert b_eff_a3 < 4.0, f"Expected b_eff(alpha=3) < 4 (collapse), got {b_eff_a3:.2f}"


def test_alpha_zero_preserves_diversity():
    """alpha=0 (Common Sense) -> b_eff stays high (no collapse to greedy)."""
    rng = np.random.default_rng(0)
    N, M = 64, 30

    def run(alpha_value: float, seed: int) -> float:
        local_rng = np.random.default_rng(seed)
        positions = local_rng.standard_normal(N) * 5.0
        labels = local_rng.integers(0, 9, size=N)
        for _ in range(M):
            partners = local_rng.permutation(N)
            for i in range(N):
                if partners[i] == i:
                    partners[i] = (i + 1) % N
            rewards = -np.abs(positions - 3.0)
            states = positions.reshape(-1, 1)
            vr = virtual_reward(rewards, states, partners, alpha=alpha_value, beta=1.0)
            idx = clone_step(vr, local_rng)
            positions = positions[idx]
            labels = labels[idx]
            positions += local_rng.standard_normal(N) * 0.1
        return effective_branching_factor(labels)

    b_eff_a0 = np.mean([run(0.0, s) for s in range(20)])
    b_eff_a1 = np.mean([run(1.0, s) for s in range(20)])
    # Common Sense should preserve more branching than greedy.
    assert b_eff_a0 > b_eff_a1, (
        f"Expected alpha=0 to preserve more diversity than alpha=1, "
        f"got b_eff(alpha=0)={b_eff_a0:.2f} vs b_eff(alpha=1)={b_eff_a1:.2f}"
    )
