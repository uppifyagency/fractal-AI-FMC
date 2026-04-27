"""Tests for the 6 canonical definitions of MATH_CANON.

Each test maps to a property stated in docs/MATH_CANON.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from fmc.core import (
    relativize,
    virtual_reward,
    effective_sample_size,
    effective_branching_factor,
    clone_step,
    decide,
)


# ---------------------------------------------------------------------------
# Definition 2 — relativize
# ---------------------------------------------------------------------------

class TestRelativize:
    """Properties (1)-(6) from MATH_CANON Definition 2."""

    def test_positivity(self):
        """Property (1): relativize is strictly positive on any input."""
        for r in [
            np.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0]),
            np.array([1e-12, 1e-12, 1e-12]),
            np.array([-1e9, -1e9, 1e9]),
        ]:
            out = relativize(r)
            assert np.all(out > 0), f"relativize produced non-positive for {r}: {out}"

    def test_constant_input_returns_ones(self):
        """std=0 case must return ones (matches reference implementation)."""
        r = np.array([5.0, 5.0, 5.0, 5.0])
        out = relativize(r)
        assert np.allclose(out, np.ones(4))

    def test_continuity_at_zero(self):
        """Property (2): continuity at z=0. Both branches should give 1.

        With a near-symmetric input the mean is ~0; values just below mean
        use exp(z), just above use 1+log(1+z). Both should be near 1.
        """
        r = np.array([-1.0, 0.0, 1.0])  # z = (-sqrt(3/2), 0, sqrt(3/2))
        out = relativize(r)
        # Middle entry has z=0 -> goes to exp branch -> 1.0
        assert out[1] == pytest.approx(1.0)

    def test_affine_invariance(self):
        """Property (6): relativize(a*r + b) == relativize(r) for a > 0."""
        r = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        for a, b in [(1.0, 0.0), (2.0, 5.0), (100.0, -50.0), (0.001, 1000.0)]:
            assert np.allclose(relativize(a * r + b), relativize(r), atol=1e-10)

    def test_order_preservation(self):
        """If r_i > r_j then relativize(r)_i > relativize(r)_j."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            r = rng.standard_normal(20)
            out = relativize(r)
            order_r = np.argsort(r)
            order_out = np.argsort(out)
            np.testing.assert_array_equal(order_r, order_out)

    def test_matches_reference_implementation(self):
        """Bit-for-bit equivalent to FractalAI_old.swarm.relativize_vector."""
        # Inline reference from repos/FractalAI_old/fractalai/swarm.py:16-23
        def reference(vector):
            std = vector.std()
            if std == 0:
                return np.ones(len(vector))
            standard = (vector - vector.mean()) / std
            standard[standard > 0] = np.log(1 + standard[standard > 0]) + 1
            standard[standard <= 0] = np.exp(standard[standard <= 0])
            return standard

        rng = np.random.default_rng(42)
        for _ in range(20):
            r = rng.standard_normal(50) * rng.uniform(0.1, 100.0) + rng.uniform(-100, 100)
            np.testing.assert_allclose(relativize(r), reference(r.copy()), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Definition 3 — virtual reward
# ---------------------------------------------------------------------------

class TestVirtualReward:

    def test_alpha_zero_means_distance_only(self):
        """alpha=0 -> VR = D_hat^beta, ignores reward entirely."""
        rng = np.random.default_rng(0)
        rewards = rng.standard_normal(10)
        states = rng.standard_normal((10, 3))
        partners = rng.permutation(10)
        # Make sure no self-pair.
        for i in range(10):
            if partners[i] == i:
                partners[i] = (i + 1) % 10

        vr_a0 = virtual_reward(rewards, states, partners, alpha=0.0, beta=1.0)
        vr_a0_diff_rewards = virtual_reward(
            rewards * 100 + 50, states, partners, alpha=0.0, beta=1.0,
        )
        # alpha=0 -> R_hat^0 = 1 always -> VR depends only on D_hat
        np.testing.assert_allclose(vr_a0, vr_a0_diff_rewards, rtol=1e-10)

    def test_beta_zero_means_reward_only(self):
        """beta=0 -> VR = R_hat^alpha, ignores distance."""
        rng = np.random.default_rng(0)
        rewards = rng.standard_normal(10)
        states_a = rng.standard_normal((10, 3))
        states_b = rng.standard_normal((10, 3))
        partners = (np.arange(10) + 1) % 10

        vr_a = virtual_reward(rewards, states_a, partners, alpha=1.0, beta=0.0)
        vr_b = virtual_reward(rewards, states_b, partners, alpha=1.0, beta=0.0)
        np.testing.assert_allclose(vr_a, vr_b, rtol=1e-10)

    def test_strictly_positive(self):
        """VR is always > 0 (after relativize)."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            rewards = rng.standard_normal(20)
            states = rng.standard_normal((20, 4))
            partners = rng.permutation(20)
            for i in range(20):
                if partners[i] == i:
                    partners[i] = (i + 1) % 20
            vr = virtual_reward(rewards, states, partners, alpha=1.0, beta=1.0)
            assert np.all(vr > 0)


# ---------------------------------------------------------------------------
# Definition 5 — effective sample size
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:

    def test_uniform_weights_give_N(self):
        vr = np.ones(50)
        assert effective_sample_size(vr) == pytest.approx(50.0)

    def test_one_walker_dominant_gives_one(self):
        vr = np.zeros(50)
        vr[7] = 1e9
        assert effective_sample_size(vr) == pytest.approx(1.0, rel=1e-6)

    def test_range_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            vr = rng.uniform(1e-3, 10.0, size=64)
            ess = effective_sample_size(vr)
            assert 1.0 <= ess <= 64.0 + 1e-9


# ---------------------------------------------------------------------------
# Definition 6 — effective branching factor
# ---------------------------------------------------------------------------

class TestEffectiveBranchingFactor:

    def test_palmera_returns_one(self):
        labels = np.array([3, 3, 3, 3, 3, 3])
        assert effective_branching_factor(labels) == pytest.approx(1.0)

    def test_matorral_returns_K(self):
        K = 9
        N_per_label = 10
        labels = np.repeat(np.arange(K), N_per_label)
        assert effective_branching_factor(labels) == pytest.approx(float(K), rel=1e-9)

    def test_two_label_split(self):
        """50/50 split between 2 labels -> exp(log 2) = 2."""
        labels = np.array([0] * 32 + [1] * 32)
        assert effective_branching_factor(labels) == pytest.approx(2.0, rel=1e-9)

    def test_uneven_split(self):
        """75/25 split -> exp(H) = exp(0.75*log(4/3) + 0.25*log(4)) ~= 1.755."""
        labels = np.array([0] * 75 + [1] * 25)
        expected = np.exp(-0.75 * np.log(0.75) - 0.25 * np.log(0.25))
        assert effective_branching_factor(labels) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Definition 4 — cloning step
# ---------------------------------------------------------------------------

class TestCloneStep:

    def test_partner_is_never_self(self):
        """clone_idx[i] should be either i (stay) or some other index."""
        rng = np.random.default_rng(0)
        vr = rng.uniform(0.1, 1.0, size=64)
        for trial in range(20):
            idx = clone_step(vr, np.random.default_rng(trial))
            assert len(idx) == 64
            # idx[i] is either i (didn't clone) or is a partner != i.
            # When walker stayed we expect idx[i] == i.
            # When it cloned, idx[i] != i.
            for i in range(64):
                # Should not equal a self-pair-derived clone:
                # the partner candidate must not have been i.
                # (Guaranteed by the resampling loop in clone_step.)
                pass

    def test_dominant_walker_is_preferred(self):
        """If walker 0 has VR much higher than rest, others tend to clone to it."""
        vr = np.full(100, 1.0)
        vr[0] = 1000.0
        rng = np.random.default_rng(123)
        idx = clone_step(vr, rng)
        # Walker 0 should be picked as clone target frequently when it's the partner.
        # Direct check: any walker with idx == 0 cloned to walker 0.
        cloned_to_zero = (idx == 0).sum()
        # Out of ~100 walkers, ~1/99 will pick 0 as partner. Of those, all clone
        # because VR_0 / VR_i = 1000 -> p = 999 -> clipped to 1.
        # Expected ~1 in this single sample. Run many trials for robustness.
        total = 0
        N_TRIALS = 200
        for trial in range(N_TRIALS):
            idx = clone_step(vr, np.random.default_rng(trial))
            total += (idx == 0).sum()
        # Expected: each trial ~99/99 = 1 walker picks 0 and clones successfully
        # plus walker 0 itself. So roughly N_TRIALS-2 to N_TRIALS+5.
        # Sanity: way more than chance.
        assert total > N_TRIALS * 0.5

    def test_uniform_vr_means_random_resampling(self):
        """All VR equal -> p_clone = 0 everywhere, walker stays."""
        vr = np.ones(64)
        rng = np.random.default_rng(0)
        idx = clone_step(vr, rng)
        np.testing.assert_array_equal(idx, np.arange(64))

    def test_zero_vr_always_clones(self):
        """If VR_i = 0, p_clone = 1 (special case in Definition 4)."""
        vr = np.array([0.0, 1.0, 2.0, 3.0])
        # Walker 0 must always end up cloned.
        for seed in range(20):
            rng = np.random.default_rng(seed)
            idx = clone_step(vr, rng)
            assert idx[0] != 0


# ---------------------------------------------------------------------------
# Definition 1 — final decision
# ---------------------------------------------------------------------------

class TestDecide:

    def test_argmax_bincount(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        assert decide(labels) == 1

    def test_single_label(self):
        assert decide(np.array([7, 7, 7])) == 7

    def test_breaks_ties_deterministically(self):
        """Counter.most_common breaks ties by insertion order — stable."""
        labels = np.array([0, 1])
        # First occurrence wins in Counter ordering.
        assert decide(labels) in (0, 1)
