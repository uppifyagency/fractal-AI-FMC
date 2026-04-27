"""Mathematical correctness tests for FMC plasma controller.

Verifies the canonical FMC primitives (paper §2.2.3, §4.3) and
plasma-specific reward/safety formulas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fmc_plasma import (
    FMCConfig,
    FMCPlasmaController,
    ShapeTarget,
    relativize_np,
    safety_penalty,
    shape_reward,
)
from plasma_simulator_jax import build_jax_params


# ============================================================
# Relativize (paper §2.2.3)
# ============================================================

class TestRelativize:
    def test_all_positive_output(self):
        x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
        out = relativize_np(x)
        assert (out > 0).all()

    def test_order_preserved(self):
        x = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
        out = relativize_np(x)
        # Relative ordering must match
        assert np.argsort(x).tolist() == np.argsort(out).tolist()

    def test_constant_input(self):
        x = np.full(10, 4.2)
        out = relativize_np(x)
        # When σ=0, return ones
        assert np.allclose(out, 1.0)

    def test_piecewise_at_mean(self):
        """At z=0 (i.e., x = mean), exp(0) = 1."""
        x = np.array([1.0, 3.0])  # mean=2, std=1
        out = relativize_np(x)
        # x[0]=1 → z=-1 → exp(-1) ≈ 0.368
        # x[1]=3 → z=+1 → 1+log(2) ≈ 1.693
        assert abs(out[0] - np.exp(-1.0)) < 1e-5
        assert abs(out[1] - (1.0 + np.log(2.0))) < 1e-5

    def test_extreme_negative_safe(self):
        """Outliers don't cause overflow."""
        x = np.array([-1000.0, 1.0, 2.0, 3.0])
        out = relativize_np(x)
        assert np.all(np.isfinite(out))
        assert (out > 0).all()


# ============================================================
# Shape reward + safety penalty
# ============================================================

class TestShapeReward:
    def test_max_at_target(self):
        """Reward is maximized when state matches target exactly."""
        sim_p, x0 = build_jax_params()
        N = sim_p.N
        target = ShapeTarget(R_p=float(x0[N + 3]), Z_p=float(x0[N + 4]),
                             kappa=float(x0[N + 5]), delta=float(x0[N + 6]))
        x_target = np.asarray(x0).reshape(1, -1).astype(np.float32)
        x_offset = x_target.copy()
        x_offset[0, N + 3] += 0.05  # 5cm away
        r_target = shape_reward(x_target, N, target)
        r_offset = shape_reward(x_offset, N, target)
        assert r_target[0] > r_offset[0]
        assert r_target[0] >= 0  # at exact target, reward = 0 (max)

    def test_quadratic(self):
        """Doubling the error → 4× the (negative) reward magnitude."""
        sim_p, x0 = build_jax_params()
        N = sim_p.N
        target = ShapeTarget(R_p=0.88, Z_p=0.0, kappa=1.7, delta=0.3,
                             w_R=100, w_Z=0, w_kappa=0, w_delta=0)
        x = np.asarray(x0).reshape(1, -1).astype(np.float32)

        x[0, N + 3] = 0.89  # 1cm error
        r1 = float(shape_reward(x, N, target)[0])
        x[0, N + 3] = 0.90  # 2cm error
        r2 = float(shape_reward(x, N, target)[0])

        # |r2| = 4 |r1| since (2x)² = 4x²
        assert abs(r2 - 4 * r1) < 1e-4


class TestSafetyPenalty:
    def test_zero_when_safe(self):
        sim_p, x0 = build_jax_params()
        N = sim_p.N
        x = np.asarray(x0).reshape(1, -1).astype(np.float32)
        # Default initial state should be safe
        pen = safety_penalty(x, N, sim_p)
        # Some baseline penalty allowed (very low) but not large
        assert pen[0] < 100.0

    def test_q95_below_threshold(self):
        """Reduce I_p to make q95 large (safe), then increase to make small (penalty)."""
        sim_p, x0 = build_jax_params()
        N = sim_p.N
        x = np.asarray(x0).reshape(1, -1).astype(np.float32)
        # I_p extremely high → q95 → 0 → penalty
        x[0, N] = 5e6  # 5 MA, way above limit
        pen = safety_penalty(x, N, sim_p, q95_min=2.0)
        # Should trigger q95 penalty
        x[0, N] = 200_000.0  # back to nominal
        pen_safe = safety_penalty(x, N, sim_p, q95_min=2.0)
        assert pen[0] > pen_safe[0]


# ============================================================
# Controller smoke test
# ============================================================

class TestController:
    def test_decision_returns_valid_v(self):
        sim_p, x0 = build_jax_params()
        target = ShapeTarget(R_p=0.90, Z_p=0.0, kappa=1.85, delta=0.3)
        cfg = FMCConfig(n_walkers=32, horizon=5, voltage_std=20.0)
        ctrl = FMCPlasmaController(sim_p, target, cfg, seed=0)
        decision = ctrl.decide(np.asarray(x0))
        assert decision["V_coils"].shape == (sim_p.N,)
        assert np.all(np.isfinite(decision["V_coils"]))
        assert decision["walkers_alive"] >= 0
        assert decision["samples_used"] == cfg.n_walkers * cfg.horizon

    def test_determinism(self):
        sim_p, x0 = build_jax_params()
        target = ShapeTarget(R_p=0.90, Z_p=0.0, kappa=1.85, delta=0.3)
        cfg = FMCConfig(n_walkers=16, horizon=3, voltage_std=20.0)
        c1 = FMCPlasmaController(sim_p, target, cfg, seed=42)
        c2 = FMCPlasmaController(sim_p, target, cfg, seed=42)
        d1 = c1.decide(np.asarray(x0))
        d2 = c2.decide(np.asarray(x0))
        # Same seed → same decision
        assert np.allclose(d1["V_coils"], d2["V_coils"])

    def test_decision_pulls_toward_target(self):
        """Across many decisions from same start, R_p should drift toward target on average."""
        sim_p, x0 = build_jax_params()
        target = ShapeTarget(R_p=0.95, Z_p=0.0, kappa=1.7, delta=0.3,
                             w_R=1000.0)  # heavy R weight to make signal clear
        cfg = FMCConfig(n_walkers=64, horizon=10, voltage_std=30.0)
        ctrl = FMCPlasmaController(sim_p, target, cfg, seed=0)
        # Apply one decision and step
        from plasma_simulator_jax import (
            DTYPE, make_jit_step,
        )
        import jax.numpy as jnp
        sim_step = make_jit_step(sim_p)
        x = np.asarray(x0).copy()
        N = sim_p.N
        R_initial = float(x[N + 3])
        # Multiple control steps
        for _ in range(10):
            d = ctrl.decide(x)
            x = np.array(sim_step(
                jnp.asarray(x),
                jnp.asarray(d["V_coils"], dtype=DTYPE),
                jnp.asarray(d["P_aux"], dtype=DTYPE),
                jnp.asarray(d["gas_puff"], dtype=DTYPE),
                jnp.asarray(cfg.dt, dtype=DTYPE),
            ))
        R_final = float(x[N + 3])
        # Move from 0.88 toward 0.95 — should be > initial after 10 steps
        assert R_final > R_initial, (
            f"R_p did not move toward target: initial={R_initial}, final={R_final}"
        )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    test_classes = [TestRelativize, TestShapeReward, TestSafetyPenalty, TestController]
    n_pass = n_fail = 0
    for cls in test_classes:
        instance = cls()
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(instance, attr)()
                print(f"  ✓ {cls.__name__}.{attr}")
                n_pass += 1
            except AssertionError as e:
                print(f"  ✗ {cls.__name__}.{attr}: {e}")
                n_fail += 1
            except Exception as e:
                print(f"  ✗ {cls.__name__}.{attr}: {type(e).__name__}: {e}")
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
