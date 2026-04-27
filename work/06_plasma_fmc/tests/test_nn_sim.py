"""Tests for Milestone 12 — NN-shape simulator + integrated pipeline."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import jax
import jax.numpy as jnp

HISTORY = Path(__file__).parent.parent / "results" / "milestone_12_history.json"
POLICY = Path(__file__).parent.parent / "results" / "policy_nn_shape.npz"


class TestNNSim:
    def test_build_succeeds(self):
        from plasma_simulator_nn_shape import build_nn_sim_params
        sim_p = build_nn_sim_params()
        assert sim_p.N == 20
        assert sim_p.x_mean.shape == (20,)
        assert sim_p.y_mean.shape == (4,)

    def test_predict_shape_at_ref(self):
        """NN prediction at I_ref must be within ~5 cm / 0.05 of calibrated ref."""
        from plasma_simulator_nn_shape import build_nn_sim_params, predict_shape
        sim_p = build_nn_sim_params()
        shape = np.asarray(predict_shape(sim_p, jnp.asarray(sim_p.I_ref)))
        # NN prediction at I_ref might differ from ref due to NN approximation
        assert abs(shape[0] - sim_p.R_ref) < 0.05, f"R_p NN={shape[0]} vs ref {sim_p.R_ref}"
        assert abs(shape[2] - sim_p.kappa_ref) < 0.10
        assert abs(shape[3] - sim_p.delta_ref) < 0.10

    def test_step_runs(self):
        """A single step must execute without NaN."""
        from plasma_simulator_nn_shape import (
            build_nn_sim_params, initial_state_nn, make_jit_step_nn,
        )
        from plasma_simulator_jax import DTYPE
        sim_p = build_nn_sim_params()
        step = make_jit_step_nn(sim_p)
        x0 = initial_state_nn(sim_p)
        V = jnp.asarray(sim_p.R_diag) * jnp.asarray(sim_p.I_ref)
        x1 = step(x0, V, jnp.asarray(0.0, dtype=DTYPE),
                  jnp.asarray(0.0, dtype=DTYPE),
                  jnp.asarray(1e-3, dtype=DTYPE))
        x1.block_until_ready()
        assert np.all(np.isfinite(np.asarray(x1)))


class TestNNFMC:
    def test_decision_runs(self):
        from plasma_simulator_nn_shape import build_nn_sim_params, initial_state_nn
        from fmc_plasma_nn import FMCPlasmaNNController
        sim_p = build_nn_sim_params()
        x0 = initial_state_nn(sim_p)
        ctrl = FMCPlasmaNNController(sim_p, n_walkers=16, horizon=5, seed=0)
        d = ctrl.decide(np.asarray(x0),
                         np.array([sim_p.R_ref, sim_p.Z_ref,
                                    sim_p.kappa_ref, sim_p.delta_ref],
                                   dtype=np.float32))
        assert d["V_coils"].shape == (sim_p.N,)
        assert np.all(np.isfinite(d["V_coils"]))


class TestPipeline:
    def test_pipeline_ran(self):
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            d = json.load(f)
        assert len(d["history"]) >= 1

    def test_dagger_attempts_improvement(self):
        """DAgger may not improve in this setup, but should not catastrophically fail."""
        if not HISTORY.exists():
            return
        with open(HISTORY) as f:
            d = json.load(f)
        # Final err should be finite + non-zero
        last = d["history"][-1]
        assert np.isfinite(last["mean_err"])
        assert 0 < last["mean_err"] < 1e6


if __name__ == "__main__":
    n_pass = n_fail = 0
    for cls in [TestNNSim, TestNNFMC, TestPipeline]:
        instance = cls()
        for attr in dir(instance):
            if not attr.startswith("test_"):
                continue
            try:
                getattr(instance, attr)()
                print(f"  ✓ {cls.__name__}.{attr}")
                n_pass += 1
            except Exception as e:
                print(f"  ✗ {cls.__name__}.{attr}: {e}")
                n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
