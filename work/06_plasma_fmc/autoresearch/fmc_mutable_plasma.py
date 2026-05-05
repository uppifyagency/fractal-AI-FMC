"""fmc_mutable_plasma.py — agent's mutable file for the autoresearch loop.

The driver imports `make_controller(sim_p, seed=0)` and calls
`controller.decide(state, target_4)`. The agent edits THIS FILE and only
this file between experiments. Each experiment changes ONE thing
(Karpathy discipline).

Initial state: VANILLA FMC (FMCPlasmaJaxController), the same controller
M15/M16 used for FMC online evaluation. This is the baseline.

Allowed mutation surfaces (as discovered through M18 failure analysis):
  1. Hyperparameters: N_WALKERS, HORIZON, VOLTAGE_STD, SHAPE_WEIGHTS
  2. Reward shaping: SHAPE_WEIGHTS, SAFETY_WEIGHTS, intrinsic bonuses
  3. FMC dynamics: alpha, beta in relativize, distance metric
  4. Action perturbation: voltage_std envelope, gas/P_aux jitter
  5. Structural: swap controller class entirely (e.g., import
     FMCPlasmaHierarchicalController from m18_hierarchical/scripts/)

Forbidden:
  - Editing prepare_plasma.py (the harness)
  - Editing tcv_published_targets.py (the eval surface)
  - Editing the simulator (plasma_simulator_jax.py)

The eval is intentionally a black-box from this file's perspective.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / "scripts"))
sys.path.insert(0, str(PARENT / "m18_hierarchical" / "scripts"))

from fmc_plasma_jax import FMCPlasmaJaxController
from fmc_plasma_hierarchical import (
    FMCPlasmaHierarchicalController,
    DEFAULT_TIER_WEIGHTS,
)

# === MUTATION SURFACE: edit values below between experiments ===

N_WALKERS = 128  # exp14 NEW BEST: V_STD 100 + W asym + N=128 (+0.230)
HORIZON = 10
DT = 1e-3
VOLTAGE_STD = 100.0  # exp14 BEST: 50->100 (sweet spot for walker divergence)
P_AUX = 5e5
GAS_PUFF = 1e21

SHAPE_WEIGHTS = [400.0, 400.0, 200.0, 200.0]  # exp14 BEST: W asym kappa/delta 5x

USE_HIERARCHICAL = False  # reverted from exp05 (-0.012 score)
TIER_WEIGHTS = list(map(float, DEFAULT_TIER_WEIGHTS.tolist()))

# === END MUTATION SURFACE ===


def make_controller(sim_p, seed: int = 0):
    """Returns an object with .decide(state, target_4) → dict.

    The dict must contain at least key "V_coils" (np.ndarray of length N=20).
    """
    import jax.numpy as jnp
    import numpy as np
    from plasma_simulator_jax import DTYPE
    if USE_HIERARCHICAL:
        ctrl = FMCPlasmaHierarchicalController(
            sim_p,
            n_walkers=N_WALKERS,
            horizon=HORIZON,
            dt=DT,
            voltage_std=VOLTAGE_STD,
            P_aux=P_AUX,
            gas_puff=GAS_PUFF,
            seed=seed,
            tier_weights=np.asarray(TIER_WEIGHTS, dtype=np.float32),
            ach_alpha=1.0,
        )
    else:
        ctrl = FMCPlasmaJaxController(
            sim_p,
            n_walkers=N_WALKERS,
            horizon=HORIZON,
            dt=DT,
            voltage_std=VOLTAGE_STD,
            P_aux=P_AUX,
            gas_puff=GAS_PUFF,
            seed=seed,
        )
    ctrl._weights = jnp.asarray(SHAPE_WEIGHTS, dtype=DTYPE)
    return ctrl
