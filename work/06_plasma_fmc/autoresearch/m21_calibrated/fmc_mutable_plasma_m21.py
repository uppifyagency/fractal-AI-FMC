"""fmc_mutable_plasma_m21.py — agent's mutable file for M21 calibrated loop.

Initial state: VANILLA FMC (FMCPlasmaJaxController) with M19 exp00 hyperparams.
This is the calibrated baseline. The Karpathy autoresearch driver edits
THIS FILE only between experiments (one change per experiment).

Allowed mutation surfaces:
  1. Hyperparameters: N_WALKERS, HORIZON, VOLTAGE_STD, SHAPE_WEIGHTS
  2. Reward shaping: SHAPE_WEIGHTS, intrinsic bonuses
  3. FMC dynamics: alpha, beta in relativize, distance metric
  4. Action perturbation: voltage_std envelope, gas/P_aux jitter

Forbidden:
  - Editing prepare_plasma_calibrated.py (the harness)
  - Editing tcv_published_targets.py (the eval surface)
  - Editing the simulator (calibrated_sim.py / plasma_simulator_jax.py)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent
sys.path.insert(0, str(PARENT / "scripts"))

from fmc_plasma_jax import FMCPlasmaJaxController

# === MUTATION SURFACE: edit values below between experiments ===

# exp02: V_STD 50 -> 25 (half baseline). Calibrated S × 10 means equivalent
# physical perturbation needs ~10× lower V_STD. Start half-baseline.
N_WALKERS = 64
HORIZON = 10
DT = 1e-3
VOLTAGE_STD = 12.0
P_AUX = 5e5
GAS_PUFF = 1e21

SHAPE_WEIGHTS = [100.0, 100.0, 10.0, 10.0]

# === END MUTATION SURFACE ===


def make_controller(sim_p, seed: int = 0):
    """Returns a controller with .decide(state, target_4) -> dict.
    The dict must contain key 'V_coils' (np.ndarray length N=20).
    """
    import jax.numpy as jnp
    from plasma_simulator_jax import DTYPE
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
