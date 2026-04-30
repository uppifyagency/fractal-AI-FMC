"""fmc_mutable.py — THE FILE THE AGENT EDITS.

Initial state: a thin wrapper around fmc_craftax_v4 with the run_007 SOTA config:
  N=512, M=40, intrinsic_inv_alpha=0.5, proximity_alpha=0.2, sigma=10, mode='delta'

This achieves 29.27% Crafter (30 seeds, +/- 1.04 CI95). The agent should edit
this file freely to try to improve it. It MUST keep the run_episode signature
intact since prepare_craftax.evaluate() calls it.

REQUIRED INTERFACE (do not break):
    run_episode(seed: int, max_steps: int = 500,
                env_name: str = "Craftax-Classic-Symbolic-v1") -> dict

    The returned dict MUST have these keys (others are fine):
      - 'reward': float
      - 'achievements_list': list[str]   (subset of CRAFTAX_CLASSIC_ACHIEVEMENTS)
      - 'achievements_unlocked': int     (= len(achievements_list))
      - 'n_steps_decisions': int
      - 'wall_time_s': float
      - 'decisions_per_sec': float

LIBERTY (full freedom under level B):
  - Modify any function, the FMC core, the reward shaping, the scanning policy
  - Add helper functions, classes, modules
  - Import anything already in the project's pyproject (no new deps)
  - Replace FMC entirely with a different planner if you want — the metric is what matters

CONSTRAINTS:
  - Cannot modify prepare_craftax.py (FROZEN evaluation harness)
  - Cannot modify test_fmc_theory.py (FROZEN theory invariants)
  - If your modified algorithm still claims to be "FMC" it should pass the
    15 unit tests; if you break the FMC invariants intentionally, document
    that you've moved to a non-FMC algorithm in the description tag.

Read these for context BEFORE editing:
  - ../../docs/MATH_CANON.md — the math invariants of FMC (Def 2-4)
  - ../docs/run_005_wigner_memory_negative.md — naive memory FAILED (-9.5pp)
  - ../docs/run_006_long_episode_and_vitality_negative.md — vitality FAILED
  - ../docs/run_007_NM_sweep_GPU.md — N x M scaling FAILED to crack diamond chain
  - ../docs/run_007_addendum_fragile_port_analysis.md — why GPU port doesn't help
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force CPU before JAX import (Metal blocks Craftax)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add the parent scripts dir so we can import fmc_craftax_v4 verbatim
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fmc_craftax_v4 import FMCConfig, run_episode as _v4_run_episode  # noqa: E402


# Initial config = run_007 SOTA at (N=512, M=40).
# The agent can change these or replace the whole pipeline below.
CONFIG = FMCConfig(
    n_walkers=512,
    time_horizon=40,
    alpha=1.0,
    beta=1.0,
    action_repeat=1,
    intrinsic_inv_alpha=0.5,
    proximity_alpha=0.2,
    proximity_sigma=10.0,
    proximity_mode="delta",
)


def run_episode(seed: int, max_steps: int = 500,
                env_name: str = "Craftax-Classic-Symbolic-v1") -> dict:
    """Run one episode with the current CONFIG. The agent typically EITHER
    edits CONFIG above OR replaces this function body.

    Must return a dict with the keys listed in the module docstring.
    """
    return _v4_run_episode(
        seed=seed, cfg=CONFIG, max_steps=max_steps, verbose=False,
        env_name=env_name,
    )


if __name__ == "__main__":
    # Quick local check: 1 seed, prints summary
    import json
    t0 = time.time()
    r = run_episode(seed=42)
    print(json.dumps({
        "seed": 42,
        "achievements_unlocked": r["achievements_unlocked"],
        "reward": r["reward"],
        "n_steps_decisions": r["n_steps_decisions"],
        "wall_s": time.time() - t0,
    }, indent=2))
