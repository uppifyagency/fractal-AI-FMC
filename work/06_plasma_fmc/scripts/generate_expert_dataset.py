"""Generate FMC expert dataset for behavioral cloning (Milestone 5).

For each sample:
  1. Pick a random initial plasma state (perturbation of x_ref)
  2. Pick a random target shape (within TCV operating envelope)
  3. Run FMC to get optimal V_coils for one control tick
  4. Save (state, target, V_optimal)

Mathematical justification (paper §4.3 + Ross-Bagnell DAgger 2010):
  FMC produces approximately π*(s, target) = argmax_V Q(s, V; target).
  We collect samples from a distribution over (s, target) and train
  a parametric policy π_θ(s, target) → V via MSE on V* targets.
  Expected gain: π_θ inference at NN forward pass speed (~ms),
  vs FMC's N×M×T = 4000-15000 simulator calls.

Output: results/expert_dataset.npz with arrays:
  states     (N_samples, 27)   plasma state
  targets    (N_samples, 4)    target shape (R_p, Z_p, κ, δ)
  voltages   (N_samples, 20)   FMC-recommended V_coils
  rewards    (N_samples,)      FMC's expected reward (for diagnostic)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent))

from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget
from plasma_simulator_jax import build_jax_params


# ---- Domain randomization ranges (within TCV operating envelope) ----

STATE_PERTURB = {
    # Initial state ~ ref + uniform(low, high)
    "I_coils_std": 200.0,    # ±200 A around I_ref
    "I_p_kA": (150.0, 250.0),  # plasma current in kA
    "T_e_keV": (0.5, 2.0),     # T_e (controls W via fixed n)
    "n_bar_e19": (3.0, 7.0),
    "R_p_cm": (-2.0, +2.0),  # offset around 0.88 m
    "Z_p_cm": (-1.5, +1.5),
    "kappa": (1.5, 2.0),
    "delta": (-0.4, 0.6),
}

TARGET_RANGES = {
    "R_p": (0.85, 0.92),     # ±2% around nominal
    "Z_p": (-0.05, +0.05),   # 5cm vertical range
    "kappa": (1.4, 2.2),
    "delta": (-0.5, +0.7),
}


def sample_initial_state(rng: np.random.Generator, sim_p) -> np.ndarray:
    """Draw a randomized initial state."""
    N = sim_p.N
    I_ref = np.asarray(sim_p.I_ref)
    I_coils = I_ref + rng.normal(0, STATE_PERTURB["I_coils_std"], N)

    I_p_kA = rng.uniform(*STATE_PERTURB["I_p_kA"])
    T_e_keV = rng.uniform(*STATE_PERTURB["T_e_keV"])
    n_bar_e19 = rng.uniform(*STATE_PERTURB["n_bar_e19"])
    R_p = sim_p.R_ref + rng.uniform(*STATE_PERTURB["R_p_cm"]) / 100.0
    Z_p = rng.uniform(*STATE_PERTURB["Z_p_cm"]) / 100.0
    kappa = rng.uniform(*STATE_PERTURB["kappa"])
    delta = rng.uniform(*STATE_PERTURB["delta"])

    # W from T_e: W = 3 n V T  with V = 2π² R a² κ
    n_bar = n_bar_e19 * 1e19
    a_eff = sim_p.a_eff
    V_plasma = 2 * np.pi**2 * R_p * a_eff**2 * kappa
    T_J = T_e_keV * 1e3 * 1.602176634e-19
    W = 3 * n_bar * V_plasma * T_J
    I_p = I_p_kA * 1e3

    return np.array([
        *I_coils,
        I_p, W, n_bar, R_p, Z_p, kappa, delta,
    ], dtype=np.float32)


def sample_target(rng: np.random.Generator) -> ShapeTarget:
    return ShapeTarget(
        R_p=rng.uniform(*TARGET_RANGES["R_p"]),
        Z_p=rng.uniform(*TARGET_RANGES["Z_p"]),
        kappa=rng.uniform(*TARGET_RANGES["kappa"]),
        delta=rng.uniform(*TARGET_RANGES["delta"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=500,
                    help="number of (state, target, V*) samples to generate")
    ap.add_argument("--n_walkers", type=int, default=64,
                    help="FMC walker pool size (smaller = faster, lower quality)")
    ap.add_argument("--horizon", type=int, default=15,
                    help="FMC lookahead tick count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim_p, _ = build_jax_params()
    print(f"Generating {args.n_samples} expert samples "
          f"(M={args.n_walkers}, H={args.horizon})...")

    states = np.zeros((args.n_samples, sim_p.N + 7), dtype=np.float32)
    targets = np.zeros((args.n_samples, 4), dtype=np.float32)
    voltages = np.zeros((args.n_samples, sim_p.N), dtype=np.float32)
    rewards = np.zeros((args.n_samples,), dtype=np.float32)
    walkers_alive = np.zeros((args.n_samples,), dtype=np.int32)

    t0 = time.perf_counter()
    last_print = t0
    failed = 0
    for i in range(args.n_samples):
        s = sample_initial_state(rng, sim_p)
        tgt = sample_target(rng)

        # Build a fresh controller each time so the FMC RNG is reseeded —
        # this gives reproducible per-sample stochasticity.
        cfg = FMCConfig(
            n_walkers=args.n_walkers, horizon=args.horizon,
            voltage_std=50.0, dt=1e-3,
        )
        ctrl = FMCPlasmaController(sim_p, tgt, cfg, seed=args.seed * 10000 + i)
        try:
            decision = ctrl.decide(s)
        except Exception as e:
            failed += 1
            print(f"  sample {i}: failed ({e})")
            continue

        states[i] = s
        targets[i] = [tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta]
        voltages[i] = decision["V_coils"]
        rewards[i] = decision["expected_reward"]
        walkers_alive[i] = decision["walkers_alive"]

        # Progress
        now = time.perf_counter()
        if now - last_print > 5.0:
            elapsed = now - t0
            rate = (i + 1) / elapsed
            eta = (args.n_samples - i - 1) / rate
            print(f"  sample {i+1}/{args.n_samples} | "
                  f"rate {rate:.1f}/s | elapsed {elapsed:.1f}s | "
                  f"eta {eta:.1f}s | "
                  f"alive avg {walkers_alive[:i+1].mean():.0f}")
            last_print = now

    elapsed = time.perf_counter() - t0
    out = Path(args.out) if args.out else (
        Path(__file__).parent.parent / "results" / "expert_dataset.npz"
    )
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        states=states, targets=targets, voltages=voltages,
        rewards=rewards, walkers_alive=walkers_alive,
    )

    print(f"\n✓ Saved {args.n_samples - failed} samples ({failed} failed) "
          f"in {elapsed:.1f}s ({(args.n_samples - failed) / elapsed:.1f}/s)")
    print(f"  → {out}")
    print(f"  States    : {states.shape}, dtype={states.dtype}")
    print(f"  Targets   : {targets.shape}")
    print(f"  Voltages  : {voltages.shape}, range [{voltages.min():.1f}, {voltages.max():.1f}] V")
    print(f"  Rewards   : mean={rewards.mean():.2f}, "
          f"min={rewards.min():.2f}, max={rewards.max():.2f}")
    print(f"  Alive avg : {walkers_alive.mean():.1f}/{args.n_walkers}")


if __name__ == "__main__":
    main()
