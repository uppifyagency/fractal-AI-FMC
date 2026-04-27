"""End-to-end calibrated pipeline (Milestone 10).

Builds a fresh dataset/policy with the M9-calibrated SimParams, then
benchmarks tracking quality vs the original M8 baseline.

Steps:
1. Build calibrated SimParams (M9 ref state + S × 10)
2. Generate 1000 expert samples with calibrated JIT FMC
3. Train MLP 64×64 via behavioral cloning
4. (Optional) 5 DAgger iterations × 500 samples
5. Benchmark on calibrated eval set + ORIGINAL eval set (cross-comparison)
6. Save: policy_calibrated.npz, calibrated_dataset.npz, calibrated_log.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp

from calibrated_sim import (
    M9_BASELINE, build_calibrated_jax_params, calibrated_target_ranges,
)
from fmc_plasma import ShapeTarget
from fmc_plasma_jax import FMCPlasmaJaxController
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import (
    DTYPE as PDTYPE, Normalizer, TrainedPolicy,
    build_features, rescale_state,
)
from train_policy import train_loop

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ============================================================
# Domain randomization for calibrated simulator
# ============================================================

def sample_calibrated_initial_state(rng, sim_p) -> np.ndarray:
    """Initial state perturbations around M9 calibrated reference."""
    N = sim_p.N
    I_ref = np.asarray(sim_p.I_ref)
    I_coils = I_ref + rng.normal(0, 200.0, N)

    I_p = rng.uniform(150e3, 250e3)
    T_e_keV = rng.uniform(0.5, 2.0)
    n_bar = rng.uniform(3e19, 7e19)
    R_p = sim_p.R_ref + rng.uniform(-0.02, 0.02)
    Z_p = sim_p.Z_ref + rng.uniform(-0.02, 0.02)
    kappa = rng.uniform(1.5, 1.8)
    delta = rng.uniform(-0.2, 0.2)

    a_eff = sim_p.a_eff
    V_plasma = 2 * np.pi**2 * R_p * a_eff**2 * kappa
    W = 3 * n_bar * V_plasma * (T_e_keV * 1e3 * 1.602176634e-19)

    return np.array([
        *I_coils, I_p, W, n_bar, R_p, Z_p, kappa, delta,
    ], dtype=np.float32)


def sample_calibrated_target(rng) -> ShapeTarget:
    ranges = calibrated_target_ranges()
    return ShapeTarget(
        R_p=rng.uniform(*ranges["R_p"]),
        Z_p=rng.uniform(*ranges["Z_p"]),
        kappa=rng.uniform(*ranges["kappa"]),
        delta=rng.uniform(*ranges["delta"]),
    )


# ============================================================
# Dataset gen + train + eval
# ============================================================

def generate_dataset(sim_p, n_samples: int, fmc_walkers=32, fmc_horizon=8,
                     seed=0):
    """Use calibrated FMC to label random scenarios."""
    rng = np.random.default_rng(seed)
    jx = FMCPlasmaJaxController(
        sim_p, n_walkers=fmc_walkers, horizon=fmc_horizon, seed=seed,
    )
    # Warmup
    jx.decide(np.zeros(27, dtype=np.float32),
              np.array([sim_p.R_ref, sim_p.Z_ref, sim_p.kappa_ref, sim_p.delta_ref],
                       dtype=np.float32))

    states = np.zeros((n_samples, sim_p.N + 7), dtype=np.float32)
    targets = np.zeros((n_samples, 4), dtype=np.float32)
    voltages = np.zeros((n_samples, sim_p.N), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_samples):
        s = sample_calibrated_initial_state(rng, sim_p)
        tgt = sample_calibrated_target(rng)
        tgt_arr = np.array([tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta], dtype=np.float32)
        d = jx.decide(s, tgt_arr)
        states[i] = s
        targets[i] = tgt_arr
        voltages[i] = d["V_coils"]
    elapsed = time.perf_counter() - t0
    print(f"  Generated {n_samples} samples in {elapsed:.1f}s "
          f"({n_samples/elapsed:.0f}/s)")
    return states, targets, voltages, elapsed


def train_and_save(sim_p, states, targets, voltages, hidden=(64, 64),
                    epochs=200, seed=0, out_path: Path | None = None):
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
    states_r = rescale_state(states)
    I_ref_proxy = V_ref / max(V_ref.max(), 1.0)
    X = build_features(states_r, targets, I_ref_proxy)
    Y = (voltages - V_ref).astype(np.float32)

    params, x_norm, y_norm, log = train_loop(
        X, Y, hidden_sizes=hidden, lr=3e-3,
        n_epochs=epochs, batch_size=64, seed=seed,
        weight_decay=1e-3, patience=30,
    )
    if out_path is not None:
        np.savez(
            out_path, params=np.array(params, dtype=object),
            x_mean=x_norm.mean, x_std=x_norm.std,
            y_mean=y_norm.mean, y_std=y_norm.std,
            V_ref=V_ref, hidden_sizes=np.array(hidden),
        )
    return params, x_norm, y_norm, log


def eval_policy(policy, sim_p, sim_step, n_eval=20, episode_length=30, seed=99,
                target_sampler=None):
    """Closed-loop eval. Returns mean shape error + quench count.

    target_sampler: function(rng) → ShapeTarget. If None, uses calibrated ranges.
    """
    if target_sampler is None:
        target_sampler = sample_calibrated_target
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_calibrated_initial_state(rng, sim_p)
        target = target_sampler(rng)
        tgt_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                           dtype=np.float32)
        ep_err = []
        for t in range(episode_length):
            V = policy(x, tgt_arr)
            x_new = sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                jnp.asarray(5e5, dtype=DTYPE),
                jnp.asarray(1e21, dtype=DTYPE),
                jnp.asarray(1e-3, dtype=DTYPE),
            )
            x = np.array(x_new)
            N = sim_p.N
            err = (target.w_R * (x[N+3] - target.R_p)**2
                   + target.w_Z * (x[N+4] - target.Z_p)**2
                   + target.w_kappa * (x[N+5] - target.kappa)**2
                   + target.w_delta * (x[N+6] - target.delta)**2)
            ep_err.append(float(err))
            if abs(x[N]) / 1e6 < 0.05:
                quenches += 1
                break
        errors.append(np.mean(ep_err))
    return {"mean_err": float(np.mean(errors)),
            "median_err": float(np.median(errors)),
            "quench": quenches, "n_eval": n_eval}


def main():
    print("=" * 70)
    print("Milestone 10 — calibrated end-to-end pipeline")
    print("=" * 70)

    # ---- 1. Build calibrated sim ----
    sim_p, x0 = build_calibrated_jax_params()
    sim_step = make_jit_step(sim_p)
    print(f"\n[1] Calibrated SimParams: ref κ={sim_p.kappa_ref:.3f}, "
          f"δ={sim_p.delta_ref:+.3f}, R_p={sim_p.R_ref:.3f}")
    print(f"    S max coeff: {float(jnp.max(jnp.abs(sim_p.S))):.2e} (×10 vs M3)")

    # ---- 2. Dataset gen ----
    print("\n[2] Generating expert dataset (1000 samples, JIT FMC M=32 H=8)...")
    states, targets, voltages, t_gen = generate_dataset(
        sim_p, n_samples=1000, fmc_walkers=32, fmc_horizon=8, seed=42,
    )

    # Save dataset
    out_ds = RESULTS_DIR / "calibrated_dataset.npz"
    np.savez_compressed(out_ds, states=states, targets=targets, voltages=voltages)
    print(f"  → {out_ds}")

    # ---- 3. Train MLP ----
    print("\n[3] Training MLP 64×64 (BC only, no DAgger for first comparison)...")
    out_pol = RESULTS_DIR / "policy_calibrated.npz"
    t0 = time.perf_counter()
    params, x_norm, y_norm, log = train_and_save(
        sim_p, states, targets, voltages, hidden=(64, 64),
        epochs=200, seed=0, out_path=out_pol,
    )
    print(f"  Trained in {time.perf_counter()-t0:.1f}s, val={log[-1]['val_loss']:.4f}")

    # ---- 4. DAgger iterations ----
    print("\n[4] DAgger iterations (5 × 500 samples)...")
    policy = TrainedPolicy.load(out_pol)
    rng = np.random.default_rng(0)

    history = [{"iter": 0, "n_samples": int(states.shape[0])}]
    history[0].update(eval_policy(policy, sim_p, sim_step, n_eval=20, seed=99))
    print(f"  iter 0 (BC): err={history[0]['mean_err']:.2f}, "
          f"quench={history[0]['quench']}/{history[0]['n_eval']}")

    jx = FMCPlasmaJaxController(sim_p, n_walkers=32, horizon=8, seed=0)
    jx.decide(np.zeros(27, dtype=np.float32),
              np.array([sim_p.R_ref, sim_p.Z_ref, sim_p.kappa_ref, sim_p.delta_ref],
                       dtype=np.float32))

    all_states = states.copy()
    all_targets = targets.copy()
    all_voltages = voltages.copy()

    for it in range(1, 6):
        # Roll out current policy
        new_states, new_targets, new_voltages = [], [], []
        n_episodes = 25  # 25 ep × 20 tick = 500 sample
        for ep in range(n_episodes):
            x = sample_calibrated_initial_state(rng, sim_p)
            target = sample_calibrated_target(rng)
            tgt_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                               dtype=np.float32)
            for t in range(20):
                new_states.append(x.copy())
                new_targets.append(tgt_arr.copy())
                # Query expert
                d = jx.decide(x, tgt_arr)
                new_voltages.append(d["V_coils"])
                # Step with current policy
                V = policy(x, tgt_arr)
                x = np.array(sim_step(
                    jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                    jnp.asarray(5e5, dtype=DTYPE),
                    jnp.asarray(1e21, dtype=DTYPE),
                    jnp.asarray(1e-3, dtype=DTYPE),
                ))
                if np.isnan(x).any() or np.isinf(x).any():
                    break
        new_states = np.array(new_states, dtype=np.float32)
        new_targets = np.array(new_targets, dtype=np.float32)
        new_voltages = np.array(new_voltages, dtype=np.float32)

        all_states = np.concatenate([all_states, new_states])
        all_targets = np.concatenate([all_targets, new_targets])
        all_voltages = np.concatenate([all_voltages, new_voltages])

        # Retrain
        params, x_norm, y_norm, log = train_and_save(
            sim_p, all_states, all_targets, all_voltages, hidden=(64, 64),
            epochs=200, seed=it, out_path=out_pol,
        )
        policy = TrainedPolicy.load(out_pol)
        ev = eval_policy(policy, sim_p, sim_step, n_eval=20, seed=99)
        history.append({"iter": it, "n_samples": int(all_states.shape[0]), **ev})
        print(f"  iter {it}: |D|={all_states.shape[0]:5d}, "
              f"err={ev['mean_err']:.2f}, quench={ev['quench']}/20")

    # ---- 5. Save log ----
    log_path = RESULTS_DIR / "calibrated_history.json"
    with open(log_path, "w") as f:
        json.dump({"history": history,
                   "calibrated_baseline": M9_BASELINE,
                   "s_scale": 10.0,
                   "dataset_gen_time_s": t_gen}, f, indent=2)
    print(f"\n✓ Saved: {log_path}")

    # ---- 6. Summary ----
    print(f"\nFinal summary:")
    print(f"  M5 BC (original sim, original eval) : err 36.00")
    print(f"  M6 DAgger×3 (original)              : err  3.55")
    print(f"  M8 DAgger×N (original)              : err  3.45")
    print(f"  M10 BC (calibrated)                 : err {history[0]['mean_err']:6.2f}")
    print(f"  M10 DAgger×5 (calibrated)           : err {history[-1]['mean_err']:6.2f}")
    print(f"\n  M10 vs M8 (lower better): "
          f"{history[-1]['mean_err'] / 3.45:.2f}× the M8 floor")


if __name__ == "__main__":
    main()
