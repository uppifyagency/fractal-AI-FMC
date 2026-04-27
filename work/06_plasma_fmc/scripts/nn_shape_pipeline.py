"""End-to-end pipeline with NN-shape simulator (M12).

Mirrors calibrated_pipeline.py (M10) but uses the NN-shape JIT simulator
of M12. Compares closed-loop tracking floor against M10 baseline.
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

from fmc_plasma import ShapeTarget
from fmc_plasma_nn import FMCPlasmaNNController
from plasma_simulator_jax import DTYPE, pack_state
from plasma_simulator_nn_shape import (
    build_nn_sim_params, initial_state_nn, make_jit_step_nn,
)
from policy import (
    DTYPE as PDTYPE, Normalizer, TrainedPolicy,
    build_features, rescale_state,
)
from train_policy import train_loop

RESULTS_DIR = Path(__file__).parent.parent / "results"


def sample_initial_state_nn(rng, sim_p) -> np.ndarray:
    """Random initial state around NN-cal reference."""
    N = sim_p.N
    I_ref = np.asarray(sim_p.I_ref)
    I_coils = I_ref + rng.normal(0, 200.0, N).astype(np.float32)

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


def sample_target_nn(rng, sim_p) -> ShapeTarget:
    """Targets in plausible reachable envelope around NN-cal ref."""
    return ShapeTarget(
        R_p=rng.uniform(sim_p.R_ref - 0.03, sim_p.R_ref + 0.03),
        Z_p=rng.uniform(sim_p.Z_ref - 0.05, sim_p.Z_ref + 0.05),
        kappa=rng.uniform(sim_p.kappa_ref - 0.15, sim_p.kappa_ref + 0.30),
        delta=rng.uniform(sim_p.delta_ref - 0.30, sim_p.delta_ref + 0.30),
    )


def eval_policy_nn(policy, sim_p, sim_step, n_eval=20, episode_length=30, seed=99):
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_initial_state_nn(rng, sim_p)
        target = sample_target_nn(rng, sim_p)
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
    print("Milestone 12 — NN-shape integrated pipeline")
    print("=" * 70)

    sim_p = build_nn_sim_params()
    sim_step = make_jit_step_nn(sim_p)
    print(f"\nNN-shape SimParams: ref κ={sim_p.kappa_ref:.3f}, δ={sim_p.delta_ref:+.3f}")

    # ---- 1. Generate dataset ----
    print("\n[1] Generating expert dataset (500 samples, NN-shape FMC)...")
    rng = np.random.default_rng(42)
    ctrl = FMCPlasmaNNController(sim_p, n_walkers=32, horizon=8, seed=0)
    # Warmup
    ctrl.decide(np.zeros(27, dtype=np.float32),
                np.array([sim_p.R_ref, sim_p.Z_ref, sim_p.kappa_ref, sim_p.delta_ref],
                         dtype=np.float32))

    n_samples = 500
    states = np.zeros((n_samples, sim_p.N + 7), dtype=np.float32)
    targets = np.zeros((n_samples, 4), dtype=np.float32)
    voltages = np.zeros((n_samples, sim_p.N), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_samples):
        s = sample_initial_state_nn(rng, sim_p)
        tgt = sample_target_nn(rng, sim_p)
        tgt_arr = np.array([tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta], dtype=np.float32)
        d = ctrl.decide(s, tgt_arr)
        states[i] = s
        targets[i] = tgt_arr
        voltages[i] = d["V_coils"]
    t_gen = time.perf_counter() - t0
    print(f"  Generated {n_samples} samples in {t_gen:.1f}s "
          f"({n_samples/t_gen:.0f}/s)")

    # ---- 2. Train MLP ----
    print("\n[2] Training MLP 64×64...")
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
    states_r = rescale_state(states)
    I_ref_proxy = V_ref / max(V_ref.max(), 1.0)
    X = build_features(states_r, targets, I_ref_proxy)
    Y = (voltages - V_ref).astype(np.float32)

    params, x_norm, y_norm, log = train_loop(
        X, Y, hidden_sizes=(64, 64), lr=3e-3, n_epochs=200, batch_size=64,
        seed=0, weight_decay=1e-3, patience=30,
    )
    out_pol = RESULTS_DIR / "policy_nn_shape.npz"
    np.savez(
        out_pol, params=np.array(params, dtype=object),
        x_mean=x_norm.mean, x_std=x_norm.std,
        y_mean=y_norm.mean, y_std=y_norm.std,
        V_ref=V_ref, hidden_sizes=np.array([64, 64]),
    )
    print(f"  Saved: {out_pol}")
    print(f"  Final val loss: {log[-1]['val_loss']:.4f}")

    # ---- 3. Eval BC ----
    print("\n[3] Evaluating BC policy on NN-shape sim (20 random scenarios)...")
    policy = TrainedPolicy.load(out_pol)
    bc_eval = eval_policy_nn(policy, sim_p, sim_step, n_eval=20, seed=99)
    print(f"  BC on NN-shape: mean err {bc_eval['mean_err']:.2f}, "
          f"quench {bc_eval['quench']}/20")

    # ---- 4. Quick DAgger 3 iter ----
    print("\n[4] DAgger × 3 iter...")
    history = [{"iter": 0, "n_samples": int(states.shape[0]), **bc_eval}]
    all_states, all_targets, all_voltages = states.copy(), targets.copy(), voltages.copy()

    for it in range(1, 4):
        new_s, new_t, new_v = [], [], []
        n_episodes = 25  # 25 ep × 20 tick = 500 sample
        for ep in range(n_episodes):
            x = sample_initial_state_nn(rng, sim_p)
            tgt = sample_target_nn(rng, sim_p)
            tgt_arr = np.array([tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta], dtype=np.float32)
            for t in range(20):
                new_s.append(x.copy())
                new_t.append(tgt_arr.copy())
                d = ctrl.decide(x, tgt_arr)
                new_v.append(d["V_coils"])
                V = policy(x, tgt_arr)
                x = np.array(sim_step(
                    jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                    jnp.asarray(5e5, dtype=DTYPE),
                    jnp.asarray(1e21, dtype=DTYPE),
                    jnp.asarray(1e-3, dtype=DTYPE),
                ))
                if np.isnan(x).any() or np.isinf(x).any():
                    break
        new_s = np.array(new_s, dtype=np.float32)
        new_t = np.array(new_t, dtype=np.float32)
        new_v = np.array(new_v, dtype=np.float32)
        all_states = np.concatenate([all_states, new_s])
        all_targets = np.concatenate([all_targets, new_t])
        all_voltages = np.concatenate([all_voltages, new_v])

        # Retrain
        states_r = rescale_state(all_states)
        X = build_features(states_r, all_targets, I_ref_proxy)
        Y = (all_voltages - V_ref).astype(np.float32)
        params, x_norm, y_norm, log = train_loop(
            X, Y, hidden_sizes=(64, 64), lr=3e-3, n_epochs=200, batch_size=64,
            seed=it, weight_decay=1e-3, patience=30,
        )
        np.savez(
            out_pol, params=np.array(params, dtype=object),
            x_mean=x_norm.mean, x_std=x_norm.std,
            y_mean=y_norm.mean, y_std=y_norm.std,
            V_ref=V_ref, hidden_sizes=np.array([64, 64]),
        )
        policy = TrainedPolicy.load(out_pol)
        ev = eval_policy_nn(policy, sim_p, sim_step, n_eval=20, seed=99)
        history.append({"iter": it, "n_samples": int(all_states.shape[0]), **ev})
        print(f"  iter {it}: |D|={all_states.shape[0]:5d}, "
              f"err={ev['mean_err']:.2f}, quench={ev['quench']}/20")

    # ---- Save ----
    log_path = RESULTS_DIR / "milestone_12_history.json"
    with open(log_path, "w") as f:
        json.dump({"history": history, "dataset_gen_time_s": t_gen}, f, indent=2)

    print(f"\n[Summary]")
    print(f"  M10 (linear S, calibrated)  DAgger×5 floor: 3.47")
    print(f"  M12 (NN shape) BC           : {history[0]['mean_err']:.2f}")
    print(f"  M12 (NN shape) DAgger×3     : {history[-1]['mean_err']:.2f}")
    if history[-1]["mean_err"] < 3.47:
        print(f"  ✓ M12 IMPROVES floor by {3.47 / history[-1]['mean_err']:.2f}×")
    else:
        print(f"  ⚠ M12 NOT better ({history[-1]['mean_err']:.2f} vs 3.47)")


if __name__ == "__main__":
    main()
