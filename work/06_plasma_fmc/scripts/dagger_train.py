"""DAgger — Dataset Aggregation algorithm (Ross-Bagnell-Gordon 2011).

Closes the covariate-shift gap from M5 behavioral cloning:

  Iter 0: train π_0 on FMC expert dataset D_0 (= M5's policy)
  Iter k (k=1..K):
    1. Run π_{k-1} in simulator on randomized scenarios → visit states s_t
    2. Query FMC expert at each visited state → V*_t
    3. D_k = D_{k-1} ∪ {(s_t, target_t, V*_t)}
    4. Re-train π_k on D_k

Theorem (Ross et al. 2011, Thm 4.1, AISTATS):
  After K iterations of DAgger, the policy's expected loss is bounded by
  ε_K ≤ ε_N + O(T·γ_N/N) where γ_N is the no-regret rate.
  → Convergence to optimal expert performance under online no-regret.

In practice: 3-10 iterations close the BC quality gap on most tasks.

This script:
  - Loads existing policy from M5
  - Runs K=3 DAgger iterations (each adds N_per_iter samples)
  - Saves intermediate datasets + policies for analysis
  - Final policy → results/policy_dagger.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))

from fmc_plasma import FMCConfig, FMCPlasmaController, ShapeTarget
from generate_expert_dataset import sample_initial_state, sample_target
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import (
    DTYPE as PDTYPE,
    Normalizer,
    PolicyMLP,
    TrainedPolicy,
    build_features,
    rescale_state,
)
from train_policy import train_loop

RESULTS_DIR = Path(__file__).parent.parent / "results"


def collect_visited_states(
    policy: TrainedPolicy,
    sim_p,
    sim_step,
    n_episodes: int,
    episode_length: int,
    rng: np.random.Generator,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[ShapeTarget]]:
    """Roll out current policy and collect visited (state, target) pairs.

    Returns:
        states_visited: (n_episodes * episode_length, 27)
        targets_visited: (n_episodes * episode_length, 4)
        targets_obj_list: list of ShapeTarget (for re-querying FMC)
    """
    states_list = []
    targets_list = []
    targets_obj = []

    dt = jnp.asarray(1e-3, dtype=DTYPE)
    P_aux = jnp.asarray(5e5, dtype=DTYPE)
    gas = jnp.asarray(1e21, dtype=DTYPE)

    for ep in range(n_episodes):
        # Random initial state + target
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
        target_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                              dtype=np.float32)

        for t in range(episode_length):
            # Save (state, target) at the moment we make a decision
            states_list.append(x.copy())
            targets_list.append(target_arr.copy())
            targets_obj.append(target)

            # Apply policy
            V = policy(x, target_arr)
            x_new = sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                P_aux, gas, dt,
            )
            x = np.array(x_new)

            # Stop if NaN (defensive — shouldn't happen with clip)
            if np.isnan(x).any() or np.isinf(x).any():
                break

    return (np.array(states_list, dtype=np.float32),
            np.array(targets_list, dtype=np.float32),
            targets_obj)


def query_expert(
    states: np.ndarray, targets_obj: list[ShapeTarget],
    sim_p, n_walkers: int = 32, horizon: int = 8, seed_base: int = 0,
) -> np.ndarray:
    """Query FMC expert at each (state, target) → V*. Returns (N, 20)."""
    cfg = FMCConfig(n_walkers=n_walkers, horizon=horizon,
                    voltage_std=50.0, dt=1e-3)
    voltages = np.zeros((len(states), sim_p.N), dtype=np.float32)

    for i, (s, tgt) in enumerate(zip(states, targets_obj)):
        ctrl = FMCPlasmaController(sim_p, tgt, cfg, seed=seed_base + i)
        try:
            d = ctrl.decide(s)
            voltages[i] = d["V_coils"]
        except Exception as e:
            # Fallback: use V_ref
            voltages[i] = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)

    return voltages


def build_xy(states, targets, voltages, sim_p):
    """Build (X, Y) feature/target matrices for training (mirrors train_policy.py)."""
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
    states_r = rescale_state(states)
    I_ref_proxy = V_ref / max(V_ref.max(), 1.0)
    X = build_features(states_r, targets, I_ref_proxy)
    Y = (voltages - V_ref).astype(np.float32)
    return X, Y, V_ref


def evaluate_policy(policy: TrainedPolicy, sim_p, sim_step,
                    n_eval: int = 5, episode_length: int = 50,
                    seed: int = 99) -> dict:
    """Closed-loop tracking quality on held-out scenarios."""
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
        target_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                              dtype=np.float32)
        ep_err = []
        for t in range(episode_length):
            V = policy(x, target_arr)
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
    return {
        "mean_err": float(np.mean(errors)),
        "median_err": float(np.median(errors)),
        "quench_count": quenches,
        "n_eval": n_eval,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=3)
    ap.add_argument("--samples_per_iter", type=int, default=200,
                    help="(rollouts) — n_episodes × episode_length")
    ap.add_argument("--episode_length", type=int, default=20)
    ap.add_argument("--fmc_walkers", type=int, default=32)
    ap.add_argument("--fmc_horizon", type=int, default=8)
    ap.add_argument("--epochs_per_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("DAgger — iterative dataset aggregation")
    print(f"  n_iter            : {args.n_iter}")
    print(f"  samples per iter  : {args.samples_per_iter}")
    print(f"  episode length    : {args.episode_length}")
    print(f"  FMC config        : M={args.fmc_walkers}, H={args.fmc_horizon}")
    print("=" * 70)

    sim_p, _ = build_jax_params()
    sim_step = make_jit_step(sim_p)

    # Iteration 0 = load existing M5 dataset + policy
    d0 = np.load(RESULTS_DIR / "expert_dataset.npz")
    all_states = d0["states"].copy()
    all_targets = d0["targets"].copy()
    all_voltages = d0["voltages"].copy()
    print(f"\n[iter 0] Loaded D_0: {all_states.shape[0]} samples (from M5)")

    policy = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    eval0 = evaluate_policy(policy, sim_p, sim_step,
                             n_eval=10, episode_length=30, seed=99)
    print(f"          policy eval (M5 baseline): mean err {eval0['mean_err']:.2f}, "
          f"quenches {eval0['quench_count']}/{eval0['n_eval']}")

    history = [{"iter": 0, "n_samples": int(all_states.shape[0]), **eval0}]

    rng = np.random.default_rng(args.seed)

    for it in range(1, args.n_iter + 1):
        print(f"\n[iter {it}] ─" + "─" * 60)

        # 1. Roll out policy → collect visited states
        n_episodes = args.samples_per_iter // args.episode_length
        print(f"  rolling out current policy: "
              f"{n_episodes} episodes × {args.episode_length} ticks")
        t0 = time.perf_counter()
        states_v, targets_v, targets_obj = collect_visited_states(
            policy, sim_p, sim_step,
            n_episodes=n_episodes, episode_length=args.episode_length,
            rng=rng, seed_offset=it * 10000,
        )
        print(f"  → collected {states_v.shape[0]} visited states "
              f"({time.perf_counter() - t0:.1f}s)")

        # 2. Query FMC at each visited state
        t0 = time.perf_counter()
        voltages_v = query_expert(
            states_v, targets_obj, sim_p,
            n_walkers=args.fmc_walkers, horizon=args.fmc_horizon,
            seed_base=it * 100000,
        )
        print(f"  → expert labels generated ({time.perf_counter() - t0:.1f}s)")

        # 3. Aggregate
        all_states = np.concatenate([all_states, states_v], axis=0)
        all_targets = np.concatenate([all_targets, targets_v], axis=0)
        all_voltages = np.concatenate([all_voltages, voltages_v], axis=0)
        print(f"  D_{it} now has {all_states.shape[0]} samples")

        # 4. Retrain (from scratch — common DAgger choice; warm-start also valid)
        X, Y, V_ref = build_xy(all_states, all_targets, all_voltages, sim_p)
        t0 = time.perf_counter()
        params, x_norm, y_norm, log = train_loop(
            X, Y,
            hidden_sizes=(32, 32), lr=3e-3,
            n_epochs=args.epochs_per_iter, batch_size=32,
            seed=args.seed + it,
            weight_decay=1e-3, patience=30,
        )
        print(f"  retrained in {time.perf_counter() - t0:.1f}s "
              f"(final val={log[-1]['val_loss']:.4f})")

        # 5. Save + reload as TrainedPolicy
        out = RESULTS_DIR / f"policy_dagger_iter{it}.npz"
        np.savez(
            out,
            params=np.array(params, dtype=object),
            x_mean=x_norm.mean, x_std=x_norm.std,
            y_mean=y_norm.mean, y_std=y_norm.std,
            V_ref=V_ref,
            hidden_sizes=np.array([32, 32]),
        )
        policy = TrainedPolicy.load(out)

        # Evaluate
        ev = evaluate_policy(policy, sim_p, sim_step,
                             n_eval=10, episode_length=30, seed=99)
        print(f"  closed-loop eval: mean err {ev['mean_err']:.2f}, "
              f"quenches {ev['quench_count']}/{ev['n_eval']}")
        history.append({"iter": it, "n_samples": int(all_states.shape[0]),
                        "train_loss": log[-1]["train_loss"],
                        "val_loss": log[-1]["val_loss"], **ev})

    # Save final
    np.savez_compressed(
        RESULTS_DIR / "dagger_dataset.npz",
        states=all_states, targets=all_targets, voltages=all_voltages,
    )
    final_path = RESULTS_DIR / "policy_dagger.npz"
    np.savez(
        final_path,
        params=np.array(params, dtype=object),
        x_mean=x_norm.mean, x_std=x_norm.std,
        y_mean=y_norm.mean, y_std=y_norm.std,
        V_ref=V_ref, hidden_sizes=np.array([32, 32]),
    )

    with open(RESULTS_DIR / "dagger_history.json", "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)

    print(f"\n✓ Saved: {final_path}")
    print(f"\nDAgger summary:")
    print(f"  {'iter':>5} {'samples':>8} {'mean err':>10} {'quench':>8}")
    for h in history:
        print(f"  {h['iter']:5d} {h['n_samples']:8d} "
              f"{h['mean_err']:10.2f} {h['quench_count']:5d}/{h['n_eval']}")


if __name__ == "__main__":
    main()
