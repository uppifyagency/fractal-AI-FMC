"""DAgger with JIT FMC backbone (Milestone 8).

Drop-in replacement for dagger_train.py that uses FMCPlasmaJaxController
for the expert oracle. Same algorithm, ~200× faster expert labeling.

Enables:
  - 20-50 DAgger iterations in minutes (vs days)
  - 1000+ samples per iteration (vs 200)
  - Convergence study to plateau

Result expected: closes the 1.5× residual gap between M6 DAgger and FMC online.
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

from fmc_plasma_jax import FMCPlasmaJaxController
from generate_expert_dataset import sample_initial_state, sample_target
from plasma_simulator_jax import DTYPE, build_jax_params, make_jit_step
from policy import (
    Normalizer, TrainedPolicy, build_features, rescale_state,
)
from train_policy import train_loop

RESULTS_DIR = Path(__file__).parent.parent / "results"


def collect_visited_states_jax(
    policy: TrainedPolicy,
    sim_p, sim_step,
    n_episodes: int, episode_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Roll out current policy → collect (state, target) pairs at decision points."""
    states_list, targets_list, targets_obj = [], [], []

    dt = jnp.asarray(1e-3, dtype=DTYPE)
    P_aux = jnp.asarray(5e5, dtype=DTYPE)
    gas = jnp.asarray(1e21, dtype=DTYPE)

    for ep in range(n_episodes):
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
        tgt_arr = np.array([target.R_p, target.Z_p, target.kappa, target.delta],
                           dtype=np.float32)

        for t in range(episode_length):
            states_list.append(x.copy())
            targets_list.append(tgt_arr.copy())
            targets_obj.append(target)

            V = policy(x, tgt_arr)
            x = np.array(sim_step(
                jnp.asarray(x), jnp.asarray(V, dtype=DTYPE),
                P_aux, gas, dt,
            ))
            if np.isnan(x).any() or np.isinf(x).any():
                break

    return (np.array(states_list, dtype=np.float32),
            np.array(targets_list, dtype=np.float32),
            targets_obj)


def query_expert_jax(
    states: np.ndarray, targets_obj: list,
    sim_p, jx_ctrl: FMCPlasmaJaxController,
) -> np.ndarray:
    """Query JIT FMC at each (state, target) → V*. Reuses single warm controller."""
    voltages = np.zeros((len(states), sim_p.N), dtype=np.float32)
    for i, (s, tgt) in enumerate(zip(states, targets_obj)):
        tgt_arr = np.array([tgt.R_p, tgt.Z_p, tgt.kappa, tgt.delta], dtype=np.float32)
        d = jx_ctrl.decide(s, tgt_arr)
        voltages[i] = d["V_coils"]
    return voltages


def evaluate_policy(policy, sim_p, sim_step, n_eval=10, episode_length=30, seed=99):
    rng = np.random.default_rng(seed)
    errors = []
    quenches = 0
    for ep in range(n_eval):
        x = sample_initial_state(rng, sim_p)
        target = sample_target(rng)
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
    return {"mean_err": float(np.mean(errors)), "quench": quenches, "n_eval": n_eval}


def build_xy(states, targets, voltages, sim_p):
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)
    states_r = rescale_state(states)
    I_ref_proxy = V_ref / max(V_ref.max(), 1.0)
    X = build_features(states_r, targets, I_ref_proxy)
    Y = (voltages - V_ref).astype(np.float32)
    return X, Y, V_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=20)
    ap.add_argument("--samples_per_iter", type=int, default=1000)
    ap.add_argument("--episode_length", type=int, default=20)
    ap.add_argument("--fmc_walkers", type=int, default=32)
    ap.add_argument("--fmc_horizon", type=int, default=8)
    ap.add_argument("--epochs_per_iter", type=int, default=200)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("DAgger (JIT backbone) — extended run")
    print(f"  n_iter            : {args.n_iter}")
    print(f"  samples per iter  : {args.samples_per_iter}")
    print(f"  hidden            : {args.hidden}")
    print(f"  FMC config        : M={args.fmc_walkers}, H={args.fmc_horizon}")
    print("=" * 70)

    sim_p, _ = build_jax_params()
    sim_step = make_jit_step(sim_p)

    # Persistent JIT FMC controller (single jit cache, reused across all queries)
    jx_ctrl = FMCPlasmaJaxController(
        sim_p, n_walkers=args.fmc_walkers, horizon=args.fmc_horizon, seed=args.seed,
    )
    # Warmup
    jx_ctrl.decide(np.zeros(27, dtype=np.float32),
                   np.array([0.88, 0, 1.7, 0.3], dtype=np.float32))

    # Load M5 dataset as D_0
    d0 = np.load(RESULTS_DIR / "expert_dataset.npz")
    all_states = d0["states"].copy()
    all_targets = d0["targets"].copy()
    all_voltages = d0["voltages"].copy()

    # M5 BC policy as starting π_0
    policy = TrainedPolicy.load(RESULTS_DIR / "policy_params.npz")
    eval0 = evaluate_policy(policy, sim_p, sim_step, n_eval=10, seed=99)
    print(f"\n[iter 0] D_0: {all_states.shape[0]} samples (M5)")
    print(f"          eval: mean err {eval0['mean_err']:.2f}, "
          f"quench {eval0['quench']}/{eval0['n_eval']}")
    history = [{"iter": 0, "n_samples": int(all_states.shape[0]), **eval0}]

    rng = np.random.default_rng(args.seed)
    grand_t0 = time.perf_counter()

    for it in range(1, args.n_iter + 1):
        t0 = time.perf_counter()
        n_episodes = max(args.samples_per_iter // args.episode_length, 1)

        # 1. Roll out
        states_v, targets_v, targets_obj = collect_visited_states_jax(
            policy, sim_p, sim_step,
            n_episodes=n_episodes, episode_length=args.episode_length, rng=rng,
        )
        t_rollout = time.perf_counter() - t0

        # 2. JIT FMC labeling (the real win of M8)
        t1 = time.perf_counter()
        voltages_v = query_expert_jax(states_v, targets_obj, sim_p, jx_ctrl)
        t_label = time.perf_counter() - t1

        # 3. Aggregate
        all_states = np.concatenate([all_states, states_v], axis=0)
        all_targets = np.concatenate([all_targets, targets_v], axis=0)
        all_voltages = np.concatenate([all_voltages, voltages_v], axis=0)

        # 4. Retrain
        X, Y, V_ref = build_xy(all_states, all_targets, all_voltages, sim_p)
        t2 = time.perf_counter()
        params, x_norm, y_norm, log = train_loop(
            X, Y, hidden_sizes=tuple(args.hidden),
            lr=3e-3, n_epochs=args.epochs_per_iter, batch_size=64,
            seed=args.seed + it, weight_decay=1e-3, patience=30,
        )
        t_train = time.perf_counter() - t2

        # 5. Save + reload
        out = RESULTS_DIR / f"policy_dagger_jax_iter{it}.npz"
        np.savez(
            out, params=np.array(params, dtype=object),
            x_mean=x_norm.mean, x_std=x_norm.std,
            y_mean=y_norm.mean, y_std=y_norm.std,
            V_ref=V_ref, hidden_sizes=np.array(args.hidden),
        )
        policy = TrainedPolicy.load(out)

        ev = evaluate_policy(policy, sim_p, sim_step, n_eval=10, seed=99)
        history.append({
            "iter": it, "n_samples": int(all_states.shape[0]),
            "train_loss": log[-1]["train_loss"], "val_loss": log[-1]["val_loss"],
            **ev,
            "t_rollout_s": t_rollout, "t_label_s": t_label, "t_train_s": t_train,
        })
        # Concise per-iter line
        print(f"[iter {it:2d}] |D|={all_states.shape[0]:5d} | "
              f"err {ev['mean_err']:6.2f} | quench {ev['quench']}/10 | "
              f"label {t_label:.1f}s | train {t_train:.1f}s | "
              f"val {log[-1]['val_loss']:.3f}")

    grand_total = time.perf_counter() - grand_t0
    print(f"\n✓ Total wall-clock: {grand_total:.1f} sec")

    # Save final
    np.savez_compressed(
        RESULTS_DIR / "dagger_jax_dataset.npz",
        states=all_states, targets=all_targets, voltages=all_voltages,
    )
    final = RESULTS_DIR / "policy_dagger_jax.npz"
    np.savez(
        final, params=np.array(params, dtype=object),
        x_mean=x_norm.mean, x_std=x_norm.std,
        y_mean=y_norm.mean, y_std=y_norm.std,
        V_ref=V_ref, hidden_sizes=np.array(args.hidden),
    )
    with open(RESULTS_DIR / "dagger_jax_history.json", "w") as f:
        json.dump({"config": vars(args), "history": history,
                   "total_wall_s": grand_total}, f, indent=2)
    print(f"✓ Saved: {final}")
    print(f"\nFinal summary:")
    print(f"  iter 0  err={history[0]['mean_err']:.2f} (M5 BC baseline)")
    print(f"  iter 1  err={history[1]['mean_err']:.2f}")
    print(f"  iter {args.n_iter}  err={history[-1]['mean_err']:.2f} "
          f"(after {history[-1]['n_samples']} samples)")


if __name__ == "__main__":
    main()
