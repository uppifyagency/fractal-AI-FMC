"""Compare NN shape surrogate vs linear S model on FreeGS held-out data."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))

from calibrated_sim import M9_BASELINE
from plasma_simulator_jax import build_jax_params
from policy import Normalizer
from train_shape_surrogate import ShapeMLP

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    print("=" * 70)
    print("Milestone 11 — NN shape surrogate vs linear S")
    print("=" * 70)

    # Load freegs dataset
    d = np.load(RESULTS_DIR / "freegs_shape_dataset.npz")
    I_coils = d["I_coils"]
    shape_true = d["shape"]
    coil_order = d["coil_order"]

    # Reproduce same train/val split (rng seed = 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(I_coils))
    n_val = len(I_coils) // 5
    val_idx = perm[:n_val]
    I_va = I_coils[val_idx]
    Y_va = shape_true[val_idx]

    # ---- 1. NN prediction ----
    nn_data = np.load(RESULTS_DIR / "shape_surrogate.npz", allow_pickle=True)
    params = nn_data["params"].item()
    x_mean = nn_data["x_mean"]; x_std = nn_data["x_std"]
    y_mean = nn_data["y_mean"]; y_std = nn_data["y_std"]
    model = ShapeMLP(hidden_sizes=(64, 64))

    X_va_kA = (I_va.astype(np.float32) / 1000.0)
    X_n = (X_va_kA - x_mean) / x_std
    Y_pred_n = np.asarray(model.apply(params, jnp.asarray(X_n)))
    Y_pred_nn = Y_pred_n * y_std + y_mean

    # ---- 2. Linear S prediction ----
    # Build the same calibrated simulator (M10): same S × 10 around M9 baseline
    from calibrated_sim import build_calibrated_jax_params
    sim_p, _ = build_calibrated_jax_params(s_scale=10.0)
    S = np.asarray(sim_p.S)        # (4, 20)
    I_ref = np.asarray(sim_p.I_ref)  # (20,)
    shape_ref = np.array([sim_p.R_ref, sim_p.Z_ref,
                           sim_p.kappa_ref, sim_p.delta_ref])

    # NB: I_coils from FreeGS uses coil_order (E1..E8, F1..F8, T1..T3, C1)
    # Our sim uses the same first 19 + OH lumped at index 19 — check alignment.
    # For the FreeGS dataset, the OH circuit is represented by C1 only (proxy).
    # In our sim, index 19 is the multi-turn solenoid. Different scale!
    # → we should compare only on E + F coils + T (indices 0..18)
    n_match = 19
    Y_pred_lin = []
    for I_va_row in I_va:
        # Only first 19 indices match (E1..E8, F1..F8, T1..T3)
        dI = np.zeros(20)
        dI[:n_match] = I_va_row[:n_match] - I_ref[:n_match]
        delta_shape = S @ dI
        Y_pred_lin.append(shape_ref + delta_shape)
    Y_pred_lin = np.array(Y_pred_lin)

    # ---- 3. Compute RMSE per dimension ----
    print(f"\n  Per-dim RMSE on {n_val} held-out FreeGS samples:")
    print(f"  {'param':6s}  {'linear S':>14s}  {'NN surrogate':>14s}  {'NN gain':>10s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")
    out = {"per_dim": {}}
    for j, name in enumerate(["R_p", "Z_p", "kappa", "delta"]):
        rmse_lin = float(np.sqrt(np.mean((Y_pred_lin[:, j] - Y_va[:, j])**2)))
        rmse_nn = float(np.sqrt(np.mean((Y_pred_nn[:, j] - Y_va[:, j])**2)))
        gain = rmse_lin / max(rmse_nn, 1e-10)
        print(f"  {name:6s}  {rmse_lin:14.4f}  {rmse_nn:14.4f}  {gain:10.1f}×")
        out["per_dim"][name] = {"rmse_linear": rmse_lin, "rmse_nn": rmse_nn,
                                 "nn_gain": gain}

    # Aggregate
    all_lin = np.sqrt(np.mean((Y_pred_lin - Y_va)**2))
    all_nn = np.sqrt(np.mean((Y_pred_nn - Y_va)**2))
    print(f"\n  Total RMSE: linear S = {all_lin:.4f}, NN = {all_nn:.4f}")
    print(f"  NN aggregate gain: {all_lin/all_nn:.1f}×")
    out["aggregate"] = {"rmse_linear": float(all_lin), "rmse_nn": float(all_nn)}

    # Save
    out_path = RESULTS_DIR / "milestone_11_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
