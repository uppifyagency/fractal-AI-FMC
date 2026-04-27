"""Train MLP shape surrogate: I_coils → shape.

Replaces the linearized S matrix with a non-linear model.
Compare prediction quality vs the linear baseline (M2/M3 synthetic S).
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

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).parent))

from policy import DTYPE, Normalizer

RESULTS_DIR = Path(__file__).parent.parent / "results"


class ShapeMLP(nn.Module):
    """Map I_coils (20) → shape parameters (R_p, Z_p, κ, δ)."""
    hidden_sizes: tuple = (64, 64)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(4, kernel_init=nn.initializers.he_normal())(x)


def main():
    print("=" * 70)
    print("Milestone 11 — train shape surrogate")
    print("=" * 70)

    # Load
    d = np.load(RESULTS_DIR / "freegs_shape_dataset.npz")
    I_coils = d["I_coils"]  # (B, 20) in Amps
    shape = d["shape"]      # (B, 4)
    print(f"Loaded {I_coils.shape[0]} samples")

    # Rescale: I_coils to kA for numerics
    X = I_coils.astype(np.float32) / 1000.0
    Y = shape.astype(np.float32)

    # Normalize input/output
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n_val = len(X) // 5
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]

    x_norm = Normalizer.fit(X_tr)
    y_norm = Normalizer.fit(Y_tr)

    X_tr_n = x_norm.transform(X_tr).astype(np.float32)
    Y_tr_n = y_norm.transform(Y_tr).astype(np.float32)
    X_va_n = x_norm.transform(X_va).astype(np.float32)
    Y_va_n = y_norm.transform(Y_va).astype(np.float32)

    print(f"  Train: {X_tr_n.shape[0]} | Val: {X_va_n.shape[0]}")

    # Model
    model = ShapeMLP(hidden_sizes=(64, 64))
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 20)))
    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"  Model: ShapeMLP(64,64) → {n_params} params")

    schedule = optax.cosine_decay_schedule(3e-3, 200)
    optimizer = optax.adamw(schedule, weight_decay=1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        return jnp.mean((model.apply(params, x) - y) ** 2)

    @jax.jit
    def update(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    log = []
    best_val = float("inf")
    best_params = params
    no_improve = 0
    bsz = 32
    t0 = time.perf_counter()
    for epoch in range(300):
        order = rng.permutation(X_tr_n.shape[0])
        losses = []
        for k in range(0, len(order), bsz):
            b = order[k:k+bsz]
            xb = jnp.asarray(X_tr_n[b])
            yb = jnp.asarray(Y_tr_n[b])
            params, opt_state, lo = update(params, opt_state, xb, yb)
            losses.append(float(lo))
        train_loss = float(np.mean(losses))
        val_loss = float(loss_fn(params, jnp.asarray(X_va_n), jnp.asarray(Y_va_n)))
        log.append({"epoch": epoch+1, "train": train_loss, "val": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_params = jax.tree.map(lambda x: x, params)
            no_improve = 0
        else:
            no_improve += 1
        if (epoch+1) % 30 == 0:
            print(f"  epoch {epoch+1:4d}: train {train_loss:.4f} | val {val_loss:.4f} (best {best_val:.4f})")
        if no_improve >= 30:
            print(f"  early stop at epoch {epoch+1}")
            break

    print(f"\n  Trained in {time.perf_counter()-t0:.1f}s, best val MSE = {best_val:.4f}")

    # Compare physical (un-normalized) prediction quality vs linear S baseline
    pred_va_n = np.asarray(model.apply(best_params, jnp.asarray(X_va_n)))
    pred_va = y_norm.inverse(pred_va_n)
    err_per_dim = np.mean((pred_va - Y_va) ** 2, axis=0)
    print(f"\n  Per-dim MSE on val (physical units):")
    print(f"    R_p   : {err_per_dim[0]:.4e}  RMSE = {np.sqrt(err_per_dim[0]):.4f} m")
    print(f"    Z_p   : {err_per_dim[1]:.4e}  RMSE = {np.sqrt(err_per_dim[1]):.4f} m")
    print(f"    kappa : {err_per_dim[2]:.4e}  RMSE = {np.sqrt(err_per_dim[2]):.4f}")
    print(f"    delta : {err_per_dim[3]:.4e}  RMSE = {np.sqrt(err_per_dim[3]):.4f}")

    # Compare to "predict mean" baseline
    Y_mean = Y_tr.mean(axis=0)
    err_mean = np.mean((Y_mean - Y_va) ** 2, axis=0)
    print(f"\n  Baseline 'predict mean' for comparison:")
    for j, name in enumerate(["R_p", "Z_p", "kappa", "delta"]):
        improvement = err_mean[j] / max(err_per_dim[j], 1e-10)
        print(f"    {name:6s}: NN MSE {err_per_dim[j]:.4e} vs mean {err_mean[j]:.4e}  "
              f"({improvement:.1f}× better)")

    # Save
    out = RESULTS_DIR / "shape_surrogate.npz"
    np.savez(
        out, params=np.array(best_params, dtype=object),
        x_mean=x_norm.mean, x_std=x_norm.std,
        y_mean=y_norm.mean, y_std=y_norm.std,
        hidden_sizes=np.array([64, 64]),
        per_dim_rmse=np.sqrt(err_per_dim),
        per_dim_rmse_baseline=np.sqrt(err_mean),
    )
    print(f"\n✓ Saved: {out}")

    # Save log
    with open(RESULTS_DIR / "shape_surrogate_log.json", "w") as f:
        json.dump({"log": log, "best_val": best_val,
                   "per_dim_rmse": err_per_dim.tolist(),
                   "per_dim_rmse_baseline": err_mean.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
