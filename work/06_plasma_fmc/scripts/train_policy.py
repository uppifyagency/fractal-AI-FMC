"""Train MLP policy via behavioral cloning on FMC expert dataset.

Inputs : results/expert_dataset.npz
Outputs: results/policy_params.npz (loadable via TrainedPolicy.load)
         results/training_log.json (loss curves)

Loss: MSE between predicted ΔV and (V*_FMC - V_ref).
Optimizer: Adam, cosine LR schedule.
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

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).parent))
from plasma_simulator_jax import build_jax_params
from policy import DTYPE, Normalizer, PolicyMLP, build_features

RESULTS_DIR = Path(__file__).parent.parent / "results"


def make_dataset(dataset_path: Path) -> tuple:
    d = np.load(dataset_path)
    states = d["states"]
    targets = d["targets"]
    voltages = d["voltages"]
    rewards = d["rewards"]
    walkers_alive = d["walkers_alive"]
    print(f"Loaded {states.shape[0]} samples from {dataset_path}")
    print(f"  rewards: mean={rewards.mean():.2f} (higher = better)")
    print(f"  alive  : mean={walkers_alive.mean():.0f}")

    # Filter out failed FMC decisions (no walkers alive or extremely low reward)
    keep = (walkers_alive > 0)
    if keep.sum() < states.shape[0]:
        print(f"  filtering {(~keep).sum()} samples with 0 walkers alive")
        states = states[keep]
        targets = targets[keep]
        voltages = voltages[keep]

    sim_p, _ = build_jax_params()
    V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)

    # ΔV = V* - V_ref (target output)
    dV = voltages - V_ref

    # Rescale to natural units (avoids float32 overflow during normalization)
    from policy import rescale_state
    states_rescaled = rescale_state(states)

    # Build feature vectors. We use the OPERATING POINT proxy (V_ref / max(V_ref))
    # for the I_ref part, so the policy stays unit-consistent.
    I_ref_proxy = V_ref / max(V_ref.max(), 1.0)
    X = build_features(states_rescaled, targets, I_ref_proxy)
    Y = dV.astype(np.float32)

    return X, Y, V_ref


def train_loop(
    X: np.ndarray, Y: np.ndarray,
    hidden_sizes: tuple = (32, 32),
    lr: float = 3e-3, n_epochs: int = 300, batch_size: int = 32,
    val_frac: float = 0.15, seed: int = 0,
    weight_decay: float = 1e-3,
    patience: int = 30,
) -> tuple:
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")

    # Normalize
    x_norm = Normalizer.fit(X_tr)
    y_norm = Normalizer.fit(Y_tr)

    X_tr_n = x_norm.transform(X_tr).astype(np.float32)
    Y_tr_n = y_norm.transform(Y_tr).astype(np.float32)
    X_va_n = x_norm.transform(X_va).astype(np.float32)
    Y_va_n = y_norm.transform(Y_va).astype(np.float32)

    # Init model
    model = PolicyMLP(hidden_sizes=hidden_sizes, output_dim=Y.shape[1])
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, X.shape[1]), dtype=DTYPE))

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"Model: PolicyMLP{hidden_sizes} → {n_params} params")

    # Optimizer with cosine schedule + weight decay (AdamW)
    n_steps = (len(train_idx) // batch_size) * n_epochs
    schedule = optax.cosine_decay_schedule(lr, n_steps)
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        pred = model.apply(params, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def update(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # AdamW needs params for the weight-decay term
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    log = []
    best_val = float("inf")
    best_params = params
    epochs_no_improve = 0
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        order = rng.permutation(len(train_idx))
        epoch_losses = []
        for k in range(0, len(order), batch_size):
            b = order[k:k + batch_size]
            xb = jnp.asarray(X_tr_n[b])
            yb = jnp.asarray(Y_tr_n[b])
            params, opt_state, loss = update(params, opt_state, xb, yb)
            epoch_losses.append(float(loss))

        train_loss = float(np.mean(epoch_losses))
        val_loss = float(loss_fn(
            params, jnp.asarray(X_va_n), jnp.asarray(Y_va_n),
        ))
        log.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_params = jax.tree.map(lambda x: x, params)  # snapshot
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"  epoch {epoch+1:4d}/{n_epochs} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} (best {best_val:.4f}) | "
                  f"elapsed {time.perf_counter()-t0:.1f}s")
        if epochs_no_improve >= patience:
            print(f"  early stop at epoch {epoch+1} (no val improvement for {patience})")
            break

    return best_params, x_norm, y_norm, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default=str(RESULTS_DIR / "expert_dataset.npz"))
    ap.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    X, Y, V_ref = make_dataset(Path(args.dataset))
    params, x_norm, y_norm, log = train_loop(
        X, Y,
        hidden_sizes=tuple(args.hidden),
        lr=args.lr, n_epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
        weight_decay=args.weight_decay, patience=args.patience,
    )

    final_train = log[-1]["train_loss"]
    final_val = log[-1]["val_loss"]
    print(f"\n✓ Final: train MSE = {final_train:.5f}, val MSE = {final_val:.5f}")

    # Save params
    out = RESULTS_DIR / "policy_params.npz"
    np.savez(
        out,
        params=np.array(params, dtype=object),
        x_mean=x_norm.mean, x_std=x_norm.std,
        y_mean=y_norm.mean, y_std=y_norm.std,
        V_ref=V_ref,
        hidden_sizes=np.array(args.hidden),
    )
    print(f"  → {out}")

    log_out = RESULTS_DIR / "training_log.json"
    with open(log_out, "w") as f:
        json.dump({
            "config": vars(args),
            "log": log,
            "final_train_loss": final_train,
            "final_val_loss": final_val,
        }, f, indent=2)
    print(f"  → {log_out}")


if __name__ == "__main__":
    main()
