"""NN policy for plasma control — Flax MLP.

Architecture (small + interpretable):
    input  : [state(27) | target(4) | I_ref(20)] = 51 features
    hidden : 2 × 128 ReLU
    output : ΔV (20) — *delta* from V_ref, not absolute V

Why ΔV rather than V?
- V_ref = R · I_ref is well-defined; the policy learns deviations
- Centers the output near 0 → easier optimization
- At inference: V_command = V_ref + π_θ(state, target)

Why include I_ref in input?
- The policy must know the reference operating point to make
  meaningful corrections; otherwise it would have to memorize V_ref.

Mathematical form:
    π_θ(x) = W₃ · ReLU(W₂ · ReLU(W₁ · x + b₁) + b₂) + b₃
    where x = [state | target | I_ref]

Loss (behavioral cloning, Ross-Bagnell 2010):
    L(θ) = E_{(s,τ,V*) ~ D} [ ||π_θ(s, τ) - (V* - V_ref)||² ]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

DTYPE = jnp.float32


# ---- Network ----

class PolicyMLP(nn.Module):
    hidden_sizes: Sequence[int] = (128, 128)
    output_dim: int = 20

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.Dense(h, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim,
                        kernel_init=nn.initializers.he_normal())(x)


# ---- Feature builder ----

def rescale_state(state: np.ndarray) -> np.ndarray:
    """Convert raw plasma state to natural units (avoids float32 overflow).

    State layout: [I_coils(20), I_p, W, n_bar, R_p, Z_p, kappa, delta]
        I_coils: A   → kA   (÷ 1e3)
        I_p    : A   → MA   (÷ 1e6)
        W      : J   → kJ   (÷ 1e3)
        n_bar  : m⁻³ → 10¹⁹ (÷ 1e19)
    """
    s = state.astype(np.float32, copy=True)
    if s.ndim == 1:
        s[:20] /= 1e3
        s[20] /= 1e6
        s[21] /= 1e3
        s[22] /= 1e19
    else:
        s[:, :20] /= 1e3
        s[:, 20] /= 1e6
        s[:, 21] /= 1e3
        s[:, 22] /= 1e19
    return s


def build_features(state: np.ndarray, target: np.ndarray, I_ref: np.ndarray) -> np.ndarray:
    """Concatenate inputs into the 51-D feature vector.

    Args:
        state: (27,) or (B, 27) — assumed already rescaled
        target: (4,) or (B, 4)
        I_ref: (20,) — proxy of operating point (unit-normalized)

    Returns:
        (51,) or (B, 51)
    """
    if state.ndim == 1:
        return np.concatenate([state, target, I_ref])
    B = state.shape[0]
    I_ref_b = np.broadcast_to(I_ref, (B, I_ref.shape[0]))
    return np.concatenate([state, target, I_ref_b], axis=1)


# ---- Normalization ----

@dataclass
class Normalizer:
    """Per-feature mean/std normalization. Learned from training data."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mu = x.mean(axis=0)
        sd = x.std(axis=0) + 1e-6
        return cls(mean=mu.astype(np.float32), std=sd.astype(np.float32))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=d["mean"], std=d["std"])


# ---- Policy wrapper for inference ----

class TrainedPolicy:
    """Bundles the network params + normalizers + V_ref for inference.

    Loaded from the .npz produced by `train_policy.py`.
    """

    def __init__(self, params, x_norm: Normalizer, y_norm: Normalizer,
                 V_ref: np.ndarray, hidden_sizes: tuple = (128, 128)):
        self.params = params
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.V_ref = V_ref.astype(np.float32)
        self.model = PolicyMLP(hidden_sizes=hidden_sizes, output_dim=V_ref.shape[0])
        self._jit_apply = jax.jit(self.model.apply)
        # Warm jit
        dummy = jnp.zeros((1, len(x_norm.mean)), dtype=DTYPE)
        self._jit_apply(self.params, dummy).block_until_ready()

    def __call__(self, state: np.ndarray, target: np.ndarray,
                 max_dV: float = 500.0) -> np.ndarray:
        """Return V_coils command for one (state, target) pair (or batch).

        Clips ΔV to ±max_dV (default 500 V) — defensive measure against
        out-of-distribution states (covariate shift, Ross-Bagnell 2010).
        Training data showed |ΔV| up to ~660 V, so 500 V is conservative.
        """
        state_r = rescale_state(np.asarray(state))
        I_ref_proxy = self.V_ref / max(self.V_ref.max(), 1.0)
        features = build_features(state_r, np.asarray(target, dtype=np.float32), I_ref_proxy)
        z = self.x_norm.transform(features)
        z = jnp.asarray(z, dtype=DTYPE)
        if z.ndim == 1:
            z = z[None]
        dV_norm = self._jit_apply(self.params, z)
        dV = self.y_norm.inverse(np.asarray(dV_norm))
        if dV.shape[0] == 1:
            dV = dV[0]
        dV = np.clip(dV, -max_dV, max_dV)
        return self.V_ref + dV

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        params = d["params"].item()
        x_norm = Normalizer.from_dict({"mean": d["x_mean"], "std": d["x_std"]})
        y_norm = Normalizer.from_dict({"mean": d["y_mean"], "std": d["y_std"]})
        V_ref = d["V_ref"]
        hidden = tuple(int(h) for h in d["hidden_sizes"])
        return cls(params, x_norm, y_norm, V_ref, hidden_sizes=hidden)
