"""2D continuous toy environments for FMC mathematical tests.

The environment is a bounded square [0, L]^2. A walker state is a point in R^2.
The perturbation kernel is a small isotropic Gaussian step (the random-action
analog of a discrete-action environment). Reward is a user-specified function
R(x, y).

For F12 (cross-entropy collapse) we use STRICTLY POSITIVE rewards (paper section 2.2.3).
For F11 (relativize ablation) we use MIXED-SIGN rewards.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BoundedDomain:
    L: float = 10.0
    step_sigma: float = 0.30  # Gaussian perturbation std

    def step_fn(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One-step Gaussian random-walk perturbation, reflected at boundaries."""
        delta = rng.normal(0.0, self.step_sigma, size=states.shape)
        new = states + delta
        # reflect at boundaries (no walker dies for boundary leaves)
        below = new < 0
        above = new > self.L
        new = np.where(below, -new, new)
        new = np.where(above, 2 * self.L - new, new)
        new = np.clip(new, 0.0, self.L)
        return new

    def random_init(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, self.L, size=(n, 2))


def gaussian_mixture_reward(centers: list[tuple[float, float]],
                            sigmas: list[float],
                            weights: list[float],
                            baseline: float = 0.05) -> callable:
    """All-positive reward: sum of K Gaussians plus a small baseline so R > 0 everywhere.

    R(x, y) = baseline + sum_k weights_k * exp(-||(x,y) - c_k||^2 / (2 sigma_k^2))
    """
    centers_arr = np.array(centers, dtype=np.float64)
    sigmas_arr = np.array(sigmas, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    K = len(centers_arr)

    def R(states: np.ndarray) -> np.ndarray:
        # states: (N, 2)
        d2 = ((states[:, None, :] - centers_arr[None, :, :]) ** 2).sum(axis=2)  # (N, K)
        gauss = np.exp(-d2 / (2.0 * sigmas_arr[None, :] ** 2))
        return baseline + (weights_arr[None, :] * gauss).sum(axis=1)

    R.centers = centers_arr
    R.sigmas = sigmas_arr
    R.weights = weights_arr
    R.baseline = baseline
    R.name = "gaussian_mixture_positive"
    return R


def mixed_sign_reward(pos_centers: list[tuple[float, float]],
                      pos_sigmas: list[float],
                      pos_weights: list[float],
                      neg_centers: list[tuple[float, float]],
                      neg_sigmas: list[float],
                      neg_weights: list[float],
                      global_offset: float = 0.0) -> callable:
    """Mixed-sign reward landscape: positive peaks plus negative wells, plus a constant offset.

    R(x, y) = global_offset
            + sum_pos w_k exp(-d^2/2sigma_k^2)
            - sum_neg w_k exp(-d^2/2sigma_k^2)

    With ``global_offset`` < 0 the entire domain has R < 0 except near positive peaks,
    which is exactly the regime in which Sergio claims raw rewards yield fearful agents.
    """
    pc = np.array(pos_centers, dtype=np.float64)
    ps = np.array(pos_sigmas, dtype=np.float64)
    pw = np.array(pos_weights, dtype=np.float64)
    nc = np.array(neg_centers, dtype=np.float64)
    ns = np.array(neg_sigmas, dtype=np.float64)
    nw = np.array(neg_weights, dtype=np.float64)

    def R(states: np.ndarray) -> np.ndarray:
        d2_p = ((states[:, None, :] - pc[None, :, :]) ** 2).sum(axis=2)
        d2_n = ((states[:, None, :] - nc[None, :, :]) ** 2).sum(axis=2)
        pos = (pw[None, :] * np.exp(-d2_p / (2.0 * ps[None, :] ** 2))).sum(axis=1)
        neg = (nw[None, :] * np.exp(-d2_n / (2.0 * ns[None, :] ** 2))).sum(axis=1)
        return global_offset + pos - neg

    R.name = "mixed_sign"
    R.pos_centers = pc
    R.neg_centers = nc
    R.global_offset = global_offset
    return R


def smooth_gradient_negative_reward(slope: float = 0.10,
                                    offset: float = -0.7,
                                    L: float = 10.0) -> callable:
    """All-negative reward with monotone x-gradient toward the right.

    R(x, y) = offset + slope * x

    With offset = -0.7 and slope = 0.10 and L = 10, R ranges over [-0.7, +0.3]
    but most of the start half [0, 5] is strictly negative. This is the regime
    in which Sergio's claim should bite hardest: relativize should *normalize
    away* the global negativity and recover a usable gradient signal.
    """
    def R(states: np.ndarray) -> np.ndarray:
        return offset + slope * states[:, 0]

    R.name = "smooth_gradient_negative"
    R.slope = slope
    R.offset = offset
    R.L = L
    return R


def grid_evaluate(R, L: float, n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate R on a regular grid for visualization and target-density computation."""
    xs = np.linspace(0, L, n_grid)
    ys = np.linspace(0, L, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = R(pts).reshape(n_grid, n_grid)
    return X, Y, Z
