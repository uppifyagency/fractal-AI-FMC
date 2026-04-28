"""Reference FMC implementation for mathematical tests.

Faithful to Hernandez-Cerezo & Duran-Ballester 2020 (arXiv:1803.05049v5)
sections 2.2.3 (relativize), 4.4 (virtual reward + pairwise clone), 4.5 (O(N) distance).
Cross-checked against repos/FractalAI_old/fractalai/swarm.py:16-23,451-531.

Numpy only, no GPU, no env coupling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


def relativize(v: np.ndarray) -> np.ndarray:
    """Paper section 2.2.3 reshape: z-score then asymmetric exp/log.

        R_N = (R - mu) / sigma
        R_hat = exp(R_N)         if R_N <= 0
        R_hat = 1 + ln(1 + R_N)  if R_N > 0

    Guarantees R_hat > 0, preserves order, compresses positive outliers
    via log and expands negative outliers via exp.
    Identical to FractalAI_old/fractalai/swarm.py:16-23.
    """
    v = np.asarray(v, dtype=np.float64)
    std = v.std()
    if std == 0:
        return np.ones_like(v)
    z = (v - v.mean()) / std
    out = np.empty_like(z)
    pos = z > 0
    out[pos] = np.log1p(z[pos]) + 1.0
    out[~pos] = np.exp(z[~pos])
    return out


@dataclass
class FMCConfig:
    """Configuration for the reference FMC swarm."""

    n_walkers: int = 200
    balance: float = 1.0  # alpha exponent on rewards (exploitation strength)
    use_relativize_reward: bool = True
    use_relativize_distance: bool = True
    # When use_relativize_reward=False, negative rewards can either be clipped
    # to 0 in the VR computation (the implementation default, similar to a
    # "valid reward" gate) or propagated as-is (Sergio's fearful-agent regime).
    clip_negative_reward_at_zero: bool = True
    rng_seed: int = 0
    eps: float = 1e-12


@dataclass
class FMCSwarm:
    """Minimal FMC swarm operating on R^d states.

    The environment is encoded by two callbacks supplied at construction:
        step_fn(state)  -> next_state   (one walker rollout step)
        reward_fn(state) -> scalar reward (>=0 for canonical FMC)
    """

    step_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    reward_fn: Callable[[np.ndarray], np.ndarray]
    init_states: np.ndarray  # shape (N, d)
    config: FMCConfig = field(default_factory=FMCConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.rng_seed)
        self.states = np.array(self.init_states, dtype=np.float64, copy=True)
        assert self.states.ndim == 2 and self.states.shape[0] == self.config.n_walkers
        self.rewards = self.reward_fn(self.states)  # instantaneous reward at current state
        self.history_states: list[np.ndarray] = [self.states.copy()]
        self.history_rewards: list[np.ndarray] = [self.rewards.copy()]

    # paper section 4.5 stochastic O(N) distance
    def _distance_to_random_partner(self) -> np.ndarray:
        N = self.config.n_walkers
        idx = self.rng.permutation(N)
        diff = self.states[idx] - self.states
        return np.linalg.norm(diff, axis=1)

    # paper section 4.4 virtual reward (instantaneous reward, faithful to swarm.py:469-480)
    def _virtual_reward(self) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        d = self._distance_to_random_partner()
        d_norm = relativize(d) if cfg.use_relativize_distance else d
        r_norm = relativize(self.rewards) if cfg.use_relativize_reward else self.rewards
        # FractalAI_old uses vir_reward = dist * scores ** balance with relativize on (scores > 0).
        # When relativize is off we either clip negatives at 0 (clean threshold variant) or let
        # negatives propagate via sign-preserving power (Sergio's fearful-agent regime).
        with np.errstate(invalid="ignore"):
            if cfg.use_relativize_reward or cfg.clip_negative_reward_at_zero:
                r_pow = np.where(r_norm > 0,
                                 np.power(np.maximum(r_norm, cfg.eps), cfg.balance), 0.0)
            else:
                # sign-preserving power: sgn(r) * |r|^alpha
                r_pow = np.sign(r_norm) * np.power(np.abs(r_norm) + cfg.eps, cfg.balance)
        vr = d_norm * r_pow
        return vr, d_norm

    # paper section 4.4 pairwise stochastic clone
    def _clone_step(self, vr: np.ndarray):
        N = self.config.n_walkers
        partners = self.rng.integers(0, N, size=N)
        vr_partner = vr[partners]
        denom = np.where(vr > 0, vr, self.config.eps)
        clone_prob = (vr_partner - vr) / denom
        clone_prob = np.clip(clone_prob, 0.0, 1.0)
        will_clone = self.rng.random(N) < clone_prob
        new_states = self.states.copy()
        new_rewards = self.rewards.copy()
        new_states[will_clone] = self.states[partners[will_clone]]
        new_rewards[will_clone] = self.rewards[partners[will_clone]]
        return will_clone, new_states, new_rewards

    def step(self) -> dict:
        """Single FMC iteration: perturb, score, clone, record."""
        # 1) perturbation (rollout one step per walker)
        self.states = self.step_fn(self.states, self.rng)
        # 2) instantaneous reward at new positions
        self.rewards = self.reward_fn(self.states)
        # 3) virtual reward
        vr, d_norm = self._virtual_reward()
        # 4) pairwise stochastic clone (transports both state AND reward)
        will_clone, new_states, new_rewards = self._clone_step(vr)
        self.states = new_states
        self.rewards = new_rewards
        # record
        self.history_states.append(self.states.copy())
        self.history_rewards.append(self.rewards.copy())
        return {
            "n_cloned": int(will_clone.sum()),
            "vr_mean": float(vr.mean()),
            "vr_std": float(vr.std()),
            "dist_mean": float(d_norm.mean()),
            "reward_mean": float(self.rewards.mean()),
            "reward_max": float(self.rewards.max()),
        }

    def run(self, n_steps: int) -> list[dict]:
        return [self.step() for _ in range(n_steps)]
