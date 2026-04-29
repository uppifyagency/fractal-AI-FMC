"""Standalone NumPy FMC swarm for any plangym PlanEnv.

Adapted from work/03_atari_replication/scripts/fmc_minimal.py — same
algorithm (paper §4.3) but generic over the env interface: uses
`env.get_state()/set_state()` instead of ALE-specific `cloneState`,
and observes via the env's observation vector instead of ALE RAM.

The dependency surface is just NumPy + gymnasium + plangym.core. This
makes the rocket_hook demo runnable without installing the full fragile
+ torch + param + panel stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plangym.core import PlanEnv


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten any-shape observation to a 1-D float32 vector for distance comp.

    For RGB Atari frames (210, 160, 3) we sub-sample to keep the L2 norm cheap.
    """
    arr = np.asarray(obs)
    if arr.ndim >= 3:
        # Sub-sample image obs to a small thumbnail before flattening.
        arr = arr[::8, ::8].astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr.reshape(-1)


def relativize(x: np.ndarray) -> np.ndarray:
    """Reshape vector to positive values preserving order (paper §2.2.3).

    R_N = (R - μ) / σ
    R̂  = exp(R_N)         if R_N ≤ 0
    R̂  = 1 + ln(1 + R_N)  if R_N > 0
    """
    std = float(x.std())
    if std == 0:
        return np.ones_like(x, dtype=np.float32)
    standard = (x - x.mean()) / std
    out = np.where(
        standard <= 0,
        np.exp(np.clip(standard, -50, 0)),
        1 + np.log1p(np.maximum(standard, 0)),
    )
    return out.astype(np.float32)


@dataclass
class FMCConfig:
    n_walkers: int = 60
    time_horizon: int = 20             # M tick of forward planning
    balance: float = 1.0               # used when alpha/beta omitted
    alpha: float | None = None         # reward exponent (defaults to balance)
    beta: float | None = None          # distance exponent (defaults to balance)
    dt: int = 1                        # action frameskip per tick

    @property
    def reward_exp(self) -> float:
        return self.alpha if self.alpha is not None else self.balance

    @property
    def dist_exp(self) -> float:
        return self.beta if self.beta is not None else self.balance


class FMCSwarm:
    """Generic FMC swarm operating on any PlanEnv with discrete actions.

    The swarm holds N independent (state, init_action, cum_reward, alive)
    tuples. Each iteration:
      1. Forward-step every alive walker by `dt` ticks with its own action
         policy (init_action at t=0, random thereafter).
      2. Compute virtual reward `VR = R̂^α * D̂^β` where D̂ is relativized
         pairwise random distance over observations.
      3. Clone walkers via pairwise comparison: walker i clones to partner j
         with probability `(VR_j - VR_i)^+ / VR_j` (paper eq. 12).
      4. After M ticks, return the action whose initial bucket has the most
         alive walkers.
    """

    def __init__(self, env: PlanEnv, config: FMCConfig, rng_seed: int | None = None):
        self.env = env
        self.cfg = config
        self.n_actions = int(env.action_space.n)
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    def _snapshot(self):
        return self.env.get_state()

    def _restore(self, state):
        self.env.set_state(state)

    def _step_dt(self, action: int) -> tuple[float, bool, np.ndarray]:
        """Apply `action` for `dt` env ticks.

        Returns
            (reward_sum, terminal_or_truncated, last_observation_flat)
        """
        total = 0.0
        last_obs: np.ndarray | None = None
        for _ in range(self.cfg.dt):
            obs, r, term, trunc, _info = self.env.apply_action(int(action))
            total += float(r)
            last_obs = np.asarray(obs)
            if term or trunc:
                return total, True, _flatten_obs(last_obs)
        assert last_obs is not None
        return total, False, _flatten_obs(last_obs)

    # ------------------------------------------------------------------
    def decide(self, root_state: Any) -> tuple[int, dict]:
        """Plan one decision from `root_state` using a forward causal cone.

        Returns
            (best_action, info_dict)
        """
        N = self.cfg.n_walkers
        M = self.cfg.time_horizon

        init_actions = self.rng.integers(0, self.n_actions, size=N)
        cum_rewards = np.zeros(N, dtype=np.float64)
        is_dead = np.zeros(N, dtype=bool)

        # Each walker starts with its own copy of root_state. We hold states
        # opaquely (could be ALEState C++ objects) — only env knows how to
        # serialize them. For numpy-state envs we still .copy() to avoid aliasing.
        def _copy_state(s):
            return s.copy() if hasattr(s, "copy") else s

        walker_states: list[Any] = [_copy_state(root_state) for _ in range(N)]

        # Distance buffer: each walker's flattened observation.
        obs_dim_probe = self._step_dt(0)[2].size  # one wasted call, env reset below
        self._restore(_copy_state(root_state))
        obs_buf = np.zeros((N, obs_dim_probe), dtype=np.float32)
        last_obs_per_walker: list[np.ndarray | None] = [None] * N

        for t in range(M):
            for i in range(N):
                if is_dead[i]:
                    if last_obs_per_walker[i] is not None:
                        obs_buf[i] = last_obs_per_walker[i]
                    continue
                self._restore(walker_states[i])
                a = int(init_actions[i]) if t == 0 else int(self.rng.integers(0, self.n_actions))
                r, terminal, last_obs = self._step_dt(a)
                cum_rewards[i] += r
                last_obs_per_walker[i] = last_obs
                obs_buf[i] = last_obs
                if terminal:
                    is_dead[i] = True
                else:
                    walker_states[i] = self._snapshot()

            alive_count = int((~is_dead).sum())
            if alive_count == 0:
                break  # no recovery possible — every walker dead in same tick

            # Pairwise random distance estimator (paper §4.5, F8 video)
            partners = self.rng.permutation(N)
            same = partners == np.arange(N)
            partners[same] = (partners[same] + 1) % N
            distances = np.linalg.norm(obs_buf - obs_buf[partners], axis=1)

            R_norm = relativize(cum_rewards)
            D_norm = relativize(distances)
            R_norm[is_dead] = 0.0
            D_norm[is_dead] = 0.0

            VR = (R_norm ** self.cfg.reward_exp) * (D_norm ** self.cfg.dist_exp)

            # Clone pairwise (paper eq. 12). Dead walkers always want to clone;
            # we route them to a random ALIVE partner so they actually resurrect.
            alive_ix = np.where(~is_dead)[0]
            clone_partners = self.rng.permutation(N)
            same = clone_partners == np.arange(N)
            clone_partners[same] = (clone_partners[same] + 1) % N
            # Override: dead walkers pick a random alive partner.
            for i in np.where(is_dead)[0]:
                clone_partners[i] = int(self.rng.choice(alive_ix))
            # Also override: alive walkers whose partner is dead → re-roll to alive.
            bad = np.array([is_dead[clone_partners[i]] and not is_dead[i] for i in range(N)])
            for i in np.where(bad)[0]:
                clone_partners[i] = int(self.rng.choice(alive_ix))

            VR_self = VR
            VR_other = VR[clone_partners]
            denom = np.where(VR_self > 1e-8, VR_self, 1e-8)
            clone_prob = np.clip((VR_other - VR_self) / denom, 0.0, 1.0)
            clone_prob[is_dead] = 1.0  # dead walkers always clone (resurrection)

            draws = self.rng.random(N)
            will_clone = draws < clone_prob

            for i in np.where(will_clone)[0]:
                k = int(clone_partners[i])
                if not is_dead[k]:
                    walker_states[i] = _copy_state(walker_states[k])
                    init_actions[i] = init_actions[k]
                    cum_rewards[i] = cum_rewards[k]
                    last_obs_per_walker[i] = (
                        last_obs_per_walker[k].copy() if last_obs_per_walker[k] is not None else None
                    )
                    is_dead[i] = False

        alive_ids = init_actions[~is_dead]
        if len(alive_ids) == 0:
            best_action = int(self.rng.integers(0, self.n_actions))
        else:
            counts = np.bincount(alive_ids, minlength=self.n_actions)
            best_action = int(counts.argmax())
        info = {
            "alive_walkers": int((~is_dead).sum()),
            "best_walker_reward": float(cum_rewards.max() if N > 0 else 0.0),
            "action_votes": np.bincount(alive_ids, minlength=self.n_actions).tolist()
            if len(alive_ids)
            else [0] * self.n_actions,
        }
        return best_action, info


def run_episode(
    env: PlanEnv,
    config: FMCConfig,
    seed: int = 0,
    max_decisions: int = 200,
    verbose: bool = False,
) -> dict:
    """Run one episode of FMC planning on `env`. Returns aggregate stats."""
    swarm = FMCSwarm(env, config, rng_seed=seed)

    state, obs, info = env.reset(return_state=True)
    cum_reward = 0.0
    n_decisions = 0
    samples = 0
    final_phase = 0
    delivered = False

    for step in range(max_decisions):
        # Snapshot the "real" env state, plan, restore, apply.
        root_state = env.get_state()
        action, info_step = swarm.decide(root_state)
        env.set_state(root_state)
        _obs, r, term, trunc, info = env.apply_action(action)
        cum_reward += float(r)
        n_decisions += 1
        samples += config.n_walkers * config.time_horizon * config.dt

        phase_now = int(round(env.get_state()[14]))
        final_phase = phase_now
        if phase_now >= 2:
            delivered = True

        if verbose and (step % 10 == 0 or term or trunc):
            print(
                f"  step={step:3d} action={action} reward_step={r:+.3f} "
                f"cum={cum_reward:+.2f} phase={phase_now} alive={info_step['alive_walkers']}"
            )

        if term or trunc:
            break

    return {
        "cum_reward": float(cum_reward),
        "decisions": n_decisions,
        "samples_total": samples,
        "delivered": bool(delivered),
        "final_phase": int(final_phase),
    }
