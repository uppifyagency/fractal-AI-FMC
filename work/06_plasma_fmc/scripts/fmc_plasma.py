"""FMC controller for TCV plasma — continuous-action adaptation.

Adapts the canonical FMC (Hernández-Cerezo & Duran-Ballester 2020, §4.3)
from discrete-action Atari to continuous-action plasma control.

Key differences vs the Atari reference impl (work/03_atari_replication/):

| Aspect          | Atari FMC          | Plasma FMC                          |
|-----------------|--------------------|-------------------------------------|
| State           | ALE clone (RAM)    | Packed 27-vector (I_coils + plasma) |
| Action          | Discrete int       | Continuous V_coils ∈ ℝ²⁰ (+ P_aux)  |
| Sampling        | Uniform over int   | Gaussian around V_ref               |
| Distance        | L2 on RAM (128 B)  | L2 on shape obs (R_p, Z_p, κ, δ)    |
| Reward          | Game score delta   | -‖shape − target‖² − safety penalty |
| Tick step       | Skipframe (5)      | dt = 1 ms                           |
| Aggregation     | bincount.argmax    | Weighted mean of initial V vectors  |

Paper sections referenced:
- §2.2.3 Relativize (composite reward construction)
- §4.3 The FMC algorithm
- §5.1.3 Choice of distance metric
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# JAX setup MUST happen before importing jax
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np

from plasma_simulator_jax import (
    DTYPE,
    SimParams,
    build_jax_params,
    make_batched_step,
    make_jit_step,
    pack_state,
    unpack_state,
)


# ============================================================
# Composite reward (paper §2.2.3) — relativize
# ============================================================

def relativize_np(x: np.ndarray) -> np.ndarray:
    """R_N = (R - μ)/σ, then exp/log piecewise. Returns >0 array preserving order.

        out = exp(R_N)        if R_N ≤ 0
        out = 1 + log(1+R_N)  if R_N > 0
    """
    sigma = float(x.std())
    if sigma == 0:
        return np.ones_like(x, dtype=np.float32)
    z = (x - x.mean()) / sigma
    return np.where(
        z <= 0,
        np.exp(np.clip(z, -50, 0)),
        1.0 + np.log1p(np.maximum(z, 0)),
    ).astype(np.float32)


# ============================================================
# Reward & safety
# ============================================================

@dataclass
class ShapeTarget:
    """Target plasma shape for tracking."""
    R_p: float
    Z_p: float
    kappa: float
    delta: float
    # Weights (m and dimensionless are mixed — these scale errors to comparable units)
    w_R: float = 100.0   # 1 cm error → 1 reward unit
    w_Z: float = 100.0
    w_kappa: float = 10.0  # κ error of 0.1 → 1 reward unit
    w_delta: float = 10.0


def shape_reward(state_batch: np.ndarray, N: int, target: ShapeTarget) -> np.ndarray:
    """Negative weighted MSE between current shape and target.

    Args:
        state_batch: (B, N+7) array of packed states
        N: number of coil channels
        target: ShapeTarget

    Returns:
        (B,) array of rewards (less-negative = better)
    """
    R_p = state_batch[:, N + 3]
    Z_p = state_batch[:, N + 4]
    kappa = state_batch[:, N + 5]
    delta = state_batch[:, N + 6]
    err = (
        target.w_R * (R_p - target.R_p) ** 2
        + target.w_Z * (Z_p - target.Z_p) ** 2
        + target.w_kappa * (kappa - target.kappa) ** 2
        + target.w_delta * (delta - target.delta) ** 2
    )
    return -err


def safety_penalty(
    state_batch: np.ndarray,
    N: int,
    sim_p: SimParams,
    q95_min: float = 2.0,
    n_over_nGW_max: float = 0.9,
    I_max_kA: float = 7.7,  # E/F coil limit
) -> np.ndarray:
    """Soft barrier penalty for unsafe states.

    Triggered when:
      q95 < q95_min  (kink instability)
      n_bar / n_GW > n_over_nGW_max  (Greenwald density limit)
      |I_coil| > I_max_kA · 1000  (engineering current limit)
    """
    I_coils = state_batch[:, :N]
    I_p = state_batch[:, N]
    R_p = state_batch[:, N + 3]
    n_bar = state_batch[:, N + 2]
    kappa = state_batch[:, N + 5]
    a_eff = float(sim_p.a_eff)
    B_T = float(sim_p.B_T)

    I_p_MA = np.abs(I_p) / 1e6
    I_p_MA_safe = np.maximum(I_p_MA, 1e-3)
    n_GW = (I_p_MA / (np.pi * a_eff**2)) * 1e20
    q95 = (5.0 * a_eff**2 * B_T * (1 + kappa**2) / 2.0) / (R_p * I_p_MA_safe)
    n_ratio = n_bar / np.maximum(n_GW, 1e10)

    pen = np.zeros(state_batch.shape[0], dtype=np.float32)
    pen += 100.0 * np.maximum(q95_min - q95, 0) ** 2
    pen += 100.0 * np.maximum(n_ratio - n_over_nGW_max, 0) ** 2
    I_excess = np.maximum(np.abs(I_coils) - I_max_kA * 1000.0, 0)
    pen += 1e-4 * I_excess.sum(axis=1)
    # Plasma quench penalty
    pen += 1000.0 * (I_p_MA < 0.05).astype(np.float32)
    return pen


# ============================================================
# FMC controller config
# ============================================================

@dataclass
class FMCConfig:
    n_walkers: int = 200
    horizon: int = 20             # M tick lookahead
    dt: float = 1e-3              # 1 ms (matches control rate 1 kHz)
    balance_alpha: float = 1.0    # exponent on R in virtual reward
    balance_beta: float = 1.0     # exponent on D in virtual reward
    voltage_std: float = 50.0     # V — Gaussian noise around V_ref
    P_aux_nominal: float = 5e5    # 0.5 MW
    gas_puff_nominal: float = 1e21


# ============================================================
# FMC plasma controller
# ============================================================

class FMCPlasmaController:
    """Continuous-action FMC for plasma shape tracking.

    Decision protocol (per control tick):
    1. Initialize N walker states = current plasma state (replicated)
    2. Sample N initial V_coils vectors ~ N(V_ref, σ²I)
    3. For tick t in 0..horizon-1:
       a. Walker i steps with V_initial[i] (always — keeps initial action info)
       b. Or: with random perturbation (better exploration vs canonical FMC)
       c. Compute reward (shape_reward - safety_penalty)
       d. Compute pairwise distance, relativize, virtual reward
       e. Cloning step
    4. Return weighted mean of V_initial across surviving walkers (weighted by VR)
    """

    def __init__(
        self,
        sim_p: SimParams,
        target: ShapeTarget,
        config: FMCConfig | None = None,
        seed: int = 0,
    ):
        self.p = sim_p
        self.target = target
        self.cfg = config if config is not None else FMCConfig()
        self.rng = np.random.default_rng(seed)
        self.N = sim_p.N
        # JIT-compiled batched step for inner rollouts
        self._batched_step = make_batched_step(sim_p)
        # V reference (steady-state holding voltage for I_ref)
        self.V_ref = np.asarray(sim_p.R_diag) * np.asarray(sim_p.I_ref)

    def _sample_initial_voltages(self) -> np.ndarray:
        """Draw N voltage vectors around V_ref."""
        return self.V_ref + self.rng.normal(
            0, self.cfg.voltage_std, size=(self.cfg.n_walkers, self.N),
        )

    def _shape_distance(self, state_batch: np.ndarray) -> np.ndarray:
        """L2 distance between each walker and a random partner — paper §5.1.3.
        Uses shape-relevant observables only (R_p, Z_p, κ, δ — scaled)."""
        N = self.N
        obs = np.column_stack([
            state_batch[:, N + 3] * 100.0,     # R_p in cm
            state_batch[:, N + 4] * 100.0,     # Z_p in cm
            state_batch[:, N + 5] * 10.0,      # κ × 10
            state_batch[:, N + 6] * 10.0,      # δ × 10
        ])
        B = obs.shape[0]
        partners = self.rng.permutation(B)
        same = partners == np.arange(B)
        partners[same] = (partners[same] + 1) % B
        return np.linalg.norm(obs - obs[partners], axis=1)

    def decide(self, current_state: np.ndarray) -> dict:
        """Return one control vector + diagnostics.

        Args:
            current_state: shape (N+7,), packed plasma state

        Returns:
            dict with:
                V_coils: shape (N,) recommended voltage
                P_aux:   scalar
                gas_puff:scalar
                expected_reward: estimate (mean across surviving walkers)
                walkers_alive: count
                samples_used: total simulator step calls
        """
        cfg = self.cfg
        B = cfg.n_walkers
        N = self.N

        # 1. Replicate root state across walkers
        x = np.broadcast_to(current_state, (B, N + 7)).astype(np.float32).copy()

        # 2. Sample initial voltage vectors
        V_init = self._sample_initial_voltages().astype(np.float32)
        P_aux_arr = np.full(B, cfg.P_aux_nominal, dtype=np.float32)
        gas_arr = np.full(B, cfg.gas_puff_nominal, dtype=np.float32)
        cum_reward = np.zeros(B, dtype=np.float32)
        is_dead = np.zeros(B, dtype=bool)

        # 3. Rollout loop with cloning
        for t in range(cfg.horizon):
            # Step (use V_init at every tick — closest to "this is what I commit to")
            # Adds small fresh perturbation each tick for exploration
            V_t = V_init + self.rng.normal(
                0, cfg.voltage_std * 0.3, size=(B, N),
            ).astype(np.float32) if t > 0 else V_init

            x_jax = jnp.asarray(x)
            V_jax = jnp.asarray(V_t)
            P_jax = jnp.asarray(P_aux_arr)
            g_jax = jnp.asarray(gas_arr)
            dt_jax = jnp.asarray(cfg.dt, dtype=DTYPE)

            # np.array (not asarray) → writable copy
            x = np.array(self._batched_step(x_jax, V_jax, P_jax, g_jax, dt_jax))

            # Mark "dead" walkers — those with quenched plasma
            quenched = np.abs(x[:, N]) / 1e6 < 0.05
            is_dead = is_dead | quenched

            # Step rewards
            r_shape = shape_reward(x, N, self.target)
            r_safety = -safety_penalty(x, N, self.p)
            cum_reward += (r_shape + r_safety)
            cum_reward[is_dead] -= 1000.0  # heavy penalty for quench

            # Virtual reward + cloning (skip on last tick — no need to spread further)
            if t < cfg.horizon - 1:
                R = relativize_np(cum_reward)
                D = relativize_np(self._shape_distance(x))
                R[is_dead] = 0
                D[is_dead] = 0
                VR = (R ** cfg.balance_alpha) * (D ** cfg.balance_beta)

                # Cloning: each walker considers a random partner
                partners = self.rng.permutation(B)
                same = partners == np.arange(B)
                partners[same] = (partners[same] + 1) % B
                VR_other = VR[partners]
                denom = np.where(VR > 1e-8, VR, 1e-8)
                clone_prob = np.clip((VR_other - VR) / denom, 0, 1)
                clone_prob[is_dead] = 1.0  # dead walkers always clone

                draws = self.rng.random(B)
                will_clone = draws < clone_prob

                idx_alive = np.where(will_clone & ~is_dead[partners])[0]
                src = partners[idx_alive]
                # Apply clone
                x[idx_alive] = x[src]
                V_init[idx_alive] = V_init[src]
                cum_reward[idx_alive] = cum_reward[src]
                P_aux_arr[idx_alive] = P_aux_arr[src]
                gas_arr[idx_alive] = gas_arr[src]
                is_dead[idx_alive] = False

        # 4. Aggregate decision: weighted mean of V_init by exp(cum_reward)
        alive = ~is_dead
        if not np.any(alive):
            # All quenched — fall back to V_ref
            return {
                "V_coils": self.V_ref.copy(),
                "P_aux": cfg.P_aux_nominal,
                "gas_puff": cfg.gas_puff_nominal,
                "expected_reward": float(cum_reward.mean()),
                "walkers_alive": 0,
                "samples_used": B * cfg.horizon,
            }

        # Softmax-weighted mean
        r_alive = cum_reward[alive]
        r_shifted = r_alive - r_alive.max()
        w = np.exp(r_shifted)
        w_sum = w.sum()
        if w_sum < 1e-8:
            V_decision = V_init[alive].mean(axis=0)
        else:
            V_decision = (V_init[alive] * w[:, None]).sum(axis=0) / w_sum

        return {
            "V_coils": V_decision,
            "P_aux": cfg.P_aux_nominal,
            "gas_puff": cfg.gas_puff_nominal,
            "expected_reward": float(r_alive.mean()),
            "walkers_alive": int(alive.sum()),
            "samples_used": B * cfg.horizon,
        }


# ============================================================
# Self-test: tracking demo
# ============================================================

if __name__ == "__main__":
    print("FMC Plasma Controller — tracking demo")
    print("=" * 60)
    sim_p, x0 = build_jax_params()
    print(f"Sim channels: {sim_p.N}, state dim: {x0.shape[0]}")

    # Target: nudge R_p outward by 2 cm and increase elongation
    target = ShapeTarget(
        R_p=0.90,    # +2 cm vs reference 0.88
        Z_p=0.0,
        kappa=1.85,  # +0.15 vs reference 1.7
        delta=0.3,
    )
    print(f"Target shape: R_p={target.R_p} m, Z_p={target.Z_p} m, "
          f"κ={target.kappa}, δ={target.delta}")

    config = FMCConfig(
        n_walkers=200, horizon=20, dt=1e-3,
        voltage_std=50.0,
    )
    print(f"FMC config: M={config.n_walkers} walkers, H={config.horizon} tick")

    controller = FMCPlasmaController(sim_p, target, config, seed=42)

    # Single decision benchmark
    import time
    t0 = time.perf_counter()
    decision = controller.decide(np.asarray(x0))
    elapsed = time.perf_counter() - t0
    print(f"\nSingle decision: {elapsed*1e3:.1f} ms")
    print(f"  Walkers alive  : {decision['walkers_alive']}/{config.n_walkers}")
    print(f"  Expected reward: {decision['expected_reward']:.3f}")
    print(f"  V_coils[:5]    : {decision['V_coils'][:5]}")
    print(f"  ‖V - V_ref‖    : {np.linalg.norm(decision['V_coils'] - controller.V_ref):.2f} V")

    # Closed-loop tracking: 50 control steps (50 ms)
    print(f"\n--- Closed-loop tracking (50 ms = 50 control steps) ---")
    x = np.asarray(x0).copy()
    log = []
    sim_step = make_jit_step(sim_p)

    for k in range(50):
        decision = controller.decide(x)
        # Apply decision to "real" simulator
        x_jax = sim_step(
            jnp.asarray(x),
            jnp.asarray(decision["V_coils"], dtype=DTYPE),
            jnp.asarray(decision["P_aux"], dtype=DTYPE),
            jnp.asarray(decision["gas_puff"], dtype=DTYPE),
            jnp.asarray(config.dt, dtype=DTYPE),
        )
        x = np.asarray(x_jax)
        N = sim_p.N
        log.append({
            "t_ms": (k + 1) * 1.0,
            "R_p": float(x[N + 3]),
            "Z_p": float(x[N + 4]),
            "kappa": float(x[N + 5]),
            "delta": float(x[N + 6]),
            "I_p_kA": float(x[N]) / 1e3,
            "alive": decision["walkers_alive"],
        })
        if k % 10 == 0:
            print(f"  t={log[-1]['t_ms']:5.1f} ms | "
                  f"R_p={log[-1]['R_p']:.4f} (target {target.R_p}) | "
                  f"κ={log[-1]['kappa']:.3f} (target {target.kappa}) | "
                  f"I_p={log[-1]['I_p_kA']:6.1f} kA | "
                  f"alive={log[-1]['alive']}")

    # Save log for analysis
    import json
    out = Path(__file__).parent.parent / "results" / "milestone_3_tracking.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "target": {"R_p": target.R_p, "Z_p": target.Z_p,
                       "kappa": target.kappa, "delta": target.delta},
            "config": config.__dict__,
            "log": log,
        }, f, indent=2)
    print(f"\n  Saved tracking log: {out}")
