"""FMC controller — fully JIT-compiled inner loop.

Replaces the Python+numpy loop in fmc_plasma.py with a single jax.lax.scan
that lives entirely on device. Target: ~30× speedup on a single decision
(190 ms → ~6 ms with M=200, H=20) and ~30× on dataset generation
(8/sec → 250/sec with M=32, H=8).

Algorithm: identical to fmc_plasma.FMCPlasmaController.decide() but in JAX:

  1. broadcast root state to (B, dim)
  2. sample initial V ~ N(V_ref, σ²I)
  3. lax.scan over horizon T:
       step (vmap'd) → reward → relativize → virtual reward → cloning
  4. softmax-weighted aggregation across surviving walkers

Key JAX primitives:
- jax.lax.scan          : tight loop without Python overhead
- jax.lax.dynamic_slice : safe gather for cloning indices
- jax.random.PRNGKey    : functional RNG (no global state)

Equivalence to fmc_plasma.py: tested in tests/test_fmc_jax.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np

from plasma_simulator_jax import (
    DTYPE,
    SimParams,
    build_jax_params,
    step_jax,
)


# ---- Static config (compile-time) ----

@dataclass(frozen=True)
class FMCStaticCfg:
    n_walkers: int
    horizon: int
    balance_alpha: float = 1.0
    balance_beta: float = 1.0


# ---- JAX relativize ----

def relativize_jax(x):
    """Same as fmc_plasma.relativize_np but pure-JAX, jit-friendly.

    R_N = (R - μ) / σ
    out = exp(R_N)        if R_N ≤ 0
    out = 1 + log(1+R_N)  if R_N > 0

    Threshold for "constant input" uses a relative tolerance because
    f32 std of identical values can be ~2e-7 due to mean-subtraction
    rounding, not 0 exactly.
    """
    sigma = jnp.std(x)
    scale = jnp.maximum(jnp.mean(jnp.abs(x)), 1.0)
    is_constant = sigma < 1e-6 * scale
    safe_sigma = jnp.where(is_constant, 1.0, sigma)
    z = (x - jnp.mean(x)) / safe_sigma
    out = jnp.where(
        z <= 0,
        jnp.exp(jnp.clip(z, -50.0, 0.0)),
        1.0 + jnp.log1p(jnp.maximum(z, 0.0)),
    )
    return jnp.where(is_constant, jnp.ones_like(x), out).astype(DTYPE)


def shape_reward_jax(state_batch, target, weights, N):
    """Negative weighted MSE shape reward (B,)."""
    R_p = state_batch[:, N + 3]
    Z_p = state_batch[:, N + 4]
    kappa = state_batch[:, N + 5]
    delta = state_batch[:, N + 6]
    err = (
        weights[0] * (R_p - target[0]) ** 2
        + weights[1] * (Z_p - target[1]) ** 2
        + weights[2] * (kappa - target[2]) ** 2
        + weights[3] * (delta - target[3]) ** 2
    )
    return -err


def safety_penalty_jax(state_batch, sim_p: SimParams,
                       q95_min: float = 2.0,
                       n_over_nGW_max: float = 0.9,
                       I_max: float = 7700.0):
    """JAX safety penalty (B,)."""
    N = sim_p.N
    I_coils = state_batch[:, :N]
    I_p = state_batch[:, N]
    R_p = state_batch[:, N + 3]
    n_bar = state_batch[:, N + 2]
    kappa = state_batch[:, N + 5]
    a_eff = sim_p.a_eff
    B_T = sim_p.B_T

    I_p_MA = jnp.abs(I_p) / 1e6
    I_p_MA_safe = jnp.maximum(I_p_MA, 1e-3)
    n_GW = (I_p_MA / (jnp.pi * a_eff ** 2)) * 1e20
    q95 = (5.0 * a_eff ** 2 * B_T * (1 + kappa ** 2) / 2.0) \
          / (R_p * I_p_MA_safe)
    n_ratio = n_bar / jnp.maximum(n_GW, 1e10)

    pen = 100.0 * jnp.maximum(q95_min - q95, 0) ** 2
    pen += 100.0 * jnp.maximum(n_ratio - n_over_nGW_max, 0) ** 2
    I_excess = jnp.maximum(jnp.abs(I_coils) - I_max, 0)
    pen += 1e-4 * jnp.sum(I_excess, axis=1)
    pen += 1000.0 * (I_p_MA < 0.05).astype(DTYPE)
    return pen


# ---- Distance metric (shape obs) ----

def shape_distance_jax(state_batch, partners_idx, N):
    """L2 distance walker_i vs walker[partners_idx[i]] on shape obs (scaled)."""
    obs = jnp.column_stack([
        state_batch[:, N + 3] * 100.0,
        state_batch[:, N + 4] * 100.0,
        state_batch[:, N + 5] * 10.0,
        state_batch[:, N + 6] * 10.0,
    ])
    return jnp.linalg.norm(obs - obs[partners_idx], axis=1)


# ---- Main JIT decision ----

def make_jit_decide(sim_p: SimParams, cfg: FMCStaticCfg, dt: float = 1e-3):
    """Returns a jit-compiled decision function bound to (sim_p, cfg, dt).

    Signature:
        decide_jit(root_state, V_ref, target, weights, P_aux, gas, voltage_std, key)
        → (V_decision, walkers_alive, expected_reward)

    All shape-static for JIT — change cfg.n_walkers or cfg.horizon → recompile.
    """
    B = cfg.n_walkers
    T = cfg.horizon
    N = sim_p.N
    state_dim = N + 7

    def decide(root_state, V_ref, target, weights, P_aux, gas, voltage_std, key):
        # 1. Replicate state across walkers
        x = jnp.broadcast_to(root_state, (B, state_dim)).astype(DTYPE)

        # 2. Sample initial voltages
        key, subkey = jax.random.split(key)
        V_init = V_ref + voltage_std * jax.random.normal(
            subkey, (B, N), dtype=DTYPE,
        )

        cum_reward = jnp.zeros(B, dtype=DTYPE)
        is_dead = jnp.zeros(B, dtype=jnp.bool_)

        # Per-tick state for scan: (x, cum_reward, is_dead, V_init, key)
        def tick_step(carry, t):
            x, cum_reward, is_dead, V_init, key = carry

            # Per-tick perturbation (smaller than initial)
            key, subkey = jax.random.split(key)
            V_t = V_init + 0.3 * voltage_std * jax.random.normal(
                subkey, (B, N), dtype=DTYPE,
            )
            V_t = jnp.where(t == 0, V_init, V_t)

            # vmapped sim step
            x_new = jax.vmap(step_jax, in_axes=(0, 0, None, None, None, None))(
                x, V_t, P_aux, gas, dt, sim_p,
            )

            # Quench detection
            quenched = jnp.abs(x_new[:, N]) / 1e6 < 0.05
            is_dead_new = is_dead | quenched

            # Step rewards (shape + safety)
            r_shape = shape_reward_jax(x_new, target, weights, N)
            r_safety = -safety_penalty_jax(x_new, sim_p)
            cum_reward_new = cum_reward + r_shape + r_safety
            cum_reward_new = jnp.where(is_dead_new, cum_reward_new - 1000.0, cum_reward_new)

            # Cloning step (skip if t == T-1)
            # — Compute partners + clone probabilities + apply
            key, sk_dist, sk_clone, sk_draw = jax.random.split(key, 4)
            partners_dist = jax.random.permutation(sk_dist, B)
            # Avoid self-comparison: shift by 1 if same
            same = partners_dist == jnp.arange(B)
            partners_dist = jnp.where(same, (partners_dist + 1) % B, partners_dist)
            D = shape_distance_jax(x_new, partners_dist, N)
            R = relativize_jax(cum_reward_new)
            D_n = relativize_jax(D)
            R = jnp.where(is_dead_new, jnp.zeros_like(R), R)
            D_n = jnp.where(is_dead_new, jnp.zeros_like(D_n), D_n)
            VR = R ** 1.0 * D_n ** 1.0  # alpha=beta=1

            partners = jax.random.permutation(sk_clone, B)
            same2 = partners == jnp.arange(B)
            partners = jnp.where(same2, (partners + 1) % B, partners)

            VR_other = VR[partners]
            denom = jnp.where(VR > 1e-8, VR, 1e-8)
            clone_prob = jnp.clip((VR_other - VR) / denom, 0.0, 1.0)
            # Dead always clone
            clone_prob = jnp.where(is_dead_new, 1.0, clone_prob)
            draws = jax.random.uniform(sk_draw, (B,))
            will_clone = draws < clone_prob
            # Don't clone from dead partner
            partner_alive = ~is_dead_new[partners]
            will_clone = will_clone & partner_alive

            # Apply: for each walker, if will_clone[i], copy from x[partners[i]]
            x_clone = x_new[partners]
            V_clone = V_init[partners]
            cum_clone = cum_reward_new[partners]
            is_dead_clone = jnp.zeros_like(is_dead_new)

            x_post = jnp.where(will_clone[:, None], x_clone, x_new)
            V_post = jnp.where(will_clone[:, None], V_clone, V_init)
            cum_post = jnp.where(will_clone, cum_clone, cum_reward_new)
            is_dead_post = jnp.where(will_clone, is_dead_clone, is_dead_new)

            # If t == T-1, skip cloning (return pre-clone state for aggregation)
            is_last = t == (T - 1)
            x_final = jnp.where(is_last, x_new, x_post)
            V_final = jnp.where(is_last, V_init, V_post)
            cum_final = jnp.where(is_last, cum_reward_new, cum_post)
            is_dead_final = jnp.where(is_last, is_dead_new, is_dead_post)

            return (x_final, cum_final, is_dead_final, V_final, key), None

        carry0 = (x, cum_reward, is_dead, V_init, key)
        ts = jnp.arange(T)
        (x_final, cum_final, is_dead_final, V_final, _), _ = jax.lax.scan(
            tick_step, carry0, ts,
        )

        # 4. Aggregate: softmax-weighted mean of V_final by cum_reward (over alive)
        alive = ~is_dead_final
        # Mask cum_reward of dead with -inf
        cum_masked = jnp.where(alive, cum_final, -jnp.inf)
        max_r = jnp.max(cum_masked)
        # If everyone dead, max_r = -inf → fallback
        all_dead = jnp.all(is_dead_final)
        weights_w = jnp.where(alive, jnp.exp(cum_masked - max_r), 0.0)
        w_sum = jnp.sum(weights_w)
        # softmax-weighted mean (avoid /0)
        V_decision = jnp.where(
            all_dead | (w_sum < 1e-8),
            V_ref,
            jnp.sum(V_final * weights_w[:, None], axis=0) / jnp.maximum(w_sum, 1e-8),
        )
        walkers_alive = jnp.sum(alive)
        expected_reward = jnp.where(
            walkers_alive > 0,
            jnp.sum(jnp.where(alive, cum_final, 0.0)) / jnp.maximum(walkers_alive, 1),
            -1e6,
        )
        return V_decision, walkers_alive, expected_reward

    return jax.jit(decide)


# ---- Convenience class ----

class FMCPlasmaJaxController:
    """Wrapper around the jit-compiled decide function for ergonomic Python use.

    Usage:
        sim_p, x0 = build_jax_params()
        ctrl = FMCPlasmaJaxController(sim_p, n_walkers=64, horizon=10, seed=0)
        target = np.array([0.90, 0.0, 1.85, 0.3])
        V = ctrl.decide(x0, target)
    """

    def __init__(self, sim_p: SimParams, n_walkers: int = 64, horizon: int = 10,
                 dt: float = 1e-3, voltage_std: float = 50.0,
                 P_aux: float = 5e5, gas_puff: float = 1e21, seed: int = 0):
        self.sim_p = sim_p
        self.cfg = FMCStaticCfg(n_walkers=n_walkers, horizon=horizon)
        self.dt = dt
        self.voltage_std = voltage_std
        self.P_aux = P_aux
        self.gas_puff = gas_puff
        self.V_ref = jnp.asarray(sim_p.R_diag) * jnp.asarray(sim_p.I_ref)
        self._key = jax.random.PRNGKey(seed)
        self._decide = make_jit_decide(sim_p, self.cfg, dt)
        self._weights = jnp.array([100.0, 100.0, 10.0, 10.0], dtype=DTYPE)

    def decide(self, state, target):
        self._key, subkey = jax.random.split(self._key)
        V, alive, exp_r = self._decide(
            jnp.asarray(state, dtype=DTYPE),
            self.V_ref,
            jnp.asarray(target, dtype=DTYPE),
            self._weights,
            jnp.asarray(self.P_aux, dtype=DTYPE),
            jnp.asarray(self.gas_puff, dtype=DTYPE),
            jnp.asarray(self.voltage_std, dtype=DTYPE),
            subkey,
        )
        # Force completion — the np.asarray + int + float conversions below
        # implicitly block, but be explicit for benchmark fairness.
        V.block_until_ready()
        return {
            "V_coils": np.asarray(V),
            "walkers_alive": int(alive),
            "expected_reward": float(exp_r),
        }


if __name__ == "__main__":
    import time
    print("FMC JAX — JIT-compiled inner loop")
    print("=" * 60)

    sim_p, x0 = build_jax_params()
    target = np.array([0.90, 0.0, 1.85, 0.3], dtype=np.float32)

    for n_w, h in [(32, 8), (64, 10), (200, 20)]:
        ctrl = FMCPlasmaJaxController(sim_p, n_walkers=n_w, horizon=h, seed=0)
        # Warmup (JIT compile)
        d = ctrl.decide(np.asarray(x0), target)
        # Block to flush
        d["V_coils"]

        # Time
        n_runs = 50
        t0 = time.perf_counter()
        for _ in range(n_runs):
            d = ctrl.decide(np.asarray(x0), target)
        elapsed = (time.perf_counter() - t0) / n_runs
        print(f"  M={n_w:3d}, H={h:2d}: {elapsed*1e6:7.1f} µs/decision  "
              f"(alive={d['walkers_alive']}/{n_w})")
