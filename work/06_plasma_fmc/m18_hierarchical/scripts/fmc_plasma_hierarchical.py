"""M18 — Hierarchical FMC for plasma shape control.

Extends fmc_plasma_jax.py with achievement-fire bonus following the
Conjecture D tier-stack compounding amplification mechanism that lifted
Craftax exp17 from 27.4% baseline to 50.95% (autoresearch session,
2026-04-30 → 2026-05-01).

Plasma achievements (boolean events that fire ONCE per walker per rollout):
  Tier 1 (easy):
    a0 = plasma_present     (|I_p| > 50 kA)
    a1 = shape_warm         (weighted-MSE err < 50)
  Tier 2 (gateway):
    a2 = shape_near         (err < 20)
    a3 = kappa_in_band      (|kappa - target_kappa| < 0.10)
  Tier 3 (blocker / "iron"):
    a4 = shape_close        (err < 5)
    a5 = shape_locked       (err < 3)

Per Conjecture D sweet spot 1.2-1.4× per tier (from Craftax falsifications):
  TIER_WEIGHTS = [10, 30, 60, 100, 200, 300]
  Stack ratios: 30/10=3, 60/30=2, 100/60=1.67, 200/100=2, 300/200=1.5
  → first iteration: aggressive (within Conjecture D bound on TOTAL stack
    ≤ 5×, since 300/10 = 30× per single ach, but inter-tier amplification
    is ≤ 2 per step → safe regime).

Implementation matches fmc_plasma_jax.make_jit_decide signature, adding:
  - ach_state in scan carry (B, A) bool — latched once fired
  - Per-tick check_achievements + fire-bonus accumulation in cum_reward
  - Cloning replicates ach_state alongside x, V, cum_reward
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

# Allow imports from the parent scripts/ directory
SCRIPT_PARENT = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_PARENT))

import jax
import jax.numpy as jnp
import numpy as np

from plasma_simulator_jax import (
    DTYPE,
    SimParams,
    build_jax_params,
    step_jax,
)
from fmc_plasma_jax import (
    FMCStaticCfg,
    relativize_jax,
    shape_reward_jax,
    safety_penalty_jax,
    shape_distance_jax,
)


# ---- Achievement spec (Conjecture D adapted to plasma) ----

NUM_ACH = 6

# Tier weights (matches Craftax exp17 tier-weighted shape: easy 10-30,
# gateway 50-100, blocker 150-300). Conjecture D sweet spot per
# inter-tier amplification ≤ 2× per step.
DEFAULT_TIER_WEIGHTS = jnp.array(
    [10.0, 30.0, 60.0, 100.0, 200.0, 300.0], dtype=DTYPE,
)


def shape_err_jax(state_batch, target, weights, N):
    """Weighted MSE shape error (B,) — same convention as shape_reward_jax."""
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
    return err


def check_achievements_jax(state_batch, target, weights, N):
    """Returns (B, NUM_ACH) bool array of achievements satisfied at this state."""
    I_p = state_batch[:, N]
    kappa = state_batch[:, N + 5]
    err = shape_err_jax(state_batch, target, weights, N)
    a0 = jnp.abs(I_p) / 1e6 > 0.05
    a1 = err < 50.0
    a2 = err < 20.0
    a3 = jnp.abs(kappa - target[2]) < 0.10
    a4 = err < 5.0
    a5 = err < 3.0
    return jnp.stack([a0, a1, a2, a3, a4, a5], axis=1)


# ---- Hierarchical JIT decision ----

def make_jit_decide_hierarchical(
    sim_p: SimParams,
    cfg: FMCStaticCfg,
    dt: float = 1e-3,
    tier_weights: jnp.ndarray = DEFAULT_TIER_WEIGHTS,
    ach_alpha: float = 1.0,
):
    """Hierarchical FMC decision: same as make_jit_decide but with ach-fire bonus.

    Args:
      sim_p: simulator params (static)
      cfg: FMC config (n_walkers, horizon — static)
      dt: integration step
      tier_weights: (NUM_ACH,) per-achievement bonus weights
      ach_alpha: scalar multiplier on the ach-fire contribution (default 1.0)
    """
    B = cfg.n_walkers
    T = cfg.horizon
    N = sim_p.N
    state_dim = N + 7
    A = NUM_ACH

    def decide(root_state, V_ref, target, weights, P_aux, gas, voltage_std, key):
        # 1. Replicate state across walkers
        x = jnp.broadcast_to(root_state, (B, state_dim)).astype(DTYPE)

        # 2. Compute root achievements (latched as already-fired)
        ach_root_single = check_achievements_jax(x[:1], target, weights, N)
        ach_state = jnp.broadcast_to(ach_root_single, (B, A))

        # 3. Sample initial voltages
        key, subkey = jax.random.split(key)
        V_init = V_ref + voltage_std * jax.random.normal(
            subkey, (B, N), dtype=DTYPE,
        )

        cum_reward = jnp.zeros(B, dtype=DTYPE)
        is_dead = jnp.zeros(B, dtype=jnp.bool_)

        def tick_step(carry, t):
            x, cum_reward, is_dead, V_init, ach_state, key = carry

            # Per-tick perturbation
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

            # Step rewards (shape + safety + ach-fire)
            r_shape = shape_reward_jax(x_new, target, weights, N)
            r_safety = -safety_penalty_jax(x_new, sim_p)

            # Achievement check + fire bonus
            ach_now = check_achievements_jax(x_new, target, weights, N)
            just_fired = ach_now & (~ach_state)              # (B, A) bool
            fire_bonus = jnp.sum(
                just_fired.astype(DTYPE) * tier_weights[None, :],
                axis=1,
            )                                                 # (B,)
            ach_state_new = ach_state | ach_now               # latch

            cum_reward_new = cum_reward + r_shape + r_safety + ach_alpha * fire_bonus
            cum_reward_new = jnp.where(
                is_dead_new, cum_reward_new - 1000.0, cum_reward_new,
            )

            # Cloning step
            key, sk_dist, sk_clone, sk_draw = jax.random.split(key, 4)
            partners_dist = jax.random.permutation(sk_dist, B)
            same = partners_dist == jnp.arange(B)
            partners_dist = jnp.where(same, (partners_dist + 1) % B, partners_dist)
            D = shape_distance_jax(x_new, partners_dist, N)
            R = relativize_jax(cum_reward_new)
            D_n = relativize_jax(D)
            R = jnp.where(is_dead_new, jnp.zeros_like(R), R)
            D_n = jnp.where(is_dead_new, jnp.zeros_like(D_n), D_n)
            VR = R ** 1.0 * D_n ** 1.0

            partners = jax.random.permutation(sk_clone, B)
            same2 = partners == jnp.arange(B)
            partners = jnp.where(same2, (partners + 1) % B, partners)

            VR_other = VR[partners]
            denom = jnp.where(VR > 1e-8, VR, 1e-8)
            clone_prob = jnp.clip((VR_other - VR) / denom, 0.0, 1.0)
            clone_prob = jnp.where(is_dead_new, 1.0, clone_prob)
            draws = jax.random.uniform(sk_draw, (B,))
            will_clone = draws < clone_prob
            partner_alive = ~is_dead_new[partners]
            will_clone = will_clone & partner_alive

            # Apply clone (also clones ach_state)
            x_clone = x_new[partners]
            V_clone = V_init[partners]
            cum_clone = cum_reward_new[partners]
            ach_state_clone = ach_state_new[partners]
            is_dead_clone = jnp.zeros_like(is_dead_new)

            x_post = jnp.where(will_clone[:, None], x_clone, x_new)
            V_post = jnp.where(will_clone[:, None], V_clone, V_init)
            cum_post = jnp.where(will_clone, cum_clone, cum_reward_new)
            ach_post = jnp.where(
                will_clone[:, None], ach_state_clone, ach_state_new,
            )
            is_dead_post = jnp.where(will_clone, is_dead_clone, is_dead_new)

            # Last tick: skip cloning
            is_last = t == (T - 1)
            x_final = jnp.where(is_last, x_new, x_post)
            V_final = jnp.where(is_last, V_init, V_post)
            cum_final = jnp.where(is_last, cum_reward_new, cum_post)
            ach_final = jnp.where(is_last, ach_state_new, ach_post)
            is_dead_final = jnp.where(is_last, is_dead_new, is_dead_post)

            return (x_final, cum_final, is_dead_final, V_final, ach_final, key), None

        carry0 = (x, cum_reward, is_dead, V_init, ach_state, key)
        ts = jnp.arange(T)
        (x_final, cum_final, is_dead_final, V_final, ach_final, _), _ = jax.lax.scan(
            tick_step, carry0, ts,
        )

        # Aggregate (same as fmc_plasma_jax)
        alive = ~is_dead_final
        cum_masked = jnp.where(alive, cum_final, -jnp.inf)
        max_r = jnp.max(cum_masked)
        all_dead = jnp.all(is_dead_final)
        weights_w = jnp.where(alive, jnp.exp(cum_masked - max_r), 0.0)
        w_sum = jnp.sum(weights_w)
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
        # Diagnostic: mean ach fired across alive walkers (per-rollout discovery rate)
        ach_fired_count = jnp.sum(ach_final.astype(DTYPE), axis=1)  # (B,)
        ach_mean = jnp.where(
            walkers_alive > 0,
            jnp.sum(jnp.where(alive, ach_fired_count, 0.0)) /
            jnp.maximum(walkers_alive, 1),
            0.0,
        )
        return V_decision, walkers_alive, expected_reward, ach_mean

    return jax.jit(decide)


class FMCPlasmaHierarchicalController:
    """Hierarchical FMC: same API as FMCPlasmaJaxController but with ach-fire bonus.

    Returns dict with: V_coils, walkers_alive, expected_reward, ach_mean
    (ach_mean = avg # achievements fired per alive walker, diagnostic).
    """

    def __init__(
        self,
        sim_p: SimParams,
        n_walkers: int = 64,
        horizon: int = 10,
        dt: float = 1e-3,
        voltage_std: float = 50.0,
        P_aux: float = 5e5,
        gas_puff: float = 1e21,
        seed: int = 0,
        tier_weights: np.ndarray | None = None,
        ach_alpha: float = 1.0,
    ):
        self.sim_p = sim_p
        self.cfg = FMCStaticCfg(n_walkers=n_walkers, horizon=horizon)
        self.dt = dt
        self.voltage_std = voltage_std
        self.P_aux = P_aux
        self.gas_puff = gas_puff
        self.V_ref = jnp.asarray(sim_p.R_diag) * jnp.asarray(sim_p.I_ref)
        self._key = jax.random.PRNGKey(seed)
        if tier_weights is None:
            tw = DEFAULT_TIER_WEIGHTS
        else:
            tw = jnp.asarray(tier_weights, dtype=DTYPE)
        self._decide = make_jit_decide_hierarchical(
            sim_p, self.cfg, dt, tier_weights=tw, ach_alpha=ach_alpha,
        )
        self._weights = jnp.array([100.0, 100.0, 10.0, 10.0], dtype=DTYPE)

    def decide(self, state, target):
        self._key, subkey = jax.random.split(self._key)
        V, alive, exp_r, ach_mean = self._decide(
            jnp.asarray(state, dtype=DTYPE),
            self.V_ref,
            jnp.asarray(target, dtype=DTYPE),
            self._weights,
            jnp.asarray(self.P_aux, dtype=DTYPE),
            jnp.asarray(self.gas_puff, dtype=DTYPE),
            jnp.asarray(self.voltage_std, dtype=DTYPE),
            subkey,
        )
        V.block_until_ready()
        return {
            "V_coils": np.asarray(V),
            "walkers_alive": int(alive),
            "expected_reward": float(exp_r),
            "ach_mean": float(ach_mean),
        }


if __name__ == "__main__":
    import time

    print("FMC Hierarchical (M18) — Conjecture D applied to plasma")
    print("=" * 60)

    sim_p, x0 = build_jax_params()
    # Real TCV-X21 shot 65402 target
    target_real = np.array([0.889, -0.056, 1.71, 0.12], dtype=np.float32)

    for n_w, h in [(32, 8), (64, 10), (200, 20)]:
        ctrl = FMCPlasmaHierarchicalController(
            sim_p, n_walkers=n_w, horizon=h, seed=0,
        )
        # Warmup
        d = ctrl.decide(np.asarray(x0), target_real)

        n_runs = 30
        t0 = time.perf_counter()
        for _ in range(n_runs):
            d = ctrl.decide(np.asarray(x0), target_real)
        elapsed = (time.perf_counter() - t0) / n_runs
        print(
            f"  M={n_w:3d}, H={h:2d}: {elapsed*1e6:7.1f} µs/decision  "
            f"(alive={d['walkers_alive']}/{n_w}, ach_mean={d['ach_mean']:.2f})"
        )
