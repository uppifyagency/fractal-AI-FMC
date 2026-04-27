"""FMC controller using NN-shape simulator (M12).

Mirrors fmc_plasma_jax.FMCPlasmaJaxController but the inner step_jax_nn
calls a shape NN. Otherwise identical algorithm + JIT compilation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np

from fmc_plasma_jax import (
    FMCStaticCfg, relativize_jax, shape_reward_jax, safety_penalty_jax,
    shape_distance_jax,
)
from plasma_simulator_jax import DTYPE, SimParams
from plasma_simulator_nn_shape import (
    SimParamsNN, step_jax_nn, build_nn_sim_params, initial_state_nn,
)


def make_jit_decide_nn(sim_p: SimParamsNN, cfg: FMCStaticCfg, dt: float = 1e-3):
    """JIT FMC decide using NN-shape simulator."""
    B = cfg.n_walkers
    T = cfg.horizon
    N = sim_p.N
    state_dim = N + 7

    # Build a "compatible" sim_p_for_safety that has the SimParams fields
    # needed by safety_penalty_jax (a_eff, B_T attributes).
    safety_sim_p = SimParams(
        N=sim_p.N, M_cc=sim_p.M_cc, M_pc=sim_p.M_pc, R_diag=sim_p.R_diag,
        L_p=sim_p.L_p, S=jnp.zeros((4, sim_p.N), dtype=DTYPE),
        I_ref=sim_p.I_ref, R_ref=sim_p.R_ref, Z_ref=sim_p.Z_ref,
        kappa_ref=sim_p.kappa_ref, delta_ref=sim_p.delta_ref,
        a_eff=sim_p.a_eff, B_T=sim_p.B_T, eps=sim_p.eps, H98=sim_p.H98,
        R_plasma_calib=sim_p.R_plasma_calib,
    )

    def decide(root_state, V_ref, target, weights, P_aux, gas, voltage_std, key):
        x = jnp.broadcast_to(root_state, (B, state_dim)).astype(DTYPE)
        key, sk = jax.random.split(key)
        V_init = V_ref + voltage_std * jax.random.normal(sk, (B, N), dtype=DTYPE)
        cum_reward = jnp.zeros(B, dtype=DTYPE)
        is_dead = jnp.zeros(B, dtype=jnp.bool_)

        def tick_step(carry, t):
            x, cum_reward, is_dead, V_init, key = carry
            key, sk = jax.random.split(key)
            V_t = V_init + 0.3 * voltage_std * jax.random.normal(sk, (B, N), dtype=DTYPE)
            V_t = jnp.where(t == 0, V_init, V_t)

            x_new = jax.vmap(
                step_jax_nn, in_axes=(0, 0, None, None, None, None),
            )(x, V_t, P_aux, gas, dt, sim_p)

            quenched = jnp.abs(x_new[:, N]) / 1e6 < 0.05
            is_dead_new = is_dead | quenched

            r_shape = shape_reward_jax(x_new, target, weights, N)
            r_safety = -safety_penalty_jax(x_new, safety_sim_p)
            cum_reward_new = cum_reward + r_shape + r_safety
            cum_reward_new = jnp.where(is_dead_new, cum_reward_new - 1000.0, cum_reward_new)

            key, sk1, sk2, sk3 = jax.random.split(key, 4)
            partners_d = jax.random.permutation(sk1, B)
            same = partners_d == jnp.arange(B)
            partners_d = jnp.where(same, (partners_d + 1) % B, partners_d)
            D = shape_distance_jax(x_new, partners_d, N)
            R_n = relativize_jax(cum_reward_new)
            D_n = relativize_jax(D)
            R_n = jnp.where(is_dead_new, jnp.zeros_like(R_n), R_n)
            D_n = jnp.where(is_dead_new, jnp.zeros_like(D_n), D_n)
            VR = R_n * D_n

            partners = jax.random.permutation(sk2, B)
            same2 = partners == jnp.arange(B)
            partners = jnp.where(same2, (partners + 1) % B, partners)
            VR_other = VR[partners]
            denom = jnp.where(VR > 1e-8, VR, 1e-8)
            clone_prob = jnp.clip((VR_other - VR) / denom, 0.0, 1.0)
            clone_prob = jnp.where(is_dead_new, 1.0, clone_prob)
            draws = jax.random.uniform(sk3, (B,))
            partner_alive = ~is_dead_new[partners]
            will_clone = (draws < clone_prob) & partner_alive

            x_clone = x_new[partners]
            V_clone = V_init[partners]
            cum_clone = cum_reward_new[partners]

            x_post = jnp.where(will_clone[:, None], x_clone, x_new)
            V_post = jnp.where(will_clone[:, None], V_clone, V_init)
            cum_post = jnp.where(will_clone, cum_clone, cum_reward_new)
            is_dead_post = jnp.where(will_clone, jnp.zeros_like(is_dead_new), is_dead_new)

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
        return V_decision, walkers_alive, expected_reward

    return jax.jit(decide)


class FMCPlasmaNNController:
    """FMC controller using NN-shape simulator inside."""

    def __init__(self, sim_p: SimParamsNN, n_walkers=32, horizon=8,
                 dt=1e-3, voltage_std=50.0,
                 P_aux=5e5, gas_puff=1e21, seed=0):
        self.sim_p = sim_p
        self.cfg = FMCStaticCfg(n_walkers=n_walkers, horizon=horizon)
        self.dt = dt
        self.voltage_std = voltage_std
        self.P_aux = P_aux
        self.gas_puff = gas_puff
        self.V_ref = jnp.asarray(sim_p.R_diag) * jnp.asarray(sim_p.I_ref)
        self._key = jax.random.PRNGKey(seed)
        self._decide = make_jit_decide_nn(sim_p, self.cfg, dt)
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
        V.block_until_ready()
        return {
            "V_coils": np.asarray(V),
            "walkers_alive": int(alive),
            "expected_reward": float(exp_r),
        }


if __name__ == "__main__":
    import time
    print("FMC + NN-shape simulator (M12)")
    print("=" * 60)
    sim_p = build_nn_sim_params()
    x0 = initial_state_nn(sim_p)
    target = np.array([sim_p.R_ref, sim_p.Z_ref, 1.7, 0.2], dtype=np.float32)

    ctrl = FMCPlasmaNNController(sim_p, n_walkers=32, horizon=8, seed=0)

    # Warmup + time
    d = ctrl.decide(np.asarray(x0), target)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        d = ctrl.decide(np.asarray(x0), target)
    elapsed = (time.perf_counter() - t0) / n
    print(f"Single decision (M=32, H=8): {elapsed*1e6:.0f} µs/decision")
    print(f"  Walkers alive : {d['walkers_alive']}/32")
    print(f"  V_coils[:5]   : {d['V_coils'][:5]}")
