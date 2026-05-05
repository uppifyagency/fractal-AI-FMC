"""M20 — freegs oracle validation of exp39 best policy.

Validates exp39 (autoresearch loop converged best, score +1.0440 in-sim)
against M14 freegs forward-mode oracle (real Grad-Shafranov truth).

Compares to historical baselines (M16):
  FMC_online: mean truth-err 27.92, steady 21.57, phys 63%
  M12_NNshape: mean truth-err 3.37, steady 3.47, phys 100%

If exp39 truth-err < FMC_online historical → real plasma control gain.
If exp39 truth-err >> FMC_online historical → in-sim overfit.

Surface: TCV-X21 shot 65402 (M16 real target) + 2 M15 published targets
        (iter_like_ramp, negative_triangularity).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "autoresearch"))

import numpy as np
import jax
import jax.numpy as jnp

from plasma_simulator_jax import DTYPE, make_jit_step
from calibrated_sim import build_calibrated_jax_params as build_jax_params
from plasma_simulator_nn_shape import (
    build_nn_sim_params, make_jit_step_nn, predict_shape,
)
from freegs_oracle_robust import COIL_ORDER, FreeGSOracle
from fmc_plasma_jax import FMCPlasmaJaxController
from tcv_published_targets import PUBLISHED_TARGETS

# Real TCV-X21 65402 t=1.0s shape (from M16)
TCV_X21_TARGET = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)

WEIGHTS = np.array([100.0, 100.0, 10.0, 10.0])

OUT_DIR = ROOT / "autoresearch" / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def make_exp39_controller(sim_p, seed: int = 0):
    """M22: exp39 base + HORIZON 10 → 15 (test ramp-scenario improvement)."""
    ctrl = FMCPlasmaJaxController(
        sim_p,
        n_walkers=2048,
        horizon=15,  # M22 mutation: H=10 → 15
        dt=1e-3,
        voltage_std=80.0,
        P_aux=1e6,
        gas_puff=2e21,
        seed=seed,
    )
    ctrl._weights = jnp.array(
        [400.0, 400.0, 200.0, 200.0], dtype=DTYPE,
    )
    return ctrl


def make_baseline_controller(sim_p, seed: int = 0):
    """Build the vanilla FMC online baseline (= exp00 = M16 historical FMC online).

    N=64, V_STD=50, default weights.
    """
    return FMCPlasmaJaxController(
        sim_p,
        n_walkers=64,
        horizon=10,
        dt=1e-3,
        voltage_std=50.0,
        P_aux=5e5,
        gas_puff=2e21,
        seed=seed,
    )


def truth_query(I_coils_np: np.ndarray, oracle: FreeGSOracle,
                fb_sim_p, target: np.ndarray) -> tuple[np.ndarray, str]:
    """Query freegs oracle for true shape; NN fallback if unconverged."""
    def fb(I_):
        s = np.array(predict_shape(fb_sim_p, jnp.asarray(I_, dtype=DTYPE)))
        s[0] = np.clip(s[0], 0.624, 1.136)
        s[1] = np.clip(s[1], -0.75, 0.75)
        s[2] = np.clip(s[2], 1.0, 2.8)
        s[3] = np.clip(s[3], -0.7, 1.0)
        return s

    r = oracle.shape_from_coils(np.asarray(I_coils_np), fallback_fn=fb)
    return np.array([r.R_p, r.Z_p, r.kappa, r.delta]), r.source


def run_episode_with_oracle(controller, sim_p, sim_step, x0, target, oracle,
                             fb_sim_p, n_ticks=30):
    """Closed-loop episode with per-tick freegs oracle truth eval."""
    state = np.asarray(x0).copy()
    P_aux = jnp.asarray(controller.P_aux, dtype=jnp.float32)
    gas = jnp.asarray(controller.gas_puff, dtype=jnp.float32)
    dt = controller.dt

    truth_errs, self_errs = [], []
    n_freegs = n_fb = 0
    decision_times = []

    for tick in range(n_ticks):
        t0 = time.perf_counter()
        V = controller.decide(state, target)["V_coils"]
        decision_times.append(time.perf_counter() - t0)

        x_new = sim_step(
            jnp.asarray(state), jnp.asarray(V, dtype=DTYPE),
            P_aux, gas, jnp.asarray(dt, dtype=jnp.float32),
        )
        state = np.array(x_new)
        if np.isnan(state).any():
            break

        N = sim_p.N
        I_coils = state[:N]
        true_shape, source = truth_query(I_coils, oracle, fb_sim_p, target)
        err_t = float(np.sum(WEIGHTS * (true_shape - target) ** 2))
        err_s = float(np.sum(WEIGHTS *
                              (state[N+3:N+7] - target) ** 2))
        truth_errs.append(err_t)
        self_errs.append(err_s)
        if source == "freegs":
            n_freegs += 1
        elif source == "nn_fallback":
            n_fb += 1

    n_steps = len(truth_errs)
    mean_truth = float(np.mean(truth_errs)) if truth_errs else float("inf")
    mean_self = float(np.mean(self_errs)) if self_errs else float("inf")
    steady_truth = (float(np.mean(truth_errs[-10:]))
                    if n_steps >= 10 else mean_truth)
    phys = n_freegs / max(1, n_freegs + n_fb)
    return {
        "n_steps": n_steps,
        "mean_truth_err": mean_truth,
        "mean_self_err": mean_self,
        "steady_truth_err_last10": steady_truth,
        "physicality": phys,
        "decision_us_p50": float(np.median(decision_times) * 1e6),
        "truth_errs": truth_errs,
        "self_errs": self_errs,
    }


def main():
    print("=" * 72)
    print("M20 — freegs oracle validation of exp39 best policy")
    print("=" * 72)

    # Setup
    print("\n[setup] Initializing oracle, sim, controllers...")
    oracle = FreeGSOracle(verbose=False)
    sim_p, x0 = build_jax_params()
    sim_step = make_jit_step(sim_p)
    fb_sim_p = build_nn_sim_params()  # for fallback in unconverged freegs

    # Targets: M16 real + 2 representative M15 published
    targets = {
        "M16_real_TCV_65402": TCV_X21_TARGET,
        "M15_iter_like_ramp_steady": np.array(
            PUBLISHED_TARGETS[0].waypoints[-1][1], dtype=np.float32),
        "M15_high_elongation_steady": np.array(
            PUBLISHED_TARGETS[2].waypoints[-1][1], dtype=np.float32),
    }
    print(f"\n[targets] {len(targets)} targets:")
    for name, t in targets.items():
        print(f"    {name:30s}  R={t[0]:.3f} Z={t[1]:+.3f} κ={t[2]:.3f} δ={t[3]:+.3f}")

    # Controllers to validate
    controllers = {
        "exp39_BEST": make_exp39_controller,
        "baseline_FMC_online": make_baseline_controller,
    }

    n_ticks = 30
    n_seeds = 2  # keep small for oracle expense
    print(f"\n[run] {n_ticks} ticks × {n_seeds} seeds × {len(targets)} targets × "
          f"{len(controllers)} controllers")
    print(f"[run] freegs oracle ~24 ms/query → ~{n_ticks * 24}ms oracle overhead per ep")

    out = {
        "n_ticks": n_ticks,
        "n_seeds": n_seeds,
        "weights": WEIGHTS.tolist(),
        "results": {},
    }

    for ctrl_name, factory in controllers.items():
        out["results"][ctrl_name] = {}
        print(f"\n--- Controller: {ctrl_name} ---")
        for tgt_name, target in targets.items():
            print(f"  target={tgt_name}", flush=True)
            seed_results = []
            for seed in range(n_seeds):
                ctrl = factory(sim_p, seed=seed)
                # Warmup JIT
                _ = ctrl.decide(np.asarray(x0), target)

                t_ep = time.perf_counter()
                r = run_episode_with_oracle(
                    ctrl, sim_p, sim_step, x0, target, oracle,
                    fb_sim_p, n_ticks=n_ticks,
                )
                ep_wall = time.perf_counter() - t_ep
                r["wall_s"] = ep_wall
                r["seed"] = seed
                seed_results.append(r)
                print(
                    f"    seed={seed}: truth={r['mean_truth_err']:7.2f}  "
                    f"steady={r['steady_truth_err_last10']:7.2f}  "
                    f"phys={r['physicality']*100:5.1f}%  "
                    f"wall={ep_wall:.1f}s",
                    flush=True,
                )

            # Aggregate
            mean_t = np.mean([r["mean_truth_err"] for r in seed_results])
            mean_steady = np.mean([r["steady_truth_err_last10"]
                                    for r in seed_results])
            mean_phys = np.mean([r["physicality"] for r in seed_results])
            out["results"][ctrl_name][tgt_name] = {
                "mean_truth_err_avg": float(mean_t),
                "steady_truth_err_avg": float(mean_steady),
                "physicality_avg": float(mean_phys),
                "n_seeds": n_seeds,
                "per_seed": [{k: v for k, v in r.items()
                              if k not in ["truth_errs", "self_errs"]}
                              for r in seed_results],
            }
            print(
                f"    AGG: mean_truth={mean_t:.2f}  steady={mean_steady:.2f}  "
                f"phys={mean_phys*100:.1f}%",
                flush=True,
            )

    # Save
    out_path = OUT_DIR / "m22_quad_compose_oracle.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Final summary
    print()
    print("=" * 72)
    print("FINAL VERDICT — exp39 truth-eval vs baseline FMC online")
    print("=" * 72)
    print(f"{'controller':<25s} {'target':<35s} {'truth':>8s} {'steady':>8s} {'phys%':>7s}")
    for ctrl_name, by_target in out["results"].items():
        for tgt_name, agg in by_target.items():
            print(
                f"{ctrl_name:<25s} {tgt_name:<35s} "
                f"{agg['mean_truth_err_avg']:8.2f} "
                f"{agg['steady_truth_err_avg']:8.2f} "
                f"{agg['physicality_avg']*100:6.1f}%"
            )

    # Specific comparison: exp39 vs baseline on M16 real shot
    e_real = out["results"]["exp39_BEST"]["M16_real_TCV_65402"]
    b_real = out["results"]["baseline_FMC_online"]["M16_real_TCV_65402"]
    print(f"\n--- M16 real TCV shot (key comparison) ---")
    print(f"  exp39 BEST:           steady_truth={e_real['steady_truth_err_avg']:7.2f}  "
          f"phys={e_real['physicality_avg']*100:5.1f}%")
    print(f"  baseline FMC online:  steady_truth={b_real['steady_truth_err_avg']:7.2f}  "
          f"phys={b_real['physicality_avg']*100:5.1f}%")
    print(f"  M16 historical FMC:   steady_truth=21.57  phys=63%  (Apr 27 run)")
    print(f"  M16 historical M12:   steady_truth= 3.47  phys=100%  (Apr 27 run)")


if __name__ == "__main__":
    main()
