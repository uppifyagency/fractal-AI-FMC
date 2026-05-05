"""m21_oracle_validation.py — freegs oracle validation of M21 BEST policy.

Mirrors M20's pipeline but on the M9-CALIBRATED sim:
  * Run M21 BEST configuration in closed-loop, with per-tick freegs
    Grad-Shafranov truth lookup (TCV-X21 65402 + 2 M15 published targets)
  * Compare directly to:
      - vanilla FMC online on calibrated sim (= M16 historical)
      - M19 exp39 transferred to calibrated sim (= M21 exp01)
      - M12 NN-shape historical (3.47 truth-err on M16 — deployment artifact)

Why this validates M21:
  * M16 historical FMC online: truth_err 21.57 (calibrated sim)
  * If M21 BEST < 21.57 → genuine FMC tuning improvement on real shape
  * If M21 BEST < 3.47 → M21 BEST replaces M12 as deploy artifact
  * If M21 BEST >> 21.57 → in-sim score gain doesn't transfer to truth either

Usage:
  JAX_PLATFORMS=cpu python scripts/m21_oracle_validation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # work/06_plasma_fmc
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import jax
import jax.numpy as jnp

from calibrated_sim import build_calibrated_jax_params  # ← calibrated
from plasma_simulator_jax import DTYPE, make_jit_step
from plasma_simulator_nn_shape import build_nn_sim_params, predict_shape
from freegs_oracle_robust import FreeGSOracle
from fmc_plasma_jax import FMCPlasmaJaxController
from tcv_published_targets import PUBLISHED_TARGETS

TCV_X21_TARGET = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)
WEIGHTS = np.array([100.0, 100.0, 10.0, 10.0])

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# === M21 BEST configuration (UPDATE after sweep converges) ===
# Filled in by orchestration once exp_NN keep is locked.
M21_BEST = {
    "n_walkers": 64,       # F11: N invariance — keep small for latency
    "horizon": 10,
    "dt": 1e-3,
    "voltage_std": 8.0,    # F10: V optimum at S × 10 sensitivity
    "P_aux": 5e5,
    "gas_puff": 1e21,
    "shape_weights": [100.0, 100.0, 10.0, 10.0],  # default
}


def make_m21_best_controller(sim_p, seed=0):
    cfg = M21_BEST
    ctrl = FMCPlasmaJaxController(
        sim_p,
        n_walkers=cfg["n_walkers"], horizon=cfg["horizon"],
        dt=cfg["dt"], voltage_std=cfg["voltage_std"],
        P_aux=cfg["P_aux"], gas_puff=cfg["gas_puff"],
        seed=seed,
    )
    ctrl._weights = jnp.asarray(cfg["shape_weights"], dtype=DTYPE)
    return ctrl


def make_vanilla_controller(sim_p, seed=0):
    """= M16 historical FMC online = M21 exp00 baseline."""
    return FMCPlasmaJaxController(
        sim_p, n_walkers=64, horizon=10, dt=1e-3,
        voltage_std=50.0, P_aux=5e5, gas_puff=1e21, seed=seed,
    )


def make_m19_exp39_controller(sim_p, seed=0):
    """M19 BEST hyperparams transferred to calibrated sim (= M21 exp01)."""
    ctrl = FMCPlasmaJaxController(
        sim_p, n_walkers=2048, horizon=10, dt=1e-3,
        voltage_std=120.0, P_aux=1e6, gas_puff=1e21, seed=seed,
    )
    ctrl._weights = jnp.asarray([640.0, 640.0, 320.0, 320.0], dtype=DTYPE)
    return ctrl


def truth_query(I_coils_np, oracle, fb_sim_p, target):
    def fb(I_):
        s = np.array(predict_shape(fb_sim_p, jnp.asarray(I_, dtype=DTYPE)))
        s[0] = np.clip(s[0], 0.624, 1.136)
        s[1] = np.clip(s[1], -0.75, 0.75)
        s[2] = np.clip(s[2], 1.0, 2.8)
        s[3] = np.clip(s[3], -0.7, 1.0)
        return s
    r = oracle.shape_from_coils(np.asarray(I_coils_np), fallback_fn=fb)
    return np.array([r.R_p, r.Z_p, r.kappa, r.delta]), r.source


def run_episode(controller, sim_p, sim_step, x0, target, oracle,
                fb_sim_p, n_ticks=30):
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

        x_new = sim_step(jnp.asarray(state),
                         jnp.asarray(V, dtype=DTYPE),
                         P_aux, gas, jnp.asarray(dt, dtype=jnp.float32))
        state = np.array(x_new)
        if np.isnan(state).any():
            break

        N = sim_p.N
        true_shape, source = truth_query(state[:N], oracle, fb_sim_p, target)
        err_t = float(np.sum(WEIGHTS * (true_shape - target) ** 2))
        err_s = float(np.sum(WEIGHTS * (state[N+3:N+7] - target) ** 2))
        truth_errs.append(err_t); self_errs.append(err_s)
        if source == "freegs":
            n_freegs += 1
        elif source == "nn_fallback":
            n_fb += 1

    n = len(truth_errs)
    return {
        "n_steps": n,
        "mean_truth_err": float(np.mean(truth_errs)) if truth_errs else float("inf"),
        "mean_self_err": float(np.mean(self_errs)) if self_errs else float("inf"),
        "steady_truth_err_last10": (
            float(np.mean(truth_errs[-10:])) if n >= 10
            else (float(np.mean(truth_errs)) if truth_errs else float("inf"))
        ),
        "physicality": n_freegs / max(1, n_freegs + n_fb),
        "decision_us_p50": float(np.median(decision_times) * 1e6),
    }


def main():
    print("=" * 72)
    print("M21 — freegs oracle validation of M21 BEST policy (CALIBRATED sim)")
    print("=" * 72)
    print("\n[setup] Calibrated sim + freegs oracle...")
    oracle = FreeGSOracle(verbose=False)
    sim_p, x0 = build_calibrated_jax_params()
    sim_step = make_jit_step(sim_p)
    fb_sim_p = build_nn_sim_params()

    targets = {
        "M16_real_TCV_65402": TCV_X21_TARGET,
        "M15_iter_like_ramp_steady": np.array(
            PUBLISHED_TARGETS[0].waypoints[-1][1], dtype=np.float32),
        "M15_high_elongation_steady": np.array(
            PUBLISHED_TARGETS[2].waypoints[-1][1], dtype=np.float32),
    }
    print(f"\n[targets] {len(targets)} targets:")
    for name, t in targets.items():
        print(f"    {name:35s}  R={t[0]:.3f} Z={t[1]:+.3f} κ={t[2]:.3f} δ={t[3]:+.3f}")

    controllers = {
        "M21_BEST": make_m21_best_controller,
        "vanilla_calibrated": make_vanilla_controller,
        "M19_exp39_transferred": make_m19_exp39_controller,
    }

    n_ticks = 30
    n_seeds = 2
    print(f"\n[config] M21_BEST = {M21_BEST}")
    print(f"[run] {n_ticks} ticks × {n_seeds} seeds × {len(targets)} targets × "
          f"{len(controllers)} controllers")

    out = {
        "n_ticks": n_ticks, "n_seeds": n_seeds,
        "weights": WEIGHTS.tolist(),
        "M21_BEST_config": M21_BEST,
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
                _ = ctrl.decide(np.asarray(x0), target)  # warmup
                t_ep = time.perf_counter()
                r = run_episode(ctrl, sim_p, sim_step, x0, target,
                                oracle, fb_sim_p, n_ticks=n_ticks)
                r["wall_s"] = time.perf_counter() - t_ep
                r["seed"] = seed
                seed_results.append(r)
                print(
                    f"    seed={seed}: truth={r['mean_truth_err']:7.2f}  "
                    f"steady={r['steady_truth_err_last10']:7.2f}  "
                    f"phys={r['physicality']*100:5.1f}%  "
                    f"wall={r['wall_s']:.1f}s",
                    flush=True,
                )
            mean_t = np.mean([r["mean_truth_err"] for r in seed_results])
            mean_s = np.mean([r["steady_truth_err_last10"] for r in seed_results])
            mean_p = np.mean([r["physicality"] for r in seed_results])
            out["results"][ctrl_name][tgt_name] = {
                "mean_truth_err_avg": float(mean_t),
                "steady_truth_err_avg": float(mean_s),
                "physicality_avg": float(mean_p),
                "n_seeds": n_seeds,
                "per_seed": seed_results,
            }
            print(f"    AGG: mean_truth={mean_t:.2f}  steady={mean_s:.2f}  "
                  f"phys={mean_p*100:.1f}%", flush=True)

    out_path = OUT_DIR / "m21_oracle_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary
    print("\n" + "=" * 72)
    print("M21 FINAL VERDICT — calibrated sim oracle truth")
    print("=" * 72)
    print(f"{'controller':<25s} {'target':<35s} {'truth':>8s} {'steady':>8s} {'phys%':>7s}")
    for cname, by_t in out["results"].items():
        for tname, agg in by_t.items():
            print(
                f"{cname:<25s} {tname:<35s} "
                f"{agg['mean_truth_err_avg']:8.2f} "
                f"{agg['steady_truth_err_avg']:8.2f} "
                f"{agg['physicality_avg']*100:6.1f}%"
            )

    m21r = out["results"]["M21_BEST"]["M16_real_TCV_65402"]
    vanr = out["results"]["vanilla_calibrated"]["M16_real_TCV_65402"]
    e39 = out["results"]["M19_exp39_transferred"]["M16_real_TCV_65402"]
    print(f"\n--- M16 real TCV-X21 65402 (key European tokamak benchmark) ---")
    print(f"  M21 BEST (calibrated):       steady_truth={m21r['steady_truth_err_avg']:7.2f}")
    print(f"  Vanilla FMC (calibrated):    steady_truth={vanr['steady_truth_err_avg']:7.2f}")
    print(f"  M19 exp39 transferred:       steady_truth={e39['steady_truth_err_avg']:7.2f}")
    print(f"  M16 historical FMC online:   steady_truth=21.57   (Apr 27 run)")
    print(f"  M16 historical M12 NN-shape: steady_truth= 3.47   (Apr 27 run, deploy)")


if __name__ == "__main__":
    main()
