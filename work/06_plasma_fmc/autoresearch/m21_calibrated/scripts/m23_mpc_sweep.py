"""m23_mpc_sweep.py — Karpathy-style autoresearch on MPC formulations.

Runs all variants from mpc_variants.py through the same oracle pipeline
that M21 used (calibrated linear sim + freegs Grad-Shafranov truth),
4 seeds × 3 targets per variant. Records steady_truth, mean_truth,
physicality, latency.

Goal: find the MPC tuning (if any) that beats M21 BEST 0.58 on M16.
If none does, the falsification of "MPC > FMC" thesis stands and we
update the project framing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp

from calibrated_sim import build_calibrated_jax_params
from plasma_simulator_jax import DTYPE, make_jit_step
from plasma_simulator_nn_shape import build_nn_sim_params, predict_shape
from freegs_oracle_robust import FreeGSOracle
from tcv_published_targets import PUBLISHED_TARGETS

from mpc_variants import MPCVariant, variant_specs

TCV_X21_TARGET = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)
WEIGHTS = np.array([100.0, 100.0, 10.0, 10.0])

OUT_DIR = Path(__file__).resolve().parent.parent / "results"


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
        "steady_truth_err_last10": (
            float(np.mean(truth_errs[-10:])) if n >= 10
            else (float(np.mean(truth_errs)) if truth_errs else float("inf"))
        ),
        "physicality": n_freegs / max(1, n_freegs + n_fb),
        "decision_us_p50": float(np.median(decision_times) * 1e6),
    }


def main():
    print("=" * 78)
    print("M23-evo — MPC variant sweep (autoresearch on linear-MPC tunings)")
    print("=" * 78)

    print("\n[setup] Calibrated sim + freegs oracle...")
    oracle = FreeGSOracle(verbose=False)
    sim_p, x0 = build_calibrated_jax_params()
    sim_step = make_jit_step(sim_p)
    fb_sim_p = build_nn_sim_params()

    targets = {
        "M16": TCV_X21_TARGET,
        "M15_iter": np.array(PUBLISHED_TARGETS[0].waypoints[-1][1], dtype=np.float32),
        "M15_high_elong": np.array(PUBLISHED_TARGETS[2].waypoints[-1][1], dtype=np.float32),
    }

    variants = variant_specs()
    n_seeds = 4
    n_ticks = 30
    print(f"[run] {len(variants)} variants × {len(targets)} targets × {n_seeds} seeds = "
          f"{len(variants)*len(targets)*n_seeds} episodes")
    print(f"  M21 BEST FMC reference: M16 steady_truth = 0.58 ± 0.18 (target to beat)\n")

    out = {"n_seeds": n_seeds, "n_ticks": n_ticks, "variants": {}}

    for vname, kw in variants:
        print(f"\n--- variant {vname} ---")
        out["variants"][vname] = {"config": kw, "results": {}}
        for tname, tgt in targets.items():
            seed_results = []
            for seed in range(n_seeds):
                ctrl = MPCVariant(sim_p, **kw)
                _ = ctrl.decide(np.asarray(x0), tgt)
                r = run_episode(ctrl, sim_p, sim_step, x0, tgt, oracle,
                                fb_sim_p, n_ticks=n_ticks)
                seed_results.append(r)
            mean_t = np.mean([r["mean_truth_err"] for r in seed_results])
            mean_s = np.mean([r["steady_truth_err_last10"] for r in seed_results])
            std_s = np.std([r["steady_truth_err_last10"] for r in seed_results], ddof=1)
            mean_p = np.mean([r["physicality"] for r in seed_results])
            mean_dec = np.mean([r["decision_us_p50"] for r in seed_results])
            out["variants"][vname]["results"][tname] = {
                "mean_truth_err": float(mean_t),
                "steady_truth_err": float(mean_s),
                "steady_truth_err_std": float(std_s),
                "physicality": float(mean_p),
                "decision_us_p50": float(mean_dec),
                "per_seed": seed_results,
            }
            print(
                f"  {tname:15s}: mean={mean_t:7.2f} steady={mean_s:7.2f}±{std_s:6.2f} "
                f"phys={mean_p*100:5.1f}% lat={mean_dec:6.0f}µs"
            )

    out_path = OUT_DIR / "m23_mpc_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary
    print("\n" + "=" * 90)
    print("M23-evo SUMMARY (steady_truth_err, all 3 targets, 4-seed mean)")
    print("=" * 90)
    print(f"{'variant':<25s} {'M16':>10s} {'M15_iter':>10s} {'M15_elong':>10s} {'phys%avg':>9s} {'lat':>8s}")
    print("-" * 90)
    fmc_baselines = {
        "M21 BEST FMC (V=8 N=64)":  {"M16": 0.58, "M15_iter": 1.27, "M15_high_elong": 1.25,
                                       "phys": 100.0, "lat": 1135},
        "vanilla FMC (V=50 N=64)":   {"M16": 44.48, "M15_iter": 50.77, "M15_high_elong": 48.55,
                                       "phys": 58.6, "lat": 1112},
    }
    for vname, vdata in out["variants"].items():
        r = vdata["results"]
        phys_avg = np.mean([r[t]["physicality"] for t in r]) * 100
        lat_avg = np.mean([r[t]["decision_us_p50"] for t in r])
        print(
            f"{vname:<25s} {r['M16']['steady_truth_err']:10.2f} "
            f"{r['M15_iter']['steady_truth_err']:10.2f} "
            f"{r['M15_high_elong']['steady_truth_err']:10.2f} "
            f"{phys_avg:8.1f}% {lat_avg:7.0f}µs"
        )
    print("-" * 90)
    for n, b in fmc_baselines.items():
        print(
            f"{n:<25s} {b['M16']:10.2f} {b['M15_iter']:10.2f} {b['M15_high_elong']:10.2f} "
            f"{b['phys']:8.1f}% {b['lat']:7.0f}µs"
        )
    print(f"{'M12 NN-shape (deploy)':<25s} {'3.47':>10s} {'(n/a)':>10s} {'(n/a)':>10s} {'100.0%':>9s} {'   122µs':>8s}")

    # Headline verdict
    best = min(out["variants"].items(),
               key=lambda kv: kv[1]["results"]["M16"]["steady_truth_err"])
    bn, bd = best
    bm = bd["results"]["M16"]["steady_truth_err"]
    print(f"\n--- Best MPC variant on M16: {bn} → steady_truth = {bm:.2f}")
    if bm < 0.58:
        print(f"    BEATS M21 BEST FMC (0.58) by {0.58/bm:.2f}× — MPC has a real edge.")
    else:
        print(f"    LOSES to M21 BEST FMC (0.58) by {bm/0.58:.2f}×")
        print(f"    ⇒ falsification of 'MPC > FMC' thesis HOLDS across all swept tunings.")


if __name__ == "__main__":
    main()
