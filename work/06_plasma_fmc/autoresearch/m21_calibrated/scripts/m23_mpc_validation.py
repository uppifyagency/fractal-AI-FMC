"""m23_mpc_validation.py — Falsify or confirm the MPC > FMC prediction.

Same oracle pipeline as M21 (calibrated linear sim + freegs Grad-Shafranov
truth), 4 seeds × 3 targets (TCV-X21 65402, M15 iter_like, M15 high_elong),
but adds a 4th controller: deterministic LQR baseline.

Falsifiable claim under test:
  H0 (my prior >95%): MPC beats M21 BEST on M16 truth-err with lower
                       latency and zero seed variance
  H1 (FMC has real edge): MPC truth-err > M21 BEST 0.58
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
from fmc_plasma_jax import FMCPlasmaJaxController
from tcv_published_targets import PUBLISHED_TARGETS

from mpc_baseline_controller import LinearMPCController

TCV_X21_TARGET = np.array([0.889, -0.0562, 1.7096, 0.1231], dtype=np.float32)
WEIGHTS = np.array([100.0, 100.0, 10.0, 10.0])

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def make_mpc(sim_p, seed=0):
    return LinearMPCController(sim_p, dt=1e-3, R_weight=1e-6,
                                P_aux=5e5, gas_puff=1e21, seed=seed)


def make_m21_best(sim_p, seed=0):
    """V=8 N=64 P=1e6 gas=5e20 (M21 BEST locked)."""
    ctrl = FMCPlasmaJaxController(
        sim_p, n_walkers=64, horizon=10, dt=1e-3,
        voltage_std=8.0, P_aux=1e6, gas_puff=5e20, seed=seed,
    )
    ctrl._weights = jnp.asarray([100.0, 100.0, 10.0, 10.0], dtype=DTYPE)
    return ctrl


def make_m22_h15v80(sim_p, seed=0):
    """H=15 V=80 N=2048 W=1.6× P=1e6 (parallel session M22 BEST)."""
    ctrl = FMCPlasmaJaxController(
        sim_p, n_walkers=2048, horizon=15, dt=1e-3,
        voltage_std=80.0, P_aux=1e6, gas_puff=1e21, seed=seed,
    )
    ctrl._weights = jnp.asarray([640.0, 640.0, 320.0, 320.0], dtype=DTYPE)
    return ctrl


def make_vanilla(sim_p, seed=0):
    """N=64 V=50 (= M16 historical FMC online, the original baseline)."""
    return FMCPlasmaJaxController(
        sim_p, n_walkers=64, horizon=10, dt=1e-3,
        voltage_std=50.0, P_aux=5e5, gas_puff=1e21, seed=seed,
    )


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
        "decision_us_p99": float(np.percentile(decision_times, 99) * 1e6),
    }


def main():
    print("=" * 72)
    print("M23 — MPC vs FMC oracle validation (calibrated sim, freegs truth)")
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
    print(f"\n[targets] {len(targets)} targets")
    for name, t in targets.items():
        print(f"    {name:35s}  R={t[0]:.3f} Z={t[1]:+.3f} κ={t[2]:.3f} δ={t[3]:+.3f}")

    controllers = {
        "MPC_DLQR": make_mpc,
        "M21_BEST_FMC": make_m21_best,
        "M22_H15V80_FMC": make_m22_h15v80,
        "vanilla_FMC": make_vanilla,
    }

    n_ticks = 30
    n_seeds = 4
    print(f"\n[run] {n_ticks} ticks × {n_seeds} seeds × {len(targets)} targets × "
          f"{len(controllers)} controllers = {n_ticks*n_seeds*len(targets)*len(controllers)} oracle calls")

    out = {
        "n_ticks": n_ticks, "n_seeds": n_seeds,
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
                    f"dec={r['decision_us_p50']:6.0f}µs  "
                    f"wall={r['wall_s']:.1f}s",
                    flush=True,
                )
            mean_t = np.mean([r["mean_truth_err"] for r in seed_results])
            mean_s = np.mean([r["steady_truth_err_last10"] for r in seed_results])
            std_s = np.std([r["steady_truth_err_last10"] for r in seed_results], ddof=1)
            mean_p = np.mean([r["physicality"] for r in seed_results])
            mean_dec = np.mean([r["decision_us_p50"] for r in seed_results])
            out["results"][ctrl_name][tgt_name] = {
                "mean_truth_err_avg": float(mean_t),
                "steady_truth_err_avg": float(mean_s),
                "steady_truth_err_std": float(std_s),
                "physicality_avg": float(mean_p),
                "decision_us_p50_avg": float(mean_dec),
                "n_seeds": n_seeds,
                "per_seed": seed_results,
            }
            print(f"    AGG: mean_truth={mean_t:.2f}  steady={mean_s:.2f}±{std_s:.2f}  "
                  f"phys={mean_p*100:.1f}%  dec={mean_dec:.0f}µs", flush=True)

    out_path = OUT_DIR / "m23_mpc_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")

    # ----- Final summary table -----
    print("\n" + "=" * 88)
    print("M23 FINAL VERDICT — calibrated sim + freegs oracle, 4 seeds × 3 targets")
    print("=" * 88)
    fmt_h = f"{'controller':<18s} {'target':<28s} {'truth':>8s} {'steady':>8s} {'std':>6s} {'phys%':>6s} {'lat µs':>8s}"
    print(fmt_h)
    print("-" * len(fmt_h))
    for cname, by_t in out["results"].items():
        for tname, agg in by_t.items():
            print(
                f"{cname:<18s} {tname:<28s} "
                f"{agg['mean_truth_err_avg']:8.2f} "
                f"{agg['steady_truth_err_avg']:8.2f} "
                f"{agg['steady_truth_err_std']:6.2f} "
                f"{agg['physicality_avg']*100:5.1f}% "
                f"{agg['decision_us_p50_avg']:8.0f}"
            )

    # ----- M16 head-to-head -----
    m16_data = {n: out["results"][n]["M16_real_TCV_65402"] for n in controllers}
    print("\n--- M16 TCV-X21 65402 head-to-head (THE falsifiable claim) ---")
    for cname, agg in m16_data.items():
        print(f"  {cname:<18s}  steady_truth={agg['steady_truth_err_avg']:7.3f} ± {agg['steady_truth_err_std']:.3f}  "
              f"latency={agg['decision_us_p50_avg']:>6.0f}µs  phys={agg['physicality_avg']*100:.0f}%")

    print("\nHistorical context:")
    print(f"  M16 historical FMC online       steady_truth=21.57    ~10000µs   (Apr 27)")
    print(f"  M16 historical M12 NN-shape     steady_truth= 3.47       122µs   (Apr 27, deploy)")

    mpc_steady = m16_data["MPC_DLQR"]["steady_truth_err_avg"]
    fmc_best_steady = m16_data["M21_BEST_FMC"]["steady_truth_err_avg"]
    print("\n--- VERDICT ---")
    if mpc_steady < fmc_best_steady:
        print(f"  H0 CONFIRMED: MPC ({mpc_steady:.3f}) < M21 BEST FMC ({fmc_best_steady:.3f})")
        print(f"  Margin: MPC is {fmc_best_steady/max(mpc_steady, 1e-6):.2f}× better truth-err")
        print(f"  → FMC has narrow value (offline expert generator); MPC is the right primary controller.")
    elif mpc_steady > fmc_best_steady:
        print(f"  H1 CONFIRMED (THESIS FALSIFIED): MPC ({mpc_steady:.3f}) > M21 BEST FMC ({fmc_best_steady:.3f})")
        print(f"  Margin: FMC is {mpc_steady/max(fmc_best_steady, 1e-6):.2f}× better truth-err than MPC")
        print(f"  → FMC has a real edge on this benchmark. Investigation warranted.")
    else:
        print(f"  TIE: MPC ({mpc_steady:.3f}) ≈ M21 BEST FMC ({fmc_best_steady:.3f})")
        print(f"  → Both viable; tie-breaker on latency/verification favours MPC.")


if __name__ == "__main__":
    main()
