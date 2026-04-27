"""Generate FreeGS dataset for NN shape surrogate (Milestone 11).

For a grid of constraint parameters, run FreeGS, record:
  - coil currents found by the constrain solver (input to NN)
  - resulting plasma shape (R_p, Z_p, kappa, delta) (target of NN)

Sampling strategy (constraint grid):
  - Z_xpoint     ∈ [0.55, 0.75]  (5 values)  — affects elongation
  - R_xpoint     ∈ [0.60, 0.70]  (3 values)  — affects radial position
  - R_outer_iso  ∈ [1.05, 1.15]  (3 values)  — outer midplane
  - R_inner_iso  ∈ [0.62, 0.68]  (3 values)  — inner midplane

Total = 5×3×3×3 = 135 freegs solves. At ~0.7 sec/solve = 95 sec wall-clock.

Each successful solve produces one (I_coils[20], shape[4]) data point.
Output: results/freegs_shape_dataset.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freegs
import freegs.critical
from freegs.machine import TCV

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"


def solve_with_constraints(Z_xpt, R_xpt, R_outer, R_inner,
                            I_p=2.0e5, paxis=1e3, fvac=0.5, maxits=15):
    """Try to solve a TCV equilibrium with the given DN constraint set."""
    m = TCV()
    eq = freegs.Equilibrium(
        tokamak=m, Rmin=0.3, Rmax=1.5, Zmin=-1.0, Zmax=1.0,
        nx=65, ny=65,
    )
    profiles = freegs.jtor.ConstrainPaxisIp(eq, paxis, I_p, fvac)
    xpoints = [(R_xpt, -Z_xpt), (R_xpt, +Z_xpt)]
    isoflux = [(R_outer, 0.0, R_inner, 0.0),
               (0.85, Z_xpt, 0.85, -Z_xpt)]
    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
    freegs.solve(eq, profiles, constrain, show=False, maxits=maxits)
    return eq


def extract_shape(eq) -> dict:
    psi = eq.psi()
    opt, _ = freegs.critical.find_critical(eq.R, eq.Z, psi)
    if not opt:
        raise RuntimeError("no O-point")
    R_axis, Z_axis, _ = opt[0]

    cs = plt.contour(eq.R, eq.Z, psi.T, levels=[eq.psi_bndry])
    plt.close()
    paths = cs.allsegs[0] if cs.allsegs else []
    boundary = None
    for path in paths:
        if len(path) < 20:
            continue
        if (path[:, 0].min() < R_axis < path[:, 0].max()
                and path[:, 1].min() < Z_axis < path[:, 1].max()):
            boundary = path
            break
    if boundary is None:
        raise RuntimeError("no LCFS")

    R_lcfs = boundary[:, 0]
    Z_lcfs = boundary[:, 1]
    R_p = 0.5 * (R_lcfs.max() + R_lcfs.min())
    a = 0.5 * (R_lcfs.max() - R_lcfs.min())
    Z_p = 0.5 * (Z_lcfs.max() + Z_lcfs.min())
    b = 0.5 * (Z_lcfs.max() - Z_lcfs.min())
    kappa = b / a
    idx_top = int(np.argmax(Z_lcfs))
    delta = (R_p - R_lcfs[idx_top]) / a
    return {"R_p": float(R_p), "Z_p": float(Z_p),
            "a": float(a), "kappa": float(kappa), "delta": float(delta),
            "I_p": float(eq.plasmaCurrent())}


def get_coil_currents(eq) -> dict:
    out = {}
    for label, elem in eq.tokamak.coils:
        if hasattr(elem, "current"):
            out[label] = float(elem.current)
        elif hasattr(elem, "coils"):
            for sub_label, sub_elem, factor in elem.coils:
                if hasattr(sub_elem, "current"):
                    out[sub_label] = float(sub_elem.current * factor)
    return out


COIL_ORDER = (
    [f"E{i}" for i in range(1, 9)]
    + [f"F{i}" for i in range(1, 9)]
    + ["T1", "T2", "T3"]
    + ["C1"]   # Use C1 as proxy for OH circuit (not all 4)
)


def coil_dict_to_array(d: dict) -> np.ndarray:
    return np.array([d.get(c, 0.0) for c in COIL_ORDER], dtype=np.float32)


def main():
    print("=" * 70)
    print("Milestone 11 — generate FreeGS shape dataset")
    print("=" * 70)

    # Constraint grid
    Z_xpoints = [0.55, 0.60, 0.65, 0.70, 0.75]
    R_xpoints = [0.60, 0.65, 0.70]
    R_outers = [1.05, 1.10, 1.15]
    R_inners = [0.62, 0.65, 0.68]

    total = len(Z_xpoints) * len(R_xpoints) * len(R_outers) * len(R_inners)
    print(f"  Grid size: {total} configurations")

    samples = []
    t0 = time.perf_counter()
    n_ok = 0
    n_fail = 0

    for i, Z_x in enumerate(Z_xpoints):
        for R_x in R_xpoints:
            for R_o in R_outers:
                for R_i in R_inners:
                    try:
                        eq = solve_with_constraints(Z_x, R_x, R_o, R_i)
                        shape = extract_shape(eq)
                        currents = get_coil_currents(eq)
                        I_coils = coil_dict_to_array(currents)
                        samples.append({
                            "I_coils": I_coils,
                            "shape": [shape["R_p"], shape["Z_p"],
                                      shape["kappa"], shape["delta"]],
                            "I_p": shape["I_p"],
                            "constraint": [Z_x, R_x, R_o, R_i],
                        })
                        n_ok += 1
                    except Exception as e:
                        n_fail += 1
                    # Progress
                    done = n_ok + n_fail
                    if done % 20 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate if rate > 0 else 0
                        print(f"  [{done}/{total}] ok={n_ok} fail={n_fail} | "
                              f"rate {rate:.1f}/s | eta {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {n_ok} ok, {n_fail} failed ({n_ok/(n_ok+n_fail)*100:.0f}% success) "
          f"in {elapsed:.0f}s")

    if n_ok == 0:
        print("ERROR: no successful solves")
        return

    # Stack
    I_arr = np.stack([s["I_coils"] for s in samples])
    shape_arr = np.array([s["shape"] for s in samples], dtype=np.float32)
    Ip_arr = np.array([s["I_p"] for s in samples], dtype=np.float32)

    out = RESULTS_DIR / "freegs_shape_dataset.npz"
    np.savez_compressed(
        out,
        I_coils=I_arr, shape=shape_arr, I_p=Ip_arr,
        coil_order=np.array(COIL_ORDER),
    )
    print(f"\n✓ Saved: {out}")
    print(f"  I_coils shape: {I_arr.shape} (B, 20)")
    print(f"  shape   shape: {shape_arr.shape} (B, 4)")

    print(f"\n  Shape statistics:")
    for j, name in enumerate(["R_p", "Z_p", "kappa", "delta"]):
        s = shape_arr[:, j]
        print(f"    {name:6s}: min={s.min():+.3f}, max={s.max():+.3f}, "
              f"mean={s.mean():+.3f}, std={s.std():.3f}")

    print(f"\n  I_coils statistics (kA):")
    I_kA = I_arr / 1000.0
    print(f"    min: {I_kA.min():.2f}, max: {I_kA.max():.2f}, "
          f"|max|: {np.abs(I_kA).max():.2f}")


if __name__ == "__main__":
    main()
