"""Robust FreeGS forward-mode oracle — Milestone 14.

Goal: extract a "true" plasma shape (R_p, Z_p, kappa, delta) from arbitrary
coil currents, robust to non-convergence of full Picard iteration.

M9/M13 found that direct `freegs.solve(constrain=None)` from arbitrary
I_coils diverges (0/24 success rate). The continuation method (warm-start
from baseline psi, gradually morph currents) ALSO fails — even at exact
baseline currents the unconstrained Picard does not converge.

Root cause: free-boundary GS without isoflux/X-point constraints needs the
LCFS topology to be uniquely identifiable from psi at every iteration.
For coil-current configurations far from the operating point, the field
has multiple critical points or none of them give a closed boundary.

WORKING APPROACH: Vacuum field + frozen plasma residual
--------------------------------------------------------
We solve the baseline equilibrium ONCE with full constraints. Decompose
the resulting flux as
    psi_baseline(R,Z) = psi_vacuum_baseline(R,Z) + psi_plasma_residual(R,Z)
where psi_vacuum_baseline is the contribution from the baseline coil currents
alone (Biot-Savart on the coils, no plasma) and psi_plasma_residual is the
remainder, which is the contribution from the toroidal plasma current.

For ARBITRARY new coil currents I_coils', we approximate
    psi_truth(R,Z; I_coils') ≈ psi_vacuum(R,Z; I_coils') + psi_plasma_residual

This is exact in the limit of small coil-current changes (linearization)
and remains a useful approximation for larger perturbations because the
plasma current is small (200 kA) compared to the coil net contribution to
the field.

Then we extract the LCFS by:
1. Find critical points of psi_truth via freegs.critical.find_critical.
2. Pick the O-point with largest psi inside the vessel, and the X-point
   that bounds it (lowest psi X-point above/below the O-point).
3. Trace the contour psi = psi_xpoint and compute shape from it.

If no valid LCFS is found → fallback to NN_shape proxy from M11.

Public API
----------
    oracle = FreeGSOracle()  # solves baseline once (~1 s)
    res = oracle.shape_from_coils(I_coils, fallback_fn=...)
    # OracleResult: R_p, Z_p, kappa, delta, source ("freegs"|"nn_fallback"|"failed")

Why this is faithful enough for an oracle
-----------------------------------------
- It's grounded in real GS physics (baseline psi is a true free-boundary
  solution; vacuum field is exact Biot-Savart).
- Same oracle for all policies → comparable.
- Differentiable from M11 NN_shape (which is a learned approximation).
- Fast: ~30-50 ms per shape (vs 700-1500 ms full GS).

Limitations vs. full GS
-----------------------
- Plasma current shape is frozen at baseline → cannot capture full
  non-linear plasma response.
- For very large coil-current perturbations the linearization error grows.
- Cannot detect configurations where there's NO physical equilibrium —
  it'll still extract a "shape" from artifact contours; the user should
  check the `n_lcfs_pts` and `inside_vessel` flags.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import freegs
import freegs.critical
from freegs.machine import TCV

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"

COIL_ORDER = [
    "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
]

# TCV vessel rough envelope (matches freegs TCV machine)
VESSEL_R_RANGE = (0.624, 1.136)
VESSEL_Z_RANGE = (-0.75, 0.75)


def get_baseline_currents(machine) -> dict[str, float]:
    out = {}
    for label, elem in machine.coils:
        if hasattr(elem, "current"):
            out[label] = float(elem.current)
        elif hasattr(elem, "coils"):
            for sub_label, sub_elem, factor in elem.coils:
                if hasattr(sub_elem, "current"):
                    out[sub_label] = float(sub_elem.current * factor)
    return out


def set_coil_currents(machine, currents: dict[str, float]) -> None:
    for label, elem in machine.coils:
        if label in currents and hasattr(elem, "current"):
            elem.current = float(currents[label])


def vacuum_psi(machine, R_grid: np.ndarray, Z_grid: np.ndarray) -> np.ndarray:
    """Compute vacuum (coils-only, no plasma) psi on grid.

    R_grid, Z_grid: 2D arrays from np.meshgrid(..., indexing='ij').
    """
    return machine.psi(R_grid, Z_grid)


def find_lcfs_from_psi(R: np.ndarray, Z: np.ndarray, psi: np.ndarray
                        ) -> Optional[dict]:
    """Find the LCFS contour and extract shape descriptors.

    Returns dict with R_p, Z_p, kappa, delta, a, n_lcfs_pts, psi_axis,
    psi_bndry. Returns None if no valid LCFS found.
    """
    if not np.isfinite(psi).all():
        return None

    try:
        opt, xpt = freegs.critical.find_critical(R, Z, psi)
    except Exception:
        return None
    if not opt or not xpt:
        return None

    # Pick the O-point with the largest psi (highest core flux) inside vessel
    R_min, R_max = VESSEL_R_RANGE
    Z_min, Z_max = VESSEL_Z_RANGE

    inside_opt = [(r, z, p) for (r, z, p) in opt
                  if R_min < r < R_max and Z_min < z < Z_max]
    if not inside_opt:
        return None
    R_axis, Z_axis, psi_axis = max(inside_opt, key=lambda t: t[2])

    # Try X-points in order: highest psi first (closest to axis from below).
    # For each X-point, attempt to trace a closed LCFS contour.
    # Fall back to next X-point if no valid boundary found.
    valid_xpts = sorted(
        [(r, z, p) for (r, z, p) in xpt
         if R_min - 0.1 < r < R_max + 0.1
         and Z_min - 0.15 < z < Z_max + 0.15
         and p < psi_axis - 1e-6],
        key=lambda t: -t[2],  # largest psi first
    )
    if not valid_xpts:
        return None

    boundary = None
    for (R_x, Z_x, psi_x) in valid_xpts[:5]:
        cs = plt.contour(R, Z, psi, levels=[psi_x])
        plt.close()
        paths = cs.allsegs[0] if cs.allsegs else []
        # Strict: closed loop containing axis with both R/Z extents
        for path in paths:
            if len(path) < 30:
                continue
            R_min_p, R_max_p = path[:, 0].min(), path[:, 0].max()
            Z_min_p, Z_max_p = path[:, 1].min(), path[:, 1].max()
            closure = np.linalg.norm(path[0] - path[-1])
            if (R_min_p <= R_axis <= R_max_p
                    and Z_min_p <= Z_axis <= Z_max_p
                    and (R_max_p - R_min_p) > 0.1
                    and closure < 0.3):
                boundary = path
                break
        if boundary is not None:
            break
    if boundary is None:
        # Last resort: any contour around axis
        for (R_x, Z_x, psi_x) in valid_xpts:
            cs = plt.contour(R, Z, psi, levels=[psi_x])
            plt.close()
            paths = cs.allsegs[0] if cs.allsegs else []
            for path in paths:
                if len(path) < 30:
                    continue
                if (path[:, 0].min() <= R_axis <= path[:, 0].max()
                        and path[:, 1].min() <= Z_axis <= path[:, 1].max()):
                    boundary = path
                    break
            if boundary is not None:
                break
    if boundary is None:
        return None

    R_lcfs = boundary[:, 0]
    Z_lcfs = boundary[:, 1]

    R_p = 0.5 * (R_lcfs.max() + R_lcfs.min())
    a = 0.5 * (R_lcfs.max() - R_lcfs.min())
    Z_p = 0.5 * (Z_lcfs.max() + Z_lcfs.min())
    b = 0.5 * (Z_lcfs.max() - Z_lcfs.min())
    if a < 0.05 or a > 0.5:
        return None
    kappa = b / a
    if kappa < 0.5 or kappa > 3.5:
        return None
    idx_top = int(np.argmax(Z_lcfs))
    delta = (R_p - R_lcfs[idx_top]) / a
    if abs(delta) > 1.5:
        return None

    return {
        "R_p": float(R_p), "Z_p": float(Z_p),
        "a": float(a), "kappa": float(kappa), "delta": float(delta),
        "n_lcfs_pts": int(len(R_lcfs)),
        "psi_axis": float(psi_axis), "psi_bndry": float(psi_x),
        "R_axis": float(R_axis), "Z_axis": float(Z_axis),
    }


@dataclass
class OracleResult:
    R_p: float
    Z_p: float
    kappa: float
    delta: float
    converged: bool
    solve_time_s: float
    source: str  # "freegs" | "nn_fallback" | "failed"
    psi_bndry: float = float("nan")
    psi_axis: float = float("nan")
    n_lcfs_pts: int = 0


class FreeGSOracle:
    """Forward-mode FreeGS oracle: vacuum + frozen-plasma decomposition."""

    def __init__(self,
                 I_p: float = 2.0e5, paxis: float = 1e3, fvac: float = 0.5,
                 nx: int = 65, ny: int = 65,
                 baseline_xpoints=((0.65, -0.65), (0.65, +0.65)),
                 baseline_isoflux=((1.10, 0.0, 0.65, 0.0),
                                   (0.85, 0.65, 0.85, -0.65)),
                 maxits_baseline: int = 30,
                 verbose: bool = False):
        self.I_p_target = I_p
        self.paxis = paxis
        self.fvac = fvac
        self.nx = nx
        self.ny = ny
        self.verbose = verbose

        if verbose:
            print("[oracle] solving baseline DN equilibrium...")
        t0 = time.perf_counter()

        m_baseline = TCV()
        eq = freegs.Equilibrium(
            tokamak=m_baseline, Rmin=0.3, Rmax=1.5,
            Zmin=-1.0, Zmax=1.0, nx=nx, ny=ny,
        )
        profiles = freegs.jtor.ConstrainPaxisIp(eq, paxis, I_p, fvac)
        constrain = freegs.control.constrain(
            xpoints=list(baseline_xpoints),
            isoflux=list(baseline_isoflux),
        )
        freegs.solve(eq, profiles, constrain, show=False, maxits=maxits_baseline)

        # Build R, Z grids
        R = eq.R  # already 2D from freegs
        Z = eq.Z
        self.R_grid = R
        self.Z_grid = Z

        psi_total_baseline = eq.psi().copy()
        # Vacuum psi from baseline coils
        psi_vacuum_baseline = vacuum_psi(m_baseline, R, Z)
        # Plasma residual = total - vacuum
        psi_plasma_residual = psi_total_baseline - psi_vacuum_baseline

        self.baseline_currents = get_baseline_currents(m_baseline)
        self.psi_plasma_residual = psi_plasma_residual
        self.baseline_eq = eq
        self.baseline_psi = psi_total_baseline

        sh = find_lcfs_from_psi(R, Z, psi_total_baseline)
        self.baseline_shape = sh
        self.baseline_solve_time = time.perf_counter() - t0

        if verbose:
            print(f"[oracle] baseline solved in {self.baseline_solve_time:.2f}s")
            if sh:
                print(f"[oracle] baseline R_p={sh['R_p']:.4f}, "
                      f"Z_p={sh['Z_p']:+.4f}, "
                      f"κ={sh['kappa']:.3f}, δ={sh['delta']:+.3f}, "
                      f"LCFS pts={sh['n_lcfs_pts']}")

    def _machine_with_currents(self, I_coils: np.ndarray) -> "freegs.machine.Machine":
        m = TCV()
        currents = dict(zip(COIL_ORDER, [float(x) for x in I_coils]))
        set_coil_currents(m, currents)
        return m

    def shape_from_coils(self,
                         I_coils: np.ndarray,
                         fallback_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
                         ) -> OracleResult:
        """Compute true shape for arbitrary coil currents.

        I_coils: array, length 16 (E1..E8, F1..F8).
        fallback_fn: optional callable I → [R_p, Z_p, kappa, delta] used
            when LCFS extraction fails.
        """
        # The simulator state has 20 active coils (E1-E8, F1-F8, C1-C2, D1-D2)
        # but freegs's TCV machine only has 16 (E + F) plus T/OH.
        # The C/D coils in the simulator geometry are vessel-correction coils
        # not present in freegs's TCV. We use the first 16 (E + F) which are
        # the dominant shape-determining coils and match freegs exactly.
        # We pass the FULL I_coils to the fallback (NN expects 20).
        I_full = np.asarray(I_coils)
        if len(I_full) == 20:
            I_for_freegs = I_full[:16]
        elif len(I_full) == 16:
            I_for_freegs = I_full
        else:
            raise ValueError(
                f"Expected 16 or 20 currents, got {len(I_full)}"
            )
        I_coils = I_for_freegs

        t0 = time.perf_counter()
        m = self._machine_with_currents(I_coils)
        psi_vacuum = vacuum_psi(m, self.R_grid, self.Z_grid)
        psi_total = psi_vacuum + self.psi_plasma_residual
        sh = find_lcfs_from_psi(self.R_grid, self.Z_grid, psi_total)
        elapsed = time.perf_counter() - t0

        if sh is not None:
            return OracleResult(
                R_p=sh["R_p"], Z_p=sh["Z_p"],
                kappa=sh["kappa"], delta=sh["delta"],
                converged=True, solve_time_s=elapsed,
                source="freegs",
                psi_bndry=sh["psi_bndry"],
                psi_axis=sh["psi_axis"],
                n_lcfs_pts=sh["n_lcfs_pts"],
            )

        if fallback_fn is not None:
            fb = fallback_fn(I_full)
            return OracleResult(
                R_p=float(fb[0]), Z_p=float(fb[1]),
                kappa=float(fb[2]), delta=float(fb[3]),
                converged=False, solve_time_s=elapsed,
                source="nn_fallback",
            )

        return OracleResult(
            R_p=float("nan"), Z_p=float("nan"),
            kappa=float("nan"), delta=float("nan"),
            converged=False, solve_time_s=elapsed,
            source="failed",
        )


def main():
    """Validate the oracle on a grid of coil-current perturbations."""
    print("=" * 70)
    print("Milestone 14 — Robust FreeGS forward-mode oracle")
    print("=" * 70)

    print("\n[1] Initializing oracle (solving baseline DN equilibrium)...")
    oracle = FreeGSOracle(verbose=True)
    if oracle.baseline_shape is None:
        print("FATAL: baseline failed")
        return

    baseline_I = np.array(
        [oracle.baseline_currents.get(k, 0.0) for k in COIL_ORDER]
    )

    # ---- Test grid ----
    print(f"\n[2] Testing convergence on grid of perturbations...")
    rng = np.random.default_rng(14)
    test_cases = []

    for k in range(8):
        scale = 1.0 + 0.05 * rng.standard_normal()
        I = baseline_I * scale + rng.normal(0, 100, size=16)
        test_cases.append(("small_perturb_5pct", I))

    for k in range(8):
        scale = 1.0 + 0.15 * rng.standard_normal()
        I = baseline_I * scale + rng.normal(0, 500, size=16)
        test_cases.append(("medium_perturb_15pct", I))

    for k in range(8):
        scale = 1.0 + 0.30 * rng.standard_normal()
        I = baseline_I * scale + rng.normal(0, 1000, size=16)
        test_cases.append(("large_perturb_30pct", I))

    for k in range(4):
        I = baseline_I + rng.normal(0, 3000, size=16)
        test_cases.append(("additive_3kA", I))

    test_cases.append(("zero_shaping", np.zeros(16)))
    test_cases.append(("baseline_exact", baseline_I.copy()))

    print(f"    {len(test_cases)} test cases")

    results = []
    times = []
    for i, (label, I) in enumerate(test_cases):
        res = oracle.shape_from_coils(I)
        times.append(res.solve_time_s)
        marker = "✓" if res.converged else "✗"
        if res.converged:
            print(f"  [{i+1:2d}/{len(test_cases)}] {marker} "
                  f"{label:22s} | {1000*res.solve_time_s:5.1f} ms | "
                  f"R_p={res.R_p:.3f} Z_p={res.Z_p:+.3f} "
                  f"κ={res.kappa:.3f} δ={res.delta:+.3f}")
        else:
            print(f"  [{i+1:2d}/{len(test_cases)}] {marker} "
                  f"{label:22s} | {1000*res.solve_time_s:5.1f} ms | FAILED")
        results.append({
            "label": label,
            "I_coils": I.tolist(),
            "converged": res.converged,
            "solve_time_s": res.solve_time_s,
            "R_p": res.R_p, "Z_p": res.Z_p,
            "kappa": res.kappa, "delta": res.delta,
            "source": res.source,
            "n_lcfs_pts": res.n_lcfs_pts,
        })

    n_conv = sum(r["converged"] for r in results)
    print(f"\n[3] Convergence rate: {n_conv}/{len(results)} = "
          f"{100*n_conv/len(results):.0f}%")
    print(f"    Mean solve time: {1000*np.mean(times):.1f} ms")
    print(f"    Speedup vs full GS (~700 ms): "
          f"{700/np.mean(times)/1000:.0f}× slower" if np.mean(times) > 0.7
          else f"    Speedup vs full GS (~700 ms): "
                f"{0.7/np.mean(times):.1f}×")

    out = {
        "baseline_shape": oracle.baseline_shape,
        "baseline_currents": oracle.baseline_currents,
        "n_test_cases": len(test_cases),
        "n_converged": n_conv,
        "success_rate": n_conv / len(results),
        "mean_solve_time_ms": float(1000 * np.mean(times)),
        "results": results,
    }
    out_path = RESULTS_DIR / "milestone_14_oracle_robust.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
