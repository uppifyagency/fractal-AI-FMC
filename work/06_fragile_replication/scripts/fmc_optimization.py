"""fmc_optimization.py — Fractal Monte Carlo per ottimizzazione funzioni.

Paradigma diverso da FMC-planning (Atari/Craftax):
  - Ogni walker = un PUNTO in R^d (search space), non uno state MDP
  - Reward = -f(x)  (vogliamo minimizzare f)
  - "Step" = perturbazione gaussiana del punto (gradient-free random walk)
  - Distance = L2 fra punti walker
  - Cloning: high-VR walker attira low-VR (come in FMC vanilla)

Riferimento: fragile/src/fragile/benchmarks.py (definisce le 8 funzioni test) e
fragile/src/fragile/core.py:run_swarm (entry point originale Hernández).

Funzioni test implementate:
  - sphere       — convex, min in 0
  - rastrigin    — multimodal, min in 0
  - eggholder    — 2D very hard, min ≈ -959.64 in (512, 404.23)
  - styblinski_tang — multimodal, min ≈ -39.16·d in (-2.9035)·1
  - rosenbrock   — valley, min in (1,1,...,1) value 0
  - easom        — 2D narrow basin, min in (π,π) value -1
  - holder_table — 2D, min ≈ -19.21 in (8.055, 9.665)
  - lennard_jones — molecular, min depends on n_atoms

Uso:
  python fmc_optimization.py --func rastrigin --dims 5 --n_walkers 100 \\
                              --n_iters 500 --seed 42

Output JSON: best value found, best position, walker history.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass

import numpy as np


# ============================================================================
# Test functions (numpy port di fragile/benchmarks.py)
# ============================================================================

def f_sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, axis=1)


def f_rastrigin(x: np.ndarray) -> np.ndarray:
    A = 10
    d = x.shape[1]
    return A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)


def f_eggholder(x: np.ndarray) -> np.ndarray:
    """2D only."""
    x0, x1 = x[:, 0], x[:, 1]
    first = np.sqrt(np.abs(x0 / 2.0 + (x1 + 47)))
    second = np.sqrt(np.abs(x0 - (x1 + 47)))
    return -(x1 + 47) * np.sin(first) - x0 * np.sin(second)


def f_styblinski_tang(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1) / 2.0


def f_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Generalized Rosenbrock per d>=2: sum_{i=0..d-2} 100·(x_{i+1}-x_i^2)^2 + (1-x_i)^2."""
    return np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2,
        axis=1,
    )


def f_easom(x: np.ndarray) -> np.ndarray:
    """2D only, min in (π, π)."""
    x0, x1 = x[:, 0], x[:, 1]
    exp_term = (x0 - np.pi) ** 2 + (x1 - np.pi) ** 2
    return -np.cos(x0) * np.cos(x1) * np.exp(-exp_term)


def f_holder_table(x: np.ndarray) -> np.ndarray:
    """2D only."""
    x0, x1 = x[:, 0], x[:, 1]
    expo = np.abs(1 - np.sqrt(x0 ** 2 + x1 ** 2) / np.pi)
    return -np.abs(np.sin(x0) * np.cos(x1) * np.exp(expo))


# Function registry: name -> (function, bounds_per_dim, default_dims, known_minimum)
FUNCTIONS = {
    "sphere":          (f_sphere,         (-5.12, 5.12), 5,   0.0),
    "rastrigin":       (f_rastrigin,      (-5.12, 5.12), 5,   0.0),
    "eggholder":       (f_eggholder,      (-512, 512),   2,   -959.6407),
    "styblinski_tang": (f_styblinski_tang,(-5, 5),       5,   -39.16599 * 5),
    "rosenbrock":      (f_rosenbrock,     (-5, 10),      5,   0.0),
    "easom":           (f_easom,          (-100, 100),   2,   -1.0),
    "holder_table":    (f_holder_table,   (-10, 10),     2,   -19.2085),
}


# ============================================================================
# FMC core (gradient-free swarm optimizer)
# ============================================================================

def relativize(x: np.ndarray) -> np.ndarray:
    """Paper §2.2.3 — preserva ordinamento, garantisce > 0."""
    std = float(x.std())
    if std == 0:
        return np.ones_like(x, dtype=np.float32)
    z = (x - x.mean()) / std
    return np.where(z <= 0, np.exp(np.clip(z, -50, 0)), 1 + np.log1p(np.maximum(z, 0))).astype(np.float32)


@dataclass
class FMCOptConfig:
    n_walkers: int = 100
    n_iters: int = 500
    perturbation_sigma: float = 0.1   # std deviation of gaussian step (in normalized [0,1] units)
    sigma_decay: float = 0.999         # exponential decay per iter (annealing)
    balance: float = 1.0               # alpha = beta
    distance_metric: str = "l2"        # only l2 implemented


def fmc_optimize(func, bounds: tuple[float, float], dims: int, cfg: FMCOptConfig,
                 seed: int = 42, verbose: bool = False) -> dict:
    """Run FMC optimization swarm.

    Returns dict with best value, best position, history, n_func_evals.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    extent = hi - lo

    # Init walkers uniformly in bounds
    N = cfg.n_walkers
    walkers = rng.uniform(lo, hi, size=(N, dims)).astype(np.float32)

    # Initial reward: -f(x) (we minimize f, maximize -f)
    f_vals = func(walkers).astype(np.float32)
    rewards = -f_vals
    n_evals = N

    best_idx = int(np.argmin(f_vals))
    best_val = float(f_vals[best_idx])
    best_pos = walkers[best_idx].copy()
    history_best = [best_val]

    sigma = cfg.perturbation_sigma * extent  # absolute sigma in problem units

    t_start = time.time()
    for it in range(cfg.n_iters):
        # Perturbation: gradient-free random walk in absolute units
        noise = rng.normal(0.0, sigma, size=walkers.shape).astype(np.float32)
        new_walkers = walkers + noise
        # Reflect at bounds (mantieni nel dominio)
        new_walkers = np.clip(new_walkers, lo, hi)

        # Eval new positions
        new_f = func(new_walkers).astype(np.float32)
        new_rewards = -new_f
        n_evals += N

        # Greedy keep: walker accetta perturbazione SOLO se migliora la sua reward.
        # Questo è "elitist hill-climb" per walker. Sergio fa cloning dopo;
        # qui scelgo di tenere la versione migliore per walker (combina hill-climb + cloning).
        accept = new_rewards > rewards
        walkers = np.where(accept[:, None], new_walkers, walkers)
        rewards = np.where(accept, new_rewards, rewards)
        f_vals = -rewards

        # Distance: each walker vs random partner
        partners = rng.permutation(N)
        same = partners == np.arange(N)
        partners[same] = (partners[same] + 1) % N
        distances = np.linalg.norm(walkers - walkers[partners], axis=1)

        # Virtual reward
        R_norm = relativize(rewards)
        D_norm = relativize(distances)
        VR = (R_norm ** cfg.balance) * (D_norm ** cfg.balance)

        # Cloning step
        clone_partners = rng.permutation(N)
        same = clone_partners == np.arange(N)
        clone_partners[same] = (clone_partners[same] + 1) % N
        VR_self = VR
        VR_other = VR[clone_partners]
        denom = np.where(VR_self > 1e-8, VR_self, 1e-8)
        clone_prob = np.clip((VR_other - VR_self) / denom, 0, 1)
        draws = rng.random(N)
        will_clone = draws < clone_prob

        # Apply clone (walker copies position from partner)
        walkers[will_clone] = walkers[clone_partners[will_clone]]
        rewards[will_clone] = rewards[clone_partners[will_clone]]
        f_vals = -rewards

        # Track best ever seen
        cur_best_idx = int(np.argmin(f_vals))
        if f_vals[cur_best_idx] < best_val:
            best_val = float(f_vals[cur_best_idx])
            best_pos = walkers[cur_best_idx].copy()
        history_best.append(best_val)

        # Annealing
        sigma *= cfg.sigma_decay

        if verbose and (it + 1) % 50 == 0:
            print(f"  iter {it+1}/{cfg.n_iters}  best_f={best_val:.6f}  "
                  f"sigma={sigma:.3f}  evals={n_evals}", file=sys.stderr)

    return {
        "best_value": float(best_val),
        "best_position": best_pos.tolist(),
        "n_func_evals": int(n_evals),
        "wall_time_s": float(time.time() - t_start),
        "history_best": [float(v) for v in history_best],
        "config": {
            "n_walkers": cfg.n_walkers,
            "n_iters": cfg.n_iters,
            "perturbation_sigma": cfg.perturbation_sigma,
            "sigma_decay": cfg.sigma_decay,
            "balance": cfg.balance,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True, choices=list(FUNCTIONS.keys()))
    ap.add_argument("--dims", type=int, default=None,
                    help="number of dimensions (only valid for non-2D-only functions)")
    ap.add_argument("--n_walkers", type=int, default=100)
    ap.add_argument("--n_iters", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--sigma_decay", type=float, default=0.999)
    ap.add_argument("--balance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    func, bounds, default_dims, known_min = FUNCTIONS[args.func]
    dims = args.dims if args.dims is not None else default_dims
    if args.func in ("eggholder", "easom", "holder_table") and dims != 2:
        print(f"WARN: {args.func} is 2D only — overriding dims=2", file=sys.stderr)
        dims = 2

    cfg = FMCOptConfig(
        n_walkers=args.n_walkers, n_iters=args.n_iters,
        perturbation_sigma=args.sigma, sigma_decay=args.sigma_decay,
        balance=args.balance,
    )
    result = fmc_optimize(func, bounds, dims, cfg, args.seed, args.verbose)
    result["function"] = args.func
    result["dims"] = dims
    result["bounds"] = list(bounds)
    result["known_minimum"] = known_min
    result["gap_to_optimum"] = abs(result["best_value"] - known_min)
    result["seed"] = args.seed
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
