"""Mutual & self inductance for circular coaxial filamentary loops.

References:
- Jackson, *Classical Electrodynamics* 3rd ed. §5.17
- Smythe, *Static and Dynamic Electricity* §8.06
- Wesson, *Tokamaks* 4th ed. §3.7

Formula (Neumann / Maxwell, complete elliptic form):

    M(R1, R2, dZ) = μ₀ √(R1·R2) · [ (2/k - k) K(k) - (2/k) E(k) ]

with

    k² = 4 R1 R2 / [ (R1 + R2)² + dZ² ]

K(k), E(k) = complete elliptic integrals of 1st and 2nd kind (k is the
modulus, not the parameter m=k²; scipy uses m, so we pass k**2).

Self-inductance of a thin circular loop (cross-section radius a_w):

    L_self = μ₀ R [ ln(8R/a_w) - 7/4 ]
"""
from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk

MU_0 = 4 * np.pi * 1e-7  # H/m


def mutual_inductance_pair(R1: float, R2: float, dZ: float) -> float:
    """Mutual inductance [H] between two coaxial filamentary loops.

    Args:
        R1, R2: loop radii [m]
        dZ: axial separation [m]
    """
    if R1 <= 0 or R2 <= 0:
        return 0.0
    denom = (R1 + R2) ** 2 + dZ**2
    k_sq = 4.0 * R1 * R2 / denom
    # Numerical safety: k² ∈ (0, 1)
    k_sq = min(k_sq, 1.0 - 1e-12)
    k = np.sqrt(k_sq)
    K = ellipk(k_sq)  # scipy: parameter m = k²
    E = ellipe(k_sq)
    return MU_0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * K - (2.0 / k) * E)


def self_inductance(R: float, a_wire: float = 0.01) -> float:
    """Self-inductance [H] of a thin circular loop.

    Args:
        R: loop radius [m]
        a_wire: wire/cross-section radius [m] (default 1 cm — typical TCV coil)
    """
    if R <= 0 or a_wire <= 0 or a_wire >= R:
        raise ValueError(f"Invalid loop geometry: R={R}, a_wire={a_wire}")
    return MU_0 * R * (np.log(8.0 * R / a_wire) - 7.0 / 4.0)


def mutual_matrix(
    R: np.ndarray,
    Z: np.ndarray,
    a_wire: float = 0.01,
) -> np.ndarray:
    """Full N×N mutual inductance matrix for a set of coaxial loops.

    Diagonal = self-inductance. Off-diagonal = pair Neumann formula.

    Args:
        R: shape (N,), loop radii [m]
        Z: shape (N,), loop heights [m]
        a_wire: cross-section radius [m]

    Returns:
        M: shape (N, N), symmetric, [H]
    """
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    N = R.shape[0]
    assert Z.shape == (N,), f"R,Z shape mismatch: {R.shape} vs {Z.shape}"

    M = np.zeros((N, N))
    for i in range(N):
        M[i, i] = self_inductance(R[i], a_wire)
        for j in range(i + 1, N):
            mij = mutual_inductance_pair(R[i], R[j], Z[i] - Z[j])
            M[i, j] = mij
            M[j, i] = mij
    return M


def mutual_to_plasma(
    R_coils: np.ndarray,
    Z_coils: np.ndarray,
    R_p: float,
    Z_p: float,
) -> np.ndarray:
    """Mutual inductance vector M_pc[i] = M(plasma_loop, coil_i).

    Treats plasma as a single filamentary current loop at (R_p, Z_p).
    This is a 0D approximation; real codes treat plasma as distributed.

    Returns:
        M_pc: shape (N_coils,), [H]
    """
    R_coils = np.asarray(R_coils, dtype=np.float64)
    Z_coils = np.asarray(Z_coils, dtype=np.float64)
    return np.array([
        mutual_inductance_pair(R_p, R_coils[i], Z_p - Z_coils[i])
        for i in range(R_coils.shape[0])
    ])


if __name__ == "__main__":
    # Sanity checks against textbook examples
    print("Mutual inductance sanity checks")
    print("=" * 60)

    # Two identical 1m-radius loops, separation 1m
    # Hand check: k² = 4/5 = 0.8 → M = μ₀ · [(2-k²)/k · K(k) - 2/k · E(k)]
    #          = 4π·10⁻⁷ · 0.394 ≈ 0.495 µH
    M_test = mutual_inductance_pair(1.0, 1.0, 1.0)
    print(f"Two 1m loops, ΔZ=1m: M = {M_test*1e6:.4f} µH (expected ≈ 0.495)")

    # Two identical loops, very far apart → 0
    M_far = mutual_inductance_pair(1.0, 1.0, 100.0)
    print(f"Two 1m loops, ΔZ=100m: M = {M_far*1e9:.4f} nH (should ≈ 0)")

    # Self-inductance of 1m loop, 1cm wire
    L_test = self_inductance(1.0, 0.01)
    print(f"Self L of 1m loop, 1cm wire: L = {L_test*1e6:.3f} µH")
    # Expected ≈ 6.27 µH

    # Symmetric matrix check
    R_demo = np.array([0.5, 1.0, 1.5])
    Z_demo = np.array([0.0, 0.3, -0.2])
    M_demo = mutual_matrix(R_demo, Z_demo)
    print(f"\nDemo 3x3 matrix:")
    print(M_demo * 1e6)
    print(f"Symmetric? max|M - M.T| = {np.max(np.abs(M_demo - M_demo.T)):.2e}")
    print(f"Positive definite? eigenvalues: {np.linalg.eigvalsh(M_demo) * 1e6}")
