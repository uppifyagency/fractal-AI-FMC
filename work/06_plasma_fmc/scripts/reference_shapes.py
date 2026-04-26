"""Reference plasma shapes — Miller parametric LCFS.

Generates 3 canonical TCV configurations:
1. Standard ELMy single-null  (κ=1.7, δ=0.3)
2. Snowflake divertor          (κ=2.0, δ=0.4, with secondary X-point)
3. Negative triangularity      (κ=1.5, δ=-0.4)

Math reference: Miller, R.L. et al. "Noncircular, finite aspect ratio,
local equilibrium model." Phys. Plasmas 5:973 (1998).

The Miller LCFS parameterization is:
    R(θ) = R₀ + a · cos(θ + arcsin(δ) · sin(θ))
    Z(θ) = Z₀ + κ · a · sin(θ)

This is *not* a self-consistent equilibrium — it's a target shape.
A real Grad-Shafranov solve (FreeGSNKE/freegs) is needed to find the
coil currents that produce this shape. We compute that in milestone 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tcv_geometry import TCVMachine, load_tcv


@dataclass(frozen=True)
class ReferenceShape:
    name: str
    R0: float       # geometric center R [m]
    Z0: float       # geometric center Z [m]
    a: float        # minor radius [m]
    kappa: float    # elongation
    delta: float    # triangularity (positive = D-shape, negative = inverted)
    description: str

    def lcfs(self, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Sample the Last Closed Flux Surface using Miller parameterization.

        Returns:
            (R_array, Z_array) of shape (n_points,) in meters.
        """
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        R = self.R0 + self.a * np.cos(theta + np.arcsin(self.delta) * np.sin(theta))
        Z = self.Z0 + self.kappa * self.a * np.sin(theta)
        return R, Z

    def derived(self, B_T: float, I_p: float) -> dict:
        """Compute derived plasma physics quantities.

        Args:
            B_T: toroidal field at R₀ [T]
            I_p: plasma current [A]

        Returns:
            dict with q95 (cylindrical), n_GW, beta_N_limit, surface_area, volume.
        """
        # Surface and volume of D-shape (analytical, Miller approximation):
        #   V = 2π² R₀ a² κ
        #   S = 4π² R₀ a √((1+κ²)/2)  [first-order, ignoring δ]
        # Reference: Wesson, Tokamaks 4th ed. §1.4
        volume = 2 * np.pi**2 * self.R0 * self.a**2 * self.kappa
        surface = 4 * np.pi**2 * self.R0 * self.a * np.sqrt((1 + self.kappa**2) / 2)

        # q95 cylindrical approximation (Wesson §3.6, eq. 3.6.1):
        #   q_a = (2π a² B_T) / (μ₀ R₀ I_p) · (1+κ²)/2
        # Numerical with I_p in MA:
        I_p_MA = I_p / 1e6
        if I_p_MA <= 0:
            q95 = float("inf")
        else:
            q95 = (
                (5.0 * self.a**2 * B_T * (1 + self.kappa**2) / 2.0)
                / (self.R0 * I_p_MA)
            )

        # Greenwald density limit [10²⁰ m⁻³]:
        #   n_GW = I_p[MA] / (π a²)
        # Reference: Greenwald PPCF 44:R27 (2002)
        n_GW_e20 = I_p_MA / (np.pi * self.a**2)
        n_GW = n_GW_e20 * 1e20  # [m⁻³]

        # Troyon β_N limit (β_N typically <2.8 for stability)
        # β_N = β[%] · a · B_T / I_p[MA]
        # Solving for β_max with β_N=2.8:
        if I_p_MA > 0:
            beta_max_pct = 2.8 * I_p_MA / (self.a * B_T)
        else:
            beta_max_pct = 0.0

        return {
            "volume_m3": volume,
            "surface_m2": surface,
            "q95_cylindrical": q95,
            "n_GW_m3": n_GW,
            "beta_max_pct": beta_max_pct,
            "epsilon": self.a / self.R0,
        }


def standard_single_null(tcv: TCVMachine) -> ReferenceShape:
    """Standard ELMy H-mode-like single-null shape on TCV.

    Source for parameter values: Reimerdes 2022 typical TCV scenario.
    """
    return ReferenceShape(
        name="standard_single_null",
        R0=tcv.R_major,
        Z0=0.0,
        a=tcv.a_minor * 0.96,  # leave small gap to vessel
        kappa=1.7,
        delta=0.3,
        description="Standard ELMy H-mode single-null divertor (TCV typical)",
    )


def snowflake(tcv: TCVMachine) -> ReferenceShape:
    """Snowflake divertor configuration.

    Approximated as elongated D-shape; true snowflake has 2 close X-points
    that need a 2D GS solve. This is a Miller approximation only.
    Source: Reimerdes 2022, snowflake demonstrated on TCV up to κ~2.
    """
    return ReferenceShape(
        name="snowflake",
        R0=tcv.R_major,
        Z0=0.05,  # slight upward shift typical for snowflake
        a=tcv.a_minor * 0.92,
        kappa=2.0,
        delta=0.4,
        description="Snowflake divertor (Miller approximation only)",
    )


def negative_triangularity(tcv: TCVMachine) -> ReferenceShape:
    """Negative triangularity scenario.

    TCV pioneered NT operation. Source: Sauter et al. *Phys. Plasmas* 21:055906
    (2014); Reimerdes 2022 routine NT operation on TCV.
    """
    return ReferenceShape(
        name="negative_triangularity",
        R0=tcv.R_major,
        Z0=0.0,
        a=tcv.a_minor * 0.96,
        kappa=1.5,
        delta=-0.4,
        description="Negative triangularity (TCV NT scenario)",
    )


def all_reference_shapes(tcv: TCVMachine) -> list[ReferenceShape]:
    return [
        standard_single_null(tcv),
        snowflake(tcv),
        negative_triangularity(tcv),
    ]


def verify_shapes_within_machine(tcv: TCVMachine) -> dict:
    """Sanity check: every reference LCFS must fit inside the vacuum vessel."""
    issues = {}
    R_in = tcv.vessel["inner_R"]
    R_out = tcv.vessel["outer_R"]
    Z_max = tcv.vessel["height"] / 2

    for shape in all_reference_shapes(tcv):
        R, Z = shape.lcfs(200)
        if R.min() < R_in:
            issues[shape.name] = f"LCFS R_min={R.min():.3f} < vessel inner_R={R_in}"
        if R.max() > R_out:
            issues[shape.name] = (
                f"LCFS R_max={R.max():.3f} > vessel outer_R={R_out}"
            )
        if abs(Z).max() > Z_max:
            issues[shape.name] = (
                f"LCFS |Z|_max={abs(Z).max():.3f} > vessel half-height={Z_max}"
            )
    return issues


if __name__ == "__main__":
    tcv = load_tcv()
    print("Reference shapes verification")
    print("=" * 60)
    issues = verify_shapes_within_machine(tcv)
    if issues:
        print("⚠ Geometry violations:")
        for name, msg in issues.items():
            print(f"  {name}: {msg}")
    else:
        print("✓ All shapes fit within vacuum vessel")
    print()

    # Sample physics for typical TCV operating point
    B_T_typical = 1.43  # T (typical TCV)
    I_p_typical = 200_000.0  # 200 kA flat-top

    print(f"Derived quantities at B_T={B_T_typical} T, I_p={I_p_typical/1e3:.0f} kA")
    print("-" * 60)
    for shape in all_reference_shapes(tcv):
        d = shape.derived(B_T_typical, I_p_typical)
        print(f"\n{shape.name} (κ={shape.kappa}, δ={shape.delta}):")
        print(f"  Volume      : {d['volume_m3']:.4f} m³")
        print(f"  Surface     : {d['surface_m2']:.4f} m²")
        print(f"  q95 (cyl.)  : {d['q95_cylindrical']:.3f}")
        print(f"  n_GW        : {d['n_GW_m3']:.3e} m⁻³")
        print(f"  β_max       : {d['beta_max_pct']:.3f} %")
        print(f"  ε = a/R     : {d['epsilon']:.4f}")
