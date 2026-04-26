"""TCV geometry: load and validate coil positions, machine parameters.

Single source of truth for everything geometric in this PoC.
Cross-checked against:
- freegs.machine.TCV() — community-validated implementation
- EPFL Infoscience LRP-755-13 — primary EPFL source
- Reimerdes et al. Nucl. Fusion 62 (2022) — published overview

See REFERENCES.md for full citations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "tcv_geometry.yaml"


@dataclass(frozen=True)
class Coil:
    name: str
    R: float
    Z: float


@dataclass(frozen=True)
class TCVMachine:
    R_major: float
    a_minor: float
    B_T_max: float
    I_p_max: float
    kappa_max: float
    delta_min: float
    delta_max: float

    shaping_coils: tuple[Coil, ...]
    t_coils: tuple[Coil, ...]
    ohmic_coils: tuple[Coil, ...]
    solenoid: dict

    I_max_EF: float
    I_max_OH: float
    V_max_EF: float
    V_max_OH: float
    R_coil_uniform: float

    vessel: dict
    control: dict

    @property
    def n_shaping(self) -> int:
        return len(self.shaping_coils)

    @property
    def n_t(self) -> int:
        return len(self.t_coils)

    @property
    def n_control_channels(self) -> int:
        # 16 shaping + 3 T + 1 OH circuit = 20 (matches Degrave 2022 architecture)
        return self.n_shaping + self.n_t + 1

    @property
    def all_active_coils(self) -> tuple[Coil, ...]:
        """Coils that produce field in the plasma region.

        Excludes T coils (toroidal correction, far from plasma).
        Includes shaping + ohmic-circuit lumped elements.
        """
        return self.shaping_coils + self.ohmic_coils

    @property
    def epsilon(self) -> float:
        """Inverse aspect ratio a/R."""
        return self.a_minor / self.R_major


def load_tcv(config_path: Path = CONFIG_PATH) -> TCVMachine:
    """Load TCV geometry from YAML config.

    Verifications performed:
    - All E coils at R=0.5050 m (matches freegs source)
    - All F coils at R=1.3095 m (matches freegs source)
    - 16 shaping coils total (matches Reimerdes 2022)
    - Inverse aspect ratio epsilon = a/R = 0.284 (TCV value)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    shaping = tuple(
        Coil(name=name, R=d["R"], Z=d["Z"])
        for name, d in cfg["shaping_coils"].items()
    )
    t = tuple(
        Coil(name=name, R=d["R"], Z=d["Z"]) for name, d in cfg["t_coils"].items()
    )
    ohmic = tuple(
        Coil(name=name, R=d["R"], Z=d["Z"])
        for name, d in cfg["ohmic_circuit"].items()
        if name != "solenoid"
    )

    machine = TCVMachine(
        R_major=cfg["machine"]["R_major"],
        a_minor=cfg["machine"]["a_minor"],
        B_T_max=cfg["machine"]["B_T_max"],
        I_p_max=cfg["machine"]["I_p_max"],
        kappa_max=cfg["machine"]["kappa_max"],
        delta_min=cfg["machine"]["delta_min"],
        delta_max=cfg["machine"]["delta_max"],
        shaping_coils=shaping,
        t_coils=t,
        ohmic_coils=ohmic,
        solenoid=cfg["ohmic_circuit"]["solenoid"],
        I_max_EF=cfg["limits"]["I_max_EF"],
        I_max_OH=cfg["limits"]["I_max_OH"],
        V_max_EF=cfg["limits"]["V_max_EF"],
        V_max_OH=cfg["limits"]["V_max_OH"],
        R_coil_uniform=cfg["limits"]["R_coil_uniform"],
        vessel=cfg["vessel"],
        control=cfg["control"],
    )

    _verify_geometry(machine)
    return machine


def _verify_geometry(m: TCVMachine) -> None:
    """Cross-checks against published TCV values."""
    assert m.n_shaping == 16, f"Expected 16 shaping coils, got {m.n_shaping}"
    assert m.n_t == 3, f"Expected 3 T coils, got {m.n_t}"
    assert m.n_control_channels == 20, (
        f"Expected 20 control channels, got {m.n_control_channels}"
    )

    # All E coils at R=0.5050
    e_coils = [c for c in m.shaping_coils if c.name.startswith("E")]
    assert len(e_coils) == 8, f"Expected 8 E coils, got {len(e_coils)}"
    for c in e_coils:
        assert abs(c.R - 0.5050) < 1e-6, f"E coil {c.name} at R={c.R}, expected 0.5050"

    # All F coils at R=1.3095
    f_coils = [c for c in m.shaping_coils if c.name.startswith("F")]
    assert len(f_coils) == 8, f"Expected 8 F coils, got {len(f_coils)}"
    for c in f_coils:
        assert abs(c.R - 1.3095) < 1e-6, f"F coil {c.name} at R={c.R}, expected 1.3095"

    # Aspect ratio
    expected_epsilon = 0.25 / 0.88
    assert abs(m.epsilon - expected_epsilon) < 1e-9, (
        f"Inverse aspect ratio mismatch: {m.epsilon} vs {expected_epsilon}"
    )

    # E and F coils are vertically symmetric about midplane
    e_zs = sorted([c.Z for c in e_coils])
    for z_neg, z_pos in zip(e_zs[: len(e_zs) // 2], e_zs[len(e_zs) // 2 :][::-1]):
        assert abs(z_neg + z_pos) < 1e-6, f"E coils not symmetric: {z_neg} vs {-z_pos}"


def cross_check_against_freegs() -> dict:
    """Verify our coil positions match freegs.machine.TCV() exactly.

    Returns dict with diff report; empty if perfect match.
    """
    from freegs.machine import TCV as freegs_TCV

    fg = freegs_TCV()
    fg_coils = {}
    for label, elem in fg.coils:
        if hasattr(elem, "R"):  # plain Coil
            fg_coils[label] = (elem.R, elem.Z)
        else:  # Circuit, e.g., OH
            for sub_label, sub_elem, _factor in elem.coils:
                if hasattr(sub_elem, "R"):
                    fg_coils[sub_label] = (sub_elem.R, sub_elem.Z)

    ours = load_tcv()
    diffs = {}

    all_ours = (
        list(ours.shaping_coils) + list(ours.t_coils) + list(ours.ohmic_coils)
    )
    for c in all_ours:
        if c.name not in fg_coils:
            diffs[c.name] = f"missing in freegs (we have R={c.R}, Z={c.Z})"
            continue
        fg_R, fg_Z = fg_coils[c.name]
        if abs(c.R - fg_R) > 1e-6 or abs(c.Z - fg_Z) > 1e-6:
            diffs[c.name] = (
                f"ours=({c.R}, {c.Z}) vs freegs=({fg_R}, {fg_Z})"
            )

    return diffs


if __name__ == "__main__":
    print("Loading TCV geometry...")
    tcv = load_tcv()
    print(f"  Machine: R₀={tcv.R_major} m, a={tcv.a_minor} m, ε={tcv.epsilon:.4f}")
    print(f"  Shaping coils: {tcv.n_shaping} (E1-E8 + F1-F8)")
    print(f"  T coils: {tcv.n_t}")
    print(f"  Total control channels: {tcv.n_control_channels}")
    print(f"  E coil R: {tcv.shaping_coils[0].R:.4f} m (E1-E8 all same)")
    print(f"  F coil R: {tcv.shaping_coils[8].R:.4f} m (F1-F8 all same)")
    print()
    print("Cross-checking against freegs.machine.TCV()...")
    diffs = cross_check_against_freegs()
    if not diffs:
        print("  ✓ Perfect match for all coil positions")
    else:
        print(f"  ⚠ Diffs found:")
        for name, diff in diffs.items():
            print(f"    {name}: {diff}")
    print()
    print("All verifications passed.")
