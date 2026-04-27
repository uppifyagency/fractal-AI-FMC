"""Cross-language bit-for-bit test: Python and JS must produce identical
relativize and virtual_reward output on shared fixtures.

Strategy:
1. Python generates fixture data (input vectors, partners, alpha/beta) and
   computes the expected outputs.
2. Fixture is written to tests/_fixture_relativize.json.
3. We invoke `node js/_test.js` which reads the fixture and verifies its own
   output matches the Python expectation to 1e-12.

Bit-for-bit at machine precision (1e-12 rel) is the contract; perfect
last-bit equality is not guaranteed across libm implementations, but in
practice for log/exp/pow the agreement is far below 1e-12.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fmc.core import relativize, virtual_reward


HERE = Path(__file__).parent
ROOT = HERE.parent
FIXTURE = HERE / "_fixture_relativize.json"


def _generate_fixture(rng: np.random.Generator):
    fixtures = []
    sizes = [5, 20, 50]
    for n in sizes:
        for _ in range(3):
            scale = rng.uniform(0.1, 100)
            shift = rng.uniform(-50, 50)
            inp = (rng.standard_normal(n) * scale + shift).tolist()
            expected = relativize(np.array(inp)).tolist()

            # Also build a virtual-reward fixture per input.
            states = rng.standard_normal((n, 3)).tolist()
            partners = [int(x) for x in (np.arange(n) + 1) % n]
            alpha = float(rng.uniform(0.0, 2.0))
            beta = float(rng.uniform(0.0, 2.0))
            vr = virtual_reward(
                np.array(inp),
                np.array(states),
                np.array(partners),
                alpha=alpha,
                beta=beta,
            ).tolist()

            fixtures.append({
                "input": inp,
                "expected": expected,
                "states": states,
                "partners": partners,
                "alpha": alpha,
                "beta": beta,
                "virtual_reward": vr,
            })
    return fixtures


def test_python_js_bit_equivalence():
    """Run the JS test runner and verify it reads the Python fixture without errors."""
    if shutil.which("node") is None:
        pytest.skip("node not available; skipping cross-language test")

    # Generate and dump fixture.
    rng = np.random.default_rng(42)
    fx = _generate_fixture(rng)
    FIXTURE.write_text(json.dumps(fx))

    try:
        result = subprocess.run(
            ["node", str(ROOT / "js" / "_test.js")],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        # Always clean up the fixture file so it doesn't pollute the repo.
        # Keep it on test failure for debugging.
        pass

    assert result.returncode == 0, (
        f"JS test runner failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    # Sanity: confirm the fixture-driven tests actually ran.
    assert "fixture" in result.stdout or "Python fixture" in result.stdout or "OK" in result.stdout, (
        f"Expected fixture-related output, got:\n{result.stdout}"
    )

    # Cleanup on success.
    FIXTURE.unlink(missing_ok=True)
