# fmc-core

Reference implementation of **Fractal Monte Carlo (FMC)** — the algorithm of Hernández-Cerezo & Duran-Ballester (2020), *Fractal AI: A Fragile Theory of Intelligence* (arXiv:1803.05049).

This package is the **math layer** described formally in [`../docs/MATH_CANON.md`](../docs/MATH_CANON.md): definitions 1–6, theorems 1–3. The Python implementation in `src/fmc/core.py` and the JS port in `js/fmc.js` are designed to produce **bit-for-bit identical** virtual reward vectors given the same input + seed.

## What this is

- A small, dependency-light core (~400 LOC NumPy) for `relativize`, `virtual_reward`, `effective_sample_size`, `clone_step`, `decide`.
- An environment protocol (`src/fmc/envs/base.py`) that lets you plug in any simulator with a `step` and `clone_state`.
- Identical math in Python and JS — useful for browser-based simulation and for the rocket validator.

## What this is not

- Not a research framework with GPU acceleration — for that use [`fragile`](https://github.com/FragileTech/fragile) or [`fragile-rl`](https://github.com/FragileTech/fragile-rl).
- Not a benchmarks harness — that lives in [`../work/0X_*/`](../work/) and (Level 2 of the canonical roadmap) in a future `bench/` directory.
- Not a planner with bells and whistles (no parallelization, no GPU, no checkpointing) — that's intentional. This is the reference.

## Install

```bash
pip install -e ".[test]"
pytest
```

## Usage

```python
from fmc.core import relativize, virtual_reward, clone_step, decide
from fmc.envs.base import Environment

# 1. Bring your environment (must implement Environment protocol)
env: Environment = MyEnv(...)

# 2. Plan
from fmc.core import plan
action = plan(env, x0=env.reset(), N=64, M=30, alpha=1.0, beta=1.0, seed=42)
```

See `tests/test_envs.py` for examples.

## Cross-references

- Math definitions and theorems: [`../docs/MATH_CANON.md`](../docs/MATH_CANON.md)
- NumPy reference: [`../repos/FractalAI_old/fractalai/swarm.py`](../repos/FractalAI_old/fractalai/swarm.py)
- PyTorch reference: [`../repos/fragile/src/fragile/fractalai.py`](../repos/fragile/src/fragile/fractalai.py)
- JS port for browser: `js/fmc.js`

## Versioning

`0.1.0` — initial scaffolding. No API stability promises.
