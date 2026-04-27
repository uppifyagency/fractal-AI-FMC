"""Benchmark runner core.

Produces a uniform JSON record per benchmark cell:

    {
      "benchmark": "rocket_alpha_beta_sweep",
      "env": "rocket",
      "params": {"alpha": ..., "beta": ..., "N": ..., "M": ...},
      "metric": "b_eff" | "reward" | "...",
      "n_seeds": int,
      "values": [...],          # one per seed
      "mean": float,
      "ci95_low": float,
      "ci95_high": float,
      "sd": float,
      "hardware": {...},
      "fmc_core_version": "...",
      "timestamp_utc": "..."
    }

The CI95 is bootstrap-based to avoid normality assumptions on metrics like
b_eff which are bounded and skewed.
"""

from __future__ import annotations

import json
import platform
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

import fmc as _fmc


@dataclass
class BenchResult:
    benchmark: str
    env: str
    params: Dict[str, Any]
    metric: str
    n_seeds: int
    values: List[float]
    mean: float
    ci95_low: float
    ci95_high: float
    sd: float
    hardware: Dict[str, Any]
    fmc_core_version: str
    timestamp_utc: str
    duration_seconds: float
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
    }


def bootstrap_ci95(values: np.ndarray, n_resample: int = 5000, rng_seed: int = 0) -> tuple:
    """Percentile bootstrap 95% CI on the mean."""
    if len(values) < 2:
        v = float(values[0]) if len(values) == 1 else 0.0
        return v, v
    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_resample)
    n = len(values)
    for i in range(n_resample):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_cell(
    benchmark: str,
    env_name: str,
    params: Dict[str, Any],
    metric: str,
    sample_fn,
    seeds: List[int],
    notes: str = "",
) -> BenchResult:
    """Run a single benchmark cell across multiple seeds.

    Parameters
    ----------
    sample_fn : Callable[[int], float]
        Given a seed, runs the experiment and returns the metric value.
    seeds : List[int]
        Seeds to run.
    """
    t0 = time.time()
    values = np.array([float(sample_fn(s)) for s in seeds])
    duration = time.time() - t0

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci95(values)

    return BenchResult(
        benchmark=benchmark,
        env=env_name,
        params=params,
        metric=metric,
        n_seeds=len(seeds),
        values=values.tolist(),
        mean=mean,
        ci95_low=lo,
        ci95_high=hi,
        sd=sd,
        hardware=_hardware_info(),
        fmc_core_version=_fmc.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=duration,
        notes=notes,
    )


def write_jsonl(results: List[BenchResult], path: str) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
