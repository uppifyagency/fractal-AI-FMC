"""Tests for the benchmark runner skeleton."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bench.runner import bootstrap_ci95, run_cell, write_jsonl


def test_bootstrap_ci95_uniform():
    """Constant values produce CI = (mean, mean)."""
    v = np.array([5.0] * 10)
    lo, hi = bootstrap_ci95(v)
    assert lo == pytest.approx(5.0)
    assert hi == pytest.approx(5.0)


def test_bootstrap_ci95_brackets_mean():
    """CI95 should bracket the empirical mean."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(50) * 2.0 + 7.0
    lo, hi = bootstrap_ci95(v, n_resample=2000, rng_seed=42)
    assert lo < v.mean() < hi


def test_run_cell_produces_valid_record(tmp_path):
    """Running a cell should produce a complete BenchResult with all fields."""
    def sample(seed):
        return float(seed) * 0.5 + 1.0

    result = run_cell(
        benchmark="dummy",
        env_name="dummy_env",
        params={"x": 1},
        metric="value",
        sample_fn=sample,
        seeds=[0, 1, 2, 3, 4],
        notes="test cell",
    )
    assert result.benchmark == "dummy"
    assert result.n_seeds == 5
    assert len(result.values) == 5
    # values: seed*0.5 + 1.0 for seeds 0..4 -> [1.0, 1.5, 2.0, 2.5, 3.0], mean 2.0
    assert result.mean == pytest.approx(2.0)
    assert result.ci95_low <= result.mean <= result.ci95_high
    assert "platform" in result.hardware
    assert result.fmc_core_version == "0.1.0"


def test_write_jsonl_roundtrip(tmp_path):
    """write_jsonl produces parseable lines."""
    def sample(seed):
        return float(seed)

    r1 = run_cell(
        benchmark="x", env_name="e", params={"a": 1},
        metric="m", sample_fn=sample, seeds=[1, 2, 3],
    )
    r2 = run_cell(
        benchmark="y", env_name="e", params={"a": 2},
        metric="m", sample_fn=sample, seeds=[1, 2, 3],
    )

    out = tmp_path / "out.jsonl"
    write_jsonl([r1, r2], str(out))

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["benchmark"] == "x"
    assert parsed[1]["benchmark"] == "y"
