"""Reproducibility: a fixed seed must give identical results."""

import numpy as np

from br_simulation import (
    make_default_scenarios,
    simulate_brown_resnick_exact,
    simulate_brown_resnick_approx,
)
from spatial_utils import StationPanel
from simulation_pipeline import make_context, run_replication


def _toy_panel(n=6, seed=0):
    rng = np.random.default_rng(seed)
    lat = 52.0 + rng.uniform(-0.6, 0.6, n)
    lon = 5.0 + rng.uniform(-0.9, 0.9, n)
    return StationPanel(ids=tuple(f"S{i}" for i in range(n)),
                        names=tuple(f"st{i}" for i in range(n)), lat=lat, lon=lon)


def test_exact_simulation_is_seed_reproducible():
    D = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) * 30.0
    a = simulate_brown_resnick_exact(D, 120.0, 1.0, n_obs=200, rng=7)
    b = simulate_brown_resnick_exact(D, 120.0, 1.0, n_obs=200, rng=7)
    assert np.array_equal(a, b)
    c = simulate_brown_resnick_exact(D, 120.0, 1.0, n_obs=200, rng=8)
    assert not np.array_equal(a, c)


def test_approx_simulation_is_seed_reproducible():
    D = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) * 30.0
    a = simulate_brown_resnick_approx(D, 120.0, 1.0, n_obs=200, n_factors=30, rng=3)
    b = simulate_brown_resnick_approx(D, 120.0, 1.0, n_obs=200, n_factors=30, rng=3)
    assert np.array_equal(a, b)


def test_replication_is_seed_reproducible():
    panel = _toy_panel()
    scen = make_default_scenarios()["alternative"]
    ctx = make_context(panel, n_obs=300, n_years=10, n_boot=15, n_grid=12)
    r1 = run_replication(2024, ctx, scen)
    r2 = run_replication(2024, ctx, scen)
    # equal_nan: curves may carry NaN at grid points outside the kernel
    # support on this sparse toy panel; identical NaN positions are fine.
    assert np.allclose(r1["curve_w"], r2["curve_w"], equal_nan=True)
    assert np.allclose(r1["curve_s"], r2["curve_s"], equal_nan=True)
    assert np.allclose(r1["band_d"][0], r2["band_d"][0], equal_nan=True)
    assert np.allclose(r1["bw"], r2["bw"], equal_nan=True)


def test_different_seed_changes_replication():
    panel = _toy_panel()
    scen = make_default_scenarios()["alternative"]
    ctx = make_context(panel, n_obs=300, n_years=10, n_boot=0, n_grid=12)
    r1 = run_replication(1, ctx, scen)
    r2 = run_replication(2, ctx, scen)
    assert not np.allclose(r1["curve_w"], r2["curve_w"])
