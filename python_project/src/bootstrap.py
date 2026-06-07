"""Seasonal-block (pseudo-season-year) bootstrap (methodology sec:meth-bootstrap).

The resampling unit is one season-year. In the empirical analysis whole
winter-years and summer-years are resampled with replacement
*independently within each season*; in the simulation each regime's ``n``
independent daily-max fields are arranged into a fixed number of
pseudo-season-years (default 35) and whole pseudo-years are resampled.
Because the simulated fields are independent across "days", this block
resample reduces to a near-i.i.d. resample, which -- as noted in
simulation.tex (sec:sim-estimation) -- is the favourable case for the
bootstrap's coverage.

This module provides the resampling primitives. The estimator + smoother
recomputation on each resample lives in ``simulation_pipeline`` so that
the exact empirical pipeline is reused unchanged.
"""

from __future__ import annotations

import numpy as np


def make_pseudo_year_blocks(n: int, n_years: int) -> list[np.ndarray]:
    """Partition field indices 0..n-1 into ``n_years`` contiguous blocks.

    Blocks are as equal in size as possible (sizes differ by at most one),
    mirroring the ~90-day season-years of the data. With n = 3150 and
    n_years = 35 every block has exactly 90 indices.
    """
    if n_years < 1:
        raise ValueError(f"n_years must be >= 1, got {n_years}")
    if n_years > n:
        raise ValueError(f"n_years ({n_years}) cannot exceed n ({n}).")
    edges = np.linspace(0, n, n_years + 1).astype(int)
    return [np.arange(edges[k], edges[k + 1]) for k in range(n_years)]


def resample_block_indices(
    blocks: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Resample whole pseudo-years with replacement and concatenate indices.

    Returns a 1-D index array (length ~= n) selecting fields for one
    bootstrap replication of a single regime.
    """
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[c] for c in chosen])


def percentile_band(
    samples: np.ndarray, alpha: float = 0.05, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided pointwise percentile band at confidence 1 - alpha.

    Returns (lower, upper) arrays; NaNs in ``samples`` are ignored.
    """
    lo = np.nanpercentile(samples, 100.0 * (alpha / 2.0), axis=axis)
    hi = np.nanpercentile(samples, 100.0 * (1.0 - alpha / 2.0), axis=axis)
    return lo, hi
