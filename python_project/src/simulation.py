"""Synthetic data generator for the simulation-design notebook.

The simulation is a methodological-validation exercise: it generates
two seasonal dependence regimes on a synthetic station grid, runs the
same pipeline that the empirical analysis uses, and checks whether
the pipeline recovers the (stronger) winter dependence.

This module only sets up the simulation; it does not run it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimSpec:
    """Specification for one seasonal regime."""

    n_stations: int = 30
    domain_km: float = 300.0          # square domain side length
    n_blocks: int = 200               # number of block maxima per regime
    range_km: float = 100.0           # spatial dependence range
    seed: int = 0


def sample_station_locations(
    spec: SimSpec, *, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``n_stations`` 2D locations uniformly on the square."""
    rng = rng or np.random.default_rng(spec.seed)
    return rng.uniform(0.0, spec.domain_km, size=(spec.n_stations, 2))


def distance_matrix(locations: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance in km."""
    diff = locations[:, None, :] - locations[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def simulate_block_maxima(spec: SimSpec) -> np.ndarray:
    """Simulate block maxima from a max-stable-like field.

    TODO: implement. A simple option is a Smith-type max-stable
    process or a Gaussian copula with Frechet margins. The goal is
    not realism but to give the pipeline a known
    chi-vs-distance ground truth.
    """
    raise NotImplementedError("TODO: implement simulate_block_maxima")
