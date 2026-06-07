"""Validate that the Brown-Resnick simulators reproduce the closed-form curves.

This is the central correctness check: an exact draw must have exact
unit-Frechet margins and a pairwise extremal-dependence structure matching
theta_BR(d) and chi_u_BR(d). Tolerances are set for the Monte Carlo sample
size used here.
"""

import numpy as np
import pytest

from br_simulation import (
    br_chi_u,
    br_theta,
    simulate_brown_resnick_approx,
    simulate_brown_resnick_exact,
)
from dependence_estimators import (
    chi_u_pairwise,
    empirical_uniform_ranks,
    fmadogram_pairwise,
    theta_from_fmadogram,
)

# A small line of stations with a spread of distances.
COORDS = np.array([0.0, 40.0, 80.0, 140.0, 200.0, 280.0])
DIST = np.abs(COORDS[:, None] - COORDS[None, :])
RHO, ALPHA = 120.0, 1.0
N = 50_000
SEED = 12345
# chi_u variance grows as u -> 1 (thinner tail sample), so the tolerance does too.
CHI_TOL = {0.95: 0.03, 0.99: 0.045}


@pytest.fixture(scope="module")
def exact_sample():
    return simulate_brown_resnick_exact(DIST, RHO, ALPHA, n_obs=N, rng=SEED)


def test_exact_unit_frechet_margins(exact_sample):
    Z = exact_sample
    # P(Z <= z) = exp(-1/z) at every station; check three quantile levels.
    for z, target in ((1.0, np.exp(-1.0)), (2.0, np.exp(-0.5)), (5.0, np.exp(-0.2))):
        emp = (Z <= z).mean(axis=0)
        assert np.all(np.abs(emp - target) < 0.012), (z, emp)


def test_exact_recovers_theta(exact_sample):
    U = empirical_uniform_ranks(exact_sample)
    theta = theta_from_fmadogram(fmadogram_pairwise(U))
    iu = np.triu_indices(len(COORDS), 1)
    err = np.abs(theta[iu] - br_theta(DIST[iu], RHO, ALPHA))
    assert err.max() < 0.02, err.max()


def test_exact_recovers_chi_u(exact_sample):
    U = empirical_uniform_ranks(exact_sample)
    iu = np.triu_indices(len(COORDS), 1)
    for u in (0.95, 0.99):
        chi = chi_u_pairwise(U, u)[iu]
        err = np.abs(chi - br_chi_u(DIST[iu], RHO, ALPHA, u))
        assert err.max() < CHI_TOL[u], (u, err.max())


def test_approx_preserves_theta():
    # The approximate simulator preserves pairwise theta (looser margins).
    Z = simulate_brown_resnick_approx(DIST, RHO, ALPHA, n_obs=N, n_factors=60, rng=SEED)
    U = empirical_uniform_ranks(Z)
    theta = theta_from_fmadogram(fmadogram_pairwise(U))
    iu = np.triu_indices(len(COORDS), 1)
    err = np.abs(theta[iu] - br_theta(DIST[iu], RHO, ALPHA))
    assert err.max() < 0.03, err.max()


def test_exact_alternative_ordering():
    # Winter (rho=180) must be MORE dependent than summer (rho=60): theta_W < theta_S.
    iu = np.triu_indices(len(COORDS), 1)
    Uw = empirical_uniform_ranks(simulate_brown_resnick_exact(DIST, 180.0, ALPHA, N, rng=1))
    Us = empirical_uniform_ranks(simulate_brown_resnick_exact(DIST, 60.0, ALPHA, N, rng=2))
    tw = theta_from_fmadogram(fmadogram_pairwise(Uw))[iu]
    ts = theta_from_fmadogram(fmadogram_pairwise(Us))[iu]
    assert np.all(tw < ts)
