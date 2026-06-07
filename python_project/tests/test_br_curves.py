"""Closed-form Brown-Resnick target curves (simulation.tex / methodology.tex)."""

import numpy as np
import pytest
from scipy.stats import norm

from br_simulation import br_chi_u, br_chi_u_from_theta, br_theta, br_variogram


def test_variogram_power_law():
    d = np.array([0.0, 60.0, 120.0, 240.0])
    g = br_variogram(d, rho=120.0, alpha=1.0)
    assert np.allclose(g, d / 120.0)
    # alpha = 2 -> squared
    assert np.isclose(br_variogram(120.0, 120.0, 2.0), 1.0)


def test_variogram_rejects_bad_params():
    with pytest.raises(ValueError):
        br_variogram(10.0, rho=0.0, alpha=1.0)
    with pytest.raises(ValueError):
        br_variogram(10.0, rho=100.0, alpha=2.5)


def test_theta_formula_matches_definition():
    d = np.array([0.0, 50.0, 120.0, 200.0, 300.0])
    rho, alpha = 120.0, 1.0
    expected = 2.0 * norm.cdf(np.sqrt(br_variogram(d, rho, alpha)) / 2.0)
    assert np.allclose(br_theta(d, rho, alpha), expected)


def test_theta_bounds_and_monotonicity():
    d = np.linspace(0.0, 500.0, 200)
    th = br_theta(d, rho=120.0, alpha=1.0)
    assert np.isclose(th[0], 1.0)                 # perfect dependence at d=0
    assert np.all(th >= 1.0 - 1e-12) and np.all(th <= 2.0 + 1e-12)
    assert np.all(np.diff(th) >= -1e-12)          # nondecreasing in distance


def test_chi_u_formula():
    d = np.array([50.0, 120.0, 200.0])
    rho, alpha, u = 120.0, 1.0, 0.99
    th = br_theta(d, rho, alpha)
    expected = (1.0 - 2.0 * u + u ** th) / (1.0 - u)
    assert np.allclose(br_chi_u(d, rho, alpha, u), expected)


def test_chi_u_special_thetas():
    # theta = 1 (perfect dependence) -> chi_u = 1 for any u
    assert np.isclose(br_chi_u_from_theta(1.0, 0.99), 1.0)
    # theta = 2 (independence) -> chi_u = 1 - u
    assert np.isclose(br_chi_u_from_theta(2.0, 0.99), 1.0 - 0.99)


def test_chi_u_approaches_limit_as_u_to_one():
    # chi_u -> 2 - theta as u -> 1
    th = 1.5
    vals = [br_chi_u_from_theta(th, u) for u in (0.9, 0.99, 0.999, 0.9999)]
    assert np.all(np.diff(vals) < 0)              # decreasing toward the limit
    assert np.isclose(vals[-1], 2.0 - th, atol=1e-3)


def test_chi_u_exceeds_limit_for_finite_u():
    d = np.array([50.0, 150.0, 300.0])
    th = br_theta(d, 120.0, 1.0)
    assert np.all(br_chi_u(d, 120.0, 1.0, 0.99) > (2.0 - th))


def test_chi_u_rejects_bad_threshold():
    with pytest.raises(ValueError):
        br_chi_u(50.0, 120.0, 1.0, u=1.0)
