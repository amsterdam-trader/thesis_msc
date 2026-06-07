"""Pairwise extremal-dependence estimators (methodology sec:meth-chi/madogram)."""

import numpy as np
import pytest

from dependence_estimators import (
    chi_u_pairwise,
    empirical_uniform_ranks,
    fmadogram_pairwise,
    theta_from_fmadogram,
)


def test_rank_transform_strictly_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    U = empirical_uniform_ranks(X)
    assert U.shape == X.shape
    assert np.all(U > 0.0) and np.all(U < 1.0)


def test_rank_transform_values():
    # column with distinct values -> ranks 1..n over (n+1)
    X = np.array([[3.0], [1.0], [2.0]])
    U = empirical_uniform_ranks(X)
    assert np.allclose(np.sort(U[:, 0]), np.array([1, 2, 3]) / 4.0)


def test_rank_transform_handles_ties_by_average():
    X = np.array([[1.0], [1.0], [2.0], [3.0]])
    U = empirical_uniform_ranks(X)
    # the two tied 1.0's get average rank (1+2)/2 = 1.5 -> 1.5/5
    assert np.isclose(U[0, 0], 1.5 / 5.0) and np.isclose(U[1, 0], 1.5 / 5.0)


def test_fmadogram_symmetry_and_diagonal():
    rng = np.random.default_rng(1)
    U = empirical_uniform_ranks(rng.standard_normal((500, 4)))
    nu = fmadogram_pairwise(U)
    assert np.allclose(nu, nu.T)
    assert np.allclose(np.diag(nu), 0.0)
    assert np.all(nu >= 0.0)


def test_fmadogram_independent_limit():
    # independent uniforms: nu_F -> 1/6, theta -> 2
    rng = np.random.default_rng(2)
    U = rng.random((200_000, 2))
    nu = fmadogram_pairwise(U)[0, 1]
    assert abs(nu - 1.0 / 6.0) < 0.01
    theta = theta_from_fmadogram(np.array([[0.0, nu], [nu, 0.0]]))[0, 1]
    assert abs(theta - 2.0) < 0.05


def test_fmadogram_perfect_dependence_limit():
    # identical columns: nu_F = 0 -> theta = 1
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1000, 1))
    U = empirical_uniform_ranks(np.hstack([x, x]))
    nu = fmadogram_pairwise(U)
    assert np.isclose(nu[0, 1], 0.0, atol=1e-12)
    assert np.isclose(theta_from_fmadogram(nu)[0, 1], 1.0)


def test_theta_in_unit_to_two_range():
    rng = np.random.default_rng(4)
    U = empirical_uniform_ranks(rng.standard_normal((300, 6)))
    theta = theta_from_fmadogram(fmadogram_pairwise(U))
    off = theta[~np.eye(6, dtype=bool)]
    assert np.all(off >= 1.0) and np.all(off <= 2.0)


def test_chi_u_symmetric_and_bounded():
    rng = np.random.default_rng(5)
    U = empirical_uniform_ranks(rng.standard_normal((2000, 4)))
    chi = chi_u_pairwise(U, u=0.95)
    assert np.allclose(chi, chi.T)
    off = chi[~np.eye(4, dtype=bool)]
    assert np.all(off >= -1e-12) and np.all(off <= 1.0 + 1e-12)


def test_chi_u_perfect_dependence_is_one():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2000, 1))
    U = empirical_uniform_ranks(np.hstack([x, x]))
    assert np.isclose(chi_u_pairwise(U, 0.95)[0, 1], 1.0)


def test_chi_u_independence_near_one_minus_u():
    rng = np.random.default_rng(7)
    U = rng.random((400_000, 2))
    u = 0.95
    chi = chi_u_pairwise(U, u)[0, 1]
    assert abs(chi - (1.0 - u)) < 0.02


def test_chi_u_rejects_bad_threshold():
    U = np.random.default_rng(0).random((10, 2))
    with pytest.raises(ValueError):
        chi_u_pairwise(U, u=1.5)
