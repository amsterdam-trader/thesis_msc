"""Brown-Resnick benchmark simulation and closed-form target curves.

Generates Brown-Resnick max-stable fields with a known power-law
variogram on the empirical KNMI station geometry. The simulation is the
controlled data-generating process of the thesis Simulation Study
(simulation.tex): it is used only to check whether the F-madogram,
theta, and finite-level chi_u estimators recover a known
dependence-distance structure in finite samples, NOT as an empirical
model for Dutch wind gusts.

Variogram and closed-form target curves
---------------------------------------
The Brown-Resnick process is governed by the power-law variogram

    gamma(d) = (d / rho) ** alpha,                 d in km, 0 < alpha <= 2,

interpreted as the full increment variance Var(W(s+h) - W(s)) of the
underlying intrinsically stationary Gaussian process W. The three
closed-form curves the estimators are measured against (and which match
methodology.tex / simulation.tex exactly) are:

    theta_BR(d)  = 2 * Phi( sqrt(gamma(d)) / 2 )                in [1, 2]
    chi_u_BR(d)  = ( 1 - 2u + u ** theta_BR(d) ) / ( 1 - u )    finite level u
    chi_lim(d)   = 2 - theta_BR(d)                              limit u -> 1

where Phi is the standard-normal CDF (Kabluchko, Schlather & de Haan
2009). theta_BR is monotone in d: 1 at d = 0 (perfect dependence),
approaching 2 as d grows (asymptotic independence). chi_u_BR exceeds the
limit 2 - theta_BR for u < 1 and decreases to it as u -> 1, so the
finite-level chi_u_BR(d) -- not the limit -- is the correct target for
the finite-level estimator (simulation.tex, sec:sim-br).

Simulation methods
------------------
Two simulators are provided via ``simulate_brown_resnick_field(method=...)``:

* ``method="exact"`` (DEFAULT) -- exact simulation by the extremal-functions
  algorithm of Dombry, Engelke & Oesting (Biometrika, 2016, Algorithm 2).
  Each location s_k is visited in turn; the spectral functions "anchored"
  at s_k (a log-Gaussian process with value 1 at s_k) are drawn with their
  value at s_k following a Poisson process of intensity r^{-2} dr, and a
  drawn function is added to the running pointwise maximum only if it is
  not already dominated at any earlier-visited location. This yields the
  EXACT finite-dimensional law of the Brown-Resnick process at the
  station set, with exact unit-Frechet margins. The implementation is
  vectorised across the n_obs independent fields. This matches the
  "exact simulation" claim in methodology.tex / simulation.tex.

* ``method="approx"`` -- the older Schlather-type truncated spectral
  representation, kept for speed/comparison and fully documented below.
  Z(s) = max_{k=1..n_factors} xi_k * exp(W_k(s) - gamma(s, s_0)/2) with
  xi_k = 1 / cumsum(Exp(1)) truncated at n_factors and W_k anchored at the
  first station. The pairwise variance Var(W(s_i) - W(s_j)) = gamma(s_i, s_j)
  is preserved exactly, so the pairwise theta_BR(d) is preserved, but the
  truncation makes the margins only approximately unit Frechet. This bias
  is absorbed by the subsequent empirical rank transform in the
  estimators; do NOT treat this method as an exact simulator.

Both methods are validated against the closed-form curves in
``tests/test_br_simulation_recovers_curves.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from spatial_utils import StationPanel, pairwise_distance_matrix


# ---------------------------------------------------------------------------
# Theoretical curves
# ---------------------------------------------------------------------------

def br_variogram(
    h: np.ndarray | float, rho: np.ndarray | float, alpha: float
) -> np.ndarray | float:
    """Brown-Resnick variogram gamma(h) = (h / rho)**alpha. Broadcasts."""
    rho_arr = np.asarray(rho, dtype=float)
    if np.any(rho_arr <= 0):
        raise ValueError(f"rho must be positive, got {rho}")
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    return np.power(np.asarray(h, dtype=float) / rho_arr, alpha)


def br_theta(h: np.ndarray | float, rho: float, alpha: float) -> np.ndarray | float:
    """Theoretical pairwise extremal coefficient theta(h) = 2*Phi(sqrt(gamma)/2)."""
    g = br_variogram(h, rho, alpha)
    return 2.0 * norm.cdf(np.sqrt(g) / 2.0)


def br_chi_from_theta(theta: np.ndarray | float) -> np.ndarray | float:
    """Limiting chi from theta via chi = 2 - theta (level u -> 1)."""
    return 2.0 - theta


def br_chi_u_from_theta(
    theta: np.ndarray | float, u: float
) -> np.ndarray | float:
    """Finite-level tail dependence chi_u for a bivariate max-stable pair.

    chi_u = (1 - 2u + u**theta) / (1 - u). This is the correct target for
    the finite-level estimator at level u; it exceeds the limit 2 - theta
    for u < 1 and decreases to it as u -> 1.
    """
    if not 0.0 < u < 1.0:
        raise ValueError(f"Threshold u must be in (0, 1), got {u}")
    theta = np.asarray(theta, dtype=float)
    return (1.0 - 2.0 * u + np.power(u, theta)) / (1.0 - u)


def br_chi_u(
    h: np.ndarray | float, rho: float, alpha: float, u: float
) -> np.ndarray | float:
    """Closed-form finite-level chi_u_BR(h) for the Brown-Resnick process.

    chi_u_BR(h) = (1 - 2u + u**theta_BR(h)) / (1 - u), with
    theta_BR(h) = 2 Phi(sqrt(gamma(h))/2) and gamma(h) = (h/rho)**alpha.
    """
    return br_chi_u_from_theta(br_theta(h, rho, alpha), u)


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BRScenario:
    """Parameters for one regime (winter or summer) under a hypothesis."""
    name: str
    rho: float
    alpha: float

    def theta(self, h):
        return br_theta(h, self.rho, self.alpha)

    def variogram(self, h):
        return br_variogram(h, self.rho, self.alpha)


@dataclass(frozen=True)
class ScenarioPair:
    """A null- or alternative-hypothesis pair of winter and summer regimes."""
    label: Literal["null", "alternative"]
    winter: BRScenario
    summer: BRScenario


def make_default_scenarios(
    alpha: float = 1.0,
    rho_null: float = 120.0,
    rho_winter_alt: float = 180.0,
    rho_summer_alt: float = 60.0,
) -> dict[str, ScenarioPair]:
    """Build the null and alternative regime pairs used in the thesis."""
    null = ScenarioPair(
        label="null",
        winter=BRScenario("winter_null", rho=rho_null, alpha=alpha),
        summer=BRScenario("summer_null", rho=rho_null, alpha=alpha),
    )
    alt = ScenarioPair(
        label="alternative",
        winter=BRScenario("winter_alt", rho=rho_winter_alt, alpha=alpha),
        summer=BRScenario("summer_alt", rho=rho_summer_alt, alpha=alpha),
    )
    return {"null": null, "alternative": alt}


# ---------------------------------------------------------------------------
# Approximate Brown-Resnick simulation
# ---------------------------------------------------------------------------

def _psd_square_root(M: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Return L with L @ L.T approximately equal to M (PSD via eigh)."""
    n = M.shape[0]
    w, V = np.linalg.eigh((M + M.T) / 2.0 + jitter * np.eye(n))
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)[None, :]


def _anchored_gaussian_covariance(D: np.ndarray, rho: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Anchored Gaussian-process covariance Sigma and drift -g0/2.

    Anchor is the first station (index 0). Returns (Sigma, drift) where
    drift = -gamma(s_i, s_0) / 2.
    """
    G = np.power(D / rho, alpha)
    g0 = G[:, 0]
    Sigma = 0.5 * (g0[:, None] + g0[None, :] - G)
    drift = -0.5 * g0
    return Sigma, drift


def _resolve_distance_matrix(panel_or_distances: StationPanel | np.ndarray) -> np.ndarray:
    """Return a validated (n, n) distance matrix from a panel or an array."""
    if isinstance(panel_or_distances, StationPanel):
        D = pairwise_distance_matrix(panel_or_distances)
    else:
        D = np.asarray(panel_or_distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square 2D, got {D.shape}")
    return D


def simulate_brown_resnick_approx(
    panel_or_distances: StationPanel | np.ndarray,
    rho: float,
    alpha: float,
    n_obs: int,
    n_factors: int = 50,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Approximate Brown-Resnick field (truncated Schlather representation).

    Z(s) = max_{k=1..n_factors} xi_k exp(W_k(s) - gamma(s, s_0)/2), with
    xi_k = 1 / cumsum(Exp(1)) and W_k anchored at the first station. The
    pairwise variance Var(W(s_i) - W(s_j)) = gamma(s_i, s_j) is preserved
    exactly so theta_BR(d) is preserved; the truncation leaves the margins
    only approximately unit Frechet (absorbed by the rank transform). Use
    ``simulate_brown_resnick_exact`` for an exact draw.

    Parameters
    ----------
    panel_or_distances
        A StationPanel or an explicit (n, n) distance matrix in km.
    rho, alpha
        Variogram parameters: gamma(d) = (d / rho) ** alpha.
    n_obs
        Number of independent realisations (daily-max fields).
    n_factors
        Truncation of the Poisson-point spectral representation.
    rng
        Random generator or seed.

    Returns
    -------
    Z : (n_obs, n_stations) array with approximately unit-Frechet margins.
    """
    D = _resolve_distance_matrix(panel_or_distances)
    n = D.shape[0]

    rng = np.random.default_rng(rng)
    Sigma, drift = _anchored_gaussian_covariance(D, rho=rho, alpha=alpha)
    L = _psd_square_root(Sigma)

    # Poisson points xi_k = 1 / cumsum(Exp(1)), per realisation.
    E_cumsum = rng.exponential(size=(n_obs, n_factors)).cumsum(axis=1)
    xi = 1.0 / E_cumsum  # (n_obs, n_factors), decreasing in k

    Z = np.full((n_obs, n), -np.inf, dtype=float)
    drift_row = drift[None, :]
    for k in range(n_factors):
        z = rng.standard_normal((n_obs, n))
        W = z @ L.T                                # (n_obs, n)
        contribution = xi[:, k:k + 1] * np.exp(W + drift_row)
        np.maximum(Z, contribution, out=Z)
    return Z


def _extremal_function_factors(
    D: np.ndarray, rho: float, alpha: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Precompute, for each anchor location k, the spectral-function law.

    The spectral function "anchored at s_k" is the size-biased (Esscher
    tilted) log-Gaussian spectral function: Y(s_k) = 1 and log Y = V is
    Gaussian with
        mean(V_l)     = -gamma(s_l, s_k) / 2
        Cov(V_l, V_m) = 0.5 * ( gamma(s_l, s_k) + gamma(s_m, s_k)
                                - gamma(s_l, s_m) ),
    equivalently V_l = W(s_l) - W(s_k) - gamma(s_l, s_k)/2 for the
    intrinsically stationary Gaussian W with variogram gamma (Dombry,
    Engelke & Oesting 2016). Then Var(V_l) = gamma(s_l, s_k) so
    E[exp(V_l)] = 1 (exact unit-Frechet margins), and row/column k of the
    covariance and the mean vanish, so V_k = 0 exactly and Y(s_k) = 1.

    Returns
    -------
    factors : list of (n, n) lower-rank square roots L_k with L_k L_k^T = Sigma_k.
    means   : (n, n) array whose row k is the mean vector -gamma(., s_k)/2.
    """
    G = np.power(D / rho, alpha)             # full variogram matrix gamma(d_lm)
    n = G.shape[0]
    factors: list[np.ndarray] = []
    means = np.empty((n, n), dtype=float)
    for k in range(n):
        gk = G[k, :]
        Sigma = 0.5 * (gk[:, None] + gk[None, :] - G)
        factors.append(_psd_square_root(Sigma))
        means[k] = -0.5 * gk
    return factors, means


def simulate_brown_resnick_exact(
    panel_or_distances: StationPanel | np.ndarray,
    rho: float,
    alpha: float,
    n_obs: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Exact Brown-Resnick field via the extremal-functions algorithm.

    Implements Algorithm 2 of Dombry, Engelke & Oesting (Biometrika 2016).
    Each station s_k is visited once; spectral functions anchored at s_k
    are generated with their value at s_k following a Poisson process of
    intensity r^{-2} dr (decreasing), and a drawn function is merged into
    the running pointwise maximum only if it is not already dominated at
    any earlier-visited location s_l (l < k). The result is an EXACT draw
    of the Brown-Resnick max-stable process at the station set, with exact
    unit-Frechet margins. The loop over Poisson points is vectorised
    across the ``n_obs`` independent fields.

    Parameters
    ----------
    panel_or_distances
        A StationPanel or an explicit (n, n) distance matrix in km.
    rho, alpha
        Variogram parameters: gamma(d) = (d / rho) ** alpha.
    n_obs
        Number of independent realisations (daily-max fields).
    rng
        Random generator or seed.

    Returns
    -------
    Z : (n_obs, n_stations) array of exact Brown-Resnick values
        (unit-Frechet margins).
    """
    D = _resolve_distance_matrix(panel_or_distances)
    n = D.shape[0]
    rng = np.random.default_rng(rng)

    factors, means = _extremal_function_factors(D, rho=rho, alpha=alpha)
    Z = np.zeros((n_obs, n), dtype=float)

    for k in range(n):
        L = factors[k]
        m = means[k][None, :]
        # Poisson arrival times on (0, inf): T accumulates Exp(1); zeta = 1/T.
        T = rng.exponential(size=n_obs)
        while True:
            zeta = 1.0 / T
            cont = zeta > Z[:, k]                  # fields still generating at s_k
            idx = np.nonzero(cont)[0]
            if idx.size == 0:
                break
            nc = idx.size
            zz = rng.standard_normal((nc, n))
            V = zz @ L.T + m
            V[:, k] = 0.0                          # enforce Y(s_k) = 1 exactly
            cand = zeta[idx, None] * np.exp(V)     # candidate function values
            if k > 0:
                accept = np.all(cand[:, :k] < Z[idx][:, :k], axis=1)
            else:
                accept = np.ones(nc, dtype=bool)
            if accept.any():
                aidx = idx[accept]
                Z[aidx] = np.maximum(Z[aidx], cand[accept])
            # Advance the Poisson clock only for fields still generating.
            T[idx] += rng.exponential(size=nc)
    return Z


def simulate_brown_resnick_field(
    panel_or_distances: StationPanel | np.ndarray,
    rho: float,
    alpha: float,
    n_obs: int,
    method: Literal["exact", "approx"] = "exact",
    n_factors: int = 50,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate a Brown-Resnick max-stable field (dispatcher).

    method="exact" (default) uses the exact extremal-functions algorithm
    (Dombry, Engelke & Oesting 2016); method="approx" uses the truncated
    spectral representation (``n_factors`` Poisson points). See module
    docstring for the trade-offs.
    """
    if method == "exact":
        return simulate_brown_resnick_exact(
            panel_or_distances, rho=rho, alpha=alpha, n_obs=n_obs, rng=rng
        )
    if method == "approx":
        return simulate_brown_resnick_approx(
            panel_or_distances, rho=rho, alpha=alpha, n_obs=n_obs,
            n_factors=n_factors, rng=rng,
        )
    raise ValueError(f"Unknown method {method!r}; use 'exact' or 'approx'.")


def simulate_scenario_pair(
    panel: StationPanel,
    scenario: ScenarioPair,
    n_obs_per_season: int,
    method: Literal["exact", "approx"] = "exact",
    n_factors: int = 50,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one winter and one summer field under a scenario pair.

    The same RNG is threaded through both regimes so a single seed fixes
    the whole pair reproducibly. Distances are computed once from the panel.
    """
    rng = np.random.default_rng(rng)
    D = pairwise_distance_matrix(panel)
    winter = simulate_brown_resnick_field(
        D, rho=scenario.winter.rho, alpha=scenario.winter.alpha,
        n_obs=n_obs_per_season, method=method, n_factors=n_factors, rng=rng,
    )
    summer = simulate_brown_resnick_field(
        D, rho=scenario.summer.rho, alpha=scenario.summer.alpha,
        n_obs=n_obs_per_season, method=method, n_factors=n_factors, rng=rng,
    )
    return {"winter": winter, "summer": summer}