"""Brown-Resnick benchmark simulation.

Generates approximate Brown-Resnick max-stable fields with a known
power-law variogram on the empirical KNMI station geometry. The
simulation is used to validate that the F-madogram, theta, and chi_u
estimators recover the true dependence structure in finite samples.

Variogram and theoretical extremal coefficient
----------------------------------------------
We use the Brown-Resnick variogram

    gamma(h) = (h / rho) ** alpha,                 h in km, 0 < alpha <= 2

and the corresponding pairwise extremal coefficient

    theta(h) = 2 * Phi( sqrt(gamma(h)) / 2 ),

where Phi is the standard-normal CDF (Kabluchko, Schlather, de Haan
2009). theta is monotone in h: 1 at h = 0 (perfect dependence) and
approaches 2 as h grows (asymptotic independence).

Simulation
----------
The exact Brown-Resnick max-stable process has the spectral
representation

    Z(s) = max_{k >= 1}  xi_k * exp( W_k(s) - sigma_W^2(s)/2 ),

where {xi_k} are points of a Poisson process with intensity xi^{-2} d xi
and {W_k} are iid centred Gaussian processes with stationary increments
satisfying Var(W_k(s) - W_k(t)) = gamma(s - t). We implement an
APPROXIMATE version with two simplifications:

(a) Truncated Poisson points: xi_k = 1 / E_k with E_k the partial sum
    of k iid Exp(1) random variables, truncated at k = n_factors. This
    truncates the upper tail of the Poisson representation and the
    resulting margins are only approximately unit Frechet. Increasing
    n_factors reduces the bias.
(b) Anchored Gaussian processes: W_k is simulated by anchoring at the
    first station, W_k(s_0) = 0, with covariance
        Cov(W(s_i), W(s_j)) = 0.5 * ( gamma(s_i, s_0)
                                      + gamma(s_j, s_0)
                                      - gamma(s_i, s_j) ).
    The pairwise variance Var(W(s_i) - W(s_j)) = gamma(s_i, s_j) is
    preserved exactly, so the THEORETICAL pairwise extremal coefficient
    theta(h) is preserved. Marginal variances differ slightly across
    stations, but this is absorbed by the subsequent empirical rank
    transform used in the estimators.

The approximation is sufficient for the methodological question of the
thesis: whether finite-sample estimators can recover known pairwise
dependence-distance curves. It should NOT be used as an exact
Brown-Resnick simulator.
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
    """Asymptotic chi from theta via chi = 2 - theta (max-stable case)."""
    return 2.0 - theta


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


def simulate_brown_resnick_field(
    panel_or_distances: StationPanel | np.ndarray,
    rho: float,
    alpha: float,
    n_obs: int,
    n_factors: int = 50,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate an approximate Brown-Resnick max-stable field.

    Parameters
    ----------
    panel_or_distances
        Either a StationPanel (from which the haversine distance matrix
        is computed) or an explicit (n, n) distance matrix in km.
    rho, alpha
        Variogram parameters: gamma(h) = (h / rho) ** alpha.
    n_obs
        Number of independent realisations (seasonal block maxima).
    n_factors
        Truncation of the Poisson-point spectral representation. With
        n_factors = 50 the truncation is conservative for the distances
        seen in the Dutch panel.
    rng
        Random generator or seed.

    Returns
    -------
    Z : (n_obs, n_stations) array of simulated values with approximately
        unit-Frechet margins.
    """
    if isinstance(panel_or_distances, StationPanel):
        D = pairwise_distance_matrix(panel_or_distances)
    else:
        D = np.asarray(panel_or_distances, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square 2D, got {D.shape}")
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


def simulate_scenario_pair(
    panel: StationPanel,
    scenario: ScenarioPair,
    n_obs_per_season: int,
    n_factors: int = 50,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate one winter and one summer field under a scenario pair."""
    rng = np.random.default_rng(rng)
    D = pairwise_distance_matrix(panel)
    winter = simulate_brown_resnick_field(
        D, rho=scenario.winter.rho, alpha=scenario.winter.alpha,
        n_obs=n_obs_per_season, n_factors=n_factors, rng=rng,
    )
    summer = simulate_brown_resnick_field(
        D, rho=scenario.summer.rho, alpha=scenario.summer.alpha,
        n_obs=n_obs_per_season, n_factors=n_factors, rng=rng,
    )
    return {"winter": winter, "summer": summer}