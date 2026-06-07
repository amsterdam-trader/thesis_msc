"""Pairwise extremal-dependence estimators.

Implements the three estimators used in the thesis empirical analysis:
    - F-madogram
    - Extremal coefficient theta from the F-madogram
    - Finite-level chi_u (symmetric)

All estimators operate on a (n_obs, n_stations) data matrix after
empirical uniform rank transform.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def empirical_uniform_ranks(X: np.ndarray) -> np.ndarray:
    """Column-wise empirical uniform rank transform.

    For each column, returns rank(x) / (n + 1) so the transformed values
    are strictly in (0, 1). Ties are broken with the 'average' method.

    Parameters
    ----------
    X : (n_obs, n_stations) array of raw observations.

    Returns
    -------
    U : (n_obs, n_stations) array on (0, 1).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {X.shape}")
    n = X.shape[0]
    # Vectorised over columns; identical to per-column rankdata(method="average").
    return rankdata(X, method="average", axis=0) / (n + 1.0)


def fmadogram_pairwise(U: np.ndarray) -> np.ndarray:
    """Pairwise F-madogram matrix.

    nu_F(i, j) = 0.5 * mean(|U_i(t) - U_j(t)|), where U are empirical
    uniform ranks. Returns a symmetric (d, d) matrix with zeros on the
    diagonal.
    """
    n, d = U.shape
    nu = np.zeros((d, d), dtype=float)
    for i in range(d):
        diff = np.abs(U[:, i, None] - U[:, i + 1:])
        nu[i, i + 1:] = 0.5 * diff.mean(axis=0)
    nu = nu + nu.T
    return nu


def theta_from_fmadogram(nu: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert F-madogram to extremal coefficient theta.

    theta = (1 + 2*nu) / (1 - 2*nu). nu is clipped to (eps, 0.5 - eps)
    to keep the estimator finite at boundary samples. theta lies in
    [1, 2]; complete dependence at nu = 0 gives theta = 1, independence
    at nu = 1/6 (= 1/(2*(d-1)) with d=2... actually nu_F asymptote is
    1/6 for independent uniforms, giving theta = 2).
    """
    nu_c = np.clip(nu, eps, 0.5 - eps)
    theta = (1.0 + 2.0 * nu_c) / (1.0 - 2.0 * nu_c)
    # Methodology: extremal-coefficient estimates are clipped to [1, 2]
    # (sec:meth-madogram). Finite-sample F-madograms can exceed the
    # independence value 1/6, which would push theta past 2.
    theta = np.clip(theta, 1.0, 2.0)
    np.fill_diagonal(theta, 1.0)
    return theta


def chi_u_pairwise(U: np.ndarray, u: float) -> np.ndarray:
    """Symmetric finite-level chi at threshold u.

    For each pair (i, j) we average the two conditional probabilities
    P(U_i > u | U_j > u) and P(U_j > u | U_i > u). With empirical
    uniform ranks the marginal counts are nearly equal so this
    coincides with the standard chi_u estimator up to ties.

    Parameters
    ----------
    U : (n_obs, n_stations) array of uniform ranks.
    u : threshold in (0, 1).
    """
    if not 0.0 < u < 1.0:
        raise ValueError(f"Threshold u must be in (0, 1), got {u}")
    above = U > u
    counts = above.sum(axis=0).astype(float)
    n, d = U.shape
    chi = np.zeros((d, d), dtype=float)
    for i in range(d):
        joint = (above[:, i, None] & above[:, i + 1:]).sum(axis=0).astype(float)
        c_i = counts[i] if counts[i] > 0 else np.nan
        c_j = counts[i + 1:]
        c_j_safe = np.where(c_j > 0, c_j, np.nan)
        c = 0.5 * (joint / c_i + joint / c_j_safe)
        chi[i, i + 1:] = c
    chi = chi + chi.T
    np.fill_diagonal(chi, 1.0)
    return chi


def estimate_all_pairs(
    X: np.ndarray,
    pair_table: pd.DataFrame,
    u_grid: Sequence[float],
) -> pd.DataFrame:
    """Compute F-madogram, theta and chi_u (for each u) on every pair.

    Parameters
    ----------
    X : (n_obs, n_stations) raw observations for a single regime/season.
    pair_table : DataFrame with columns (i, j, dist_km) and one row per
        unordered pair.
    u_grid : tail thresholds at which to compute chi_u.

    Returns
    -------
    DataFrame with one row per pair, holding nu, theta, and chi_u_{u}
    for each u in u_grid.
    """
    U = empirical_uniform_ranks(X)
    nu = fmadogram_pairwise(U)
    theta = theta_from_fmadogram(nu)

    out = pair_table.copy()
    i_arr = out["i"].to_numpy()
    j_arr = out["j"].to_numpy()
    out["nu_F"] = nu[i_arr, j_arr]
    out["theta_hat"] = theta[i_arr, j_arr]
    for u in u_grid:
        chi = chi_u_pairwise(U, u)
        out[f"chi_u_{u:.2f}"] = chi[i_arr, j_arr]
    return out