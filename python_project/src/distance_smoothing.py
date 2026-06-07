"""Local-linear dependence--distance smoother (methodology sec:meth-distance).

The pairwise extremal-dependence estimates psi_ij (either the extremal
coefficient theta or the finite-level chi_u) are summarised against
inter-station distance by a local-linear kernel smoother evaluated on a
distance grid:

    (a_hat(d), b_hat(d)) = argmin_{a,b}  sum_{i<j} K((d_ij - d)/h)
                                          [psi_ij - a - b (d_ij - d)]^2,
    psi_hat(d) = a_hat(d),

with bandwidth h selected by leave-one-out cross-validation.

Kernel note (thesis inconsistency, flagged)
-------------------------------------------
methodology.tex (sec:meth-distance) names a *Gaussian* kernel, while
simulation.tex (sec:sim-estimation and Algorithm 1) names an
*Epanechnikov* kernel. Both describe the same local-linear / LOOCV
construction and the fitted curves are visually indistinguishable for
these data. This module therefore supports both kernels via the
``kernel`` argument; the project default is set in
``config``-independent form here as Epanechnikov (matching the simulation
section being implemented). Switch with ``kernel="gaussian"`` to match
methodology.tex. The two sections should be reconciled in the prose.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

Kernel = Literal["epanechnikov", "gaussian"]

DEFAULT_KERNEL: Kernel = "epanechnikov"


def kernel_weights(t: np.ndarray, kernel: Kernel = DEFAULT_KERNEL) -> np.ndarray:
    """Kernel weights K(t). Epanechnikov has compact support |t| < 1."""
    t = np.asarray(t, dtype=float)
    if kernel == "epanechnikov":
        return np.where(np.abs(t) < 1.0, 1.0 - t * t, 0.0)
    if kernel == "gaussian":
        return np.exp(-0.5 * t * t)
    raise ValueError(f"Unknown kernel {kernel!r}; use 'epanechnikov' or 'gaussian'.")


def default_bandwidth_grid(
    x: np.ndarray, n: int = 18, lo_frac: float = 0.04, hi_frac: float = 0.5
) -> np.ndarray:
    """A geometric grid of candidate bandwidths scaled to the data range."""
    x = np.asarray(x, dtype=float)
    span = float(np.nanmax(x) - np.nanmin(x))
    if span <= 0:
        return np.array([1.0])
    return np.geomspace(lo_frac * span, hi_frac * span, n)


def _local_linear_weight_rows(
    x: np.ndarray, eval_pts: np.ndarray, h: float, kernel: Kernel
) -> np.ndarray:
    """Return the (n_eval, n) hat rows l(d) so that psi_hat(d) = l(d) . y.

    Row m gives the linear weights producing the local-linear intercept at
    eval_pts[m]. Falls back to local-constant (Nadaraya--Watson) weights
    where the local design is rank-deficient (too few points in the window).
    """
    x = np.asarray(x, dtype=float)
    eval_pts = np.asarray(eval_pts, dtype=float)
    n_eval = eval_pts.size
    n = x.size
    rows = np.zeros((n_eval, n), dtype=float)
    for m in range(n_eval):
        dx = x - eval_pts[m]
        w = kernel_weights(dx / h, kernel)
        s0 = w.sum()
        if s0 <= 0:
            rows[m] = np.nan
            continue
        s1 = (w * dx).sum()
        s2 = (w * dx * dx).sum()
        det = s0 * s2 - s1 * s1
        # Require a well-conditioned local design with >= 2 effective points.
        if not np.isfinite(det) or abs(det) < 1e-12 * (s0 * s2 + 1e-12):
            rows[m] = w / s0                       # local constant
        else:
            rows[m] = w * (s2 - s1 * dx) / det     # local linear intercept weights
    return rows


def local_linear_predict(
    x: np.ndarray,
    y: np.ndarray,
    eval_pts: np.ndarray,
    h: float,
    kernel: Kernel = DEFAULT_KERNEL,
) -> np.ndarray:
    """Local-linear estimate psi_hat(eval_pts) of y on x with bandwidth h."""
    rows = _local_linear_weight_rows(x, eval_pts, h, kernel)
    y = np.asarray(y, dtype=float)
    return rows @ y


def loocv_score(
    x: np.ndarray, y: np.ndarray, h: float, kernel: Kernel = DEFAULT_KERNEL
) -> float:
    """Leave-one-out CV residual sum of squares for bandwidth h.

    Uses the closed-form local-linear LOO residual
    r_i = (y_i - psi_hat(x_i)) / (1 - S_ii), with S_ii the self-weight.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = _local_linear_weight_rows(x, x, h, kernel)   # (n, n) hat matrix
    if not np.isfinite(rows).all():
        return np.inf
    yhat = rows @ y
    s_ii = np.diag(rows)
    denom = 1.0 - s_ii
    # Guard against leverage == 1 (degenerate window).
    bad = np.abs(denom) < 1e-8
    denom = np.where(bad, np.nan, denom)
    resid = (y - yhat) / denom
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return np.inf
    return float(np.mean(resid ** 2))


def select_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    candidates: Sequence[float] | None = None,
    kernel: Kernel = DEFAULT_KERNEL,
) -> float:
    """Select the LOOCV-optimal bandwidth over a grid of candidates."""
    x = np.asarray(x, dtype=float)
    if candidates is None:
        candidates = default_bandwidth_grid(x)
    scores = [loocv_score(x, y, float(h), kernel) for h in candidates]
    best = int(np.argmin(scores))
    return float(candidates[best])


def select_common_bandwidth(
    x: np.ndarray,
    ys: Sequence[np.ndarray],
    candidates: Sequence[float] | None = None,
    kernel: Kernel = DEFAULT_KERNEL,
) -> float:
    """LOOCV bandwidth shared across several response vectors on the same x.

    Implements "the same rule for both seasons": the single bandwidth that
    minimises the summed leave-one-out CV score over the supplied response
    vectors (e.g. the winter and summer pairwise estimates). Using one
    bandwidth for both seasons ensures the seasonal difference curve is not
    an artefact of unequal smoothing.
    """
    x = np.asarray(x, dtype=float)
    if candidates is None:
        candidates = default_bandwidth_grid(x)
    total = np.array(
        [sum(loocv_score(x, y, float(h), kernel) for y in ys) for h in candidates]
    )
    best = int(np.argmin(total))
    return float(candidates[best])


def smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    bandwidth: float | None = None,
    candidates: Sequence[float] | None = None,
    kernel: Kernel = DEFAULT_KERNEL,
) -> tuple[np.ndarray, float]:
    """Smooth (x, y) onto ``grid``; select bandwidth by LOOCV if not given.

    Returns (curve_on_grid, bandwidth_used).
    """
    if bandwidth is None:
        bandwidth = select_bandwidth(x, y, candidates=candidates, kernel=kernel)
    return local_linear_predict(x, y, grid, bandwidth, kernel=kernel), bandwidth
