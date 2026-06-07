"""Per-replication estimation pipeline and Monte Carlo aggregation.

Ties together the controlled Brown--Resnick DGP (``br_simulation``), the
pairwise extremal-dependence estimators (``dependence_estimators``), the
local-linear dependence--distance smoother (``distance_smoothing``) and
the seasonal-block bootstrap (``bootstrap``) into the exact per-replication
procedure of Algorithm 1 in simulation.tex, and aggregates the
replications into the performance measures of sec:sim-performance (bias,
RMSE, bootstrap coverage, and band-based size/power).

Estimand naming
---------------
``"theta"``           -> extremal coefficient, target br_theta(d).
``"chi_<u>"``         -> finite-level chi_u, target br_chi_u(d, u), e.g.
                        "chi_0.95", "chi_0.975", "chi_0.99".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from br_simulation import BRScenario, ScenarioPair, br_theta, br_chi_u, simulate_brown_resnick_field
from bootstrap import make_pseudo_year_blocks, percentile_band, resample_block_indices
from dependence_estimators import (
    chi_u_pairwise,
    empirical_uniform_ranks,
    fmadogram_pairwise,
    theta_from_fmadogram,
)
from distance_smoothing import (
    DEFAULT_KERNEL,
    Kernel,
    default_bandwidth_grid,
    local_linear_predict,
    select_common_bandwidth,
)
from spatial_utils import StationPanel, pairwise_distance_matrix


# ---------------------------------------------------------------------------
# Estimands and their closed-form Brown--Resnick targets
# ---------------------------------------------------------------------------

def build_estimands(u_grid: Sequence[float]) -> list[str]:
    """Estimand keys: the extremal coefficient and one chi_u per level."""
    return ["theta"] + [f"chi_{u:g}" for u in u_grid]


def true_curve(estimand: str, d: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    """Closed-form Brown--Resnick target curve for an estimand at distances d."""
    if estimand == "theta":
        return np.asarray(br_theta(d, rho, alpha), dtype=float)
    if estimand.startswith("chi_"):
        u = float(estimand.split("_", 1)[1])
        return np.asarray(br_chi_u(d, rho, alpha, u), dtype=float)
    raise ValueError(f"Unknown estimand {estimand!r}")


def pairwise_psi(
    X: np.ndarray, u_grid: Sequence[float], iu: tuple[np.ndarray, np.ndarray]
) -> dict[str, np.ndarray]:
    """Pairwise estimates for every estimand from one rank transform.

    Returns a dict estimand -> length-P vector aligned with the upper-
    triangle pair index ``iu``.
    """
    U = empirical_uniform_ranks(X)
    theta = theta_from_fmadogram(fmadogram_pairwise(U))
    out: dict[str, np.ndarray] = {"theta": theta[iu]}
    for u in u_grid:
        out[f"chi_{u:g}"] = chi_u_pairwise(U, u)[iu]
    return out


# ---------------------------------------------------------------------------
# Replication context (geometry + design + numerical settings)
# ---------------------------------------------------------------------------

@dataclass
class SimContext:
    """Everything fixed across replications for a given design."""
    panel: StationPanel
    D: np.ndarray                         # (N, N) distance matrix
    iu: tuple[np.ndarray, np.ndarray]     # upper-triangle indices
    d_pairs: np.ndarray                   # (P,) pairwise distances
    grid: np.ndarray                      # (G,) evaluation distances
    ref_idx: np.ndarray                   # indices into grid for reference distances
    ref_distances: np.ndarray             # (Rd,) reference distances
    u_grid: tuple[float, ...]
    estimands: list[str]
    alpha: float
    n_obs: int
    n_years: int
    method: str
    kernel: Kernel
    bw_candidates: np.ndarray
    n_boot: int
    boot_alpha: float                     # bootstrap band level (e.g. 0.05)


def make_context(
    panel: StationPanel,
    *,
    grid: np.ndarray | None = None,
    ref_distances: Sequence[float] = (50.0, 100.0, 200.0),
    u_grid: Sequence[float] = (0.95, 0.975, 0.99),
    alpha: float = 1.0,
    n_obs: int = 3150,
    n_years: int = 35,
    method: str = "exact",
    kernel: Kernel = DEFAULT_KERNEL,
    n_grid: int = 60,
    n_boot: int = 999,
    boot_alpha: float = 0.05,
) -> SimContext:
    """Precompute geometry, evaluation grid and numerical settings."""
    D = pairwise_distance_matrix(panel)
    iu = np.triu_indices(panel.n_stations, k=1)
    d_pairs = D[iu]
    ref_distances = np.asarray(ref_distances, dtype=float)
    if grid is None:
        base = np.linspace(d_pairs.min(), d_pairs.max(), n_grid)
        grid = np.unique(np.concatenate([base, ref_distances]))
    else:
        grid = np.unique(np.concatenate([np.asarray(grid, float), ref_distances]))
    ref_idx = np.array([int(np.argmin(np.abs(grid - d0))) for d0 in ref_distances])
    return SimContext(
        panel=panel, D=D, iu=iu, d_pairs=d_pairs, grid=grid,
        ref_idx=ref_idx, ref_distances=ref_distances,
        u_grid=tuple(u_grid), estimands=build_estimands(u_grid),
        alpha=alpha, n_obs=n_obs, n_years=n_years, method=method, kernel=kernel,
        bw_candidates=default_bandwidth_grid(d_pairs),
        n_boot=n_boot, boot_alpha=boot_alpha,
    )


# ---------------------------------------------------------------------------
# One Monte Carlo replication
# ---------------------------------------------------------------------------

def run_replication(seed: int, ctx: SimContext, scenario: ScenarioPair) -> dict:
    """Run one replication of the full pipeline for one design.

    Returns a dict of numpy arrays keyed by estimand-major layout
    (M estimands x G grid points), holding the smoothed point-estimate
    curves, the bootstrap percentile bands (if ``ctx.n_boot > 0``), the
    selected bandwidths and the raw (unsmoothed) pairwise bias. Plain
    dict/array return keeps the result picklable for joblib.
    """
    rng = np.random.default_rng(seed)
    M, G = len(ctx.estimands), ctx.grid.size

    # --- draw one winter and one summer sample of n_obs fields -------------
    Xw = simulate_brown_resnick_field(
        ctx.D, scenario.winter.rho, ctx.alpha, ctx.n_obs, method=ctx.method, rng=rng)
    Xs = simulate_brown_resnick_field(
        ctx.D, scenario.summer.rho, ctx.alpha, ctx.n_obs, method=ctx.method, rng=rng)

    psi_w = pairwise_psi(Xw, ctx.u_grid, ctx.iu)
    psi_s = pairwise_psi(Xs, ctx.u_grid, ctx.iu)

    curve_w = np.empty((M, G)); curve_s = np.empty((M, G))
    bw = np.empty(M)
    rawbias_w = np.empty(M); rawbias_s = np.empty(M)
    true_w_pairs = {e: true_curve(e, ctx.d_pairs, scenario.winter.rho, ctx.alpha) for e in ctx.estimands}
    true_s_pairs = {e: true_curve(e, ctx.d_pairs, scenario.summer.rho, ctx.alpha) for e in ctx.estimands}

    for m, e in enumerate(ctx.estimands):
        # one common LOOCV bandwidth for both seasons (same rule for both)
        h = select_common_bandwidth(
            ctx.d_pairs, [psi_w[e], psi_s[e]], candidates=ctx.bw_candidates, kernel=ctx.kernel)
        bw[m] = h
        curve_w[m] = local_linear_predict(ctx.d_pairs, psi_w[e], ctx.grid, h, ctx.kernel)
        curve_s[m] = local_linear_predict(ctx.d_pairs, psi_s[e], ctx.grid, h, ctx.kernel)
        rawbias_w[m] = float(np.mean(psi_w[e] - true_w_pairs[e]))
        rawbias_s[m] = float(np.mean(psi_s[e] - true_s_pairs[e]))

    result = {
        "curve_w": curve_w, "curve_s": curve_s,
        "bw": bw, "rawbias_w": rawbias_w, "rawbias_s": rawbias_s,
    }

    # --- seasonal-block bootstrap bands ------------------------------------
    if ctx.n_boot > 0:
        blocks = make_pseudo_year_blocks(ctx.n_obs, ctx.n_years)
        bw_dict = {e: bw[m] for m, e in enumerate(ctx.estimands)}
        boot_w = np.empty((ctx.n_boot, M, G))
        boot_s = np.empty((ctx.n_boot, M, G))
        for b in range(ctx.n_boot):
            idx_w = resample_block_indices(blocks, rng)
            idx_s = resample_block_indices(blocks, rng)
            bpsi_w = pairwise_psi(Xw[idx_w], ctx.u_grid, ctx.iu)
            bpsi_s = pairwise_psi(Xs[idx_s], ctx.u_grid, ctx.iu)
            for m, e in enumerate(ctx.estimands):
                boot_w[b, m] = local_linear_predict(ctx.d_pairs, bpsi_w[e], ctx.grid, bw_dict[e], ctx.kernel)
                boot_s[b, m] = local_linear_predict(ctx.d_pairs, bpsi_s[e], ctx.grid, bw_dict[e], ctx.kernel)
        boot_d = boot_w - boot_s
        a = ctx.boot_alpha
        result["band_w"] = percentile_band(boot_w, a)            # (lo (M,G), hi (M,G))
        result["band_s"] = percentile_band(boot_s, a)
        result["band_d"] = percentile_band(boot_d, a)
    return result


# ---------------------------------------------------------------------------
# Aggregation across replications -> performance measures
# ---------------------------------------------------------------------------

@dataclass
class DesignResults:
    """Aggregated Monte Carlo results for one design."""
    design: str
    estimands: list[str]
    grid: np.ndarray
    ref_idx: np.ndarray
    ref_distances: np.ndarray
    # MC-averaged curves and envelopes (M, G)
    mean_curve_w: np.ndarray
    mean_curve_s: np.ndarray
    lo_curve_w: np.ndarray
    hi_curve_w: np.ndarray
    lo_curve_s: np.ndarray
    hi_curve_s: np.ndarray
    mean_diff: np.ndarray
    lo_diff: np.ndarray
    hi_diff: np.ndarray
    true_curve_w: np.ndarray
    true_curve_s: np.ndarray
    true_diff: np.ndarray
    table: "list[dict]" = field(default_factory=list)   # tidy per-(estimand,season,d0) rows
    median_bw: dict = field(default_factory=dict)
    n_rep: int = 0
    n_boot: int = 0


def aggregate(
    reps: list[dict], ctx: SimContext, scenario: ScenarioPair, design: str
) -> DesignResults:
    """Turn a list of per-replication results into performance measures."""
    M, G = len(ctx.estimands), ctx.grid.size
    R = len(reps)
    cw = np.stack([r["curve_w"] for r in reps])      # (R, M, G)
    cs = np.stack([r["curve_s"] for r in reps])
    bw = np.stack([r["bw"] for r in reps])           # (R, M)
    rbw = np.stack([r["rawbias_w"] for r in reps])
    rbs = np.stack([r["rawbias_s"] for r in reps])
    has_boot = "band_w" in reps[0]

    tw = np.stack([true_curve(e, ctx.grid, scenario.winter.rho, ctx.alpha) for e in ctx.estimands])  # (M,G)
    ts = np.stack([true_curve(e, ctx.grid, scenario.summer.rho, ctx.alpha) for e in ctx.estimands])
    td = tw - ts

    mean_cw = cw.mean(0); mean_cs = cs.mean(0)
    lo_cw, hi_cw = np.nanpercentile(cw, 2.5, 0), np.nanpercentile(cw, 97.5, 0)
    lo_cs, hi_cs = np.nanpercentile(cs, 2.5, 0), np.nanpercentile(cs, 97.5, 0)
    cd = cw - cs
    mean_cd = cd.mean(0)
    lo_cd, hi_cd = np.nanpercentile(cd, 2.5, 0), np.nanpercentile(cd, 97.5, 0)

    if has_boot:
        bwl = np.stack([r["band_w"][0] for r in reps]); bwh = np.stack([r["band_w"][1] for r in reps])
        bsl = np.stack([r["band_s"][0] for r in reps]); bsh = np.stack([r["band_s"][1] for r in reps])
        bdl = np.stack([r["band_d"][0] for r in reps]); bdh = np.stack([r["band_d"][1] for r in reps])

    rows: list[dict] = []
    for m, e in enumerate(ctx.estimands):
        # directional detection rule for the difference (winter vs summer)
        # theta: winter MORE dependent => Delta_theta < 0; chi: Delta_chi > 0.
        is_theta = (e == "theta")
        for k, d0 in zip(ctx.ref_idx, ctx.ref_distances):
            base = {"design": design, "estimand": e, "d0_km": float(d0)}
            for season, c, t in (("winter", cw, tw), ("summer", cs, ts)):
                est = c[:, m, k]
                truth = t[m, k]
                row = dict(base)
                row["season"] = season
                row["true"] = float(truth)
                row["mc_mean"] = float(np.nanmean(est))
                row["bias"] = float(np.nanmean(est) - truth)
                row["rmse"] = float(np.sqrt(np.nanmean((est - truth) ** 2)))
                if has_boot:
                    lo = (bwl if season == "winter" else bsl)[:, m, k]
                    hi = (bwh if season == "winter" else bsh)[:, m, k]
                    row["coverage"] = float(np.nanmean((lo <= truth) & (truth <= hi)))
                    row["mean_band_width"] = float(np.nanmean(hi - lo))
                rows.append(row)
            # difference row
            estd = cw[:, m, k] - cs[:, m, k]
            truthd = td[m, k]
            rowd = dict(base); rowd["season"] = "diff"
            rowd["true"] = float(truthd)
            rowd["mc_mean"] = float(np.nanmean(estd))
            rowd["bias"] = float(np.nanmean(estd) - truthd)
            rowd["rmse"] = float(np.sqrt(np.nanmean((estd - truthd) ** 2)))
            if has_boot:
                lo = bdl[:, m, k]; hi = bdh[:, m, k]
                rowd["coverage"] = float(np.nanmean((lo <= truthd) & (truthd <= hi)))
                rowd["mean_band_width"] = float(np.nanmean(hi - lo))
                excl_zero = (hi < 0) | (lo > 0)
                detect = (hi < 0) if is_theta else (lo > 0)   # correct-direction exclusion
                rowd["size_or_power"] = float(np.nanmean(detect))
                rowd["size_excl_zero_2sided"] = float(np.nanmean(excl_zero))
            rows.append(rowd)

    return DesignResults(
        design=design, estimands=ctx.estimands, grid=ctx.grid,
        ref_idx=ctx.ref_idx, ref_distances=ctx.ref_distances,
        mean_curve_w=mean_cw, mean_curve_s=mean_cs,
        lo_curve_w=lo_cw, hi_curve_w=hi_cw, lo_curve_s=lo_cs, hi_curve_s=hi_cs,
        mean_diff=mean_cd, lo_diff=lo_cd, hi_diff=hi_cd,
        true_curve_w=tw, true_curve_s=ts, true_diff=td,
        table=rows,
        median_bw={e: float(np.median(bw[:, m])) for m, e in enumerate(ctx.estimands)},
        n_rep=R, n_boot=(ctx.n_boot if has_boot else 0),
    )
