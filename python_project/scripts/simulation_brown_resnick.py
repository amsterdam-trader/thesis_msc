#!/usr/bin/env python3
"""
Simulation study for season-specific Brown--Resnick extremal dependence.

Main design:
- Brown--Resnick DGP on the empirical KNMI station geometry.
- Null:        rho_W = rho_S = 0.5*dmax, alpha_W = alpha_S = 1.
- Alternative: rho_W = dmax, rho_S = 0.25*dmax, alpha_W = alpha_S = 1.
- Estimate rho and alpha freely in both seasons.
- Pairwise composite likelihood using all station pairs.
- Season-year block bootstrap.
- M = 250 Monte Carlo replications.
- B = 500 bootstrap replications inside each Monte Carlo replication.

Outputs:
- CSV summaries.
- LaTeX summary table.
- Plots for parameter estimates, Delta_rho, theta(d), and coverage/rejection.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from joblib import Parallel, delayed

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class SimConfig:
    M: int = 250
    B: int = 500
    n_years: int = 35
    days_per_season: int = 90
    max_spectral_terms: int = 800
    spectral_chunk_size: int = 32
    pair_chunk_size: int = 128
    seed: int = 12345
    n_jobs: int = 1
    optimizer_maxiter: int = 250


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------


def local_xy_from_latlon(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Convert latitude/longitude to local Cartesian kilometres using an
    equirectangular approximation. This is adequate for the Netherlands-scale
    KNMI network.
    """
    radius_km = 6371.0
    lat0 = np.deg2rad(np.mean(lat))
    lon0 = np.deg2rad(np.mean(lon))
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = radius_km * np.cos(lat0) * (lon_rad - lon0)
    y = radius_km * (lat_rad - lat0)
    return np.column_stack([x, y])


def load_station_coordinates(path: str | Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load station coordinates.

    Accepted formats:
    1. station, lat, lon
    2. station, x_km, y_km
    """
    df = pd.read_csv(path)

    lower_cols = {c.lower(): c for c in df.columns}

    if {"lat", "lon"}.issubset(lower_cols):
        lat = df[lower_cols["lat"]].to_numpy(float)
        lon = df[lower_cols["lon"]].to_numpy(float)
        coords = local_xy_from_latlon(lat, lon)
    elif {"x_km", "y_km"}.issubset(lower_cols):
        x = df[lower_cols["x_km"]].to_numpy(float)
        y = df[lower_cols["y_km"]].to_numpy(float)
        coords = np.column_stack([x, y])
        coords = coords - coords.mean(axis=0)
    else:
        raise ValueError("Station file must contain either columns 'lat, lon' or 'x_km, y_km'.")

    dist_mat = squareform(pdist(coords))
    return df, coords, dist_mat


def pair_information(n_sites: int, dist_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_i, pair_j = np.triu_indices(n_sites, k=1)
    d_pair = dist_mat[pair_i, pair_j]
    if np.any(d_pair <= 0):
        raise ValueError("Some station pairs have zero distance. Check duplicate coordinates.")
    return pair_i, pair_j, d_pair


# ---------------------------------------------------------------------
# Brown--Resnick functions
# ---------------------------------------------------------------------


def gamma_variogram(d: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    return np.power(np.maximum(d, 0.0) / rho, alpha)


def a_from_distance(d: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    return np.power(np.maximum(d, 0.0) / rho, alpha / 2.0)


def theta_br(d: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    return 2.0 * norm.cdf(0.5 * a_from_distance(d, rho, alpha))


def brown_resnick_gaussian_cov(coords: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    """
    Covariance of the underlying Gaussian process W with stationary increments
    and W(0)=0:
        Cov(W(s_i), W(s_j))
        = 0.5 * [gamma(s_i) + gamma(s_j) - gamma(s_i - s_j)].
    """
    coords = np.asarray(coords, float)
    r0 = np.linalg.norm(coords, axis=1)
    dist_mat = squareform(pdist(coords))

    gamma0 = gamma_variogram(r0, rho, alpha)
    gammaij = gamma_variogram(dist_mat, rho, alpha)

    cov = 0.5 * (gamma0[:, None] + gamma0[None, :] - gammaij)
    cov = 0.5 * (cov + cov.T)
    return cov


def cholesky_psd(cov: np.ndarray) -> np.ndarray:
    """
    Cholesky with jitter fallback. Returns L such that approximately L L' = cov.
    """
    n = cov.shape[0]
    jitter = 1e-10

    for _ in range(10):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(np.sqrt(vals))


def simulate_br_fields(
    coords: np.ndarray,
    rho: float,
    alpha: float,
    n_fields: int,
    rng: np.random.Generator,
    max_terms: int = 800,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Simulate Brown--Resnick fields with unit-Fréchet margins using a finite
    spectral approximation:
        Z(s) = max_k xi_k exp(W_k(s) - Var{W(s)}/2),
    where xi_k = 1 / Gamma_k and Gamma_k are cumulative sums of Exp(1).

    This is an approximation. Increase max_terms to reduce truncation error.
    """
    n_sites = coords.shape[0]
    cov = brown_resnick_gaussian_cov(coords, rho, alpha)
    var = np.diag(cov)
    L = cholesky_psd(cov)

    out = np.empty((n_fields, n_sites), dtype=float)

    for start in range(0, n_fields, chunk_size):
        end = min(start + chunk_size, n_fields)
        m = end - start

        exp_draws = rng.exponential(scale=1.0, size=(m, max_terms))
        gamma_points = np.cumsum(exp_draws, axis=1)
        xi = 1.0 / gamma_points

        normals = rng.standard_normal(size=(m * max_terms, n_sites)) @ L.T
        normals = normals.reshape(m, max_terms, n_sites)

        with np.errstate(over="ignore", invalid="ignore"):
            spectral = np.exp(normals - 0.5 * var[None, None, :])

        spectral = np.nan_to_num(spectral, nan=0.0, posinf=1e300, neginf=0.0)
        out[start:end, :] = np.max(xi[:, :, None] * spectral, axis=1)

    return out


def simulate_season_years(
    coords: np.ndarray,
    rho: float,
    alpha: float,
    n_years: int,
    days_per_season: int,
    rng: np.random.Generator,
    max_terms: int,
    chunk_size: int,
) -> np.ndarray:
    n_fields = n_years * days_per_season
    z = simulate_br_fields(
        coords=coords,
        rho=rho,
        alpha=alpha,
        n_fields=n_fields,
        rng=rng,
        max_terms=max_terms,
        chunk_size=chunk_size,
    )
    return z.reshape(n_years, days_per_season, coords.shape[0])


# ---------------------------------------------------------------------
# Bivariate Brown--Resnick density and composite likelihood
# ---------------------------------------------------------------------


def br_pair_logpdf(z_i: np.ndarray, z_j: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Log-density for the Brown--Resnick / Hüsler--Reiss bivariate distribution
    with unit-Fréchet margins.

    z_i, z_j: arrays of observations.
    a: sqrt(gamma(d)), broadcastable to z_i and z_j.
    """
    eps = 1e-12
    x = np.maximum(z_i, eps)
    y = np.maximum(z_j, eps)
    a = np.maximum(a, 1e-8)

    log_yx = np.log(y / x)
    q1 = 0.5 * a + log_yx / a
    q2 = 0.5 * a - log_yx / a

    Phi1 = norm.cdf(q1)
    Phi2 = norm.cdf(q2)
    phi1 = norm.pdf(q1)
    phi2 = norm.pdf(q2)

    V = Phi1 / x + Phi2 / y

    Vx = (-Phi1 - phi1 / a) / (x**2) + phi2 / (a * x * y)
    Vy = phi1 / (a * x * y) + (-Phi2 - phi2 / a) / (y**2)

    Vxy = phi1 / (x**2 * y) * (q1 / (a**2) - 1.0 / a) + phi2 / (a * x * y**2) * (q2 / a - 1.0)

    density_factor = Vx * Vy - Vxy

    with np.errstate(divide="ignore", invalid="ignore"):
        log_density = np.log(np.maximum(density_factor, 1e-300)) - V

    return np.nan_to_num(log_density, nan=-1e300, posinf=-1e300, neginf=-1e300)


def composite_loglik(
    z: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    d_pair: np.ndarray,
    rho: float,
    alpha: float,
    pair_chunk_size: int = 128,
) -> float:
    """
    Pairwise composite log-likelihood using all station pairs.
    """
    total = 0.0
    n_pairs = len(d_pair)

    for start in range(0, n_pairs, pair_chunk_size):
        end = min(start + pair_chunk_size, n_pairs)

        ii = pair_i[start:end]
        jj = pair_j[start:end]
        d = d_pair[start:end]

        zi = z[:, ii]
        zj = z[:, jj]
        a = a_from_distance(d, rho, alpha)[None, :]

        total += float(np.sum(br_pair_logpdf(zi, zj, a)))

    return total


def make_start_grid(dmax: float) -> List[Tuple[float, float]]:
    rho_grid = np.array([0.1 * dmax, 0.25 * dmax, 0.5 * dmax, dmax, 2.0 * dmax])
    alpha_grid = np.array([0.5, 1.0, 1.5, 2.0])
    return [(float(r), float(a)) for r in rho_grid for a in alpha_grid]


def estimate_br_pairwise(
    z: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    d_pair: np.ndarray,
    dmax: float,
    config: SimConfig,
) -> Dict[str, float]:
    """
    Estimate (rho, alpha) by bounded pairwise composite likelihood.
    """
    bounds = [(1.0, 3.0 * dmax), (0.05, 2.0)]
    starts = make_start_grid(dmax)

    def objective(par: np.ndarray) -> float:
        rho, alpha = float(par[0]), float(par[1])
        if rho <= 0 or alpha <= 0 or alpha > 2:
            return 1e300
        ll = composite_loglik(
            z=z,
            pair_i=pair_i,
            pair_j=pair_j,
            d_pair=d_pair,
            rho=rho,
            alpha=alpha,
            pair_chunk_size=config.pair_chunk_size,
        )
        if not np.isfinite(ll):
            return 1e300
        return -ll

    best_res = None

    for start in starts:
        try:
            res = minimize(
                objective,
                x0=np.array(start, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": config.optimizer_maxiter, "ftol": 1e-6},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res
        except Exception as exc:
            warnings.warn(f"Optimisation failed from start {start}: {exc}")

    if best_res is None:
        return {
            "rho": np.nan,
            "alpha": np.nan,
            "loglik": np.nan,
            "converged": 0,
            "boundary": 0,
        }

    rho_hat, alpha_hat = best_res.x
    loglik = -best_res.fun

    tol = 1e-4
    boundary = int(
        abs(rho_hat - bounds[0][0]) < tol
        or abs(rho_hat - bounds[0][1]) < tol
        or abs(alpha_hat - bounds[1][0]) < tol
        or abs(alpha_hat - bounds[1][1]) < tol
    )

    return {
        "rho": float(rho_hat),
        "alpha": float(alpha_hat),
        "loglik": float(loglik),
        "converged": int(bool(best_res.success)),
        "boundary": boundary,
    }


# ---------------------------------------------------------------------
# Bootstrap and Monte Carlo
# ---------------------------------------------------------------------


def flatten_years(z_years: np.ndarray) -> np.ndarray:
    return z_years.reshape(-1, z_years.shape[-1])


def bootstrap_two_season_delta(
    winter_years: np.ndarray,
    summer_years: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    d_pair: np.ndarray,
    dmax: float,
    config: SimConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Season-year block bootstrap. Resample whole season-years independently
    within winter and summer.
    """
    n_w_years = winter_years.shape[0]
    n_s_years = summer_years.shape[0]

    rho_w = np.empty(config.B)
    rho_s = np.empty(config.B)
    alpha_w = np.empty(config.B)
    alpha_s = np.empty(config.B)
    delta = np.empty(config.B)

    for b in range(config.B):
        idx_w = rng.integers(0, n_w_years, size=n_w_years)
        idx_s = rng.integers(0, n_s_years, size=n_s_years)

        z_w_star = flatten_years(winter_years[idx_w])
        z_s_star = flatten_years(summer_years[idx_s])

        est_w = estimate_br_pairwise(z_w_star, pair_i, pair_j, d_pair, dmax, config)
        est_s = estimate_br_pairwise(z_s_star, pair_i, pair_j, d_pair, dmax, config)

        rho_w[b] = est_w["rho"]
        rho_s[b] = est_s["rho"]
        alpha_w[b] = est_w["alpha"]
        alpha_s[b] = est_s["alpha"]
        delta[b] = est_w["rho"] - est_s["rho"]

    return {
        "rho_w": rho_w,
        "rho_s": rho_s,
        "alpha_w": alpha_w,
        "alpha_s": alpha_s,
        "delta": delta,
    }


def scenario_definitions(dmax: float) -> Dict[str, Dict[str, float]]:
    return {
        "null": {
            "rho_w_true": 0.5 * dmax,
            "rho_s_true": 0.5 * dmax,
            "alpha_w_true": 1.0,
            "alpha_s_true": 1.0,
        },
        "alternative": {
            "rho_w_true": 1.0 * dmax,
            "rho_s_true": 0.25 * dmax,
            "alpha_w_true": 1.0,
            "alpha_s_true": 1.0,
        },
    }


def run_one_replication(
    rep: int,
    scenario_name: str,
    scenario: Dict[str, float],
    coords: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    d_pair: np.ndarray,
    dmax: float,
    config: SimConfig,
) -> Dict[str, float]:
    rng = np.random.default_rng(config.seed + 100000 * hash(scenario_name) % 100000 + rep)

    winter_years = simulate_season_years(
        coords=coords,
        rho=scenario["rho_w_true"],
        alpha=scenario["alpha_w_true"],
        n_years=config.n_years,
        days_per_season=config.days_per_season,
        rng=rng,
        max_terms=config.max_spectral_terms,
        chunk_size=config.spectral_chunk_size,
    )

    summer_years = simulate_season_years(
        coords=coords,
        rho=scenario["rho_s_true"],
        alpha=scenario["alpha_s_true"],
        n_years=config.n_years,
        days_per_season=config.days_per_season,
        rng=rng,
        max_terms=config.max_spectral_terms,
        chunk_size=config.spectral_chunk_size,
    )

    z_w = flatten_years(winter_years)
    z_s = flatten_years(summer_years)

    est_w = estimate_br_pairwise(z_w, pair_i, pair_j, d_pair, dmax, config)
    est_s = estimate_br_pairwise(z_s, pair_i, pair_j, d_pair, dmax, config)

    delta_hat = est_w["rho"] - est_s["rho"]
    delta_true = scenario["rho_w_true"] - scenario["rho_s_true"]

    boot = bootstrap_two_season_delta(
        winter_years=winter_years,
        summer_years=summer_years,
        pair_i=pair_i,
        pair_j=pair_j,
        d_pair=d_pair,
        dmax=dmax,
        config=config,
        rng=rng,
    )

    delta_boot = boot["delta"]

    delta_ci_low, delta_ci_high = np.nanpercentile(delta_boot, [2.5, 97.5])
    delta_lower_one_sided = np.nanpercentile(delta_boot, 5.0)

    rho_w_ci_low, rho_w_ci_high = np.nanpercentile(boot["rho_w"], [2.5, 97.5])
    rho_s_ci_low, rho_s_ci_high = np.nanpercentile(boot["rho_s"], [2.5, 97.5])

    alpha_w_ci_low, alpha_w_ci_high = np.nanpercentile(boot["alpha_w"], [2.5, 97.5])
    alpha_s_ci_low, alpha_s_ci_high = np.nanpercentile(boot["alpha_s"], [2.5, 97.5])

    reject_one_sided = int(delta_lower_one_sided > 0.0)
    p_boot = (1.0 + np.sum(delta_boot <= 0.0)) / (len(delta_boot) + 1.0)

    d_grid = np.linspace(0.0, dmax, 100)
    theta_w_true = theta_br(d_grid, scenario["rho_w_true"], scenario["alpha_w_true"])
    theta_s_true = theta_br(d_grid, scenario["rho_s_true"], scenario["alpha_s_true"])
    theta_w_hat = theta_br(d_grid, est_w["rho"], est_w["alpha"])
    theta_s_hat = theta_br(d_grid, est_s["rho"], est_s["alpha"])

    theta_w_mse = float(np.mean((theta_w_hat - theta_w_true) ** 2))
    theta_s_mse = float(np.mean((theta_s_hat - theta_s_true) ** 2))

    return {
        "scenario": scenario_name,
        "rep": rep,
        "rho_w_true": scenario["rho_w_true"],
        "rho_s_true": scenario["rho_s_true"],
        "alpha_w_true": scenario["alpha_w_true"],
        "alpha_s_true": scenario["alpha_s_true"],
        "delta_true": delta_true,
        "rho_w_hat": est_w["rho"],
        "rho_s_hat": est_s["rho"],
        "alpha_w_hat": est_w["alpha"],
        "alpha_s_hat": est_s["alpha"],
        "delta_hat": delta_hat,
        "rho_w_ci_low": rho_w_ci_low,
        "rho_w_ci_high": rho_w_ci_high,
        "rho_s_ci_low": rho_s_ci_low,
        "rho_s_ci_high": rho_s_ci_high,
        "alpha_w_ci_low": alpha_w_ci_low,
        "alpha_w_ci_high": alpha_w_ci_high,
        "alpha_s_ci_low": alpha_s_ci_low,
        "alpha_s_ci_high": alpha_s_ci_high,
        "delta_ci_low": delta_ci_low,
        "delta_ci_high": delta_ci_high,
        "delta_lower_one_sided": delta_lower_one_sided,
        "cover_rho_w": int(rho_w_ci_low <= scenario["rho_w_true"] <= rho_w_ci_high),
        "cover_rho_s": int(rho_s_ci_low <= scenario["rho_s_true"] <= rho_s_ci_high),
        "cover_alpha_w": int(alpha_w_ci_low <= scenario["alpha_w_true"] <= alpha_w_ci_high),
        "cover_alpha_s": int(alpha_s_ci_low <= scenario["alpha_s_true"] <= alpha_s_ci_high),
        "cover_delta": int(delta_ci_low <= delta_true <= delta_ci_high),
        "reject_one_sided": reject_one_sided,
        "p_boot": p_boot,
        "theta_w_mse": theta_w_mse,
        "theta_s_mse": theta_s_mse,
        "loglik_w": est_w["loglik"],
        "loglik_s": est_s["loglik"],
        "converged_w": est_w["converged"],
        "converged_s": est_s["converged"],
        "boundary_w": est_w["boundary"],
        "boundary_s": est_s["boundary"],
    }


def run_scenario(
    scenario_name: str,
    scenario: Dict[str, float],
    coords: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    d_pair: np.ndarray,
    dmax: float,
    config: SimConfig,
) -> pd.DataFrame:
    if config.n_jobs == 1:
        rows = [
            run_one_replication(
                rep=m,
                scenario_name=scenario_name,
                scenario=scenario,
                coords=coords,
                pair_i=pair_i,
                pair_j=pair_j,
                d_pair=d_pair,
                dmax=dmax,
                config=config,
            )
            for m in range(config.M)
        ]
    else:
        rows = Parallel(n_jobs=config.n_jobs, verbose=10)(
            delayed(run_one_replication)(
                rep=m,
                scenario_name=scenario_name,
                scenario=scenario,
                coords=coords,
                pair_i=pair_i,
                pair_j=pair_j,
                d_pair=d_pair,
                dmax=dmax,
                config=config,
            )
            for m in range(config.M)
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------


def rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(x) ** 2)))


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario_name, g in results.groupby("scenario"):
        row0 = g.iloc[0]

        rows.append(
            {
                "scenario": scenario_name,
                "bias_rho_w": float(np.nanmean(g["rho_w_hat"] - row0["rho_w_true"])),
                "bias_rho_s": float(np.nanmean(g["rho_s_hat"] - row0["rho_s_true"])),
                "bias_alpha_w": float(np.nanmean(g["alpha_w_hat"] - row0["alpha_w_true"])),
                "bias_alpha_s": float(np.nanmean(g["alpha_s_hat"] - row0["alpha_s_true"])),
                "bias_delta": float(np.nanmean(g["delta_hat"] - row0["delta_true"])),
                "rmse_rho_w": rmse(g["rho_w_hat"] - row0["rho_w_true"]),
                "rmse_rho_s": rmse(g["rho_s_hat"] - row0["rho_s_true"]),
                "rmse_alpha_w": rmse(g["alpha_w_hat"] - row0["alpha_w_true"]),
                "rmse_alpha_s": rmse(g["alpha_s_hat"] - row0["alpha_s_true"]),
                "rmse_delta": rmse(g["delta_hat"] - row0["delta_true"]),
                "coverage_rho_w": float(np.nanmean(g["cover_rho_w"])),
                "coverage_rho_s": float(np.nanmean(g["cover_rho_s"])),
                "coverage_alpha_w": float(np.nanmean(g["cover_alpha_w"])),
                "coverage_alpha_s": float(np.nanmean(g["cover_alpha_s"])),
                "coverage_delta": float(np.nanmean(g["cover_delta"])),
                "rejection_rate": float(np.nanmean(g["reject_one_sided"])),
                "mean_p_boot": float(np.nanmean(g["p_boot"])),
                "mean_theta_w_mse": float(np.nanmean(g["theta_w_mse"])),
                "mean_theta_s_mse": float(np.nanmean(g["theta_s_mse"])),
                "convergence_rate_w": float(np.nanmean(g["converged_w"])),
                "convergence_rate_s": float(np.nanmean(g["converged_s"])),
                "boundary_rate_w": float(np.nanmean(g["boundary_w"])),
                "boundary_rate_s": float(np.nanmean(g["boundary_s"])),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def plot_delta_distribution(results: pd.DataFrame, scenario: str, outdir: Path) -> None:
    g = results[results["scenario"] == scenario]
    delta_true = g["delta_true"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(g["delta_hat"], bins=30, alpha=0.8)
    ax.axvline(delta_true, linestyle="--", label="True Delta rho")
    ax.axvline(0.0, linestyle=":", label="Null value")
    ax.set_title(f"Monte Carlo distribution of Delta rho: {scenario}")
    ax.set_xlabel(r"$\widehat{\Delta}_\rho$")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{scenario}_delta_distribution.png", dpi=300)
    plt.close(fig)


def plot_parameter_boxplots(results: pd.DataFrame, scenario: str, outdir: Path) -> None:
    g = results[results["scenario"] == scenario]
    row0 = g.iloc[0]

    labels = [
        r"$\rho_W$",
        r"$\rho_S$",
        r"$\alpha_W$",
        r"$\alpha_S$",
    ]
    values = [
        g["rho_w_hat"].to_numpy(),
        g["rho_s_hat"].to_numpy(),
        g["alpha_w_hat"].to_numpy(),
        g["alpha_s_hat"].to_numpy(),
    ]
    true_vals = [
        row0["rho_w_true"],
        row0["rho_s_true"],
        row0["alpha_w_true"],
        row0["alpha_s_true"],
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(values, labels=labels, showfliers=False)
    for k, tv in enumerate(true_vals, start=1):
        ax.scatter(k, tv, marker="x", s=60, label="True value" if k == 1 else None)

    ax.set_title(f"Parameter recovery: {scenario}")
    ax.set_ylabel("Estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{scenario}_parameter_boxplots.png", dpi=300)
    plt.close(fig)


def plot_theta_curves(results: pd.DataFrame, scenario: str, dmax: float, outdir: Path) -> None:
    g = results[results["scenario"] == scenario]
    row0 = g.iloc[0]
    d_grid = np.linspace(0.0, dmax, 200)

    true_w = theta_br(d_grid, row0["rho_w_true"], row0["alpha_w_true"])
    true_s = theta_br(d_grid, row0["rho_s_true"], row0["alpha_s_true"])

    theta_w = np.vstack([theta_br(d_grid, r, a) for r, a in zip(g["rho_w_hat"], g["alpha_w_hat"])])
    theta_s = np.vstack([theta_br(d_grid, r, a) for r, a in zip(g["rho_s_hat"], g["alpha_s_hat"])])

    w_mean = np.nanmean(theta_w, axis=0)
    s_mean = np.nanmean(theta_s, axis=0)

    w_low, w_high = np.nanpercentile(theta_w, [5, 95], axis=0)
    s_low, s_high = np.nanpercentile(theta_s, [5, 95], axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(d_grid, true_w, linestyle="--", label="True winter")
    ax.plot(d_grid, true_s, linestyle="--", label="True summer")

    ax.plot(d_grid, w_mean, label="Mean estimated winter")
    ax.plot(d_grid, s_mean, label="Mean estimated summer")

    ax.fill_between(d_grid, w_low, w_high, alpha=0.2)
    ax.fill_between(d_grid, s_low, s_high, alpha=0.2)

    ax.set_title(f"Extremal-coefficient curves: {scenario}")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(r"$\theta(d)$")
    ax.set_ylim(1.0, 2.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{scenario}_theta_curves.png", dpi=300)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(summary))
    width = 0.35

    ax.bar(x - width / 2, summary["coverage_delta"], width, label="Delta coverage")
    ax.bar(x + width / 2, summary["rejection_rate"], width, label="Rejection rate")

    ax.axhline(0.95, linestyle="--", linewidth=1, label="0.95 reference")
    ax.axhline(0.05, linestyle=":", linewidth=1, label="0.05 reference")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Bootstrap coverage and rejection rates")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "coverage_rejection_summary.png", dpi=300)
    plt.close(fig)


def make_all_plots(results: pd.DataFrame, summary: pd.DataFrame, dmax: float, outdir: Path) -> None:
    for scenario in results["scenario"].unique():
        plot_delta_distribution(results, scenario, outdir)
        plot_parameter_boxplots(results, scenario, outdir)
        plot_theta_curves(results, scenario, dmax, outdir)

    plot_summary_bars(summary, outdir)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=str, required=True, help="CSV with station coordinates.")
    parser.add_argument("--out", type=str, default="outputs/simulation")
    parser.add_argument("--M", type=int, default=250)
    parser.add_argument("--B", type=int, default=500)
    parser.add_argument("--n-years", type=int, default=35)
    parser.add_argument("--days-per-season", type=int, default=90)
    parser.add_argument("--max-spectral-terms", type=int, default=800)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    config = SimConfig(
        M=args.M,
        B=args.B,
        n_years=args.n_years,
        days_per_season=args.days_per_season,
        max_spectral_terms=args.max_spectral_terms,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    if args.smoke_test:
        config.M = 2
        config.B = 5
        config.n_years = min(config.n_years, 5)
        config.days_per_season = min(config.days_per_season, 20)
        config.max_spectral_terms = min(config.max_spectral_terms, 150)
        print("Running smoke test with reduced settings.")

    station_df, coords, dist_mat = load_station_coordinates(args.stations)
    n_sites = coords.shape[0]
    pair_i, pair_j, d_pair = pair_information(n_sites, dist_mat)
    dmax = float(np.max(dist_mat))

    print(f"Loaded {n_sites} stations.")
    print(f"Number of station pairs: {len(d_pair)}.")
    print(f"dmax = {dmax:.3f} km.")
    print(f"M = {config.M}, B = {config.B}.")
    print("Warning: full simulation can be computationally expensive.")

    scenarios = scenario_definitions(dmax)

    all_results = []

    for scenario_name, scenario in scenarios.items():
        print(f"\nRunning scenario: {scenario_name}")
        print(scenario)

        scenario_results = run_scenario(
            scenario_name=scenario_name,
            scenario=scenario,
            coords=coords,
            pair_i=pair_i,
            pair_j=pair_j,
            d_pair=d_pair,
            dmax=dmax,
            config=config,
        )

        scenario_results.to_csv(outdir / f"{scenario_name}_mc_results.csv", index=False)
        all_results.append(scenario_results)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(outdir / "all_mc_results.csv", index=False)

    summary = summarize_results(results)
    summary.to_csv(outdir / "simulation_summary.csv", index=False)

    with open(outdir / "simulation_summary.tex", "w", encoding="utf-8") as f:
        f.write(
            summary.to_latex(
                index=False,
                float_format="%.3f",
                caption="Monte Carlo simulation summary for Brown--Resnick seasonal dependence estimation.",
                label="tab:simulation-summary",
            )
        )

    make_all_plots(results, summary, dmax, outdir)

    station_df.to_csv(outdir / "stations_used.csv", index=False)

    print("\nDone.")
    print(f"Outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
