"""Seasonal extreme-value construction from hourly FX.

Two routes are supported:
  * Seasonal block maxima (default, main route in methodology.tex):
        M_{i,b}^{(c)} = max FX over season c of year b at station i,
        with coverage tracked alongside.
  * Threshold exceedances (robustness route):
        Y_{i,t}^{(c)} = FX - u_i^{(c)} for FX > u_i^{(c)},
        with the threshold u_i^{(c)} a fixed empirical quantile per
        (station, season).

The empirical-rank transform is also provided here because it is the
marginal standardisation used by the dependence estimators on the
hourly series.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

import config
import seasons


logger = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Block maxima
# ---------------------------------------------------------------------------

def seasonal_block_maxima(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    station_col: str = "station",
    value_col: str = "FX",
    min_value_share: float | None = None,
) -> pd.DataFrame:
    """Compute one seasonal block maximum per (station, season, season_year).

    Returns a tidy DataFrame with columns:
        station, season, season_year, block_max,
        n_hours_present, n_value_present, n_hours_expected,
        present_share, value_share, time_of_max

    Blocks whose ``value_share`` falls below ``min_value_share`` are
    dropped (default = ``config.MIN_WITHIN_SEASON_COVERAGE``).
    """
    if min_value_share is None:
        min_value_share = config.MIN_WITHIN_SEASON_COVERAGE

    work = df[[time_col, station_col, value_col]].copy()
    work["season"] = seasons.assign_season(work[time_col])
    work = work.loc[work["season"] != "other"].copy()
    work["season_year"] = seasons.assign_season_year(work[time_col]).astype("Int64")
    work = work.dropna(subset=["season_year"])

    # Coverage table per (station, season, season_year).
    coverage = seasons.station_season_coverage(
        df, time_col=time_col, station_col=station_col, value_col=value_col,
    )

    # Block max + time of max.
    grouped = work.groupby([station_col, "season", "season_year"], dropna=False)
    block_max = grouped[value_col].max().rename("block_max")
    idx_max = grouped[value_col].idxmax()
    time_of_max = work.loc[idx_max.dropna()].set_index(
        [station_col, "season", "season_year"]
    )[time_col].rename("time_of_max")

    out = pd.concat([block_max, time_of_max], axis=1).reset_index()
    out = out.merge(coverage, on=[station_col, "season", "season_year"], how="left")

    before = len(out)
    out = out.loc[out["value_share"] >= min_value_share].reset_index(drop=True)
    logger.info(
        "seasonal_block_maxima: kept %d / %d blocks (coverage >= %.2f)",
        len(out), before, min_value_share,
    )
    return out


# ---------------------------------------------------------------------------
# Threshold exceedances
# ---------------------------------------------------------------------------

def station_season_thresholds(
    df: pd.DataFrame,
    *,
    quantile: float = config.DEFAULT_TAIL_LEVELS[-1],
    time_col: str = "time",
    station_col: str = "station",
    value_col: str = "FX",
) -> pd.DataFrame:
    """Empirical (station, season) threshold u_i^{(c)} at given quantile.

    Returns columns: station, season, quantile, threshold.
    """
    work = df[[time_col, station_col, value_col]].copy()
    work["season"] = seasons.assign_season(work[time_col])
    work = work.loc[work["season"] != "other"]
    rows = []
    for (station, season), grp in work.groupby([station_col, "season"], observed=True):
        u = grp[value_col].quantile(quantile)
        rows.append({
            station_col: station,
            "season": season,
            "quantile": quantile,
            "threshold": float(u) if pd.notna(u) else float("nan"),
        })
    return pd.DataFrame(rows)


def threshold_exceedances(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    time_col: str = "time",
    station_col: str = "station",
    value_col: str = "FX",
) -> pd.DataFrame:
    """Return exceedances (FX > threshold) using per-(station, season) u.

    ``thresholds`` is the DataFrame returned by
    ``station_season_thresholds``. Output columns:
        time, station, season, season_year, value, threshold, excess.
    """
    work = df[[time_col, station_col, value_col]].copy()
    work["season"] = seasons.assign_season(work[time_col])
    work = work.loc[work["season"] != "other"].copy()
    work["season_year"] = seasons.assign_season_year(work[time_col]).astype("Int64")
    merged = work.merge(
        thresholds[[station_col, "season", "threshold"]],
        on=[station_col, "season"], how="left",
    )
    mask = merged[value_col] > merged["threshold"]
    out = merged.loc[mask, [
        time_col, station_col, "season", "season_year",
        value_col, "threshold",
    ]].copy()
    out = out.rename(columns={value_col: "value"})
    out["excess"] = out["value"] - out["threshold"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Empirical-rank transform
# ---------------------------------------------------------------------------

def empirical_rank_transform(
    series: pd.Series,
    *,
    ties: str = "average",
) -> pd.Series:
    """Empirical-CDF transform of ``series`` to (0, 1).

    Returns ``rank(method=ties) / (n_non_null + 1)`` per Genest's
    convention. NaN inputs propagate to NaN outputs.
    """
    n = int(series.notna().sum())
    if n == 0:
        return series.astype(float).copy()
    return series.rank(method=ties) / (n + 1.0)


def rank_transform_per_station_season(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    station_col: str = "station",
    value_col: str = "FX",
) -> pd.DataFrame:
    """Apply the empirical-CDF transform within each (station, season).

    Returns the input with two new columns:
        season  (str: "W", "S", "other"),
        U       (float in (0, 1) within each (station, season)).
    """
    out = df[[time_col, station_col, value_col]].copy()
    out["season"] = seasons.assign_season(out[time_col])
    out["U"] = np.nan
    for (station, season), idx in out.groupby([station_col, "season"], observed=True).groups.items():
        if season == "other":
            continue
        out.loc[idx, "U"] = empirical_rank_transform(out.loc[idx, value_col])
    return out


# ---------------------------------------------------------------------------
# Convenience saver
# ---------------------------------------------------------------------------

def save_block_maxima(
    bmx: pd.DataFrame,
    path: str | None = None,
) -> str:
    """Persist a seasonal-block-maxima table to parquet."""
    config.ensure_output_dirs()
    target = path or str(
        config.INTERMEDIATE_DIR / config.OUTPUTS.seasonal_block_maxima_parquet
    )
    bmx.to_parquet(target, index=False)
    logger.info("block maxima -> %s (rows=%d)", target, len(bmx))
    return target
