"""Seasonal stratification with the DJF winter-year convention.

The DJF winter year is named by its January/February calendar year:
December of calendar year ``y`` is attached to winter year ``y+1``.
So winter 1967 = {Dec 1966, Jan 1967, Feb 1967}.

JJA summer year equals the calendar year.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

import config


logger = config.get_logger(__name__)

Season = Literal["W", "S"]
_SEASON_OTHER = "other"


# ---------------------------------------------------------------------------
# Per-row labels
# ---------------------------------------------------------------------------

def assign_season(timestamps: pd.Series) -> pd.Series:
    """Return a Series of seasonal labels in {"W", "S", "other"}.

    "W" = December, January, February.
    "S" = June, July, August.
    """
    ts = pd.to_datetime(timestamps)
    months = ts.dt.month
    label = pd.Series(_SEASON_OTHER, index=timestamps.index, dtype="object")
    label[months.isin(config.WINTER_MONTHS)] = "W"
    label[months.isin(config.SUMMER_MONTHS)] = "S"
    return label


def assign_winter_year(timestamps: pd.Series) -> pd.Series:
    """Return the winter-year label for each timestamp, ``Int64`` NA-aware.

    For December of calendar year y, returns y + 1.
    For January and February of calendar year y, returns y.
    For other months returns ``pd.NA``.
    """
    ts = pd.to_datetime(timestamps)
    year = ts.dt.year
    month = ts.dt.month
    wy = pd.Series(pd.NA, index=timestamps.index, dtype="Int64")
    is_dec = month == 12
    is_jf = month.isin([1, 2])
    wy.loc[is_dec] = (year[is_dec] + 1).astype("Int64")
    wy.loc[is_jf] = year[is_jf].astype("Int64")
    return wy


def assign_summer_year(timestamps: pd.Series) -> pd.Series:
    """Summer-year label (= calendar year) for JJA, NA otherwise."""
    ts = pd.to_datetime(timestamps)
    sy = pd.Series(pd.NA, index=timestamps.index, dtype="Int64")
    mask = ts.dt.month.isin(config.SUMMER_MONTHS)
    sy.loc[mask] = ts.dt.year[mask].astype("Int64")
    return sy


def assign_season_year(timestamps: pd.Series) -> pd.Series:
    """Single season-year label, NA for non-DJF, non-JJA hours."""
    w = assign_winter_year(timestamps)
    s = assign_summer_year(timestamps)
    return w.fillna(s)


# ---------------------------------------------------------------------------
# Expected hour counts (leap-aware)
# ---------------------------------------------------------------------------

def expected_hours_winter(winter_year: int) -> int:
    """Hours in DJF of the named winter year.

    winter_year = y means {Dec (y-1), Jan y, Feb y}. Feb has 29 days in
    a leap calendar year y.
    """
    feb = 29 if _is_leap(winter_year) else 28
    return 24 * (31 + 31 + feb)


def expected_hours_summer(summer_year: int) -> int:
    """Hours in JJA of summer year y, calendar year. Independent of
    leap year (June+July+August = 30+31+31)."""
    del summer_year
    return 24 * (30 + 31 + 31)


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


# ---------------------------------------------------------------------------
# Drop helpers
# ---------------------------------------------------------------------------

def keep_djf_jja(df: pd.DataFrame, *, time_col: str = "time") -> pd.DataFrame:
    """Drop rows whose timestamp is not in DJF or JJA."""
    labels = assign_season(df[time_col])
    return df.loc[labels != _SEASON_OTHER].reset_index(drop=True)


def split_by_season(
    df: pd.DataFrame, *, time_col: str = "time",
) -> dict[Season, pd.DataFrame]:
    """Split ``df`` into winter and summer DataFrames (other months dropped)."""
    labels = assign_season(df[time_col])
    return {
        "W": df.loc[labels == "W"].reset_index(drop=True),
        "S": df.loc[labels == "S"].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Coverage tables
# ---------------------------------------------------------------------------

def station_season_coverage(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    station_col: str = "station",
    value_col: str = "FX",
) -> pd.DataFrame:
    """Return (station, season, season_year) coverage of ``value_col``.

    Columns of the returned DataFrame:
        station, season, season_year,
        n_hours_present, n_value_present, n_hours_expected,
        present_share, value_share
    """
    if df.empty:
        return pd.DataFrame(columns=[
            station_col, "season", "season_year",
            "n_hours_present", "n_value_present",
            "n_hours_expected", "present_share", "value_share",
        ])

    work = df[[time_col, station_col, value_col]].copy()
    work["season"] = assign_season(work[time_col])
    work = work.loc[work["season"] != _SEASON_OTHER].copy()
    work["season_year"] = assign_season_year(work[time_col]).astype("Int64")

    grp = (
        work.groupby([station_col, "season", "season_year"], dropna=False)
            .agg(
                n_hours_present=(time_col, "size"),
                n_value_present=(value_col, lambda s: int(s.notna().sum())),
            )
            .reset_index()
    )
    grp["n_hours_expected"] = grp.apply(
        lambda r: (
            expected_hours_winter(int(r["season_year"]))
            if r["season"] == "W"
            else expected_hours_summer(int(r["season_year"]))
        ),
        axis=1,
    )
    grp["present_share"] = grp["n_hours_present"] / grp["n_hours_expected"]
    grp["value_share"] = grp["n_value_present"] / grp["n_hours_expected"]
    return grp


def filter_coverage(
    coverage: pd.DataFrame,
    *,
    min_value_share: float = config.MIN_WITHIN_SEASON_COVERAGE,
) -> pd.DataFrame:
    """Drop rows whose value-share falls below the threshold."""
    return coverage.loc[coverage["value_share"] >= min_value_share].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Inline self-checks (cheap, runs on import only if explicitly invoked)
# ---------------------------------------------------------------------------

def _self_check() -> None:
    """Assert-style sanity checks on the season-year logic."""
    ts = pd.to_datetime([
        "1966-12-01 00:00", "1966-12-31 23:00",   # belong to winter 1967
        "1967-01-15 12:00", "1967-02-28 23:00",   # belong to winter 1967
        "1967-06-15 00:00",                        # belongs to summer 1967
        "1967-10-15 00:00",                        # other
    ])
    s = pd.Series(ts)
    assert assign_winter_year(s).tolist() == [1967, 1967, 1967, 1967, pd.NA, pd.NA]
    assert assign_summer_year(s).tolist() == [pd.NA, pd.NA, pd.NA, pd.NA, 1967, pd.NA]
    assert assign_season(s).tolist() == ["W", "W", "W", "W", "S", "other"]
    # Feb 1968 is a leap February -> 29 days.
    assert expected_hours_winter(1968) == 24 * (31 + 31 + 29)
    assert expected_hours_winter(1967) == 24 * (31 + 31 + 28)
    assert expected_hours_summer(2000) == 24 * (30 + 31 + 31)


if __name__ == "__main__":
    _self_check()
    print("seasons._self_check passed.")
