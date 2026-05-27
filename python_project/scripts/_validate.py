"""Lightweight pipeline validation.

Imports every src module, runs cheap checks on one month of data,
confirms station-id normalisation, DJF season-year assignment, pair
distance computation, block maxima on a small subset, and the
empirical-rank transform. Prints PASS/FAIL per check.

Designed to run in <30 seconds.
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import math
import sys
import traceback

import numpy as np
import pandas as pd


def check(label: str):
    def deco(fn):
        def wrap():
            try:
                fn()
                print(f"PASS  {label}")
            except Exception as e:
                print(f"FAIL  {label}: {e}")
                traceback.print_exc(limit=2, file=sys.stdout)
                wrap.failed = True
        wrap.failed = False
        return wrap
    return deco


@check("imports")
def _imports():
    import config  # noqa: F401
    import data_loading  # noqa: F401
    import seasons  # noqa: F401
    import station_metadata  # noqa: F401
    import station_pairs  # noqa: F401
    import extremes  # noqa: F401
    import dependence_estimators  # noqa: F401
    import plotting  # noqa: F401


@check("config sanity")
def _config():
    import config
    assert config.PRIMARY_VARIABLE == "FX"
    assert config.MAIN_SAMPLE_START_YEAR <= config.MAIN_SAMPLE_END_YEAR
    assert 0.0 < config.DEFAULT_REFERENCE_TAIL_LEVEL < 1.0
    assert config.HOURLY_PARQUET_DIR.exists()


@check("station_id_to_5digit")
def _wigos():
    from data_loading import station_id_to_5digit
    assert station_id_to_5digit("0-20000-0-06210") == "06210"
    assert station_id_to_5digit("06210") == "06210"
    try:
        station_id_to_5digit("abc")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on 'abc'")


@check("DJF winter-year assignment")
def _djf():
    import seasons as S
    ts = pd.to_datetime([
        "1966-12-15 12:00",   # winter 1967
        "1967-01-15 12:00",   # winter 1967
        "1967-02-28 12:00",   # winter 1967
        "1967-06-15 12:00",   # summer 1967
        "1967-10-15 12:00",   # other
    ])
    s = pd.Series(ts)
    assert S.assign_season(s).tolist() == ["W", "W", "W", "S", "other"]
    wy = S.assign_winter_year(s).tolist()
    assert wy[:3] == [1967, 1967, 1967]
    assert pd.isna(wy[3]) and pd.isna(wy[4])
    assert S.expected_hours_winter(1968) == 24 * (31 + 31 + 29)
    assert S.expected_hours_winter(1967) == 24 * (31 + 31 + 28)


@check("haversine + pair table")
def _pairs():
    from station_pairs import haversine_km, all_pairs
    # Amsterdam Schiphol vs De Bilt, ~40 km
    d = float(haversine_km(52.30, 4.78, 52.10, 5.18))
    assert 30.0 < d < 60.0, f"unexpected haversine distance {d}"
    meta = pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "lat": [52.3, 52.1, 53.0],
        "lon": [4.8, 5.2, 4.7],
    })
    pairs = all_pairs(meta)
    assert len(pairs) == 3
    assert (pairs["distance_km"] > 0).all()


@check("load one month + schema")
def _load_one_month():
    import data_loading
    parts = data_loading.available_partitions()
    if not parts:
        raise AssertionError("no partitions on disk")
    y, m = parts[0]
    df = data_loading.load_month(y, m, columns=("time", "station", "FX"))
    assert "station" in df.columns
    # IDs should have been normalised to 5 digits.
    sample = df["station"].head(5).tolist()
    for s in sample:
        assert s.isdigit() and len(s) == 5, f"non-normalised id {s!r}"


@check("seasonal_block_maxima on subset")
def _bmx():
    import data_loading
    import extremes
    # Use one recent year that the audit shows has good coverage.
    df = data_loading.load_year(
        2010, columns=("time", "station", "FX"), skip_missing=True,
    )
    if df.empty:
        raise AssertionError("year 2010 has no data")
    # Pick 3 stations with most observations to keep this fast.
    top3 = (df.dropna(subset=["FX"])["station"]
              .value_counts().head(3).index.tolist())
    df = df.loc[df["station"].isin(top3)]
    bmx = extremes.seasonal_block_maxima(df, min_value_share=0.0)
    # We expect 1 winter + 1 summer per station for 2010, but winter 2010
    # only has Jan+Feb of 2010 from this single-year load (no Dec 2009),
    # so its expected hours -> partial. We just check that the function
    # runs and returns the correct columns.
    expected_cols = {"station", "season", "season_year", "block_max",
                     "value_share"}
    assert expected_cols.issubset(bmx.columns)


@check("rank transform")
def _rank():
    import extremes
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    u = extremes.empirical_rank_transform(s)
    finite = u.dropna()
    assert ((finite > 0) & (finite < 1)).all()


@check("chi_at_level_pair")
def _chi():
    import dependence_estimators as D
    rng = np.random.default_rng(0)
    u = pd.Series(rng.uniform(0, 1, 1000), name="A")
    v = pd.Series(rng.uniform(0, 1, 1000), name="B")
    r = D.chi_at_level_pair(u, v, 0.95)
    # Independent uniforms: chi_u at 0.95 should sit near 0.05.
    assert 0.0 <= r.chi_hat <= 0.5


def main() -> int:
    checks = [_imports, _config, _wigos, _djf, _pairs,
              _load_one_month, _bmx, _rank, _chi]
    for c in checks:
        c()
    any_failed = any(getattr(c, "failed", False) for c in checks)
    print()
    print("ALL CHECKS PASSED" if not any_failed else "SOME CHECKS FAILED")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
