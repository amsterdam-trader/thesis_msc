"""Station metadata extraction and offshore/mainland classification.

The single source of truth for station metadata is the parquet store
itself, where every monthly partition reports
``station, stationname, lat, lon, height`` for each station that
recorded that month. We aggregate these per (station_id) and
reconcile inconsistencies (e.g. a station was relocated) by taking
the most-recent non-null value.

The companion CSV at ``data/station_metadata.csv`` is read only as a
secondary check, since its contents are not always aligned with the
station id form used in the parquet.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

import config
import data_loading


logger = config.get_logger(__name__)

StationType = Literal["mainland", "offshore", "outside-NL", "unknown"]


# ---------------------------------------------------------------------------
# Extraction from the parquet store
# ---------------------------------------------------------------------------

def get_station_metadata(
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build the station metadata table from the parquet partitions.

    For each unique 5-digit ``station_id`` we report:
        stationname, lat, lon, height,
        first_seen, last_seen, n_partitions

    "Stationname / lat / lon / height" are taken from the most recent
    partition in which that station appears; "first_seen" / "last_seen"
    are the earliest and latest partition (year, month) months
    (formatted YYYY-MM).
    """
    cache = config.TABLES_DIR / config.OUTPUTS.station_metadata_csv
    if use_cache and cache.exists():
        logger.info("loading cached station metadata: %s", cache)
        df = pd.read_csv(cache, dtype={"station_id": str})
        return df

    records: list[pd.DataFrame] = []
    for year, month, df in data_loading.iter_partitions(
        start_year=start_year, end_year=end_year,
        columns=("time", "station", "stationname", "lat", "lon", "height"),
    ):
        meta = df.groupby("station", as_index=False).agg(
            stationname=("stationname", "last"),
            lat=("lat", "last"),
            lon=("lon", "last"),
            height=("height", "last"),
        )
        meta["year"] = year
        meta["month"] = month
        records.append(meta)

    if not records:
        return pd.DataFrame(columns=[
            "station_id", "stationname", "lat", "lon", "height",
            "first_seen", "last_seen", "n_partitions",
        ])

    long = pd.concat(records, ignore_index=True)
    long = long.rename(columns={"station": "station_id"})
    # Use most-recent observation for each (lat, lon, height, stationname).
    long = long.sort_values(["station_id", "year", "month"])
    most_recent = long.groupby("station_id", as_index=False).agg(
        stationname=("stationname", "last"),
        lat=("lat", "last"),
        lon=("lon", "last"),
        height=("height", "last"),
    )
    span = long.groupby("station_id").agg(
        first_year=("year", "min"),
        first_month=("month", "min"),
        last_year=("year", "max"),
        last_month=("month", "max"),
        n_partitions=("year", "size"),
    )
    span["first_seen"] = span.apply(
        lambda r: f"{int(r['first_year']):04d}-{int(r['first_month']):02d}", axis=1,
    )
    span["last_seen"] = span.apply(
        lambda r: f"{int(r['last_year']):04d}-{int(r['last_month']):02d}", axis=1,
    )
    span = span[["first_seen", "last_seen", "n_partitions"]].reset_index()
    out = most_recent.merge(span, on="station_id", how="left").sort_values("station_id")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_station_type(
    meta: pd.DataFrame,
    *,
    bbox: dict[str, float] | None = None,
    offshore_keywords: Iterable[str] | None = None,
) -> pd.Series:
    """Return a Series of ``StationType`` labels per row.

    A station is labelled:
      * ``"offshore"`` if its ``stationname`` matches any of the
        configured offshore keywords;
      * ``"outside-NL"`` if its (lat, lon) falls outside the
        configured Dutch bounding box (and it is not labelled
        offshore);
      * ``"mainland"`` otherwise, provided (lat, lon) are available;
      * ``"unknown"`` if (lat, lon) are missing.
    """
    bbox = bbox or config.NL_BOUNDING_BOX
    kws = tuple(kw.upper() for kw in (offshore_keywords or config.OFFSHORE_STATIONNAME_KEYWORDS))

    name_upper = meta["stationname"].fillna("").str.upper()
    is_offshore = name_upper.apply(lambda s: any(k in s for k in kws))

    lat = pd.to_numeric(meta["lat"], errors="coerce")
    lon = pd.to_numeric(meta["lon"], errors="coerce")
    in_box = (
        lat.between(bbox["lat_min"], bbox["lat_max"]) &
        lon.between(bbox["lon_min"], bbox["lon_max"])
    )

    out = pd.Series("unknown", index=meta.index, dtype="object")
    valid = lat.notna() & lon.notna()
    out.loc[valid & in_box] = "mainland"
    out.loc[valid & ~in_box] = "outside-NL"
    out.loc[is_offshore] = "offshore"
    return out


def filter_mainland_stations(
    meta: pd.DataFrame,
    *,
    bbox: dict[str, float] | None = None,
    offshore_keywords: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Restrict ``meta`` to stations classified as mainland.

    Does not silently delete; logs the count per category before
    filtering.
    """
    meta = meta.copy()
    meta["station_type"] = classify_station_type(
        meta, bbox=bbox, offshore_keywords=offshore_keywords,
    )
    counts = meta["station_type"].value_counts().to_dict()
    logger.info("station_type counts: %s", counts)
    mainland = meta.loc[meta["station_type"] == "mainland"].reset_index(drop=True)
    return mainland


# ---------------------------------------------------------------------------
# Persist artefact
# ---------------------------------------------------------------------------

def save_station_metadata(
    meta: pd.DataFrame,
    *,
    with_classification: bool = True,
    path: str | None = None,
) -> str:
    """Save the station metadata table to ``outputs/tables/``."""
    out = meta.copy()
    if with_classification and "station_type" not in out.columns:
        out["station_type"] = classify_station_type(out)
    target = path or str(config.TABLES_DIR / config.OUTPUTS.station_metadata_csv)
    config.ensure_output_dirs()
    out.to_csv(target, index=False)
    logger.info("station metadata -> %s (rows=%d)", target, len(out))
    return target


# ---------------------------------------------------------------------------
# Secondary check against the CSV companion
# ---------------------------------------------------------------------------

def read_companion_csv() -> pd.DataFrame:
    """Read ``data/station_metadata.csv`` as a side reference.

    The companion CSV uses the bare 5-digit form of station ids and may
    cover stations not present in the parquet (and vice versa). It is
    *not* the authoritative source.
    """
    p = config.STATION_METADATA_CSV
    if not p.exists():
        logger.warning("companion CSV not found: %s", p)
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"station_id": str})
    return df
