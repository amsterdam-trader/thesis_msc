"""Load KNMI hourly wind-gust observations from the partitioned
parquet store.

The store lives at
``data/yearly_aggregated_FH_FX/year=YYYY/month=MM/knmi_fh_YYYY_MM.parquet``
and contains one row per (hour, station). Schema (audit-confirmed):

    time          datetime64[us]
    station       str    (WIGOS-style, e.g. "0-20000-0-06210")
    stationname   str
    lat, lon      float64
    height        float64
    FH            float64
    FX            float64

The functions below either operate on a single partition or stream
over a range without holding the whole archive in memory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd

import config


logger = config.get_logger(__name__)


# WIGOS-style station id format observed in the parquet, e.g.
# "0-20000-0-06210". The trailing 5-digit code matches station_metadata.csv.
_WIGOS_RE = re.compile(r".*?(\d{5})\s*$")


# ---------------------------------------------------------------------------
# Partition discovery
# ---------------------------------------------------------------------------

def parquet_path_for(year: int, month: int) -> Path:
    """Return the parquet path for one (year, month) partition.

    Does not check whether the file exists.
    """
    return (
        config.HOURLY_PARQUET_DIR
        / f"year={year}"
        / f"month={month:02d}"
        / f"knmi_fh_{year}_{month:02d}.parquet"
    )


def available_partitions() -> list[tuple[int, int]]:
    """List (year, month) partitions present on disk, sorted."""
    base = config.HOURLY_PARQUET_DIR
    out: list[tuple[int, int]] = []
    if not base.exists():
        logger.warning("Hourly parquet directory not found: %s", base)
        return out
    for year_dir in sorted(base.glob("year=*")):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            if not month_dir.is_dir():
                continue
            try:
                month = int(month_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            if parquet_path_for(year, month).exists():
                out.append((year, month))
    return out


def available_years() -> list[int]:
    """Years for which at least one monthly partition exists."""
    return sorted({y for y, _ in available_partitions()})


def missing_partitions(
    start_year: int, end_year: int,
    months: Iterable[int] | None = None,
) -> list[tuple[int, int]]:
    """List (year, month) partitions that are expected but absent."""
    present = set(available_partitions())
    months_to_check = tuple(months) if months is not None else tuple(range(1, 13))
    out: list[tuple[int, int]] = []
    for y in range(start_year, end_year + 1):
        for m in months_to_check:
            if (y, m) not in present:
                out.append((y, m))
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet file {path} is missing expected columns {missing}; "
            f"present columns are: {list(df.columns)}"
        )


def _post_process(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
    stations: Sequence[str] | None,
    normalize_station_ids: bool,
) -> pd.DataFrame:
    """Common normalisation step after reading raw parquet."""
    # Optional column projection.
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]

    # Ensure time is parsed.
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="raise")

    # Normalise station ids if requested.
    if "station" in df.columns and normalize_station_ids:
        df = df.assign(station=df["station"].map(station_id_to_5digit))

    # Optional station filter.
    if stations is not None and "station" in df.columns:
        wanted = set(stations)
        df = df[df["station"].isin(wanted)]

    return df.reset_index(drop=True)


def load_month(
    year: int,
    month: int,
    *,
    columns: Sequence[str] | None = None,
    stations: Sequence[str] | None = None,
    normalize_station_ids: bool = True,
) -> pd.DataFrame:
    """Load one monthly partition.

    Raises FileNotFoundError if the partition is absent.
    """
    path = parquet_path_for(year, month)
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet partition not on disk: {path}. "
            f"Use load_year_range(..., skip_missing=True) to ignore."
        )
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    _validate_columns(df, path) if columns is None else None
    return _post_process(df, columns, stations, normalize_station_ids)


def load_year(
    year: int,
    *,
    columns: Sequence[str] | None = None,
    stations: Sequence[str] | None = None,
    months: Iterable[int] | None = None,
    skip_missing: bool = True,
    normalize_station_ids: bool = True,
) -> pd.DataFrame:
    """Load every available monthly partition for a single calendar year."""
    months_t = tuple(months) if months is not None else tuple(range(1, 13))
    frames: list[pd.DataFrame] = []
    for m in months_t:
        path = parquet_path_for(year, m)
        if not path.exists():
            if skip_missing:
                logger.debug("skipping missing partition %s/%02d", year, m)
                continue
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, columns=list(columns) if columns else None)
        if columns is None:
            _validate_columns(df, path)
        frames.append(_post_process(df, columns, stations, normalize_station_ids))
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else list(config.EXPECTED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def load_year_range(
    start_year: int,
    end_year: int,
    *,
    columns: Sequence[str] | None = None,
    stations: Sequence[str] | None = None,
    months: Iterable[int] | None = None,
    skip_missing: bool = True,
    normalize_station_ids: bool = True,
) -> pd.DataFrame:
    """Load consecutive months in ``[start_year, end_year]``.

    By default skips missing partitions silently (e.g. June 2003).
    Always returns a single concatenated DataFrame. Use
    ``iter_partitions`` to stream year-by-year for low-memory pipelines.
    """
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} < start_year {start_year}")
    months_t = tuple(months) if months is not None else tuple(range(1, 13))
    frames: list[pd.DataFrame] = []
    n_loaded = 0
    n_skipped = 0
    for y in range(start_year, end_year + 1):
        for m in months_t:
            path = parquet_path_for(y, m)
            if not path.exists():
                if skip_missing:
                    n_skipped += 1
                    continue
                raise FileNotFoundError(path)
            df = pd.read_parquet(path, columns=list(columns) if columns else None)
            if columns is None:
                _validate_columns(df, path)
            frames.append(_post_process(df, columns, stations, normalize_station_ids))
            n_loaded += 1
    logger.info(
        "load_year_range(%d..%d): loaded %d partitions, skipped %d",
        start_year, end_year, n_loaded, n_skipped,
    )
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else list(config.EXPECTED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def iter_partitions(
    start_year: int | None = None,
    end_year: int | None = None,
    months: Iterable[int] | None = None,
    *,
    columns: Sequence[str] | None = None,
    stations: Sequence[str] | None = None,
    normalize_station_ids: bool = True,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield ``(year, month, df)`` for every available partition in range.

    Streams one partition at a time without holding more than one month
    in memory.
    """
    months_set = set(months) if months is not None else set(range(1, 13))
    for y, m in available_partitions():
        if start_year is not None and y < start_year:
            continue
        if end_year is not None and y > end_year:
            continue
        if m not in months_set:
            continue
        df = load_month(
            y, m,
            columns=columns,
            stations=stations,
            normalize_station_ids=normalize_station_ids,
        )
        yield y, m, df


# ---------------------------------------------------------------------------
# Station id normalisation
# ---------------------------------------------------------------------------

def station_id_to_5digit(s: str | None) -> str:
    """Normalise a WIGOS-style station id to its trailing 5-digit code.

    Example: ``"0-20000-0-06210" -> "06210"``. Already-5-digit
    strings pass through unchanged. Raises ``ValueError`` if no
    5-digit tail can be extracted.
    """
    if s is None:
        raise ValueError("station_id is None")
    s = str(s).strip()
    if s.isdigit() and len(s) == 5:
        return s
    m = _WIGOS_RE.match(s)
    if not m:
        raise ValueError(f"could not extract 5-digit station id from {s!r}")
    return m.group(1)
