# build_knmi_hourly_fh_parquet.py

from __future__ import annotations
import re
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_ROOT = Path(
    r"C:\Users\floris\Desktop\MSC\thesis_msc\data\hourly_data_validated"
    r"\hourly-in-situ-meteorological-observations-validated"
)

OUTPUT_DIR = Path(r"C:\Users\floris\Desktop\MSC\thesis_msc\data\yearly_aggregated_FH_FX")

EXTRACT_VARS = ["stationname", "lat", "lon", "height", "FH", "FX"]
OUTPUT_COLS = ["time", "station", "stationname", "lat", "lon", "height", "FH", "FX"]

START_YEAR = 1951
END_YEAR = 2026

# One process per month. After the NetCDF read the work is CPU-bound
# pandas/numpy code, so processes bypass the GIL and scale near-linearly.
MAX_WORKERS = 4
COMPRESSION = "snappy"


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "build_knmi_hourly_fh_parquet.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def find_files_for_month(year: int, month: int) -> list[Path]:
    month_dir = DATA_ROOT / f"{year:04d}" / f"{month:02d}"
    if not month_dir.exists():
        return []

    return sorted(month_dir.glob("hourly-observations-validated-*.nc"))


def month_parquet_path(year: int, month: int) -> Path:
    return OUTPUT_DIR / f"year={year:04d}" / f"month={month:02d}" / f"knmi_fh_{year:04d}_{month:02d}.parquet"


def clean_stationname(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return None
    return str(value).strip()


def timestamp_from_filename(path: Path) -> pd.Timestamp:

    match = re.search(r"(\d{8})-(\d{2})", path.name)
    if not match:
        return pd.NaT

    date_part, hour_part = match.groups()
    return pd.to_datetime(
        date_part + hour_part,
        format="%Y%m%d%H",
        errors="coerce",
    )


def make_nan_frame_for_file(path: Path, station_template: pd.DataFrame | None) -> pd.DataFrame:
    time_value = timestamp_from_filename(path)

    if station_template is None or station_template.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = station_template.copy()
    df["time"] = time_value
    df["FH"] = np.nan

    return df[OUTPUT_COLS]


def _broadcast_var(arr: np.ndarray, n_time: int, n_station: int) -> np.ndarray:
    """Expand a variable to shape (n_time * n_station,) by tile/repeat as needed."""
    if arr.ndim == 0:
        return np.repeat(arr, n_time * n_station)
    if arr.ndim == 1:
        if arr.size == n_station:
            return np.tile(arr, n_time)
        if arr.size == n_time:
            return np.repeat(arr, n_station)
        return np.broadcast_to(arr, (n_time, n_station)).ravel()
    # 2-D and above: assume (time, station, ...) and flatten the leading two axes.
    return arr.reshape(n_time * n_station, -1).squeeze()


def _read_time_values(ds: Dataset) -> np.ndarray:
    time_var = ds.variables["time"]
    raw_time = np.atleast_1d(time_var[:])

    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")

    if units is None:
        return pd.to_datetime(raw_time, errors="coerce").to_numpy()

    decoded = num2date(
        raw_time,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    return pd.to_datetime(decoded, errors="coerce").to_numpy()


def _read_nc_var(ds: Dataset, name: str) -> np.ndarray:
    arr = ds.variables[name][:]

    # Convert masked arrays to normal numpy arrays with NaN where possible.
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    return np.asarray(arr)


def read_one_netcdf(path: Path) -> pd.DataFrame:
    """
    Read one hourly NetCDF file and return one row per (time, station).
    Uses netCDF4.Dataset directly instead of xarray.open_dataset(), which avoids
    a lot of xarray metadata/index overhead for hundreds of thousands of files.
    """
    try:
        with Dataset(path, mode="r") as ds:
            missing = [v for v in EXTRACT_VARS if v not in ds.variables]
            if "time" not in ds.variables:
                missing.append("time")
            if "station" not in ds.variables:
                missing.append("station")

            if missing:
                raise ValueError(f"Missing variables {missing}")

            time_arr = np.atleast_1d(_read_time_values(ds))
            station_arr = np.atleast_1d(_read_nc_var(ds, "station"))

            n_time = time_arr.size
            n_station = station_arr.size

            time_col = np.repeat(time_arr, n_station)
            station_col = np.tile(station_arr, n_time).astype(str)

            cols = {
                "time": pd.to_datetime(time_col, errors="coerce"),
                "station": station_col,
            }

            for var in EXTRACT_VARS:
                arr = _read_nc_var(ds, var)
                cols[var] = _broadcast_var(arr, n_time, n_station)

            df = pd.DataFrame(cols)

            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["height"] = pd.to_numeric(df["height"], errors="coerce")
            df["FH"] = pd.to_numeric(df["FH"], errors="coerce")

            if df["stationname"].dtype == object:
                df["stationname"] = df["stationname"].map(clean_stationname)

            return df[OUTPUT_COLS]

    except Exception:
        logging.exception("Failed to read %s", path)
        return pd.DataFrame(columns=OUTPUT_COLS)


def write_month_parquet(year: int, month: int, df: pd.DataFrame) -> Path:
    out_file = month_parquet_path(year, month)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = out_file.with_suffix(".tmp.parquet")

    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_table(
        table,
        tmp_file,
        compression=COMPRESSION,
        use_dictionary=True,
        write_statistics=True,
    )

    tmp_file.replace(out_file)
    return out_file


def process_month_worker(task: tuple[int, int]) -> dict:
    year, month = task
    files = find_files_for_month(year, month)

    if not files:
        # No files at all for this month: write an empty parquet with the right schema.
        month_df = pd.DataFrame(columns=OUTPUT_COLS)
        out_file = write_month_parquet(year, month, month_df)

        return {
            "year": year,
            "month": month,
            "status": "ok_no_files",
            "rows": 0,
            "stations": 0,
            "files": 0,
            "failures": 0,
            "fh_missing": float("nan"),
            "file": str(out_file),
        }

    frames: list[pd.DataFrame] = []
    failed_paths: list[Path] = []
    failures = 0
    station_template: pd.DataFrame | None = None

    for path in files:
        df = read_one_netcdf(path)

        if df.empty:
            failures += 1
            failed_paths.append(path)
            continue

        if station_template is None:
            station_template = (
                df[["station", "stationname", "lat", "lon", "height"]]
                .drop_duplicates("station")
                .reset_index(drop=True)
            )

            # Backfill earlier failed files now that we know the station layout.
            for failed_path in failed_paths:
                nan_df = make_nan_frame_for_file(failed_path, station_template)
                if not nan_df.empty:
                    frames.append(nan_df)

            failed_paths.clear()

        frames.append(df)

    # Fill failed files that occurred after the first good file.
    if station_template is not None:
        for failed_path in failed_paths:
            nan_df = make_nan_frame_for_file(failed_path, station_template)
            if not nan_df.empty:
                frames.append(nan_df)

    if not frames:
        # Files existed, but none could be read.
        # Still write an empty parquet with the right columns.
        month_df = pd.DataFrame(columns=OUTPUT_COLS)
    else:
        month_df = pd.concat(frames, ignore_index=True)[OUTPUT_COLS]

    out_file = write_month_parquet(year, month, month_df)

    n_rows = len(month_df)
    n_stations = int(month_df["station"].nunique(dropna=True)) if n_rows else 0
    fh_missing = float(month_df["FH"].isna().mean()) if n_rows else float("nan")

    del frames, month_df
    gc.collect()

    return {
        "year": year,
        "month": month,
        "status": "ok",
        "rows": n_rows,
        "stations": n_stations,
        "files": len(files),
        "failures": failures,
        "fh_missing": fh_missing,
        "file": str(out_file),
    }


def main() -> None:
    setup_logging()

    logging.info("Starting KNMI NetCDF -> Parquet conversion")
    logging.info("DATA_ROOT=%s", DATA_ROOT)
    logging.info("OUTPUT_DIR=%s", OUTPUT_DIR)
    logging.info("MAX_WORKERS=%s", MAX_WORKERS)
    logging.info("NetCDF reader=netCDF4.Dataset")

    # Resume support: skip any month whose parquet file already exists.
    tasks: list[tuple[int, int]] = []
    skipped = 0
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            if month_parquet_path(year, month).exists():
                skipped += 1
                continue
            tasks.append((year, month))

    logging.info("Months to process: %s (skipping %s already-written)", len(tasks), skipped)

    if not tasks:
        logging.info("Nothing to do.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_month_worker, t): t for t in tasks}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="months"):
            task = futures[fut]
            try:
                result = fut.result()
            except Exception:
                logging.exception("Worker crashed for %s", task)
                continue

            if result["status"] in {"ok", "ok_no_files"}:
                logging.info(
                    "%04d-%02d %s | rows=%s | stations=%s | files=%s | failures=%s | FH_missing=%s",
                    result["year"],
                    result["month"],
                    result["status"],
                    result["rows"],
                    result["stations"],
                    result["files"],
                    result["failures"],
                    result["fh_missing"],
                )
            else:
                logging.warning(
                    "%04d-%02d %s | files=%s | failures=%s",
                    result["year"],
                    result["month"],
                    result["status"],
                    result.get("files", 0),
                    result.get("failures", 0),
                )

    logging.info("Finished conversion")


if __name__ == "__main__":
    df = read_one_netcdf(find_files_for_month(1951, 1)[0])
    print(df.head())
    print(df.dtypes)
    print(df["stationname"].head(20))
    main()
