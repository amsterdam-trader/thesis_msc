"""Build seasonal block maxima on the resolved main panel.

Reads the resolved balanced mainland panel from
``tables/balanced_panels_resolved.csv`` (produced by script 02), then
loads hourly FX for the main sample window plus December of the
year-before-start so winter (start) gets all three of its months, and
finally computes one block maximum per (station, season, season_year).

Block-level discards: a block is dropped when its value_share falls
below ``config.MIN_WITHIN_SEASON_COVERAGE`` (default 0.80). This is
the right place for the known summer 2003 gap (June 2003 missing) to
be removed.

Outputs:
    intermediate/seasonal_block_maxima.parquet
    tables/seasonal_block_maxima_summary.csv
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import pandas as pd

import config
import data_loading
import extremes


logger = config.get_logger(__name__)


def _load_resolved_panel(sample: str = "main") -> list[str]:
    path = config.TABLES_DIR / "balanced_panels_resolved.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/02_run_seasonal_sample_audit.py first."
        )
    table = pd.read_csv(path, dtype={"mainland_station_ids": str})
    row = table.loc[table["sample"] == sample]
    if row.empty:
        raise SystemExit(f"sample={sample!r} not present in {path}")
    ids = row.iloc[0]["mainland_station_ids"]
    if not isinstance(ids, str) or not ids.strip():
        raise SystemExit(f"empty mainland panel for sample={sample!r}")
    return [s for s in ids.split(",") if s]


def main() -> None:
    config.ensure_output_dirs()
    start = config.MAIN_SAMPLE_START_YEAR
    end = config.MAIN_SAMPLE_END_YEAR
    panel = _load_resolved_panel("main")
    logger.info("resolved main panel: %d stations", len(panel))

    logger.info(
        "loading hourly FX %d..%d (incl. Dec %d for winter %d), %d stations",
        start - 1, end, start - 1, start, len(panel),
    )
    df = data_loading.load_year_range(
        start - 1, end,
        columns=("time", "station", "FX"),
        stations=panel,
        skip_missing=True,
    )
    logger.info("rows loaded: %d", len(df))

    logger.info("computing seasonal block maxima (block discard threshold=%.2f)...",
                config.MIN_WITHIN_SEASON_COVERAGE)
    bmx = extremes.seasonal_block_maxima(df)

    # Restrict to the sample's season-years (drop summer (start-1) etc).
    bmx = bmx.loc[bmx["season_year"].between(start, end, inclusive="both")].reset_index(drop=True)
    logger.info("kept %d (station, season, season_year) blocks after window trim",
                len(bmx))

    extremes.save_block_maxima(bmx)

    # Per (station, season) summary.
    summary = (
        bmx.groupby(["station", "season"])
           .agg(
               n_blocks=("block_max", "size"),
               block_max_mean=("block_max", "mean"),
               block_max_median=("block_max", "median"),
               block_max_max=("block_max", "max"),
               first_season_year=("season_year", "min"),
               last_season_year=("season_year", "max"),
           )
           .reset_index()
    )
    summary.to_csv(
        config.TABLES_DIR / "seasonal_block_maxima_summary.csv", index=False,
    )
    logger.info("summary -> %s",
                config.TABLES_DIR / "seasonal_block_maxima_summary.csv")

    # Diagnostics: per-season counts.
    counts = bmx.groupby("season").size().to_dict()
    logger.info("block counts by season: %s", counts)


if __name__ == "__main__":
    main()
