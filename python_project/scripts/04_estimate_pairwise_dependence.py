"""Estimate pairwise chi_u (hourly) and theta (block maxima) per season.

Outputs:
    tables/pairwise_chi.csv     -- (station_i, station_j, u, season,
                                    chi_hat, n_joint, ...)
    tables/pairwise_theta.csv   -- (station_i, station_j, season,
                                    nu_F, theta_hat, chi_from_theta, ...)
    tables/station_pairs.csv    -- (station_i, station_j, distance_km, ...)

Bootstrap is intentionally NOT run here; this script produces point
estimates only. The bootstrap is a separate driver in
``year_resample_bootstrap_chi`` to keep this script fast.
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import pandas as pd

import config
import data_loading
import dependence_estimators
import extremes
import station_metadata
import station_pairs


logger = config.get_logger(__name__)


def main() -> None:
    config.ensure_output_dirs()
    start = config.MAIN_SAMPLE_START_YEAR
    end = config.MAIN_SAMPLE_END_YEAR

    # 1. Block maxima already saved by script 03; reload.
    bmx_path = config.INTERMEDIATE_DIR / config.OUTPUTS.seasonal_block_maxima_parquet
    if not bmx_path.exists():
        raise SystemExit(
            f"Block-maxima parquet not found at {bmx_path}. "
            "Run scripts/03_build_seasonal_extremes.py first."
        )
    bmx = pd.read_parquet(bmx_path)
    panel = sorted(bmx["station"].unique().tolist())
    logger.info("panel: %d stations from block-maxima file", len(panel))

    # 2. Station pair table (haversine distances).
    meta = station_metadata.get_station_metadata(use_cache=True)
    if "station_type" not in meta.columns:
        meta["station_type"] = station_metadata.classify_station_type(meta)
    panel_meta = meta.loc[meta["station_id"].isin(panel)].reset_index(drop=True)
    pairs = station_pairs.all_pairs(panel_meta)
    pairs.to_csv(
        config.TABLES_DIR / config.OUTPUTS.station_pairs_csv, index=False,
    )
    logger.info("pairs: %d (saved to %s)", len(pairs),
                config.OUTPUTS.station_pairs_csv)

    # 3. theta via F-madogram on block maxima per season.
    theta_rows: list[pd.DataFrame] = []
    for season in ("W", "S"):
        Mwide, Fwide = dependence_estimators.block_maxima_wide_FU(
            bmx, season=season,
        )
        tbl = dependence_estimators.theta_all_pairs(Fwide)
        tbl["season"] = season
        theta_rows.append(tbl)
    theta_all = pd.concat(theta_rows, ignore_index=True)
    theta_all.to_csv(
        config.TABLES_DIR / config.OUTPUTS.pairwise_theta_csv, index=False,
    )
    logger.info("theta table: %d rows -> %s", len(theta_all),
                config.OUTPUTS.pairwise_theta_csv)

    # 4. chi_u on hourly rank-transformed series per season.
    logger.info("loading hourly FX for rank transform...")
    df = data_loading.load_year_range(
        start, end, columns=("time", "station", "FX"),
        stations=panel, skip_missing=True,
    )
    logger.info("applying empirical rank transform per (station, season)...")
    rank_df = extremes.rank_transform_per_station_season(df)

    chi_rows: list[pd.DataFrame] = []
    for season in ("W", "S"):
        U_wide = dependence_estimators.hourly_wide_U(rank_df, season=season)
        for u in config.DEFAULT_TAIL_LEVELS:
            tbl = dependence_estimators.chi_all_pairs(U_wide, u=u)
            tbl["season"] = season
            chi_rows.append(tbl)
    chi_all = pd.concat(chi_rows, ignore_index=True)
    chi_all.to_csv(
        config.TABLES_DIR / config.OUTPUTS.pairwise_chi_csv, index=False,
    )
    logger.info("chi table: %d rows -> %s", len(chi_all),
                config.OUTPUTS.pairwise_chi_csv)


if __name__ == "__main__":
    main()
