"""Produce the headline figures for the thesis empirical chapter.

Reads outputs of scripts 01-04 from disk and writes PDFs into
``python_project/outputs/figures/``.

This script makes only point-estimate figures (no bootstrap bands).
The bootstrap is intentionally separated from this script.
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import pandas as pd

import config
import plotting
import station_metadata


logger = config.get_logger(__name__)


def main() -> None:
    config.ensure_output_dirs()
    fig_dir = config.FIGURES_DIR

    # 1. Station map.
    meta = station_metadata.get_station_metadata(use_cache=True)
    if "station_type" not in meta.columns:
        meta["station_type"] = station_metadata.classify_station_type(meta)
    fig = plotting.station_map(meta)
    plotting.save_fig(fig, config.OUTPUTS.station_map_pdf)

    # 2. Pairwise chi vs distance, winter / summer.
    pairs_path = config.TABLES_DIR / config.OUTPUTS.station_pairs_csv
    chi_path   = config.TABLES_DIR / config.OUTPUTS.pairwise_chi_csv
    theta_path = config.TABLES_DIR / config.OUTPUTS.pairwise_theta_csv
    if not (pairs_path.exists() and chi_path.exists() and theta_path.exists()):
        raise SystemExit(
            "Missing one of pairs/chi/theta tables. "
            "Run scripts/04_estimate_pairwise_dependence.py first."
        )
    pairs = pd.read_csv(pairs_path, dtype={"station_i": str, "station_j": str})
    chi = pd.read_csv(chi_path, dtype={"station_i": str, "station_j": str})
    theta = pd.read_csv(theta_path, dtype={"station_i": str, "station_j": str})

    # Restrict to the reference u for the headline panel.
    u0 = config.DEFAULT_REFERENCE_TAIL_LEVEL
    chi_u = chi.loc[chi["u"] == u0]
    chi_w = chi_u.loc[chi_u["season"] == "W"]
    chi_s = chi_u.loc[chi_u["season"] == "S"]
    fig = plotting.chi_vs_distance_winter_summer(chi_w, chi_s, pairs)
    plotting.save_fig(fig, config.OUTPUTS.chi_vs_distance_pdf)

    # 3. Theta vs distance (block-maxima route).
    theta_w = theta.loc[theta["season"] == "W"].rename(columns={"theta_hat": "chi_hat"})
    theta_s = theta.loc[theta["season"] == "S"].rename(columns={"theta_hat": "chi_hat"})
    fig = plotting.chi_vs_distance_winter_summer(
        theta_w, theta_s, pairs, value_col="chi_hat",
    )
    plotting.save_fig(fig, config.OUTPUTS.theta_vs_distance_pdf)

    # 4. Winter - summer chi difference.
    import dependence_estimators
    delta = dependence_estimators.seasonal_difference(chi_w, chi_s)
    fig = plotting.chi_difference_vs_distance(delta, pairs)
    plotting.save_fig(fig, config.OUTPUTS.chi_diff_pdf)
    logger.info("done.")


if __name__ == "__main__":
    main()
