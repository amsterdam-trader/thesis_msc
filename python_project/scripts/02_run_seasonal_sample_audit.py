"""Seasonal audit: DJF/JJA coverage diagnostic + mainland balanced panel.

This script resolves the balanced panel for the main and robustness
samples using the SAME annual-coverage criterion as script 01 (so
the balanced-panel counts here match audit_balanced_panels.csv), then
applies the mainland filter and writes a seasonal-coverage diagnostic.

Important design point: the seasonal `value_share` (FX non-null share
within DJF/JJA) is reported as a DIAGNOSTIC, not as a filter. Filtering
on every (station, season, season_year) is brittle because:

  * Loading sample [start, end] omits Dec (start-1), which truncates
    winter (start) artificially.
  * One bad season-year (e.g. a station downtime month) can knock out
    an otherwise-clean station.

We therefore resolve the panel using the per-year coverage table from
script 01, which aggregates winter and summer together and is robust
to these effects. Seasonal coverage is still computed and saved so the
methodology section can report it.

Outputs:
    tables/seasonal_coverage.csv
    tables/station_metadata_resolved.csv
    tables/balanced_panels_resolved.csv
    figures/seasonal_coverage.pdf
    outputs/seasonal_audit_report.md
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import pandas as pd

import config
import data_loading
import plotting
import seasons
import station_metadata


logger = config.get_logger(__name__)


def resolve_balanced_panel(
    coverage_pivot: pd.DataFrame,
    start_year: int,
    end_year: int,
    threshold: float,
) -> list[str]:
    """Stations with annual fx_present_share >= threshold in every year
    of [start_year, end_year]. Coverage_pivot has int-typed year columns.
    """
    cols = [y for y in coverage_pivot.columns if start_year <= int(y) <= end_year]
    if not cols:
        return []
    sub = coverage_pivot[cols]
    mask = (sub >= threshold).all(axis=1) & sub.notna().all(axis=1)
    return [str(s) for s in coverage_pivot.index[mask].tolist()]


def main() -> None:
    config.ensure_output_dirs()

    # 1. Station metadata + classification.
    logger.info("extracting station metadata...")
    meta = station_metadata.get_station_metadata(use_cache=False)
    meta["station_type"] = station_metadata.classify_station_type(meta)
    station_metadata.save_station_metadata(meta, with_classification=True)
    logger.info(
        "station_type counts: %s",
        meta["station_type"].value_counts().to_dict(),
    )

    # 2. Reuse the audit per-year coverage pivot.
    pivot_path = config.TABLES_DIR / config.OUTPUTS.audit_station_coverage_pivot_csv
    if not pivot_path.exists():
        raise SystemExit(
            f"{pivot_path} not found. Run scripts/01_run_data_audit.py first."
        )
    pv = pd.read_csv(pivot_path, dtype={"station": str}).set_index("station")
    pv.columns = pv.columns.astype(int)

    # 3. Resolve balanced panels for main and robustness windows.
    thr = config.DEFAULT_COVERAGE_THRESHOLD
    mainland_ids = set(meta.loc[meta["station_type"] == "mainland", "station_id"])

    samples = {
        "main": (config.MAIN_SAMPLE_START_YEAR, config.MAIN_SAMPLE_END_YEAR),
        "robustness": (
            config.ROBUSTNESS_SAMPLE_START_YEAR,
            config.ROBUSTNESS_SAMPLE_END_YEAR,
        ),
    }
    sample_panels: dict[str, dict] = {}
    for label, (start, end) in samples.items():
        panel = resolve_balanced_panel(pv, start, end, thr)
        panel_mainland = [s for s in panel if s in mainland_ids]
        sample_panels[label] = {
            "start": start, "end": end, "threshold": thr,
            "all_balanced": panel,
            "mainland_balanced": panel_mainland,
        }
        logger.info(
            "%s sample (%d..%d, thr=%.2f): %d balanced (%d after mainland filter)",
            label, start, end, thr, len(panel), len(panel_mainland),
        )

    # 4. Persist resolved panels for downstream scripts.
    rows = []
    for label, info in sample_panels.items():
        rows.append({
            "sample": label,
            "start_year": info["start"],
            "end_year": info["end"],
            "threshold": info["threshold"],
            "n_all_balanced": len(info["all_balanced"]),
            "n_mainland_balanced": len(info["mainland_balanced"]),
            "mainland_station_ids": ",".join(info["mainland_balanced"]),
        })
    pd.DataFrame(rows).to_csv(
        config.TABLES_DIR / "balanced_panels_resolved.csv", index=False,
    )

    # 5. Seasonal coverage DIAGNOSTIC for the main sample.
    #    Load (start-1) to end so winter (start) gets its Dec (start-1).
    start, end = config.MAIN_SAMPLE_START_YEAR, config.MAIN_SAMPLE_END_YEAR
    logger.info(
        "loading hourly FX %d..%d (incl. Dec %d for winter %d)",
        start - 1, end, start - 1, start,
    )
    df = data_loading.load_year_range(
        start - 1, end,
        columns=("time", "station", "FX"),
        skip_missing=True,
    )
    logger.info("rows loaded: %d", len(df))

    coverage = seasons.station_season_coverage(df)
    # Trim to the sample's season-years (drop summer (start-1) and any
    # boundary winters outside [start, end]).
    coverage = coverage.loc[
        coverage["season_year"].between(start, end, inclusive="both")
    ].reset_index(drop=True)
    coverage = coverage.merge(
        meta[["station_id", "stationname", "station_type"]],
        left_on="station", right_on="station_id", how="left",
    )
    coverage.to_csv(
        config.TABLES_DIR / config.OUTPUTS.seasonal_coverage_csv, index=False,
    )

    # Diagnostic: per (station, season) median value_share and worst year.
    diag = (
        coverage.groupby(["station", "season"])
                .agg(
                    median_value_share=("value_share", "median"),
                    min_value_share=("value_share", "min"),
                    n_season_years=("season_year", "size"),
                )
                .reset_index()
    )

    # 6. Figure.
    fig = plotting.seasonal_coverage_panel(coverage)
    plotting.save_fig(fig, config.OUTPUTS.seasonal_coverage_pdf)

    # 7. Report.
    lines: list[str] = []
    a = lines.append
    a("# Seasonal audit report")
    a("")
    a("Generated by `python_project/scripts/02_run_seasonal_sample_audit.py`.")
    a("")
    a("## Station-type classification")
    a("")
    for k, v in meta["station_type"].value_counts().to_dict().items():
        a(f"- **{k}**: {v}")
    a("")
    a("## Balanced panels")
    a("")
    a("Resolution rule: per-year fx_present_share >= "
      f"{config.DEFAULT_COVERAGE_THRESHOLD:.2f} in every year of the "
      "sample window (same criterion as script 01), then restrict to "
      "stations classified as mainland.")
    a("")
    for label, info in sample_panels.items():
        a(f"### `{label}` sample")
        a("")
        a(f"- Window: {info['start']}-{info['end']} "
          f"({info['end'] - info['start'] + 1} years).")
        a(f"- All balanced stations: **{len(info['all_balanced'])}**.")
        a(f"- After mainland filter: **{len(info['mainland_balanced'])}**.")
        a(f"- Stations in mainland balanced panel:")
        if info["mainland_balanced"]:
            a("  " + ", ".join(info["mainland_balanced"]))
        else:
            a("  _(empty)_")
        a("")
    a("## Seasonal coverage diagnostic (main sample)")
    a("")
    a(f"Sample window {start}-{end}, loaded with Dec {start - 1} for "
      f"winter completeness. Coverage is reported per "
      "(station, season, season_year); below is the per-(station, season) "
      "median and worst-year value_share for the mainland balanced panel.")
    a("")
    mainland_panel = sample_panels["main"]["mainland_balanced"]
    diag_mainland = diag.loc[diag["station"].isin(mainland_panel)]
    n_low_winter = int(
        (diag_mainland.loc[diag_mainland["season"] == "W", "min_value_share"]
         < config.MIN_WITHIN_SEASON_COVERAGE).sum()
    )
    n_low_summer = int(
        (diag_mainland.loc[diag_mainland["season"] == "S", "min_value_share"]
         < config.MIN_WITHIN_SEASON_COVERAGE).sum()
    )
    a(f"- Mainland balanced stations with at least one DJF season-year "
      f"below value_share = {config.MIN_WITHIN_SEASON_COVERAGE:.2f}: "
      f"**{n_low_winter}** / {len(mainland_panel)}.")
    a(f"- Same for JJA: **{n_low_summer}** / {len(mainland_panel)}.")
    a("")
    a("These flagged season-years are NOT a reason to drop the station "
      "from the panel (the per-year criterion already passed); they "
      "should be inspected as candidates for block-level discarding in "
      "the next script.")
    a("")
    a("## Files produced")
    a("- `tables/balanced_panels_resolved.csv`")
    a("- `tables/seasonal_coverage.csv`")
    a("- `tables/station_metadata_resolved.csv`")
    a("- `figures/seasonal_coverage.pdf`")
    out_path = config.OUTPUT_DIR / config.OUTPUTS.seasonal_audit_report_md
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report -> %s", out_path)


if __name__ == "__main__":
    main()
