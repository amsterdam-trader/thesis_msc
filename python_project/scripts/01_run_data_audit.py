"""Data audit: partition inventory, schema, FX coverage by station-year.

Outputs (under python_project/outputs/):
    tables/audit_years.csv
    tables/audit_station_year.csv
    tables/audit_station_coverage_pivot.csv
    tables/audit_balanced_panels.csv
    figures/station_coverage_heatmap.pdf
    data_audit_report.md

This script is the production wrapper around the audit code that was
previously in notebooks/run_data_audit.py. It is the first step in
the pipeline.
"""

from __future__ import annotations

import _bootstrap_path  # noqa: F401

import numpy as np
import pandas as pd

import config
import data_loading
import plotting


logger = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# audit reductions
# ---------------------------------------------------------------------------

def collect_audit_records() -> pd.DataFrame:
    """One pass over all partitions: one row per (station, year, month)."""
    records: list[pd.DataFrame] = []
    for year, month, df in data_loading.iter_partitions(
        columns=("station", "FH", "FX"),
    ):
        grp = df.groupby("station", observed=True, dropna=False)
        agg = grp.agg(
            n_hours=("FX", "size"),
            n_fx_present=("FX", lambda s: int(s.notna().sum())),
            n_fh_present=("FH", lambda s: int(s.notna().sum())),
        ).reset_index()
        agg["year"] = year
        agg["month"] = month
        records.append(agg)
    if not records:
        return pd.DataFrame(columns=[
            "station", "n_hours", "n_fx_present", "n_fh_present",
            "year", "month",
        ])
    return pd.concat(records, ignore_index=True)


def yearly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    by_year = monthly.groupby("year")
    out = pd.DataFrame({
        "n_months_present": by_year["month"].nunique(),
        "n_stations": by_year["station"].nunique(),
        "n_hours": by_year["n_hours"].sum(),
        "n_fx_present": by_year["n_fx_present"].sum(),
    }).reset_index()
    out["fx_missing_share"] = 1.0 - out["n_fx_present"] / out["n_hours"]
    return out


def station_year_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    grp = monthly.groupby(["station", "year"], observed=True).agg(
        n_hours=("n_hours", "sum"),
        n_fx_present=("n_fx_present", "sum"),
        n_months_present=("month", "nunique"),
    )
    grp["fx_present_share"] = grp["n_fx_present"] / grp["n_hours"].replace(0, np.nan)
    return grp.reset_index()


def coverage_pivot(station_year: pd.DataFrame) -> pd.DataFrame:
    return station_year.pivot(
        index="station", columns="year", values="fx_present_share",
    ).sort_index()


def balanced_panel_for_start(
    pv: pd.DataFrame, start_year: int, end_year: int, threshold: float,
) -> list[str]:
    cols = [y for y in pv.columns if start_year <= int(y) <= end_year]
    if not cols:
        return []
    sub = pv[cols]
    mask = (sub >= threshold).all(axis=1) & sub.notna().all(axis=1)
    return pv.index[mask].tolist()


def candidate_balanced_panels(
    pv: pd.DataFrame, candidate_starts, end_year, thresholds,
) -> pd.DataFrame:
    rows = []
    for start in candidate_starts:
        for thr in thresholds:
            stations = balanced_panel_for_start(pv, start, end_year, thr)
            rows.append({
                "start_year": start,
                "end_year": end_year,
                "fx_present_share_threshold": thr,
                "n_stations_balanced": len(stations),
                "years_in_window": end_year - start + 1,
                "station_ids": ",".join(stations),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame, *, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v))
             for v in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def write_report(out_path, *, parts, monthly, years_tbl, pv, panels):
    n_partitions = len(parts)
    years_present = sorted({y for y, _ in parts})
    first_year, last_year = years_present[0], years_present[-1]
    expected = (last_year - first_year + 1) * 12
    missing = sorted({(y, m) for y in range(first_year, last_year + 1)
                       for m in range(1, 13)} - set(parts))

    overall_hours = int(monthly["n_hours"].sum())
    overall_fx = int(monthly["n_fx_present"].sum())
    overall_miss = 1.0 - overall_fx / overall_hours if overall_hours else float("nan")

    def first_year_meeting(T, K):
        per_year = (pv >= T).sum(axis=0)
        cands = per_year.index[per_year >= K]
        return int(min(cands)) if len(cands) else None

    scoreboard = panels.pivot_table(
        index="start_year", columns="fx_present_share_threshold",
        values="n_stations_balanced",
    ).fillna(0).astype(int)

    yr = years_tbl.sort_values("year").reset_index(drop=True)
    ticks = sorted({0, len(yr) - 1, *range(0, len(yr), 10)})
    snap = yr.iloc[ticks][["year", "n_months_present", "n_stations",
                           "n_hours", "n_fx_present", "fx_missing_share"]].copy()
    snap["fx_missing_share"] = snap["fx_missing_share"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
    )

    L = []
    a = L.append
    a("# Data audit report")
    a("")
    a("Generated by `python_project/scripts/01_run_data_audit.py`.")
    a(f"Data path: `{config.HOURLY_PARQUET_DIR.relative_to(config.REPO_ROOT)}`.")
    a("")
    a("## 1. Partition inventory")
    a("")
    a(f"- Partitions on disk: **{n_partitions}**.")
    a(f"- Years: **{first_year}-{last_year}** ({len(years_present)} years).")
    a(f"- Expected: {expected}. Missing: {len(missing)}.")
    if missing:
        miss_years = {}
        for y, m in missing:
            miss_years.setdefault(y, []).append(m)
        head = ", ".join(f"{y} ({len(v)} months)"
                         for y, v in sorted(miss_years.items())[:6])
        a(f"- Years with the most missing partitions (first 6): {head}.")
    a("")
    a("## 2. Schema confirmation")
    a("")
    a("Schema: `time, station, stationname, lat, lon, height, FH, FX`.")
    a("Station ids in the parquet use the WIGOS form `0-20000-0-NNNNN`.")
    a("The 5-digit tail `NNNNN` is the canonical id; normaliser is "
      "`data_loading.station_id_to_5digit`.")
    a("")
    a("## 3. Overall FX completeness")
    a("")
    a(f"- Station-hour cells (whole archive): **{overall_hours:,}**.")
    a(f"- FX non-null cells: {overall_fx:,} "
      f"(missing share: {overall_miss:.3f}).")
    a("")
    a("## 4. Per-year summary")
    a("")
    a("Selected rows (first, last, and every 10th year):")
    a("")
    a(df_to_markdown(snap, index=False))
    a("")
    a(f"- First year with >= 10 stations at fx_present_share >= 0.50: "
      f"**{first_year_meeting(0.50, 10)}**.")
    a(f"- First year with >= 10 stations >= 0.80: "
      f"**{first_year_meeting(0.80, 10)}**.")
    a(f"- First year with >= 20 stations >= 0.80: "
      f"**{first_year_meeting(0.80, 20)}**.")
    a(f"- First year with >= 30 stations >= 0.80: "
      f"**{first_year_meeting(0.80, 30)}**.")
    a("")
    a("## 5. Candidate balanced panels")
    a("")
    a("Number of stations whose fx_present_share is at least the "
      f"threshold in every year of `[start_year, {last_year}]`.")
    a("")
    a(df_to_markdown(scoreboard, index=True))
    a("")
    out_path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    config.ensure_output_dirs()
    logger.info("enumerating partitions...")
    parts = data_loading.available_partitions()
    if not parts:
        raise SystemExit("No partitions found.")
    logger.info("%d partitions over %d years", len(parts),
                len({y for y, _ in parts}))

    logger.info("collecting per-(station, year, month) records...")
    monthly = collect_audit_records()
    logger.info("monthly records: %d", len(monthly))

    logger.info("rolling up...")
    yr_tbl = yearly_summary(monthly)
    sy_tbl = station_year_summary(monthly)
    pv = coverage_pivot(sy_tbl)
    end_year = max(y for y, _ in parts)
    panels = candidate_balanced_panels(
        pv, config.SAMPLE_START_YEAR_CANDIDATES, end_year,
        config.COVERAGE_THRESHOLDS,
    )

    # save tables
    yr_tbl.to_csv(config.TABLES_DIR / config.OUTPUTS.audit_years_csv, index=False)
    sy_tbl.to_csv(config.TABLES_DIR / config.OUTPUTS.audit_station_year_csv, index=False)
    pv.to_csv(config.TABLES_DIR / config.OUTPUTS.audit_station_coverage_pivot_csv)
    panels.to_csv(config.TABLES_DIR / config.OUTPUTS.audit_balanced_panels_csv, index=False)
    logger.info("tables saved to %s", config.TABLES_DIR)

    # heatmap
    fig = plotting.coverage_heatmap(pv)
    plotting.save_fig(fig, config.OUTPUTS.coverage_heatmap_pdf)

    # report
    report_path = config.OUTPUT_DIR / config.OUTPUTS.data_audit_report_md
    write_report(report_path, parts=parts, monthly=monthly,
                 years_tbl=yr_tbl, pv=pv, panels=panels)
    logger.info("report -> %s", report_path)
    logger.info("done.")


if __name__ == "__main__":
    main()
