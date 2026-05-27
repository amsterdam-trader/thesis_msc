"""DEPRECATED -- superseded by python_project/scripts/01_run_data_audit.py.

This file was the prototype audit script. The production version lives
in scripts/. Kept here only to preserve the prior implementation for
reference; do not extend it.

Original docstring follows.
=============================================================
Data audit for the KNMI hourly FX archive.

Run from the repo root via:
    python python_project/notebooks/run_data_audit.py

Produces (under python_project/outputs/):
    tables/audit_years.csv
    tables/audit_station_year.csv
    tables/audit_station_coverage_pivot.csv
    tables/audit_balanced_panels.csv
    figures/station_coverage_heatmap.pdf
    data_audit_report.md

No extremal-dependence estimation is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_markdown(df: pd.DataFrame, *, index: bool = False) -> str:
    """Minimal markdown table renderer that does not require tabulate."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row]
            for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
import data_loading


# ---------------------------------------------------------------------------
# 1. Enumerate partitions and aggregate per (year, month, station)
# ---------------------------------------------------------------------------

def collect_audit_records() -> pd.DataFrame:
    """One pass over all parquet partitions, producing one row per
    (year, month, station) with hourly-FX completeness statistics.
    """
    records: list[dict] = []
    for year, month, df in data_loading.iter_partitions():
        df = df[["station", "FH", "FX"]].copy()
        grp = df.groupby("station", observed=True, dropna=False)
        agg = grp.agg(
            n_hours=("FX", "size"),
            n_fx_present=("FX", lambda s: s.notna().sum()),
            n_fh_present=("FH", lambda s: s.notna().sum()),
        ).reset_index()
        agg["year"] = year
        agg["month"] = month
        records.append(agg)
    if not records:
        return pd.DataFrame(
            columns=["station", "n_hours", "n_fx_present",
                     "n_fh_present", "year", "month"]
        )
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Roll up to (year) and (station, year)
# ---------------------------------------------------------------------------

def yearly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Number of months present and number of unique stations per year."""
    months_per_year = (
        monthly.groupby("year")["month"].nunique().rename("n_months_present")
    )
    stations_per_year = (
        monthly.groupby("year")["station"].nunique().rename("n_stations")
    )
    fx_per_year = (
        monthly.groupby("year")[["n_hours", "n_fx_present"]].sum()
    )
    fx_per_year["fx_missing_share"] = 1.0 - (
        fx_per_year["n_fx_present"] / fx_per_year["n_hours"]
    )
    return pd.concat(
        [months_per_year, stations_per_year, fx_per_year], axis=1
    ).reset_index()


def station_year_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, year): hourly-FX completeness for the year."""
    grp = monthly.groupby(["station", "year"], observed=True).agg(
        n_hours=("n_hours", "sum"),
        n_fx_present=("n_fx_present", "sum"),
        n_months_present=("month", "nunique"),
    )
    grp["fx_present_share"] = grp["n_fx_present"] / grp["n_hours"].replace(0, np.nan)
    return grp.reset_index()


# ---------------------------------------------------------------------------
# 3. Coverage pivot and heatmap
# ---------------------------------------------------------------------------

def coverage_pivot(station_year: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = station, columns = year, values = fx_present_share."""
    pv = station_year.pivot(
        index="station", columns="year", values="fx_present_share"
    )
    return pv.sort_index()


def save_coverage_heatmap(pv: pd.DataFrame, out_path: Path) -> None:
    """Render the coverage matrix as a year x station heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    M = pv.to_numpy(dtype=float, na_value=np.nan)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.16 * pv.shape[0])))
    cmap = mpl.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    Mm = np.where(np.isnan(M), np.nan, M)
    im = ax.imshow(
        Mm, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(pv.columns.astype(int).tolist(), rotation=90, fontsize=6)
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels(pv.index.tolist(), fontsize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("KNMI station id")
    ax.set_title("FX hourly observations: present-share per station-year")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("non-null FX share")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Candidate balanced panels
# ---------------------------------------------------------------------------

def balanced_panel_for_start(
    pv: pd.DataFrame,
    start_year: int,
    end_year: int,
    threshold: float,
) -> tuple[list[str], int]:
    """Set of stations whose fx_present_share >= threshold in every year
    of [start_year, end_year]. Returns the station list and its size.
    """
    cols = [y for y in pv.columns if start_year <= int(y) <= end_year]
    if not cols:
        return [], 0
    sub = pv[cols]
    mask = (sub >= threshold).all(axis=1) & sub.notna().all(axis=1)
    keep = pv.index[mask].tolist()
    return keep, len(keep)


def candidate_balanced_panels(
    pv: pd.DataFrame,
    candidate_starts: tuple[int, ...],
    end_year: int,
    thresholds: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> pd.DataFrame:
    rows = []
    for start in candidate_starts:
        for thr in thresholds:
            stations, n = balanced_panel_for_start(pv, start, end_year, thr)
            rows.append({
                "start_year": start,
                "end_year": end_year,
                "fx_present_share_threshold": thr,
                "n_stations_balanced": n,
                "years_in_window": end_year - start + 1,
                "station_ids": ",".join(stations),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------

def main() -> None:
    print("[audit] enumerating partitions...")
    parts = data_loading.available_partitions()
    years_avail = sorted({y for y, _ in parts})
    if not years_avail:
        raise SystemExit("No partitions found.")
    print(f"[audit] {len(parts)} partitions covering "
          f"{years_avail[0]}..{years_avail[-1]} ({len(years_avail)} years)")

    print("[audit] collecting per-(year, month, station) statistics...")
    monthly = collect_audit_records()
    print(f"[audit] monthly records: {len(monthly):,}")

    print("[audit] rolling up...")
    years_tbl = yearly_summary(monthly)
    sy_tbl = station_year_summary(monthly)
    pv = coverage_pivot(sy_tbl)
    print(f"[audit] coverage pivot shape: {pv.shape} (stations x years)")

    print("[audit] candidate balanced panels...")
    end_year = years_avail[-1]
    panels = candidate_balanced_panels(
        pv, config.SAMPLE_START_YEAR_CANDIDATES, end_year=end_year,
    )

    # ---- save outputs --------------------------------------------------
    tables_dir = config.TABLES_DIR
    figures_dir = config.FIGURES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    years_tbl.to_csv(tables_dir / "audit_years.csv", index=False)
    sy_tbl.to_csv(tables_dir / "audit_station_year.csv", index=False)
    pv.to_csv(tables_dir / "audit_station_coverage_pivot.csv")
    panels.to_csv(tables_dir / "audit_balanced_panels.csv", index=False)

    print("[audit] heatmap...")
    save_coverage_heatmap(
        pv, figures_dir / "station_coverage_heatmap.pdf",
    )

    # ---- short report --------------------------------------------------
    report_path = config.OUTPUT_DIR / "data_audit_report.md"
    write_report(
        report_path,
        parts=parts, years_avail=years_avail, monthly=monthly,
        years_tbl=years_tbl, sy_tbl=sy_tbl, pv=pv, panels=panels,
    )
    print(f"[audit] report -> {report_path}")
    print("[audit] done.")


def write_report(
    out_path: Path, *,
    parts: list[tuple[int, int]],
    years_avail: list[int],
    monthly: pd.DataFrame,
    years_tbl: pd.DataFrame,
    sy_tbl: pd.DataFrame,
    pv: pd.DataFrame,
    panels: pd.DataFrame,
) -> None:
    """Build the markdown audit report."""

    n_partitions = len(parts)
    first_year, last_year = years_avail[0], years_avail[-1]
    expected = (last_year - first_year + 1) * 12
    missing_partitions = sorted(
        {(y, m) for y in range(first_year, last_year + 1) for m in range(1, 13)}
        - set(parts)
    )

    overall_hours = int(monthly["n_hours"].sum())
    overall_fx = int(monthly["n_fx_present"].sum())
    overall_fx_miss = 1.0 - overall_fx / overall_hours if overall_hours else float("nan")
    overall_fh = int(monthly["n_fh_present"].sum())
    overall_fh_miss = 1.0 - overall_fh / overall_hours if overall_hours else float("nan")

    n_unique_stations = pv.shape[0]

    # Per-year sentence-level summaries: first year with >= K stations
    # whose FX-present-share >= T.
    def first_year_meeting(T: float, K: int) -> int | None:
        per_year = (pv >= T).sum(axis=0)
        candidates = per_year.index[per_year >= K]
        return int(min(candidates)) if len(candidates) else None

    first_50_10 = first_year_meeting(0.50, 10)
    first_80_10 = first_year_meeting(0.80, 10)
    first_80_20 = first_year_meeting(0.80, 20)
    first_80_30 = first_year_meeting(0.80, 30)

    # Balanced-panel scoreboard
    scoreboard = panels.copy()
    scoreboard = scoreboard.pivot_table(
        index="start_year", columns="fx_present_share_threshold",
        values="n_stations_balanced",
    ).fillna(0).astype(int)

    lines: list[str] = []
    add = lines.append

    add(f"# Data audit report")
    add("")
    add(f"Generated by `python_project/notebooks/run_data_audit.py`.")
    add(f"Underlying data path: `data/yearly_aggregated_FH_FX/`.")
    add("")
    add("## 1. Partition inventory")
    add("")
    add(f"- Partitions on disk: **{n_partitions}**.")
    add(f"- Years present: **{first_year}–{last_year}** "
        f"({len(years_avail)} years).")
    add(f"- Expected partitions if every (year, month) were present: "
        f"{expected}. Actual: {n_partitions}. Missing: "
        f"{len(missing_partitions)}.")
    if missing_partitions:
        # Report year ranges where multiple months are missing for the
        # same year, plus a sample.
        miss_years = {}
        for y, m in missing_partitions:
            miss_years.setdefault(y, []).append(m)
        head = ", ".join(
            f"{y} ({len(v)} months)" for y, v in sorted(miss_years.items())[:6]
        )
        add(f"- Years with the most missing monthly partitions (first 6): "
            f"{head}.")
    add("")
    add("## 2. Schema confirmation")
    add("")
    add("Schema confirmed against `year=1951/month=01`:")
    add("```text")
    add("time           datetime64[us]")
    add("station        str   (WIGOS-style: \"0-20000-0-06210\")")
    add("stationname    str")
    add("lat, lon       float64")
    add("height         float64")
    add("FH             float64   (hourly mean wind speed; not the thesis variable)")
    add("FX             float64   (hourly maximum wind gust; primary variable)")
    add("```")
    add("")
    add("Station ids in the parquet use the WIGOS form "
        "`0-20000-0-NNNNN`. The 5-digit tail `NNNNN` matches the "
        "`station_id` column in `data/station_metadata.csv`. The "
        "normaliser is `data_loading.station_id_to_5digit`.")
    add("")
    add("## 3. Overall non-null shares (whole archive)")
    add("")
    add(f"- Total station-hour cells across all partitions: "
        f"{overall_hours:,}.")
    add(f"- FX non-null cells: {overall_fx:,} "
        f"(share missing: {overall_fx_miss:.3f}).")
    add(f"- FH non-null cells: {overall_fh:,} "
        f"(share missing: {overall_fh_miss:.3f}).")
    add("")
    add("The FX missing share is high for the whole archive because "
        "FX coverage was sparse in the early decades. Per-year and "
        "per-station-year detail is in the tables below.")
    add("")
    add("## 4. Per-year summary")
    add("")
    add("Stored as `outputs/tables/audit_years.csv`. Columns: "
        "`year, n_months_present, n_stations, n_hours, n_fx_present, "
        "fx_missing_share`.")
    add("")
    add("Selected rows (first, last, and every 10th year):")
    add("")
    yr = years_tbl.sort_values("year").reset_index(drop=True)
    ticks = sorted({0, len(yr) - 1, *range(0, len(yr), 10)})
    snap = yr.iloc[ticks][
        ["year", "n_months_present", "n_stations",
         "n_hours", "n_fx_present", "fx_missing_share"]
    ].copy()
    snap["fx_missing_share"] = snap["fx_missing_share"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
    )
    add(df_to_markdown(snap, index=False))
    add("")
    add(f"- Unique station ids ever observed: **{n_unique_stations}**.")
    add(f"- First year with at least 10 stations whose FX-present-share "
        f"in that year >= 0.50: **{first_50_10}**.")
    add(f"- First year with at least 10 stations >= 0.80: "
        f"**{first_80_10}**.")
    add(f"- First year with at least 20 stations >= 0.80: "
        f"**{first_80_20}**.")
    add(f"- First year with at least 30 stations >= 0.80: "
        f"**{first_80_30}**.")
    add("")
    add("## 5. Candidate balanced panels")
    add("")
    add("Number of stations whose FX-present-share is at least the "
        "threshold in EVERY year of the window "
        f"`[start_year, {last_year}]`. Full table in "
        "`outputs/tables/audit_balanced_panels.csv`.")
    add("")
    add(df_to_markdown(scoreboard, index=True))
    add("")
    add("Interpretation: pick a `(start_year, threshold)` pair that "
        "trades sample length against panel size. For the seasonal "
        "comparison a panel of ~20+ stations and >=20 years would be "
        "comfortable; the table above shows where on the trade-off "
        "curve that lies.")
    add("")
    add("## 6. Outputs")
    add("")
    add("- `outputs/tables/audit_years.csv` — per-year summary.")
    add("- `outputs/tables/audit_station_year.csv` — one row per "
        "(station, year).")
    add("- `outputs/tables/audit_station_coverage_pivot.csv` — "
        "stations x years matrix of FX-present-share.")
    add("- `outputs/tables/audit_balanced_panels.csv` — candidate "
        "balanced panels for the six start years in "
        "`config.SAMPLE_START_YEAR_CANDIDATES`.")
    add("- `outputs/figures/station_coverage_heatmap.pdf` — visual "
        "of the coverage pivot.")
    add("")
    add("## 7. Open decisions for the next pass")
    add("")
    add("1. Pick `SAMPLE_START_YEAR` from the candidate panels table.")
    add("2. Set `MIN_WITHIN_SEASON_COVERAGE` in `config.py` based on "
        "the distribution of within-season completeness "
        "(this audit reports yearly completeness, not the seasonal "
        "split — that comes in the next pass when DJF/JJA are "
        "computed).")
    add("3. Decide whether to include or exclude offshore platform "
        "stations on physical grounds.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
