"""Daily-maxima coverage audit for the KNMI hourly FX archive.

Purpose
-------
This script audits the data object used in the thesis: daily maximum wind gusts
constructed from hourly FX observations. It checks whether a fixed station panel
has sufficiently complete daily maxima in DJF and JJA.

Run from the repository root, for example:
    python python_project/scripts/02_run_daily_maxima_audit.py

Outputs, under python_project/outputs by default:
    tables/audit_daily_station_season.csv
    tables/audit_daily_balanced_panels.csv
    tables/audit_daily_candidate_panels_wide.csv
    figures/daily_maxima_coverage_DJF.pdf
    figures/daily_maxima_coverage_JJA.pdf
    daily_maxima_audit_report.md

Notes
-----
- FX is the primary variable: hourly maximum wind gust.
- A station-day is treated as valid when at least MIN_HOURS_PER_DAY non-null FX
  observations are available. Default: 18.
- Daily maximum = max hourly FX within a valid station-day.
- DJF is assigned to the year containing Jan/Feb. Example: Dec 1991 + Jan/Feb
  1992 is DJF season_year 1992.
- By default, the audit excludes 2026 and uses complete seasons up to 2025.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports with a fallback that works from scripts/ or notebooks/
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
CANDIDATE_SRC = [
    HERE.parents[1] / "src" if len(HERE.parents) > 1 else None,
    HERE.parents[0] / "src",
    Path.cwd() / "python_project" / "src",
    Path.cwd() / "src",
]
for src in CANDIDATE_SRC:
    if src is not None and src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

try:
    import config  # type: ignore
    import data_loading  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import project modules `config` and `data_loading`. "
        "Run this from the repository root or adjust the SRC path.\n"
        f"Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def df_to_markdown(df: pd.DataFrame, *, index: bool = False) -> str:
    """Minimal markdown table renderer that does not require tabulate."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def station_to_5digit_safe(x: object) -> str:
    """Use the project normaliser if available, else keep the final token."""
    if hasattr(data_loading, "station_id_to_5digit"):
        return str(data_loading.station_id_to_5digit(x))
    s = str(x)
    return s.split("-")[-1].zfill(5)


def assign_season_and_year(dates: pd.Series) -> pd.DataFrame:
    """Return season and season_year for dates.

    DJF is assigned to the year of Jan/Feb, so Dec belongs to year + 1.
    JJA is assigned to the calendar year.
    Other months receive NA and are dropped later.
    """
    month = dates.dt.month
    year = dates.dt.year

    season = pd.Series(pd.NA, index=dates.index, dtype="object")
    season_year = pd.Series(pd.NA, index=dates.index, dtype="Int64")

    is_djf = month.isin([12, 1, 2])
    is_jja = month.isin([6, 7, 8])

    season.loc[is_djf] = "DJF"
    season.loc[is_jja] = "JJA"

    season_year.loc[is_jja] = year.loc[is_jja].astype("int64")
    season_year.loc[is_djf] = year.loc[is_djf].astype("int64")
    season_year.loc[month.eq(12)] = (year.loc[month.eq(12)] + 1).astype("int64")

    return pd.DataFrame({"season": season, "season_year": season_year})


def expected_days_for_season(season: str, season_year: int) -> int:
    if season == "JJA":
        return 30 + 31 + 31
    if season == "DJF":
        feb_days = 29 if calendar.isleap(int(season_year)) else 28
        return 31 + 31 + feb_days  # Dec(previous year) + Jan + Feb
    raise ValueError(f"Unsupported season: {season}")


def available_partitions_in_window(start_year: int, end_year: int) -> list[tuple[int, int]]:
    """Partitions to read.

    We include Dec of start_year-1 because it is needed for DJF start_year.
    We do not need Jan/Feb of end_year+1 unless analysing DJF end_year+1.
    """
    parts = sorted(data_loading.available_partitions())
    read_start = start_year - 1
    read_end = end_year
    return [(y, m) for y, m in parts if read_start <= int(y) <= read_end]


def iter_partitions_filtered(parts_to_read: set[tuple[int, int]]) -> Iterable[tuple[int, int, pd.DataFrame]]:
    """Yield project partitions, filtered to requested (year, month) pairs."""
    for year, month, df in data_loading.iter_partitions():
        key = (int(year), int(month))
        if key in parts_to_read:
            yield int(year), int(month), df


# ---------------------------------------------------------------------------
# Daily-maxima audit
# ---------------------------------------------------------------------------


def collect_daily_station_records(
    *,
    start_year: int,
    end_year: int,
    min_hours_per_day: int,
) -> pd.DataFrame:
    """One row per station-day in DJF/JJA with daily-FX completeness.

    The returned table does not yet enforce a balanced station panel. It is the
    raw station-day audit from which seasonal coverage is computed.
    """
    parts_to_read = set(available_partitions_in_window(start_year, end_year))
    records: list[pd.DataFrame] = []

    for year, month, df in iter_partitions_filtered(parts_to_read):
        if month not in {1, 2, 6, 7, 8, 12}:
            continue

        needed = ["time", "station", "FX"]
        missing_cols = [c for c in needed if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Partition {year}-{month:02d} missing columns: {missing_cols}")

        tmp = df[needed].copy()
        tmp["time"] = pd.to_datetime(tmp["time"])
        tmp["date"] = tmp["time"].dt.floor("D")
        tmp["station"] = tmp["station"].map(station_to_5digit_safe)

        sy = assign_season_and_year(tmp["date"])
        tmp = pd.concat([tmp, sy], axis=1)
        tmp = tmp[tmp["season"].isin(["DJF", "JJA"])].copy()
        tmp = tmp[tmp["season_year"].between(start_year, end_year)].copy()
        if tmp.empty:
            continue

        grp = tmp.groupby(["station", "date", "season", "season_year"], observed=True)
        day = grp.agg(
            n_hour_rows=("FX", "size"),
            n_fx_present=("FX", lambda s: int(s.notna().sum())),
            daily_max_fx=("FX", "max"),
        ).reset_index()
        day["valid_day"] = day["n_fx_present"] >= min_hours_per_day
        day.loc[~day["valid_day"], "daily_max_fx"] = np.nan
        records.append(day)

    if not records:
        return pd.DataFrame(
            columns=[
                "station",
                "date",
                "season",
                "season_year",
                "n_hour_rows",
                "n_fx_present",
                "daily_max_fx",
                "valid_day",
            ]
        )

    out = pd.concat(records, ignore_index=True)
    out["season_year"] = out["season_year"].astype(int)
    return out


def station_season_summary(
    daily: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """One row per station x season x season_year."""
    if daily.empty:
        return pd.DataFrame()

    observed = (
        daily.groupby(["station", "season", "season_year"], observed=True)
        .agg(
            n_station_days_observed=("date", "nunique"),
            n_valid_days=("valid_day", "sum"),
            n_fx_hours_present=("n_fx_present", "sum"),
            n_hour_rows=("n_hour_rows", "sum"),
        )
        .reset_index()
    )

    # Complete grid for all stations observed at least once in the analysis window.
    stations = sorted(daily["station"].dropna().unique().tolist())
    seasons = ["DJF", "JJA"]
    years = list(range(start_year, end_year + 1))
    grid = pd.MultiIndex.from_product([stations, seasons, years], names=["station", "season", "season_year"]).to_frame(
        index=False
    )

    out = grid.merge(observed, on=["station", "season", "season_year"], how="left")
    count_cols = ["n_station_days_observed", "n_valid_days", "n_fx_hours_present", "n_hour_rows"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    out["n_days_expected"] = [expected_days_for_season(s, int(y)) for s, y in zip(out["season"], out["season_year"])]
    out["valid_day_share"] = out["n_valid_days"] / out["n_days_expected"]
    out["observed_day_share"] = out["n_station_days_observed"] / out["n_days_expected"]
    out["fx_hour_share_within_observed_days"] = np.where(
        out["n_hour_rows"] > 0,
        out["n_fx_hours_present"] / out["n_hour_rows"],
        np.nan,
    )
    return out.sort_values(["station", "season", "season_year"]).reset_index(drop=True)


def balanced_panel_for_daily(
    ss: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    threshold: float,
    seasons_required: tuple[str, ...] = ("DJF", "JJA"),
) -> list[str]:
    """Stations with valid_day_share >= threshold for every required season-year."""
    sub = ss[ss["season"].isin(seasons_required) & ss["season_year"].between(start_year, end_year)].copy()
    if sub.empty:
        return []

    expected_n = (end_year - start_year + 1) * len(seasons_required)
    ok = sub["valid_day_share"] >= threshold
    by_station = (
        sub.assign(ok=ok)
        .groupby("station", observed=True)
        .agg(
            n_rows=("ok", "size"),
            n_ok=("ok", "sum"),
            min_valid_day_share=("valid_day_share", "min"),
        )
    )
    keep = by_station[(by_station["n_rows"] == expected_n) & (by_station["n_ok"] == expected_n)]
    return keep.sort_index().index.tolist()


def candidate_daily_balanced_panels(
    ss: pd.DataFrame,
    *,
    candidate_starts: tuple[int, ...],
    end_year: int,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict] = []
    season_sets = {
        "DJF": ("DJF",),
        "JJA": ("JJA",),
        "DJF_and_JJA": ("DJF", "JJA"),
    }
    for start in candidate_starts:
        for thr in thresholds:
            for label, seasons in season_sets.items():
                stations = balanced_panel_for_daily(
                    ss,
                    start_year=start,
                    end_year=end_year,
                    threshold=thr,
                    seasons_required=seasons,
                )
                rows.append(
                    {
                        "start_year": start,
                        "end_year": end_year,
                        "min_valid_day_share": thr,
                        "season_requirement": label,
                        "n_stations_balanced": len(stations),
                        "years_in_window": end_year - start + 1,
                        "n_pairs": len(stations) * (len(stations) - 1) // 2,
                        "station_ids": ",".join(stations),
                    }
                )
    return pd.DataFrame(rows)


def save_season_heatmap(ss: pd.DataFrame, *, season: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    sub = ss[ss["season"] == season]
    pv = sub.pivot(index="station", columns="season_year", values="valid_day_share").sort_index()
    M = pv.to_numpy(dtype=float, na_value=np.nan)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.16 * pv.shape[0])))
    cmap = mpl.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels(pv.columns.astype(int).tolist(), rotation=90, fontsize=6)
    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels(pv.index.tolist(), fontsize=5)
    ax.set_xlabel("Season year")
    ax.set_ylabel("KNMI station id")
    ax.set_title(f"Daily maximum FX coverage: valid-day share per station-season ({season})")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("valid-day share")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_report(
    out_path: Path,
    *,
    start_year: int,
    end_year: int,
    min_hours_per_day: int,
    daily: pd.DataFrame,
    ss: pd.DataFrame,
    panels: pd.DataFrame,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Daily-maxima data audit report")
    add("")
    add("Generated by `python_project/scripts/02_run_daily_maxima_audit.py`.")
    add("")
    add("## 1. Design audited")
    add("")
    add(f"- Analysis window: **{start_year}--{end_year}**.")
    add("- Seasons audited: **DJF** and **JJA**.")
    add(f"- Valid station-day rule: at least **{min_hours_per_day}** non-null hourly FX observations.")
    add("- Daily maximum wind gust is computed only for valid station-days.")
    add(
        "- DJF is assigned to the year containing January and February; December belongs to the following DJF season-year."
    )
    add("")

    n_stations = ss["station"].nunique() if not ss.empty else 0
    add("## 2. Daily records")
    add("")
    add(f"- Station-day rows found in DJF/JJA: **{len(daily):,}**.")
    add(f"- Stations observed at least once in the audited window: **{n_stations}**.")
    if not daily.empty:
        add(f"- Valid station-days: **{int(daily['valid_day'].sum()):,}**.")
        add(f"- Invalid station-days under the rule: **{int((~daily['valid_day']).sum()):,}**.")
    add("")

    add("## 3. Seasonal coverage summary")
    add("")
    if not ss.empty:
        summ = (
            ss.groupby("season")
            .agg(
                n_station_seasons=("valid_day_share", "size"),
                median_valid_day_share=("valid_day_share", "median"),
                p05_valid_day_share=("valid_day_share", lambda x: x.quantile(0.05)),
                p01_valid_day_share=("valid_day_share", lambda x: x.quantile(0.01)),
                min_valid_day_share=("valid_day_share", "min"),
            )
            .reset_index()
        )
        for c in ["median_valid_day_share", "p05_valid_day_share", "p01_valid_day_share", "min_valid_day_share"]:
            summ[c] = summ[c].map(lambda x: f"{x:.3f}")
        add(df_to_markdown(summ))
    add("")

    add("## 4. Candidate fixed station panels")
    add("")
    add("Number of stations passing the minimum valid-day-share threshold in every required season-year.")
    add("")
    wide = (
        panels.pivot_table(
            index=["start_year", "season_requirement"],
            columns="min_valid_day_share",
            values="n_stations_balanced",
            aggfunc="first",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    add(df_to_markdown(wide))
    add("")

    add("## 5. Recommended rows to inspect")
    add("")
    add(
        "For your fixed-station thesis design, first inspect `season_requirement = DJF_and_JJA`, because the same stations should work for both winter and summer."
    )
    add(
        "A natural first candidate is the largest panel beginning in 1991 with a valid-day-share threshold of 0.90, ending in 2025."
    )
    add("")
    candidate = panels[
        (panels["start_year"] == 1991)
        & (panels["end_year"] == end_year)
        & (panels["min_valid_day_share"] == 0.90)
        & (panels["season_requirement"] == "DJF_and_JJA")
    ]
    if not candidate.empty:
        row = candidate.iloc[0]
        add(
            f"- 1991--{end_year}, threshold 0.90, DJF and JJA: **{int(row['n_stations_balanced'])} stations**, **{int(row['n_pairs'])} pairs**."
        )
        if row["station_ids"]:
            add(f"- Station ids: `{row['station_ids']}`")
    add("")

    add("## 6. Outputs")
    add("")
    add("- `outputs/tables/audit_daily_station_season.csv` -- one row per station, season, season-year.")
    add("- `outputs/tables/audit_daily_balanced_panels.csv` -- candidate fixed station panels.")
    add("- `outputs/tables/audit_daily_candidate_panels_wide.csv` -- compact panel-count table.")
    add("- `outputs/figures/daily_maxima_coverage_DJF.pdf` -- DJF coverage heatmap.")
    add("- `outputs/figures/daily_maxima_coverage_JJA.pdf` -- JJA coverage heatmap.")
    add("")

    add("## 7. Interpretation")
    add("")
    add(
        "Choose the largest fixed station set that passes the same threshold for both DJF and JJA over the selected sample period. If the 0.90 threshold gives too few stations, inspect 0.80; if 0.95 loses only a few stations, it is the cleaner choice. Exclude 2026 unless you explicitly want to handle partial-year data."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit daily maxima coverage for KNMI hourly FX data.")
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-hours-per-day", type=int, default=18)
    parser.add_argument(
        "--candidate-starts",
        type=int,
        nargs="*",
        default=None,
        help="Candidate panel start years. Defaults to config.SAMPLE_START_YEAR_CANDIDATES if available, else common values.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.80, 0.90, 0.95],
        help="Minimum valid-day share thresholds for balanced panels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = getattr(config, "OUTPUT_DIR", Path("python_project/outputs"))
    tables_dir = getattr(config, "TABLES_DIR", output_dir / "tables")
    figures_dir = getattr(config, "FIGURES_DIR", output_dir / "figures")
    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_starts is not None:
        candidate_starts = tuple(args.candidate_starts)
    elif hasattr(config, "SAMPLE_START_YEAR_CANDIDATES"):
        candidate_starts = tuple(int(x) for x in config.SAMPLE_START_YEAR_CANDIDATES if int(x) <= args.end_year)
    else:
        candidate_starts = (1971, 1981, 1991, 2001)

    print("[daily-audit] collecting station-day records...")
    daily = collect_daily_station_records(
        start_year=args.start_year,
        end_year=args.end_year,
        min_hours_per_day=args.min_hours_per_day,
    )
    print(f"[daily-audit] station-day rows: {len(daily):,}")

    print("[daily-audit] rolling up to station-season coverage...")
    ss = station_season_summary(daily, start_year=args.start_year, end_year=args.end_year)
    print(f"[daily-audit] station-season rows: {len(ss):,}")

    print("[daily-audit] candidate fixed panels...")
    panels = candidate_daily_balanced_panels(
        ss,
        candidate_starts=candidate_starts,
        end_year=args.end_year,
        thresholds=tuple(float(x) for x in args.thresholds),
    )

    # Save tables.
    daily_summary = daily.drop(columns=["daily_max_fx"], errors="ignore")
    daily_summary.to_csv(tables_dir / "audit_daily_station_day.csv", index=False)
    ss.to_csv(tables_dir / "audit_daily_station_season.csv", index=False)
    panels.to_csv(tables_dir / "audit_daily_balanced_panels.csv", index=False)

    wide = (
        panels.pivot_table(
            index=["start_year", "season_requirement"],
            columns="min_valid_day_share",
            values="n_stations_balanced",
            aggfunc="first",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    wide.to_csv(tables_dir / "audit_daily_candidate_panels_wide.csv", index=False)

    print("[daily-audit] heatmaps...")
    save_season_heatmap(ss, season="DJF", out_path=figures_dir / "daily_maxima_coverage_DJF.pdf")
    save_season_heatmap(ss, season="JJA", out_path=figures_dir / "daily_maxima_coverage_JJA.pdf")

    report_path = output_dir / "daily_maxima_audit_report.md"
    write_report(
        report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_hours_per_day=args.min_hours_per_day,
        daily=daily,
        ss=ss,
        panels=panels,
    )
    print(f"[daily-audit] report -> {report_path}")
    print("[daily-audit] done.")


if __name__ == "__main__":
    main()
