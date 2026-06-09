"""
KNMI wind-gust extremes thesis workflow
======================================

This script is designed for the empirical chapter of a thesis that compares
winter and summer spatial extremal dependence in Dutch wind gusts.

It does five things:
1. Read KNMI hourly or daily CSV/TXT files.
2. Convert hourly gusts to daily maximum gusts by station.
3. Split the retained station panel into DJF winter and JJA summer season-years.
4. Fit station-season GEV margins and transform to unit Frechet margins.
5. Fit a Brown--Resnick dependence model by pairwise composite likelihood and
   save simple figures/tables for the thesis.

Important:
- The script assumes your station-retention step is already proven. It does not
  drop stations for being below 95 percent coverage unless you explicitly pass a
  smaller retained station list.
- Bootstrap is optional and can be slow. Start with --bootstrap 0, check the
  figures/tables, then run --bootstrap 1000 when the workflow is correct.

Expected station metadata CSV columns by default:
    STN, NAME, LAT, LON

Expected KNMI observation columns by default:
    STN, YYYYMMDD, FX
For Floris' processed daily file, pass:
    --date-col date --station-col station --meta-station-col station_id --lon-col lon --lat-col lat --name-col stationname --wind-scale 1
where FX is already in m/s.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.special import log_ndtr, logsumexp


LOGGER = logging.getLogger("knmi_wind_extremes")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def log_step(message: str, start_time: float | None = None) -> float:
    if start_time is None:
        LOGGER.info(message)
    else:
        LOGGER.info("%s [%.1f sec]", message, time.perf_counter() - start_time)
    return time.perf_counter()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    raw_data: Path
    station_meta: Path
    output_dir: Path = Path("outputs_knmi_thesis")
    value_col: str = "FX"
    station_col: str = "STN"
    meta_station_col: Optional[str] = None
    date_col: str = "YYYYMMDD"
    lon_col: str = "LON"
    lat_col: str = "LAT"
    name_col: str = "NAME"
    wind_scale: float = 10.0
    start_date: str = "1991-01-01"
    end_date: str = "2026-02-01"  # exclusive; drops truncated winter 2026
    u_level: float = 0.95
    bootstrap: int = 0
    seed: int = 20260608
    retained_station_ids: Optional[list[str]] = None
    log_level: str = "INFO"
    log_every: int = 25
    optimizer_maxiter: int = 300
    start_grid: str = "full"


# ---------------------------------------------------------------------
# Reading KNMI files
# ---------------------------------------------------------------------

def _clean_column_names(columns: Iterable[str]) -> list[str]:
    return [str(c).strip().replace(" ", "") for c in columns]


def normalize_station_id_value(value) -> str:
    """Return a stable KNMI station code.

    Handles both numeric station IDs such as 6225 / 06225 and WIGOS-style
    IDs such as 0-20000-0-06225. The returned value is a string without
    leading zeroes, e.g. 6225.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "-" in text:
        text = text.split("-")[-1]
    text = text.replace(".0", "") if text.endswith(".0") else text
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text


def normalize_station_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_station_id_value).astype(str)


def _find_knmi_header(path: Path) -> Optional[list[str]]:
    """Find a KNMI-style commented header line such as '# STN,YYYYMMDD,HH,FX'."""
    header = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            candidate = stripped.lstrip("#").strip()
            if "STN" in candidate and "," in candidate:
                header = _clean_column_names(candidate.split(","))
    return header


def read_one_knmi_file(path: Path) -> pd.DataFrame:
    """Read either a normal CSV or the KNMI commented CSV format."""
    header = _find_knmi_header(path)
    if header is not None:
        df = pd.read_csv(
            path,
            comment="#",
            names=header,
            skipinitialspace=True,
            na_values=["", " ", "NA", "NaN", -999, -9999, 9999],
        )
    else:
        df = pd.read_csv(path, skipinitialspace=True, na_values=["", " ", "NA", "NaN", -999, -9999, 9999])
        df.columns = _clean_column_names(df.columns)
    return df


def read_knmi_observations(raw_data: Path) -> pd.DataFrame:
    """Read a single file or concatenate every CSV/TXT file in a directory."""
    raw_data = Path(raw_data)
    LOGGER.info("Reading observations from %s", raw_data)
    if raw_data.is_dir():
        files = sorted(
            p for p in raw_data.iterdir()
            if p.suffix.lower() in {".csv", ".txt", ".dat"}
        )
        if not files:
            raise FileNotFoundError(f"No CSV/TXT/DAT files found in {raw_data}")
        LOGGER.info("Found %d raw files", len(files))
        frames = []
        for k, p in enumerate(files, start=1):
            LOGGER.info("Reading file %d/%d: %s", k, len(files), p.name)
            frames.append(read_one_knmi_file(p))
        out = pd.concat(frames, ignore_index=True)
        LOGGER.info("Observation rows read: %s", f"{len(out):,}")
        return out
    out = read_one_knmi_file(raw_data)
    LOGGER.info("Observation rows read: %s; columns: %s", f"{len(out):,}", list(out.columns))
    return out


def make_daily_maxima(cfg: Config) -> pd.DataFrame:
    """
    Return long daily data with columns:
        date, station, wind_mps

    The station identifier is normalised so that WIGOS-style IDs such as
    0-20000-0-06225 match metadata IDs such as 6225.
    """
    raw = read_knmi_observations(cfg.raw_data)
    raw.columns = _clean_column_names(raw.columns)

    required = {cfg.station_col, cfg.date_col, cfg.value_col}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in observations: {sorted(missing)}")

    df = raw[[cfg.station_col, cfg.date_col, cfg.value_col]].copy()
    df["station"] = normalize_station_id_series(df[cfg.station_col])

    if cfg.retained_station_ids is not None:
        keep = {normalize_station_id_value(s) for s in cfg.retained_station_ids}
        df = df[df["station"].isin(keep)].copy()

    date_as_str = df[cfg.date_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    parsed_dates = pd.to_datetime(date_as_str, format="%Y%m%d", errors="coerce")
    needs_fallback = parsed_dates.isna()
    if needs_fallback.any():
        parsed_dates.loc[needs_fallback] = pd.to_datetime(date_as_str.loc[needs_fallback], errors="coerce")
    df["date"] = parsed_dates
    df[cfg.value_col] = pd.to_numeric(df[cfg.value_col], errors="coerce")
    df["wind_mps"] = df[cfg.value_col] / cfg.wind_scale

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    df = df[(df["date"] >= start) & (df["date"] < end)].copy()
    df = df.dropna(subset=["date", "wind_mps"])

    daily = df.groupby(["date", "station"], as_index=False)["wind_mps"].max()
    daily["station"] = daily["station"].astype(str)
    LOGGER.info(
        "Daily maxima ready: %s rows, %d stations, %s to %s",
        f"{len(daily):,}",
        daily["station"].nunique(),
        daily["date"].min().date() if len(daily) else "NA",
        daily["date"].max().date() if len(daily) else "NA",
    )
    return daily


# ---------------------------------------------------------------------
# Seasons and retained station panel
# ---------------------------------------------------------------------

def add_season_columns(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    month = df["date"].dt.month
    year = df["date"].dt.year

    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([6, 7, 8])],
        ["W", "S"],
        default="",
    )
    df["season_year"] = np.where(
        month.eq(12),
        year + 1,
        year,
    )
    df = df[df["season"].isin(["W", "S"])].copy()
    df["season_year"] = df["season_year"].astype(int)
    return df


def expected_dates_for_season_year(season: str, season_year: int) -> pd.DatetimeIndex:
    if season == "S":
        return pd.date_range(f"{season_year}-06-01", f"{season_year}-08-31", freq="D")
    if season == "W":
        return pd.date_range(f"{season_year - 1}-12-01", f"{season_year}-02-28", freq="D")
    raise ValueError(season)


def keep_complete_calendar_season_years(daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Drop incomplete endpoint season-years: winter 1991 and truncated winter 2026.
    Missing station values are allowed; this checks calendar coverage at network level.
    """
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    keep_keys: set[tuple[str, int]] = set()

    unique_dates = daily[["season", "season_year", "date"]].drop_duplicates()
    for (season, sy), group in unique_dates.groupby(["season", "season_year"]):
        expected = expected_dates_for_season_year(season, int(sy))
        if expected.min() < start or expected.max() >= end:
            continue
        actual = pd.DatetimeIndex(group["date"].sort_values().unique())
        if set(expected).issubset(set(actual)):
            keep_keys.add((season, int(sy)))

    before_keys = set(map(tuple, daily[["season", "season_year"]].drop_duplicates().to_numpy()))
    dropped = sorted(before_keys.difference(keep_keys))
    if dropped:
        LOGGER.info("Dropping incomplete endpoint/network season-years: %s", dropped)
    out = daily[
        daily.apply(lambda r: (r["season"], int(r["season_year"])) in keep_keys, axis=1)
    ].copy()
    LOGGER.info("Seasonal daily panel: %s rows after calendar season-year filtering", f"{len(out):,}")
    return out


def load_station_metadata(cfg: Config, stations: Iterable[str]) -> pd.DataFrame:
    meta = pd.read_csv(cfg.station_meta, skipinitialspace=True)
    meta.columns = _clean_column_names(meta.columns)

    meta_station_col = cfg.meta_station_col or cfg.station_col
    if meta_station_col not in meta.columns and "station_id" in meta.columns:
        meta_station_col = "station_id"

    required = {meta_station_col, cfg.lon_col, cfg.lat_col}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in station metadata: {sorted(missing)}")

    meta = meta.copy()
    meta["station"] = normalize_station_id_series(meta[meta_station_col])
    if cfg.name_col not in meta.columns:
        meta["name"] = meta["station"]
    else:
        meta["name"] = meta[cfg.name_col].astype(str).str.strip()

    stations = [normalize_station_id_value(s) for s in stations]
    meta = meta[meta["station"].isin(stations)].copy()
    meta = meta.rename(columns={cfg.lon_col: "lon", cfg.lat_col: "lat"})
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta = meta.dropna(subset=["lon", "lat"])

    missing_stations = sorted(set(stations).difference(set(meta["station"])))
    if missing_stations:
        raise ValueError(f"No metadata for stations: {missing_stations}")
    LOGGER.info("Loaded metadata for %d retained stations", len(meta))
    return meta[["station", "name", "lat", "lon"]].sort_values("station")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def make_distance_table(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = meta.sort_values("station").reset_index(drop=True)
    for i, j in itertools.combinations(range(len(meta)), 2):
        a = meta.iloc[i]
        b = meta.iloc[j]
        rows.append({
            "station_i": a["station"],
            "station_j": b["station"],
            "distance_km": haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]),
        })
    out = pd.DataFrame(rows)
    LOGGER.info("Distance table ready: %d station pairs", len(out))
    return out


def coverage_table(daily: pd.DataFrame, stations: list[str], cfg: Config) -> pd.DataFrame:
    full_dates = pd.date_range(cfg.start_date, pd.Timestamp(cfg.end_date) - pd.Timedelta(days=1), freq="D")
    daily_wide = daily.pivot_table(index="date", columns="station", values="wind_mps", aggfunc="max")
    daily_wide = daily_wide.reindex(full_dates)
    rows = []
    for station in stations:
        non_missing = int(daily_wide[station].notna().sum()) if station in daily_wide.columns else 0
        rows.append({
            "station": station,
            "non_missing_days": non_missing,
            "calendar_days": len(full_dates),
            "coverage": non_missing / len(full_dates),
        })
    return pd.DataFrame(rows).sort_values("coverage", ascending=False)


# ---------------------------------------------------------------------
# Marginal GEV fits and transformations
# ---------------------------------------------------------------------

def fit_gev_transform(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """
    Fit SciPy's genextreme distribution and return U = Ghat(x).
    SciPy's shape c equals -xi in the usual GEV notation.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        raise ValueError("Too few observations for a stable station-season GEV fit.")

    c, loc, scale = stats.genextreme.fit(x)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid GEV scale estimate.")

    u = stats.genextreme.cdf(x, c=c, loc=loc, scale=scale)
    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)
    params = {"scipy_c": c, "xi": -c, "mu": loc, "sigma": scale, "n": len(x)}
    return u, params


def standardise_margins(
    daily_seasonal: pd.DataFrame,
    date_key: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit each station-season GEV margin and add U and Z columns.
    Z = -1/log(U) has unit-Frechet margins.
    """
    out_frames = []
    param_rows = []

    groups = list(daily_seasonal.groupby(["station", "season"]))
    LOGGER.info("Fitting GEV margins for %d station-season groups", len(groups))
    for k, ((station, season), group) in enumerate(groups, start=1):
        if k == 1 or k == len(groups) or k % 10 == 0:
            LOGGER.info("GEV margin %d/%d: station=%s season=%s n=%d", k, len(groups), station, season, len(group))
        group = group.sort_values(date_key).copy()
        x = group["wind_mps"].to_numpy(dtype=float)
        mask = np.isfinite(x)
        u_values, params = fit_gev_transform(x[mask])

        group["U"] = np.nan
        group.loc[group.index[mask], "U"] = u_values
        group["Z"] = -1.0 / np.log(group["U"])
        out_frames.append(group)
        param_rows.append({"station": station, "season": season, **params})

    transformed = pd.concat(out_frames, ignore_index=True)
    params = pd.DataFrame(param_rows).sort_values(["season", "station"])
    LOGGER.info("GEV standardisation complete: %s transformed rows", f"{len(transformed):,}")
    return transformed, params


# ---------------------------------------------------------------------
# Brown--Resnick / Huesler--Reiss pairwise likelihood
# ---------------------------------------------------------------------

def theta_br(distance_km: np.ndarray | float, rho: float, alpha: float) -> np.ndarray:
    d = np.asarray(distance_km, dtype=float)
    a = np.power(np.maximum(d, 0.0) / rho, alpha / 2.0)
    return 2.0 * stats.norm.cdf(0.5 * a)


def chi_from_theta(theta: np.ndarray | float, u: float = 0.95) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return (1.0 - 2.0 * u + np.power(u, theta)) / (1.0 - u)


def hr_logpdf_unit_frechet(z1: np.ndarray, z2: np.ndarray, distance_km: float, rho: float, alpha: float) -> np.ndarray:
    """
    Huesler--Reiss bivariate density for Brown--Resnick unit-Frechet margins.

    The thesis convention is:
        gamma(d) = (d/rho)^alpha
        a(d) = sqrt(gamma(d)) = (d/rho)^(alpha/2)
        theta(d) = 2 Phi(a(d)/2)

    Exponent measure:
        V = Phi(q1)/z1 + Phi(q2)/z2
        q1 = a/2 + log(z2/z1)/a
        q2 = a/2 + log(z1/z2)/a

    Density:
        f = exp(-V) * (V_1 V_2 - V_12)
          = exp(-V) * [Phi(q1)Phi(q2)/(z1^2 z2^2) + phi(q1)/(a z1^2 z2)]
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    valid = np.isfinite(z1) & np.isfinite(z2) & (z1 > 0) & (z2 > 0)
    ans = np.full_like(z1, fill_value=-np.inf, dtype=float)
    if not np.any(valid):
        return ans

    x = z1[valid]
    y = z2[valid]
    a = (distance_km / rho) ** (alpha / 2.0)
    a = max(float(a), 1e-10)

    logx = np.log(x)
    logy = np.log(y)
    q1 = 0.5 * a + (logy - logx) / a
    q2 = 0.5 * a + (logx - logy) / a

    Phi1 = stats.norm.cdf(q1)
    Phi2 = stats.norm.cdf(q2)
    V = Phi1 / x + Phi2 / y

    log_a_term = log_ndtr(q1) + log_ndtr(q2) - 2.0 * logx - 2.0 * logy
    log_b_term = stats.norm.logpdf(q1) - np.log(a) - 2.0 * logx - logy
    log_density = -V + logsumexp(np.vstack([log_a_term, log_b_term]), axis=0)
    ans[valid] = log_density
    return ans


def prepare_pair_data(z_wide: pd.DataFrame, distances: pd.DataFrame) -> list[tuple[float, np.ndarray, np.ndarray]]:
    pairs = []
    for row in distances.itertuples(index=False):
        si = row.station_i
        sj = row.station_j
        if si not in z_wide.columns or sj not in z_wide.columns:
            continue
        both = z_wide[[si, sj]].dropna()
        if len(both) == 0:
            continue
        pairs.append((float(row.distance_km), both[si].to_numpy(), both[sj].to_numpy()))
    return pairs


def composite_loglik(pair_data: list[tuple[float, np.ndarray, np.ndarray]], rho: float, alpha: float) -> float:
    total = 0.0
    for distance, z1, z2 in pair_data:
        lp = hr_logpdf_unit_frechet(z1, z2, distance, rho, alpha)
        total += float(np.sum(lp[np.isfinite(lp)]))
    return total


def fit_brown_resnick_pairwise(
    z_wide: pd.DataFrame,
    distances: pd.DataFrame,
    dmax: float,
    season_label: str = "?",
    log_every: int = 25,
    optimizer_maxiter: int = 300,
    start_grid: str = "full",
) -> dict[str, float | bool | str]:
    pair_data = prepare_pair_data(z_wide, distances)
    if not pair_data:
        raise ValueError("No usable station pairs for pairwise likelihood.")

    total_pair_observations = sum(len(z1) for _, z1, _ in pair_data)
    bounds = [(1.0, 3.0 * dmax), (0.05, 2.0)]

    if start_grid == "fast":
        rho_starts = [0.25 * dmax, 0.75 * dmax]
        alpha_starts = [0.75, 1.5]
    elif start_grid == "standard":
        rho_starts = [0.25 * dmax, 0.50 * dmax, 1.00 * dmax]
        alpha_starts = [0.5, 1.0, 1.5]
    else:
        rho_starts = [0.10 * dmax, 0.25 * dmax, 0.50 * dmax, 1.00 * dmax, 2.00 * dmax]
        alpha_starts = [0.5, 1.0, 1.5, 2.0]

    starts = list(itertools.product(rho_starts, alpha_starts))
    LOGGER.info(
        "Season %s Brown--Resnick fit: %d pairs, %s pair-days, %d starts, maxiter=%d",
        season_label, len(pair_data), f"{total_pair_observations:,}", len(starts), optimizer_maxiter,
    )

    eval_count = 0
    last_log = time.perf_counter()

    def objective(par: np.ndarray) -> float:
        nonlocal eval_count, last_log
        eval_count += 1
        rho, alpha = float(par[0]), float(par[1])
        if rho <= 0 or alpha <= 0:
            return np.inf
        ll = composite_loglik(pair_data, rho, alpha)
        value = -ll / total_pair_observations
        if log_every > 0 and eval_count % log_every == 0:
            now = time.perf_counter()
            LOGGER.info(
                "Season %s optimisation eval %d: rho=%.2f alpha=%.3f objective=%.6f elapsed_since_last=%.1fs",
                season_label, eval_count, rho, alpha, value, now - last_log,
            )
            last_log = now
        return value

    best = None
    fit_start = time.perf_counter()
    for start_no, (rho0, alpha0) in enumerate(starts, start=1):
        x0 = np.array([
            min(max(rho0, bounds[0][0]), bounds[0][1]),
            min(max(alpha0, bounds[1][0]), bounds[1][1]),
        ])
        LOGGER.info("Season %s start %d/%d: rho0=%.2f alpha0=%.2f", season_label, start_no, len(starts), x0[0], x0[1])
        start_time = time.perf_counter()
        res = optimize.minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": optimizer_maxiter, "ftol": 1e-8},
        )
        LOGGER.info(
            "Season %s start %d/%d done in %.1fs: fun=%.6f rho=%.2f alpha=%.3f success=%s",
            season_label, start_no, len(starts), time.perf_counter() - start_time,
            float(res.fun), float(res.x[0]), float(res.x[1]), bool(res.success),
        )
        if best is None or res.fun < best.fun:
            best = res
    LOGGER.info("Season %s optimisation finished in %.1fs after %d objective evaluations", season_label, time.perf_counter() - fit_start, eval_count)

    assert best is not None
    rho_hat, alpha_hat = map(float, best.x)
    ll_hat = -float(best.fun) * total_pair_observations
    at_boundary = (
        abs(rho_hat - bounds[0][0]) < 1e-5 or
        abs(rho_hat - bounds[0][1]) < 1e-5 or
        abs(alpha_hat - bounds[1][0]) < 1e-5 or
        abs(alpha_hat - bounds[1][1]) < 1e-5
    )
    return {
        "rho": rho_hat,
        "alpha": alpha_hat,
        "loglik": ll_hat,
        "converged": bool(best.success),
        "message": str(best.message),
        "n_pair_observations": int(total_pair_observations),
        "n_objective_evaluations": int(eval_count),
        "start_grid": str(start_grid),
        "optimizer_maxiter": int(optimizer_maxiter),
        "at_boundary": bool(at_boundary),
    }


def fit_by_season(transformed: pd.DataFrame, distances: pd.DataFrame, dmax: float, date_key: str = "date", cfg: Config | None = None) -> pd.DataFrame:
    rows = []
    for season in ["W", "S"]:
        sub = transformed[transformed["season"] == season]
        z_wide = sub.pivot_table(index=date_key, columns="station", values="Z", aggfunc="first")
        fit = fit_brown_resnick_pairwise(
            z_wide,
            distances,
            dmax,
            season_label=season,
            log_every=cfg.log_every if cfg is not None else 25,
            optimizer_maxiter=cfg.optimizer_maxiter if cfg is not None else 300,
            start_grid=cfg.start_grid if cfg is not None else "full",
        )
        rows.append({"season": season, **fit})
    out = pd.DataFrame(rows)
    wide = out.set_index("season")
    delta_rho = float(wide.loc["W", "rho"] - wide.loc["S", "rho"])
    delta_alpha = float(wide.loc["W", "alpha"] - wide.loc["S", "alpha"])
    out["delta_rho_W_minus_S"] = np.where(out["season"].eq("W"), delta_rho, np.nan)
    out["delta_alpha_W_minus_S"] = np.where(out["season"].eq("W"), delta_alpha, np.nan)
    return out


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def pairwise_empirical_diagnostics(
    transformed: pd.DataFrame,
    distances: pd.DataFrame,
    u: float = 0.95,
    date_key: str = "date",
) -> pd.DataFrame:
    rows = []
    for season in ["W", "S"]:
        sub = transformed[transformed["season"] == season]
        u_wide = sub.pivot_table(index=date_key, columns="station", values="U", aggfunc="first")
        for row in distances.itertuples(index=False):
            si, sj, d = row.station_i, row.station_j, float(row.distance_km)
            if si not in u_wide.columns or sj not in u_wide.columns:
                continue
            both = u_wide[[si, sj]].dropna()
            if len(both) < 30:
                continue
            ui = both[si].to_numpy()
            uj = both[sj].to_numpy()

            denom_i = np.sum(ui > u)
            denom_j = np.sum(uj > u)
            chi_i = np.sum((ui > u) & (uj > u)) / denom_i if denom_i > 0 else np.nan
            chi_j = np.sum((ui > u) & (uj > u)) / denom_j if denom_j > 0 else np.nan
            chi_sym = np.nanmean([chi_i, chi_j])

            nu_f = 0.5 * np.mean(np.abs(ui - uj))
            theta_hat = (1.0 + 2.0 * nu_f) / (1.0 - 2.0 * nu_f)
            theta_hat = float(np.clip(theta_hat, 1.0, 2.0))

            rows.append({
                "season": season,
                "station_i": si,
                "station_j": sj,
                "distance_km": d,
                "n_pair_days": int(len(both)),
                "chi_0_95": chi_sym,
                "f_madogram": float(nu_f),
                "theta_fmadogram": theta_hat,
            })
    return pd.DataFrame(rows)


def distance_bin_summary(diag: pd.DataFrame, value_col: str, bin_width_km: int = 25) -> pd.DataFrame:
    max_d = math.ceil(diag["distance_km"].max() / bin_width_km) * bin_width_km
    bins = np.arange(0, max_d + bin_width_km, bin_width_km)
    temp = diag.copy()
    temp["distance_bin"] = pd.cut(temp["distance_km"], bins=bins, include_lowest=True)
    out = (
        temp.groupby(["season", "distance_bin"], observed=True)
            .agg(
                distance_mid=("distance_km", "mean"),
                value_mean=(value_col, "mean"),
                value_median=(value_col, "median"),
                n_pairs=(value_col, "count"),
            )
            .reset_index()
    )
    return out


def daily_extent(transformed: pd.DataFrame, u: float = 0.95) -> pd.DataFrame:
    df = transformed.copy()
    df["is_extreme"] = df["U"] > u
    out = (
        df.groupby(["season", "season_year", "date"])
          .agg(
              observed_stations=("U", "count"),
              extreme_stations=("is_extreme", "sum"),
          )
          .reset_index()
    )
    out["extreme_share"] = out["extreme_stations"] / out["observed_stations"]
    out["any_extreme"] = out["extreme_stations"] > 0
    return out


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_station_map(meta: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 8))
    plt.scatter(meta["lon"], meta["lat"], s=28)
    for row in meta.itertuples(index=False):
        label = str(row.name) if len(str(row.name)) <= 14 else str(row.station)
        plt.annotate(label, (row.lon, row.lat), xytext=(3, 3), textcoords="offset points", fontsize=6)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Retained KNMI stations")
    plt.grid(True, linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_knmi_stations_map.png")


def plot_coverage(coverage: pd.DataFrame, output_dir: Path) -> None:
    cov = coverage.sort_values("coverage")
    plt.figure(figsize=(8, 9))
    plt.barh(cov["station"], 100.0 * cov["coverage"])
    plt.axvline(95, linestyle="--", linewidth=1)
    plt.xlabel("Non-missing daily gust observations (%)")
    plt.ylabel("Station")
    plt.title("Daily data coverage by station")
    plt.xlim(90, 100.5)
    savefig(output_dir / "fig_station_coverage.png")


def plot_annual_seasonal_maxima(daily: pd.DataFrame, output_dir: Path) -> None:
    annual = (
        daily.groupby(["season", "season_year", "date"])
             .agg(network_max_mps=("wind_mps", "max"), network_mean_mps=("wind_mps", "mean"))
             .reset_index()
             .groupby(["season", "season_year"])
             .agg(seasonal_network_max_mps=("network_max_mps", "max"),
                  seasonal_mean_daily_max_mps=("network_mean_mps", "max"))
             .reset_index()
    )
    plt.figure(figsize=(8, 4.8))
    for season, group in annual.groupby("season"):
        label = "Winter DJF" if season == "W" else "Summer JJA"
        plt.plot(group["season_year"], group["seasonal_network_max_mps"], marker="o", linewidth=1, label=label)
    plt.xlabel("Season-year")
    plt.ylabel("Maximum daily gust observed anywhere (m/s)")
    plt.title("Seasonal maximum gusts across the station network")
    plt.legend(frameon=False)
    plt.grid(True, linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_seasonal_network_maxima.png")


def plot_extreme_extent(extent: pd.DataFrame, output_dir: Path) -> None:
    plot_df = extent[extent["any_extreme"]].copy()
    data = [
        100.0 * plot_df.loc[plot_df["season"].eq("W"), "extreme_share"].dropna().to_numpy(),
        100.0 * plot_df.loc[plot_df["season"].eq("S"), "extreme_share"].dropna().to_numpy(),
    ]
    plt.figure(figsize=(6.5, 4.8))
    plt.boxplot(data, labels=["Winter DJF", "Summer JJA"], showfliers=False)
    plt.ylabel("Stations above their station-season 95th percentile (%)")
    plt.title("Spatial footprint on days with at least one extreme station")
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_extreme_spatial_footprint.png")


def plot_chi_diagnostics(diag: pd.DataFrame, fits: pd.DataFrame, output_dir: Path, u: float = 0.95) -> None:
    grid = np.linspace(0, diag["distance_km"].max(), 250)
    binned = distance_bin_summary(diag.dropna(subset=["chi_0_95"]), "chi_0_95")
    fit_lookup = fits.set_index("season")

    plt.figure(figsize=(7.5, 5.0))
    for season in ["W", "S"]:
        label = "Winter DJF" if season == "W" else "Summer JJA"
        sub = diag[diag["season"].eq(season)]
        plt.scatter(sub["distance_km"], sub["chi_0_95"], s=10, alpha=0.20)
        bs = binned[binned["season"].eq(season)]
        plt.plot(bs["distance_mid"], bs["value_mean"], marker="o", linewidth=1.5, label=f"{label}: binned empirical")
        rho = float(fit_lookup.loc[season, "rho"])
        alpha = float(fit_lookup.loc[season, "alpha"])
        plt.plot(grid, chi_from_theta(theta_br(grid, rho, alpha), u=u), linewidth=2, label=f"{label}: fitted BR")
    plt.xlabel("Inter-station distance (km)")
    plt.ylabel(f"Tail-dependence coefficient at u={u:.2f}")
    plt.title("Pairwise tail dependence weakens with distance")
    plt.ylim(-0.02, 1.02)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_chi95_distance_diagnostics.png")


def plot_theta_diagnostics(diag: pd.DataFrame, fits: pd.DataFrame, output_dir: Path) -> None:
    grid = np.linspace(0, diag["distance_km"].max(), 250)
    binned = distance_bin_summary(diag.dropna(subset=["theta_fmadogram"]), "theta_fmadogram")
    fit_lookup = fits.set_index("season")

    plt.figure(figsize=(7.5, 5.0))
    for season in ["W", "S"]:
        label = "Winter DJF" if season == "W" else "Summer JJA"
        sub = diag[diag["season"].eq(season)]
        plt.scatter(sub["distance_km"], sub["theta_fmadogram"], s=10, alpha=0.20)
        bs = binned[binned["season"].eq(season)]
        plt.plot(bs["distance_mid"], bs["value_mean"], marker="o", linewidth=1.5, label=f"{label}: binned empirical")
        rho = float(fit_lookup.loc[season, "rho"])
        alpha = float(fit_lookup.loc[season, "alpha"])
        plt.plot(grid, theta_br(grid, rho, alpha), linewidth=2, label=f"{label}: fitted BR")
    plt.xlabel("Inter-station distance (km)")
    plt.ylabel("Extremal coefficient")
    plt.title("Extremal coefficient by distance")
    plt.ylim(0.98, 2.02)
    plt.legend(frameon=False, fontsize=8)
    plt.grid(True, linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_theta_distance_diagnostics.png")


def plot_bootstrap_delta(boot: pd.DataFrame, output_dir: Path) -> None:
    if boot.empty:
        return
    plt.figure(figsize=(6.5, 4.5))
    plt.hist(boot["delta_rho_W_minus_S"].dropna(), bins=35)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Bootstrap range difference, rho_W - rho_S (km)")
    plt.ylabel("Frequency")
    plt.title("Bootstrap uncertainty for winter-minus-summer range")
    plt.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    savefig(output_dir / "fig_bootstrap_delta_rho.png")


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def bootstrap_season_years(
    daily_seasonal: pd.DataFrame,
    distances: pd.DataFrame,
    dmax: float,
    B: int,
    seed: int,
    log_every: int = 0,
    optimizer_maxiter: int = 150,
    start_grid: str = "standard",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    years_by_season = {
        season: sorted(daily_seasonal.loc[daily_seasonal["season"].eq(season), "season_year"].unique())
        for season in ["W", "S"]
    }

    for b in range(1, B + 1):
        boot_parts = []
        for season in ["W", "S"]:
            years = np.asarray(years_by_season[season])
            sampled = rng.choice(years, size=len(years), replace=True)
            season_data = daily_seasonal[daily_seasonal["season"].eq(season)]
            for k, year in enumerate(sampled):
                block = season_data[season_data["season_year"].eq(int(year))].copy()
                # Unique pseudo-date prevents pivot collisions when a block is sampled twice.
                block["boot_date"] = block["date"].dt.strftime("%Y-%m-%d") + f"__b{b:04d}_{season}_{k:03d}"
                boot_parts.append(block)
        boot_daily = pd.concat(boot_parts, ignore_index=True)
        try:
            boot_transformed, _ = standardise_margins(boot_daily, date_key="boot_date")
            boot_cfg = Config(raw_data=Path("."), station_meta=Path("."), log_every=log_every, optimizer_maxiter=optimizer_maxiter, start_grid=start_grid)
            boot_fits = fit_by_season(boot_transformed, distances, dmax, date_key="boot_date", cfg=boot_cfg)
            wide = boot_fits.set_index("season")
            rows.append({
                "bootstrap": b,
                "rho_W": float(wide.loc["W", "rho"]),
                "alpha_W": float(wide.loc["W", "alpha"]),
                "rho_S": float(wide.loc["S", "rho"]),
                "alpha_S": float(wide.loc["S", "alpha"]),
                "delta_rho_W_minus_S": float(wide.loc["W", "rho"] - wide.loc["S", "rho"]),
                "delta_alpha_W_minus_S": float(wide.loc["W", "alpha"] - wide.loc["S", "alpha"]),
                "converged_W": bool(wide.loc["W", "converged"]),
                "converged_S": bool(wide.loc["S", "converged"]),
            })
        except Exception as exc:  # keep the run going but record the failed replication
            rows.append({"bootstrap": b, "error": repr(exc)})

        if b == 1 or b == B or b % 25 == 0:
            LOGGER.info("Bootstrap replication %d/%d finished", b, B)

    return pd.DataFrame(rows)


def summarise_bootstrap(boot: pd.DataFrame) -> pd.DataFrame:
    good = boot.dropna(subset=["delta_rho_W_minus_S"]).copy()
    if good.empty:
        return pd.DataFrame()

    rows = []
    for col in ["rho_W", "alpha_W", "rho_S", "alpha_S", "delta_rho_W_minus_S", "delta_alpha_W_minus_S"]:
        if col in good.columns:
            rows.append({
                "quantity": col,
                "p2_5": float(np.percentile(good[col], 2.5)),
                "p5": float(np.percentile(good[col], 5.0)),
                "median": float(np.percentile(good[col], 50.0)),
                "p95": float(np.percentile(good[col], 95.0)),
                "p97_5": float(np.percentile(good[col], 97.5)),
            })

    p_boot = (1.0 + np.sum(good["delta_rho_W_minus_S"] <= 0.0)) / (len(good) + 1.0)
    rows.append({
        "quantity": "one_sided_p_delta_rho_le_0",
        "p2_5": np.nan,
        "p5": np.nan,
        "median": float(p_boot),
        "p95": np.nan,
        "p97_5": np.nan,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run_workflow(cfg: Config) -> None:
    setup_logging(cfg.log_level)
    workflow_start = time.perf_counter()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting KNMI thesis workflow; output_dir=%s", cfg.output_dir)
    LOGGER.info("Config: value_col=%s station_col=%s date_col=%s wind_scale=%s bootstrap=%d start_grid=%s", cfg.value_col, cfg.station_col, cfg.date_col, cfg.wind_scale, cfg.bootstrap, cfg.start_grid)

    step = log_step("Step 1/8: read data and compute daily maxima")
    daily = make_daily_maxima(cfg)
    log_step("Step 1/8 complete", step)

    step = log_step("Step 2/8: assign DJF/JJA seasons and keep complete calendar season-years")
    daily = add_season_columns(daily)
    LOGGER.info("Rows after keeping only DJF/JJA: %s", f"{len(daily):,}")
    daily = keep_complete_calendar_season_years(daily, cfg)
    log_step("Step 2/8 complete", step)

    step = log_step("Step 3/8: load station metadata and compute distances")
    stations = sorted(daily["station"].unique())
    meta = load_station_metadata(cfg, stations)
    distances = make_distance_table(meta)
    dmax = float(distances["distance_km"].max())
    LOGGER.info("Network summary: stations=%d pairs=%d dmax=%.2f km", len(stations), len(distances), dmax)
    log_step("Step 3/8 complete", step)

    step = log_step("Step 4/8: compute coverage table")
    coverage = coverage_table(daily, stations, cfg)
    LOGGER.info("Coverage range: %.2f%% to %.2f%%", 100 * coverage["coverage"].min(), 100 * coverage["coverage"].max())
    log_step("Step 4/8 complete", step)

    step = log_step("Step 5/8: fit station-season GEV margins and transform to unit Frechet")
    transformed, gev_params = standardise_margins(daily)
    log_step("Step 5/8 complete", step)

    step = log_step("Step 6/8: fit Brown--Resnick dependence by pairwise composite likelihood")
    fits = fit_by_season(transformed, distances, dmax, cfg=cfg)
    log_step("Step 6/8 complete", step)

    step = log_step("Step 7/8: compute empirical diagnostics")
    diagnostics = pairwise_empirical_diagnostics(transformed, distances, u=cfg.u_level)
    extent = daily_extent(transformed, u=cfg.u_level)
    LOGGER.info("Diagnostics rows: pairwise=%s daily_extent=%s", f"{len(diagnostics):,}", f"{len(extent):,}")
    log_step("Step 7/8 complete", step)

    step = log_step("Step 8/8: save tables and figures")

    # Save tables for thesis reporting.
    daily.to_csv(cfg.output_dir / "daily_maxima_seasonal.csv", index=False)
    meta.to_csv(cfg.output_dir / "station_metadata_retained.csv", index=False)
    distances.to_csv(cfg.output_dir / "station_pair_distances.csv", index=False)
    coverage.to_csv(cfg.output_dir / "station_coverage.csv", index=False)
    gev_params.to_csv(cfg.output_dir / "gev_margin_parameters.csv", index=False)
    fits.to_csv(cfg.output_dir / "brown_resnick_pairwise_fits.csv", index=False)
    diagnostics.to_csv(cfg.output_dir / "pairwise_empirical_diagnostics.csv", index=False)
    extent.to_csv(cfg.output_dir / "daily_extreme_spatial_extent.csv", index=False)

    # Save figures.
    plot_station_map(meta, cfg.output_dir)
    plot_coverage(coverage, cfg.output_dir)
    plot_annual_seasonal_maxima(daily, cfg.output_dir)
    plot_extreme_extent(extent, cfg.output_dir)
    plot_chi_diagnostics(diagnostics, fits, cfg.output_dir, u=cfg.u_level)
    plot_theta_diagnostics(diagnostics, fits, cfg.output_dir)
    log_step("Step 8/8 complete", step)

    if cfg.bootstrap > 0:
        LOGGER.info("Starting season-year bootstrap: B=%d", cfg.bootstrap)
        boot = bootstrap_season_years(
            daily, distances, dmax, B=cfg.bootstrap, seed=cfg.seed,
            log_every=0, optimizer_maxiter=max(80, min(cfg.optimizer_maxiter, 150)), start_grid="standard",
        )
        boot_summary = summarise_bootstrap(boot)
        boot.to_csv(cfg.output_dir / "bootstrap_estimates.csv", index=False)
        boot_summary.to_csv(cfg.output_dir / "bootstrap_summary.csv", index=False)
        plot_bootstrap_delta(boot, cfg.output_dir)

    LOGGER.info("Finished full workflow in %.1f sec. Key outputs are in: %s", time.perf_counter() - workflow_start, cfg.output_dir)
    print("Finished. Key outputs are in:", cfg.output_dir)
    print("Main fit table:")
    print(fits)
    print(f"d_max = {dmax:.2f} km; stations = {len(stations)}; pairs = {len(distances)}")


def parse_station_ids(text: Optional[str]) -> Optional[list[str]]:
    if text is None or text.strip() == "":
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="KNMI wind-gust extremes thesis workflow")
    parser.add_argument("--raw-data", required=True, type=Path, help="KNMI observation CSV/TXT file or directory")
    parser.add_argument("--station-meta", required=True, type=Path, help="Station metadata CSV with station, lat, lon")
    parser.add_argument("--output-dir", default=Path("outputs_knmi_thesis"), type=Path)
    parser.add_argument("--value-col", default="FX", help="Wind-gust column; usually FX for hourly maximum gust")
    parser.add_argument("--station-col", default="STN")
    parser.add_argument("--meta-station-col", default=None, help="Station column in metadata; defaults to --station-col, with fallback to station_id")
    parser.add_argument("--date-col", default="YYYYMMDD")
    parser.add_argument("--lon-col", default="LON")
    parser.add_argument("--lat-col", default="LAT")
    parser.add_argument("--name-col", default="NAME")
    parser.add_argument("--wind-scale", default=10.0, type=float, help="Divide raw gust values by this; KNMI FX is usually 10")
    parser.add_argument("--start-date", default="1991-01-01")
    parser.add_argument("--end-date", default="2026-02-01", help="Exclusive end date")
    parser.add_argument("--u-level", default=0.95, type=float)
    parser.add_argument("--bootstrap", default=0, type=int, help="Number of season-year bootstrap replications")
    parser.add_argument("--seed", default=20260608, type=int)
    parser.add_argument("--retained-station-ids", default=None, help="Optional comma-separated list of already-retained stations")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console logging level")
    parser.add_argument("--log-every", default=25, type=int, help="Log every N objective evaluations during Brown--Resnick optimisation; use 0 to disable inner logs")
    parser.add_argument("--optimizer-maxiter", default=300, type=int, help="Maximum L-BFGS-B iterations per starting value")
    parser.add_argument("--start-grid", default="full", choices=["fast", "standard", "full"], help="Number of multistart values: fast=4, standard=9, full=20. Full is closest to the original thesis run.")
    args = parser.parse_args()

    return Config(
        raw_data=args.raw_data,
        station_meta=args.station_meta,
        output_dir=args.output_dir,
        value_col=args.value_col,
        station_col=args.station_col,
        meta_station_col=args.meta_station_col,
        date_col=args.date_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        name_col=args.name_col,
        wind_scale=args.wind_scale,
        start_date=args.start_date,
        end_date=args.end_date,
        u_level=args.u_level,
        bootstrap=args.bootstrap,
        seed=args.seed,
        retained_station_ids=parse_station_ids(args.retained_station_ids),
        log_level=args.log_level,
        log_every=args.log_every,
        optimizer_maxiter=args.optimizer_maxiter,
        start_grid=args.start_grid,
    )


if __name__ == "__main__":
    run_workflow(parse_args())
