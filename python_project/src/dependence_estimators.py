"""Pairwise extremal-dependence estimators.

Headline objects:

* Empirical tail dependence coefficient
      chi_u(i, j) = #{U_i > u and U_j > u} / #{U_i > u},
  on rank-transformed hourly observations within a season.

* Empirical extremal coefficient via the F-madogram on block maxima:
      theta(i, j) = (1 + 2 nu_F) / (1 - 2 nu_F),
  where
      nu_F = 0.5 E |F_i(M_i) - F_j(M_j)|.
  See Davison, Padoan, Ribatet (2012).

Both estimators are season-stratified by construction; the seasonal
difference is just a subtraction.

The bootstrap resamples *years*, not hours, to preserve the
within-season dependence structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

import config


logger = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# Wide-format pivot helpers (hourly and block-maxima)
# ---------------------------------------------------------------------------

def hourly_wide_U(
    rank_df: pd.DataFrame,
    *,
    season: str,
    time_col: str = "time",
    station_col: str = "station",
) -> pd.DataFrame:
    """Pivot the long-format rank-transform table to wide:
    rows = hourly timestamps in season, columns = station, values = U.
    """
    sub = rank_df.loc[rank_df["season"] == season, [time_col, station_col, "U"]]
    return sub.pivot_table(index=time_col, columns=station_col, values="U")


def block_maxima_wide_FU(
    bmx: pd.DataFrame,
    *,
    season: str,
    station_col: str = "station",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Mwide, Fwide):
    Mwide = block_max wide  (index = season_year, columns = station)
    Fwide = F_i(block_max)  per-station empirical-CDF of block maxima.
    """
    sub = bmx.loc[bmx["season"] == season,
                  [station_col, "season_year", "block_max"]]
    Mwide = sub.pivot_table(
        index="season_year", columns=station_col, values="block_max",
    )
    Fwide = Mwide.apply(
        lambda col: col.rank(method="average") / (col.notna().sum() + 1.0),
        axis=0,
    )
    return Mwide, Fwide


# ---------------------------------------------------------------------------
# chi_u at fixed level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairChi:
    station_i: str
    station_j: str
    u: float
    chi_hat: float
    n_joint: int       # n hours with both U_i and U_j observed
    n_exceed_i: int    # n hours with U_i > u (and U_j observed)


def chi_at_level_pair(U_i: pd.Series, U_j: pd.Series, u: float) -> PairChi:
    """Empirical chi_u for one station pair.

    Returns the raw counts so the caller can flag low-information pairs.
    """
    pair = pd.concat([U_i, U_j], axis=1).dropna()
    n_joint = len(pair)
    if n_joint == 0:
        return PairChi(U_i.name, U_j.name, u, float("nan"), 0, 0)
    a = pair.iloc[:, 0]
    b = pair.iloc[:, 1]
    n_exceed_i = int((a > u).sum())
    if n_exceed_i == 0:
        return PairChi(U_i.name, U_j.name, u, float("nan"), n_joint, 0)
    chi = float(((a > u) & (b > u)).sum() / n_exceed_i)
    return PairChi(U_i.name, U_j.name, u, chi, n_joint, n_exceed_i)


def chi_all_pairs(
    U_wide: pd.DataFrame,
    u: float,
    *,
    min_exceedances: int = 30,
) -> pd.DataFrame:
    """Empirical chi_u for every station pair in ``U_wide``.

    Returns columns: station_i, station_j, u, chi_hat, n_joint,
    n_exceed_i, n_exceed_j, low_information.

    Rows for which max(n_exceed_i, n_exceed_j) < ``min_exceedances``
    are flagged with low_information=True; the chi_hat value is
    retained but should be treated cautiously.
    """
    cols = list(U_wide.columns)
    rows: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_i = chi_at_level_pair(U_wide[cols[i]], U_wide[cols[j]], u)
            # Also count exceedances at j so we can flag thin pairs both ways.
            n_ex_j = int((pd.concat([U_wide[cols[i]], U_wide[cols[j]]], axis=1)
                          .dropna().iloc[:, 1] > u).sum())
            rows.append({
                "station_i": cols[i],
                "station_j": cols[j],
                "u": u,
                "chi_hat": r_i.chi_hat,
                "n_joint": r_i.n_joint,
                "n_exceed_i": r_i.n_exceed_i,
                "n_exceed_j": n_ex_j,
                "low_information": max(r_i.n_exceed_i, n_ex_j) < min_exceedances,
            })
    out = pd.DataFrame(rows)
    n_low = int(out["low_information"].sum())
    if n_low:
        logger.warning(
            "chi_all_pairs: %d / %d pairs flagged low_information at u=%.3f",
            n_low, len(out), u,
        )
    return out


def chi_grid_all_pairs(
    U_wide: pd.DataFrame,
    u_grid: Sequence[float] | None = None,
    *,
    min_exceedances: int = 30,
) -> pd.DataFrame:
    """``chi_all_pairs`` for every level in ``u_grid`` (default = config)."""
    levels = tuple(u_grid) if u_grid is not None else config.DEFAULT_TAIL_LEVELS
    return pd.concat(
        [chi_all_pairs(U_wide, u, min_exceedances=min_exceedances) for u in levels],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# theta via F-madogram on block maxima
# ---------------------------------------------------------------------------

def fmadogram_pair(F_i: pd.Series, F_j: pd.Series) -> float:
    """F-madogram of two F-transformed block-maxima series."""
    pair = pd.concat([F_i, F_j], axis=1).dropna()
    if pair.empty:
        return float("nan")
    return float(0.5 * (pair.iloc[:, 0] - pair.iloc[:, 1]).abs().mean())


def theta_from_madogram(nu_F: float) -> float:
    """Convert F-madogram nu_F to bivariate extremal coefficient theta.

    theta = (1 + 2 nu_F) / (1 - 2 nu_F) in (1, 2). Returns NaN if
    nu_F is outside (0, 0.5).
    """
    if not np.isfinite(nu_F) or nu_F <= 0.0 or nu_F >= 0.5:
        return float("nan")
    return float((1.0 + 2.0 * nu_F) / (1.0 - 2.0 * nu_F))


def theta_all_pairs(
    F_wide: pd.DataFrame,
    *,
    min_blocks: int = 5,
) -> pd.DataFrame:
    """Pairwise theta via F-madogram for every column pair in ``F_wide``.

    Rows with fewer than ``min_blocks`` jointly observed block maxima
    are returned but flagged via ``low_information``.
    """
    cols = list(F_wide.columns)
    rows: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = pd.concat([F_wide[cols[i]], F_wide[cols[j]]], axis=1).dropna()
            nu = fmadogram_pair(F_wide[cols[i]], F_wide[cols[j]])
            theta = theta_from_madogram(nu) if pd.notna(nu) else float("nan")
            chi_from_theta = 2.0 - theta if pd.notna(theta) else float("nan")
            rows.append({
                "station_i": cols[i],
                "station_j": cols[j],
                "n_blocks": int(len(pair)),
                "nu_F": nu,
                "theta_hat": theta,
                "chi_from_theta": chi_from_theta,
                "low_information": len(pair) < min_blocks,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance-bin summary
# ---------------------------------------------------------------------------

def summarise_by_distance_bin(
    pair_table: pd.DataFrame,
    pair_distances: pd.DataFrame,
    *,
    value_col: str = "chi_hat",
    edges_km: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Aggregate a pairwise estimator (e.g. chi_hat) by distance bin.

    ``pair_distances`` must include ``station_i, station_j, distance_km``.
    Returns columns: distance_bin, midpoint_km, n_pairs, mean, sd, q25,
    median, q75.
    """
    from station_pairs import assign_distance_bin, bin_midpoints

    joined = pair_table.merge(
        pair_distances[["station_i", "station_j", "distance_km"]],
        on=["station_i", "station_j"], how="left",
    )
    joined["distance_bin"] = assign_distance_bin(joined["distance_km"], edges_km)
    grp = joined.groupby("distance_bin", observed=True)[value_col]
    out = grp.agg(
        n_pairs="size",
        mean="mean",
        sd="std",
        q25=lambda s: s.quantile(0.25),
        median="median",
        q75=lambda s: s.quantile(0.75),
    ).reset_index()
    # Attach midpoints.
    out["midpoint_km"] = bin_midpoints(edges_km)[: len(out)]
    return out


# ---------------------------------------------------------------------------
# Winter - summer difference
# ---------------------------------------------------------------------------

def seasonal_difference(
    chi_winter: pd.DataFrame,
    chi_summer: pd.DataFrame,
    *,
    value_col: str = "chi_hat",
) -> pd.DataFrame:
    """Aligned per-pair difference chi_W - chi_S at the same tail level.

    Returns columns: station_i, station_j, u, chi_W, chi_S, delta_chi.
    """
    keys = ["station_i", "station_j", "u"]
    a = chi_winter[keys + [value_col]].rename(columns={value_col: "chi_W"})
    b = chi_summer[keys + [value_col]].rename(columns={value_col: "chi_S"})
    out = a.merge(b, on=keys, how="outer")
    out["delta_chi"] = out["chi_W"] - out["chi_S"]
    return out


# ---------------------------------------------------------------------------
# Year-resample bootstrap (skeleton)
# ---------------------------------------------------------------------------

def year_resample_bootstrap_chi(
    rank_df: pd.DataFrame,
    *,
    season: str,
    u: float,
    n_rep: int = config.DEFAULT_BOOTSTRAP_REPS,
    rng: np.random.Generator | None = None,
    time_col: str = "time",
    station_col: str = "station",
) -> pd.DataFrame:
    """Bootstrap pairwise chi_u by resampling whole season-years.

    For each replication we draw, with replacement, a set of
    season-years equal in number to the observed set, pool the rank-
    transformed hourly observations in those years, and recompute
    chi_u on every pair. Returns a long-format DataFrame:
        replication, station_i, station_j, u, chi_hat.

    This is the skeleton used by the analysis script; aggregation to
    percentile bands happens in the caller.
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    import seasons as _seasons  # local import to avoid name shadow at module top

    sub = rank_df.loc[rank_df["season"] == season].copy()
    sub["season_year"] = _seasons.assign_season_year(sub[time_col]).astype("Int64")
    sub = sub.dropna(subset=["season_year"])

    years = np.sort(sub["season_year"].dropna().unique().astype(int))
    if len(years) == 0:
        return pd.DataFrame(columns=[
            "replication", "station_i", "station_j", "u", "chi_hat",
        ])

    all_reps: list[pd.DataFrame] = []
    for rep in range(n_rep):
        drawn = rng.choice(years, size=len(years), replace=True)
        # Build a frame that concatenates the drawn years (with duplication).
        parts = [sub.loc[sub["season_year"] == y] for y in drawn]
        rep_df = pd.concat(parts, ignore_index=True)
        U_wide = rep_df.pivot_table(
            index=time_col, columns=station_col, values="U", aggfunc="first",
        )
        tbl = chi_all_pairs(U_wide, u=u, min_exceedances=1)
        tbl.insert(0, "replication", rep)
        all_reps.append(tbl[["replication", "station_i", "station_j", "u", "chi_hat"]])
        if (rep + 1) % max(1, n_rep // 10) == 0:
            logger.info("bootstrap progress: %d/%d", rep + 1, n_rep)
    return pd.concat(all_reps, ignore_index=True)
