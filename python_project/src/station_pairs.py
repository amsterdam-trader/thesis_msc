"""Construct station pairs and pair-level distances.

The headline distance is the great-circle (haversine) distance in
kilometres between the two stations' WGS84 latitude/longitude
coordinates. A projected-distance helper is provided for the
robustness sample.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

import config


EARTH_RADIUS_KM: float = 6371.0088


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float | np.ndarray, lon1: float | np.ndarray,
    lat2: float | np.ndarray, lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Great-circle distance in km. Vectorised; broadcasts naturally."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def projected_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> float:
    """Approximate equirectangular projection at the midpoint latitude.

    Adequate for the Dutch domain; not for global use.
    """
    lat_mid = np.radians(0.5 * (lat1 + lat2))
    dx = np.radians(lon2 - lon1) * np.cos(lat_mid) * EARTH_RADIUS_KM
    dy = np.radians(lat2 - lat1) * EARTH_RADIUS_KM
    return float(np.sqrt(dx * dx + dy * dy))


# ---------------------------------------------------------------------------
# Pair table
# ---------------------------------------------------------------------------

def all_pairs(
    station_meta: pd.DataFrame,
    *,
    id_col: str = "station_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Return a DataFrame with one row per unordered station pair.

    Columns:
        station_i, station_j, lat_i, lon_i, lat_j, lon_j, distance_km

    Requires ``station_meta`` to have unique ids and no NA in the
    coordinate columns.
    """
    needed = {id_col, lat_col, lon_col}
    if not needed.issubset(station_meta.columns):
        missing = needed - set(station_meta.columns)
        raise ValueError(f"station_meta is missing columns: {missing}")
    if station_meta[[id_col, lat_col, lon_col]].isna().any().any():
        raise ValueError("station_meta has NA in id/lat/lon columns")
    if station_meta[id_col].duplicated().any():
        dups = station_meta[id_col][station_meta[id_col].duplicated()].tolist()
        raise ValueError(f"duplicated station ids: {dups[:5]}")

    rows: list[dict] = []
    for (_, ri), (_, rj) in combinations(station_meta.iterrows(), 2):
        rows.append({
            "station_i": ri[id_col],
            "station_j": rj[id_col],
            "lat_i": ri[lat_col], "lon_i": ri[lon_col],
            "lat_j": rj[lat_col], "lon_j": rj[lon_col],
            "distance_km": float(haversine_km(
                ri[lat_col], ri[lon_col], rj[lat_col], rj[lon_col],
            )),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance bins
# ---------------------------------------------------------------------------

def assign_distance_bin(
    distances_km: pd.Series,
    edges: Iterable[float] | None = None,
) -> pd.Series:
    """Return a categorical Series of bin labels of the form "[a, b)" km."""
    edges_t = tuple(edges if edges is not None else config.DEFAULT_DISTANCE_BIN_EDGES_KM)
    return pd.cut(
        distances_km,
        bins=list(edges_t) + [float("inf")],
        right=False,
        include_lowest=True,
    )


def bin_midpoints(edges: Iterable[float] | None = None) -> np.ndarray:
    """Bin midpoints for the same bins used by ``assign_distance_bin``."""
    edges_t = list(edges if edges is not None else config.DEFAULT_DISTANCE_BIN_EDGES_KM)
    mids = [(edges_t[i] + edges_t[i + 1]) / 2.0 for i in range(len(edges_t) - 1)]
    return np.asarray(mids, dtype=float)
