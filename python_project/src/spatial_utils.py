"""Spatial geometry helpers for the simulation study.

Loads KNMI station metadata, normalises station ids to 5-digit strings,
and constructs a haversine pairwise-distance matrix and pair table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import DAILY_MAX_CSV, STATION_METADATA_CSV, TABLES_DIR

EARTH_RADIUS_KM: float = 6371.0088

# Headline empirical panel: 25 stations balanced on DJF 1991-2025 at
# fx_present_share >= 0.80 (see audit_daily_balanced_panels.csv).
# Used as the default geometry for the simulation study.
DEFAULT_PANEL_STATION_IDS: tuple[str, ...] = (
    "06225", "06235", "06240", "06260", "06269", "06270", "06273",
    "06275", "06280", "06283", "06286", "06290", "06310", "06312",
    "06316", "06320", "06330", "06344", "06348", "06350", "06356",
    "06370", "06375", "06380", "06391",
)


@dataclass(frozen=True)
class StationPanel:
    """Container for a fixed station panel used in the simulation."""
    ids: tuple[str, ...]
    names: tuple[str, ...]
    lat: np.ndarray
    lon: np.ndarray

    @property
    def n_stations(self) -> int:
        return len(self.ids)


def normalise_station_id(raw: int | str) -> str:
    """Return a 5-digit zero-padded station id string.

    KNMI station ids appear in the metadata as 4-digit integers (e.g.
    6225) but in audit outputs as 5-digit strings (e.g. "06225"). This
    function unifies the representation.
    """
    s = str(raw).strip()
    return s.zfill(5)


def load_station_metadata(path: Path | str = STATION_METADATA_CSV) -> pd.DataFrame:
    """Load station metadata and add a normalised id column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Station metadata file not found: {path}. "
            f"Ensure data/station_metadata.csv is present."
        )
    df = pd.read_csv(path)
    required = {"station_id", "stationname", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Station metadata missing required columns: {missing}")
    df["station_id_str"] = df["station_id"].apply(normalise_station_id)
    return df


def select_panel(
    ids: Sequence[str] | None = None,
    metadata: pd.DataFrame | None = None,
    n_stations: int | None = None,
) -> StationPanel:
    """Build a StationPanel from station ids.

    Parameters
    ----------
    ids
        Iterable of 5-digit station id strings. If None, uses
        DEFAULT_PANEL_STATION_IDS.
    metadata
        Pre-loaded metadata DataFrame; loaded if None.
    n_stations
        Optionally subset to the first n_stations (useful for quick
        smoke tests).
    """
    if metadata is None:
        metadata = load_station_metadata()
    if ids is None:
        ids = DEFAULT_PANEL_STATION_IDS
    ids = [normalise_station_id(x) for x in ids]
    if n_stations is not None:
        ids = ids[:n_stations]

    sub = metadata.set_index("station_id_str").reindex(ids)
    missing = sub[sub["lat"].isna()].index.tolist()
    if missing:
        raise ValueError(
            f"Station ids not found in metadata: {missing}. "
            f"Check audit outputs and metadata CSV."
        )
    return StationPanel(
        ids=tuple(ids),
        names=tuple(sub["stationname"].astype(str).tolist()),
        lat=sub["lat"].to_numpy(dtype=float),
        lon=sub["lon"].to_numpy(dtype=float),
    )


def load_empirical_panel(
    path: Path | str = DAILY_MAX_CSV,
    expected_n: int | None = 33,
) -> StationPanel:
    """Build the empirical station panel directly from the daily-max file.

    This is the authoritative geometry for the simulation study: it is the
    exact set of stations (and their lat/lon) used in the empirical
    analysis, so the simulated and observed inter-station distances
    coincide (simulation.tex, sec:sim-spatial). Station ids in the daily-max
    file are WMO strings like "0-20000-0-06225"; the trailing five digits
    are the KNMI id. lat/lon are read from the file itself (constant per
    station), so no metadata join is needed.

    Parameters
    ----------
    path
        Daily-max CSV with columns ``station, stationname, lat, lon``.
    expected_n
        If given, assert the panel has exactly this many stations (the
        thesis uses N = 33). Pass ``None`` to skip the check.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Daily-max panel file not found: {path}. "
            f"Expected data/knmi_daily_max_1991_2026.csv."
        )
    df = pd.read_csv(path, usecols=["station", "stationname", "lat", "lon"])
    df["station_id_str"] = df["station"].astype(str).str[-5:].apply(normalise_station_id)
    grouped = (
        df.groupby("station_id_str", as_index=False)
        .agg(stationname=("stationname", "first"),
             lat=("lat", "first"), lon=("lon", "first"))
        .sort_values("station_id_str")
        .reset_index(drop=True)
    )
    if expected_n is not None and len(grouped) != expected_n:
        raise ValueError(
            f"Expected {expected_n} stations in {path.name}, found {len(grouped)}. "
            f"Set expected_n=None to override."
        )
    return StationPanel(
        ids=tuple(grouped["station_id_str"]),
        names=tuple(grouped["stationname"].astype(str)),
        lat=grouped["lat"].to_numpy(dtype=float),
        lon=grouped["lon"].to_numpy(dtype=float),
    )


def haversine_distance_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in km between (lat1, lon1) and (lat2, lon2).

    Inputs are in decimal degrees. Broadcasts numpy-style.
    """
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_distance_matrix(panel: StationPanel) -> np.ndarray:
    """Return the symmetric (n, n) haversine distance matrix in km."""
    lat = panel.lat
    lon = panel.lon
    return haversine_distance_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def build_pair_table(panel: StationPanel) -> pd.DataFrame:
    """Return one row per unordered station pair with the haversine distance."""
    D = pairwise_distance_matrix(panel)
    n = panel.n_stations
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "i": i,
                "j": j,
                "station_i": panel.ids[i],
                "station_j": panel.ids[j],
                "name_i": panel.names[i],
                "name_j": panel.names[j],
                "dist_km": float(D[i, j]),
            })
    return pd.DataFrame(rows).sort_values("dist_km").reset_index(drop=True)


def save_panel_geometry(panel: StationPanel, pair_table: pd.DataFrame, out_dir: Path | str = TABLES_DIR) -> None:
    """Write the panel geometry and pairwise distance table to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = pd.DataFrame({
        "station_id": panel.ids,
        "stationname": panel.names,
        "lat": panel.lat,
        "lon": panel.lon,
    })
    geom.to_csv(out_dir / "sim_station_geometry.csv", index=False)
    pair_table.to_csv(out_dir / "sim_pair_distances.csv", index=False)