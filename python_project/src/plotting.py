"""Plotting helpers.

Thin matplotlib wrappers. Each function returns the matplotlib
``Figure`` so the caller can chain further customisation; figures are
optionally saved to a fixed location with ``save_fig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


logger = config.get_logger(__name__)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, filename: str, *, subdir: str | None = None) -> Path:
    """Save ``fig`` under ``config.FIGURES_DIR`` and return the path."""
    config.ensure_output_dirs()
    target = config.FIGURES_DIR / (subdir or "") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    logger.info("figure -> %s", target)
    return target


# ---------------------------------------------------------------------------
# Coverage heatmap
# ---------------------------------------------------------------------------

def coverage_heatmap(
    pivot: pd.DataFrame,
    *,
    title: str = "FX hourly observations: present-share per station-year",
    cmap: str = "viridis",
) -> plt.Figure:
    """Heatmap of a station x year coverage pivot in [0, 1]."""
    M = pivot.to_numpy(dtype=float, na_value=np.nan)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.16 * pivot.shape[0])))
    cm = mpl.colormaps.get_cmap(cmap).copy()
    cm.set_bad("#dddddd")
    im = ax.imshow(M, aspect="auto", cmap=cm, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(c) for c in pivot.index], fontsize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("KNMI station id")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("non-null share")
    return fig


# ---------------------------------------------------------------------------
# Station map
# ---------------------------------------------------------------------------

def station_map(
    meta: pd.DataFrame,
    *,
    id_col: str = "station_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    type_col: str | None = "station_type",
    annotate: bool = True,
    title: str = "KNMI station locations",
) -> plt.Figure:
    """Scatter of station lat/lon. Optional colour by station_type."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if type_col and type_col in meta.columns:
        groups = meta.groupby(type_col)
        cmap = {"mainland": "#1F4E79", "offshore": "#999999",
                "outside-NL": "#cc6677", "unknown": "#aaaaaa"}
        for label, sub in groups:
            ax.scatter(sub[lon_col], sub[lat_col],
                       label=str(label), s=30,
                       c=cmap.get(str(label), "black"))
    else:
        ax.scatter(meta[lon_col], meta[lat_col], s=30, c="#1F4E79")
    if annotate:
        for _, r in meta.iterrows():
            ax.annotate(str(r[id_col]),
                        (r[lon_col], r[lat_col]),
                        fontsize=5, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if type_col and type_col in meta.columns:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    return fig


# ---------------------------------------------------------------------------
# Seasonal coverage summary
# ---------------------------------------------------------------------------

def seasonal_coverage_panel(coverage: pd.DataFrame) -> plt.Figure:
    """Two-panel (winter, summer) coverage scatter: value_share vs year per station."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, season, title in zip(axes, ("W", "S"),
                                 ("Winter (DJF)", "Summer (JJA)")):
        sub = coverage.loc[coverage["season"] == season]
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue
        for station, grp in sub.groupby("station"):
            ax.plot(grp["season_year"], grp["value_share"],
                    marker=".", alpha=0.5, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Season year")
        ax.set_ylim(0.0, 1.05)
        ax.axhline(config.MIN_WITHIN_SEASON_COVERAGE,
                   linestyle="--", color="red", alpha=0.6)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("FX non-null share")
    fig.suptitle("Within-season FX coverage by station-year")
    return fig


# ---------------------------------------------------------------------------
# Dependence vs distance
# ---------------------------------------------------------------------------

def chi_vs_distance(
    pair_chi: pd.DataFrame,
    pair_distances: pd.DataFrame,
    *,
    value_col: str = "chi_hat",
    season_label: str = "",
    ax: plt.Axes | None = None,
    color: str = "#1F4E79",
) -> plt.Axes:
    """Scatter of chi_hat vs distance for one season."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    joined = pair_chi.merge(
        pair_distances[["station_i", "station_j", "distance_km"]],
        on=["station_i", "station_j"], how="inner",
    )
    ax.scatter(joined["distance_km"], joined[value_col],
               s=10, alpha=0.5, color=color)
    ax.set_xlabel("distance (km)")
    ax.set_ylabel(r"$\widehat{\chi}_u$")
    ax.set_ylim(0.0, 1.05)
    if season_label:
        ax.set_title(season_label)
    ax.grid(True, alpha=0.3)
    return ax


def chi_vs_distance_winter_summer(
    chi_winter: pd.DataFrame,
    chi_summer: pd.DataFrame,
    pair_distances: pd.DataFrame,
    *,
    value_col: str = "chi_hat",
) -> plt.Figure:
    """Two-panel winter / summer chi-vs-distance scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    chi_vs_distance(chi_winter, pair_distances, value_col=value_col,
                    season_label="Winter (DJF)", ax=axes[0], color="#1F4E79")
    chi_vs_distance(chi_summer, pair_distances, value_col=value_col,
                    season_label="Summer (JJA)", ax=axes[1], color="#cc7700")
    fig.suptitle(r"Pairwise $\widehat{\chi}_u$ vs. inter-station distance")
    return fig


# ---------------------------------------------------------------------------
# Winter - summer difference
# ---------------------------------------------------------------------------

def chi_difference_vs_distance(
    diff_table: pd.DataFrame,
    pair_distances: pd.DataFrame,
) -> plt.Figure:
    """Scatter of (chi_W - chi_S) vs distance with a zero reference line."""
    joined = diff_table.merge(
        pair_distances[["station_i", "station_j", "distance_km"]],
        on=["station_i", "station_j"], how="inner",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(joined["distance_km"], joined["delta_chi"], s=10, alpha=0.6,
               color="#1F4E79")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("distance (km)")
    ax.set_ylabel(r"$\widehat{\chi}_u^{(W)} - \widehat{\chi}_u^{(S)}$")
    ax.set_title("Winter minus summer pairwise tail dependence")
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# Robustness multi-panel
# ---------------------------------------------------------------------------

def robustness_grid(
    panels: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    pair_distances: pd.DataFrame,
    *,
    value_col: str = "chi_hat",
) -> plt.Figure:
    """Grid of chi-vs-distance panels for several robustness samples.

    ``panels`` maps a label (e.g. "main", "POT", "unbalanced") to
    a tuple ``(chi_winter_df, chi_summer_df)``.
    """
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), sharey=True)
    if n == 1:
        axes = np.atleast_2d(axes)
    for r, (label, (cw, cs)) in enumerate(panels.items()):
        chi_vs_distance(cw, pair_distances, value_col=value_col,
                        season_label=f"{label} - Winter",
                        ax=axes[r, 0], color="#1F4E79")
        chi_vs_distance(cs, pair_distances, value_col=value_col,
                        season_label=f"{label} - Summer",
                        ax=axes[r, 1], color="#cc7700")
    fig.suptitle(r"Pairwise $\widehat{\chi}_u$ under robustness perturbations")
    return fig
