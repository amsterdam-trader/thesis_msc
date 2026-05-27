"""Project-wide configuration.

All empirical choices, paths, and constants live here so that no
notebook or script ever needs to redefine them. Importing this
module performs no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
HOURLY_PARQUET_DIR: Path = DATA_DIR / "yearly_aggregated_FH_FX"
STATION_METADATA_CSV: Path = DATA_DIR / "station_metadata.csv"

PYTHON_PROJECT_DIR: Path = REPO_ROOT / "python_project"
SRC_DIR: Path = PYTHON_PROJECT_DIR / "src"
SCRIPTS_DIR: Path = PYTHON_PROJECT_DIR / "scripts"
NOTEBOOKS_DIR: Path = PYTHON_PROJECT_DIR / "notebooks"

OUTPUT_DIR: Path = PYTHON_PROJECT_DIR / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
INTERMEDIATE_DIR: Path = OUTPUT_DIR / "intermediate"

# A short-cut for "make sure all output dirs exist". Scripts call this
# at startup; importing the module does not.
def ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, INTERMEDIATE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

PRIMARY_VARIABLE: str = "FX"
SECONDARY_VARIABLE: str = "FH"   # not used in the analysis; read for completeness only

# Schema expected from each monthly parquet partition.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "time", "station", "stationname", "lat", "lon", "height", "FH", "FX",
)


# ---------------------------------------------------------------------------
# Sample windows
# ---------------------------------------------------------------------------

# Candidate start years considered in the audit. The final choice is
# pinned by MAIN_SAMPLE_START_YEAR below.
SAMPLE_START_YEAR_CANDIDATES: tuple[int, ...] = (
    1951, 1961, 1971, 1981, 1991, 2001,
)

# Headline sample: 1991-2025. Chosen from the data-audit balanced-panel
# table: 25 stations at fx_present_share >= 0.80 in every year. 2026 is
# excluded because it is partial at the audit date.
MAIN_SAMPLE_START_YEAR: int = 1991
MAIN_SAMPLE_END_YEAR: int = 2025

# Robustness sample: 2001-2025. 38 stations at fx_present_share >= 0.80
# in every year; stronger coverage at the cost of a shorter window.
ROBUSTNESS_SAMPLE_START_YEAR: int = 2001
ROBUSTNESS_SAMPLE_END_YEAR: int = 2025

# The 2026 partition exists on disk but only covers five months. By
# default we exclude it from every analysis sample.
EXCLUDE_PARTIAL_YEAR: int = 2026

# Coverage thresholds for the balanced-panel screen, applied to the
# fraction of non-null FX hours in a (station, year) cell.
COVERAGE_THRESHOLDS: tuple[float, ...] = (0.80, 0.90, 0.95)
DEFAULT_COVERAGE_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

WINTER_MONTHS: tuple[int, ...] = (12, 1, 2)   # DJF
SUMMER_MONTHS: tuple[int, ...] = (6, 7, 8)    # JJA

SEASON_LABELS: tuple[str, ...] = ("W", "S")   # winter, summer

# DJF winter-year convention: December of calendar year y belongs to
# winter year y+1.


# ---------------------------------------------------------------------------
# Marginal / dependence defaults
# ---------------------------------------------------------------------------

MIN_WITHIN_SEASON_COVERAGE: float = 0.80
MAX_BLOCK_MISSING_FRAC: float = 0.10

# Tail levels at which empirical \chi_u is reported.
DEFAULT_TAIL_LEVELS: tuple[float, ...] = (0.95, 0.97, 0.98)
DEFAULT_REFERENCE_TAIL_LEVEL: float = 0.97

# Reference distances (km) for headline scalar summaries.
DEFAULT_REFERENCE_DISTANCES_KM: tuple[float, ...] = (50.0, 100.0, 200.0)

# Distance binning for the dependence-by-distance summary.
DEFAULT_DISTANCE_BIN_EDGES_KM: tuple[float, ...] = (
    0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0,
)


# ---------------------------------------------------------------------------
# Bootstrap / reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 20260101  # arbitrary fixed seed, locked for reproducibility
DEFAULT_BOOTSTRAP_REPS: int = 1000


# ---------------------------------------------------------------------------
# Offshore-station classification
# ---------------------------------------------------------------------------

# Stations whose `stationname` matches any of these patterns
# (case-insensitive substring match) are flagged as offshore platforms
# in classify_station_type. Conservative list, derived from KNMI
# naming conventions; can be tightened during the next pass.
OFFSHORE_STATIONNAME_KEYWORDS: tuple[str, ...] = (
    "FA-",          # production platform suffix
    "AWG-",         # AWG-1 etc.
    "CPP",          # central production platform
    "PLATFORM",
    "EUROPLATFORM",
    "L9-FF",
    "K13-A",
    "K14",
    "F3",
    "F16",
    "J6",
    "P11",
    "Q1",
    "D15",
    "A12",
)
# Latitude/longitude rectangle used as a soft geographic filter for
# Dutch land/coastal stations. Stations outside this box are flagged
# as "outside-NL" by classify_station_type.
NL_BOUNDING_BOX: dict[str, float] = {
    "lat_min": 50.5,
    "lat_max": 53.7,
    "lon_min": 3.0,
    "lon_max": 7.3,
}


# ---------------------------------------------------------------------------
# Default output file names
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputNames:
    # tables
    audit_years_csv:                  str = "audit_years.csv"
    audit_station_year_csv:           str = "audit_station_year.csv"
    audit_station_coverage_pivot_csv: str = "audit_station_coverage_pivot.csv"
    audit_balanced_panels_csv:        str = "audit_balanced_panels.csv"
    seasonal_coverage_csv:            str = "seasonal_coverage.csv"
    station_metadata_csv:             str = "station_metadata_resolved.csv"
    station_pairs_csv:                str = "station_pairs.csv"
    seasonal_block_maxima_parquet:    str = "seasonal_block_maxima.parquet"
    pairwise_chi_csv:                 str = "pairwise_chi.csv"
    pairwise_theta_csv:               str = "pairwise_theta.csv"
    seasonal_summary_csv:             str = "seasonal_summary.csv"
    # figures
    coverage_heatmap_pdf:             str = "station_coverage_heatmap.pdf"
    station_map_pdf:                  str = "station_map.pdf"
    seasonal_coverage_pdf:            str = "seasonal_coverage.pdf"
    chi_vs_distance_pdf:              str = "chi_vs_distance.pdf"
    theta_vs_distance_pdf:            str = "theta_vs_distance.pdf"
    chi_diff_pdf:                     str = "chi_winter_minus_summer.pdf"
    robustness_multipanel_pdf:        str = "robustness_multipanel.pdf"
    # reports
    data_audit_report_md:             str = "data_audit_report.md"
    seasonal_audit_report_md:         str = "seasonal_audit_report.md"


OUTPUTS = OutputNames()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "thesis_wind") -> logging.Logger:
    """Return a configured logger.

    Each call returns the same logger object; the first call configures
    a single stream handler at INFO level. Scripts can override the
    level by calling ``logger.setLevel(logging.DEBUG)``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Sanity check on import
# ---------------------------------------------------------------------------

if not HOURLY_PARQUET_DIR.exists():
    # Do not raise on import; some unit tests may run without the data
    # mounted. Just log a soft warning if the logger has been created.
    pass
