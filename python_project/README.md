# python_project/

Python pipeline for the thesis *Seasonal variation in spatial
extremal dependence of extreme wind gusts at KNMI station locations
in the Netherlands*.

Primary variable: **`FX`** — hourly maximum wind gust, in m/s.
Mean wind speed (`FH`) is read from the same parquet schema but is
not used in any estimator.

## Pipeline overview

```
data/yearly_aggregated_FH_FX/year=YYYY/month=MM/knmi_fh_YYYY_MM.parquet
                       │
                       ▼
           python_project/src/data_loading.py
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    seasons.py   station_metadata.py  station_pairs.py
        │                                  │
        ▼                                  ▼
   extremes.py ──────────────► dependence_estimators.py
        │                                  │
        ▼                                  ▼
                       plotting.py
                            │
                            ▼
              python_project/outputs/
                            │
                            ▼
                   latex_project/thesis/*.tex
```

## Layout

```
python_project/
├── src/                            -- reusable library code
│   ├── config.py                   -- single source of truth for paths,
│   │                                  sample windows, thresholds, seed
│   ├── data_loading.py             -- partitioned parquet reader, WIGOS->5-digit
│   ├── seasons.py                  -- DJF/JJA + winter-year convention
│   ├── station_metadata.py         -- metadata extraction + offshore filter
│   ├── station_pairs.py            -- haversine distance + pair table
│   ├── extremes.py                 -- seasonal block maxima, POT, rank transform
│   ├── dependence_estimators.py    -- chi_u, theta via F-madogram, bootstrap
│   ├── plotting.py                 -- matplotlib helpers (PDF output)
│   └── simulation.py               -- (optional) synthetic data generator
│
├── scripts/                        -- runnable end-to-end pipeline
│   ├── _bootstrap_path.py          -- prepends src/ to sys.path
│   ├── _validate.py                -- lightweight self-checks
│   ├── 01_run_data_audit.py        -- partition inventory + coverage
│   ├── 02_run_seasonal_sample_audit.py  -- DJF/JJA coverage + panel
│   ├── 03_build_seasonal_extremes.py    -- seasonal block maxima
│   ├── 04_estimate_pairwise_dependence.py -- chi_u, theta per season
│   └── 05_make_figures.py          -- headline PDFs for the thesis
│
├── notebooks/                      -- exploratory; not the pipeline
│   ├── 01_data_exploration.ipynb
│   ├── 02_seasonal_blocks.ipynb
│   ├── 03_extremal_dependence.ipynb
│   ├── 04_simulation_design.ipynb
│   ├── 05_results_figures.ipynb
│   └── run_data_audit.py           -- DEPRECATED prototype of script 01
│
├── outputs/                        -- everything generated
│   ├── tables/
│   ├── figures/
│   └── intermediate/
│
├── pyproject.toml
└── README.md
```

## Data path

`data/yearly_aggregated_FH_FX/year=YYYY/month=MM/knmi_fh_YYYY_MM.parquet`

Schema (audit-confirmed):
```
time           datetime64[us]
station        str   (WIGOS-style "0-20000-0-NNNNN")
stationname    str
lat, lon       float64
height         float64
FH             float64
FX             float64
```

## Script execution order

Run from the repository root:

```
python python_project/scripts/_validate.py
python python_project/scripts/01_run_data_audit.py
python python_project/scripts/02_run_seasonal_sample_audit.py
python python_project/scripts/03_build_seasonal_extremes.py
python python_project/scripts/04_estimate_pairwise_dependence.py
python python_project/scripts/05_make_figures.py
```

Each script:
- finds `src/` automatically via `_bootstrap_path.py`,
- reads `config.py` for paths and thresholds,
- writes its outputs under `python_project/outputs/`,
- prints a concise progress log (INFO level),
- raises an explicit error message if a prerequisite output is
  missing.

## Output files

### Tables (`outputs/tables/`)
| File | Produced by | Description |
|---|---|---|
| `audit_years.csv` | 01 | per-year n_stations, FX completeness |
| `audit_station_year.csv` | 01 | per (station, year) FX completeness |
| `audit_station_coverage_pivot.csv` | 01 | stations × years coverage matrix |
| `audit_balanced_panels.csv` | 01 | balanced-panel sizes for 6 start years × 4 thresholds |
| `station_metadata_resolved.csv` | 02 | station metadata with mainland/offshore classification |
| `seasonal_coverage.csv` | 02 | DJF/JJA coverage per (station, season_year) |
| `seasonal_block_maxima_summary.csv` | 03 | block-maxima summary per (station, season) |
| `station_pairs.csv` | 04 | (station_i, station_j, distance_km) table |
| `pairwise_chi.csv` | 04 | empirical chi_u for every pair / season / u |
| `pairwise_theta.csv` | 04 | F-madogram theta per pair / season |

### Intermediate (`outputs/intermediate/`)
| File | Produced by | Description |
|---|---|---|
| `seasonal_block_maxima.parquet` | 03 | tidy block-maxima table used by script 04 |

### Figures (`outputs/figures/`)
| File | Produced by | Description |
|---|---|---|
| `station_coverage_heatmap.pdf` | 01 | station × year FX-coverage heatmap |
| `station_map.pdf` | 05 | station locations + type classification |
| `seasonal_coverage.pdf` | 02 | DJF/JJA coverage scatter per station |
| `chi_vs_distance.pdf` | 05 | pairwise chi_u vs distance, winter / summer |
| `theta_vs_distance.pdf` | 05 | pairwise theta vs distance, winter / summer |
| `chi_winter_minus_summer.pdf` | 05 | difference vs distance |

### Reports (`outputs/`)
| File | Produced by |
|---|---|
| `data_audit_report.md` | 01 |
| `seasonal_audit_report.md` | 02 |

## Main empirical choices (in `config.py`)

| Variable | Value | Notes |
|---|---|---|
| `PRIMARY_VARIABLE` | `"FX"` | hard constraint of the thesis |
| `MAIN_SAMPLE_START_YEAR` | `1991` | from audit balanced-panel table |
| `MAIN_SAMPLE_END_YEAR` | `2025` | last complete calendar year |
| `ROBUSTNESS_SAMPLE_START_YEAR` | `2001` | larger panel, shorter window |
| `ROBUSTNESS_SAMPLE_END_YEAR` | `2025` | |
| `EXCLUDE_PARTIAL_YEAR` | `2026` | partition is only 5 months |
| `DEFAULT_COVERAGE_THRESHOLD` | `0.80` | fraction of expected hours with non-null FX |
| `MIN_WITHIN_SEASON_COVERAGE` | `0.80` | per-season equivalent |
| `MAX_BLOCK_MISSING_FRAC` | `0.10` | for the within-block check |
| `DEFAULT_TAIL_LEVELS` | `(0.95, 0.97, 0.98)` | tail levels for chi_u |
| `DEFAULT_REFERENCE_TAIL_LEVEL` | `0.97` | u_0 for the headline panel |
| `DEFAULT_REFERENCE_DISTANCES_KM` | `(50, 100, 200)` | scalar summary distances |
| `RANDOM_SEED` | `20260101` | locked for reproducibility |
| `DEFAULT_BOOTSTRAP_REPS` | `1000` | year-resample bootstrap |

## Known caveats

1. **June 2003 partition is missing.** All readers skip absent
   partitions silently by default. Coverage statistics correctly
   reflect the gap.
2. **2026 is partial** (5 months at the audit date). Excluded from
   the main and robustness samples by default.
3. **FX is sparse before 1971.** Audit shows fx_missing_share = 0.93
   in 1951. Sample windows start no earlier than 1991.
4. **Offshore stations.** Classified by combining a stationname
   keyword list with an NL bounding box. The list is conservative;
   review `config.OFFSHORE_STATIONNAME_KEYWORDS` against the
   resolved station-metadata table.
5. **Station-id form.** Parquet uses WIGOS `0-20000-0-NNNNN`; the
   normaliser collapses to the bare 5-digit code. The `station_id`
   column in the companion CSV `data/station_metadata.csv` may not
   align with the parquet for every station; the parquet is the
   authoritative source.
6. **Sample window trade-off.** From audit: 1991-onwards gives 25
   balanced mainland-candidate stations at threshold 0.80;
   2001-onwards gives 38 stations. The thesis defaults to the
   longer-window panel; sensitivity to this choice is reported as
   a robustness check.

## Reproducibility

- Every empirical choice lives in `config.py`.
- Random draws use `config.RANDOM_SEED`.
- Outputs are deterministic given the data and the config.
- `_validate.py` exercises the critical paths in <30 seconds and
  should be the first command run on a fresh checkout.
