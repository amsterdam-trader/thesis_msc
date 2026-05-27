Prefer Python.
Use pandas, numpy, scipy, statsmodels, xarray if needed, pyarrow/parquet.
Write notebook-friendly code in clear chunks.
Separate reusable functions into src/ when building larger workflows.
Always include sanity checks for missing data, station counts, date ranges, and duplicated timestamps. 
Work in the python_project folder.

# Code location rule

All generated Python code must be placed inside the `python_project/` directory.

Do not place Python scripts inside the LaTeX thesis folder.

Preferred structure:

python_project/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_seasonal_blocks.ipynb
│   ├── 03_extremal_dependence.ipynb
│   └── 04_results_figures.ipynb
├── src/
│   ├── data_loading.py
│   ├── seasons.py
│   ├── extremes.py
│   ├── station_pairs.py
│   ├── dependence_estimators.py
│   ├── plotting.py
│   └── config.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── intermediate/
└── README.md

All figures should be saved to:

python_project/outputs/figures/

All tables should be saved to:

python_project/outputs/tables/

When a figure or table is ready for the thesis, provide the LaTeX include command, for example:

\includegraphics[width=0.8\textwidth]{../python_project/outputs/figures/winter_summer_dependence.pdf}

# Repository rule

- Thesis prose belongs in `thesis/*.tex`.
- The master LaTeX file is `thesis-main.tex`.
- Python code belongs in `python_project/`.
- Reusable Python functions belong in `python_project/src/`.
- Notebooks belong in `python_project/notebooks/`.
- Generated figures and tables should first be saved in `python_project/outputs/`.
- Final thesis figures/tables may be copied or exported to `thesis/figures/` and `thesis/tables/` if needed.
- Claude should never mix Python code into `.tex` files except for short pseudocode or algorithm environments.