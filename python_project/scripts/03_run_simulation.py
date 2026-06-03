"""Monte Carlo simulation of seasonal extremal dependence.

Generates approximate Brown-Resnick max-stable fields on the KNMI
station geometry under two regimes (null: identical winter/summer
dependence; alternative: winter has larger range than summer) and
evaluates whether the pairwise F-madogram, theta, and chi_u estimators
recover the true dependence structure in finite samples.

Run from the repository root:

    python python_project/scripts/03_run_simulation.py --quick
    python python_project/scripts/03_run_simulation.py --n-rep 200

Outputs (in python_project/outputs):
    tables/sim_station_geometry.csv
    tables/sim_pair_distances.csv
    tables/sim_pairwise_results.csv
    tables/sim_distance_bin_summary.csv
    tables/sim_detection_summary.csv
    tables/sim_parameters.csv
    figures/sim_theoretical_theta_curves.pdf
    figures/sim_estimated_theta_vs_distance.pdf
    figures/sim_theta_diff_vs_distance.pdf
    figures/sim_chi_u_diff_vs_distance.pdf
    simulation_report.md
    intermediate/sim_pairwise_results.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project imports (add src/ to sys.path; mirrors existing scripts)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config                                        # type: ignore  # noqa: E402
from spatial_utils import (                          # type: ignore  # noqa: E402
    DEFAULT_PANEL_STATION_IDS,
    StationPanel,
    build_pair_table,
    load_station_metadata,
    pairwise_distance_matrix,
    save_panel_geometry,
    select_panel,
)
from dependence_estimators import estimate_all_pairs  # type: ignore  # noqa: E402
from br_simulation import (                          # type: ignore  # noqa: E402
    BRScenario,
    ScenarioPair,
    br_theta,
    br_variogram,
    make_default_scenarios,
    simulate_brown_resnick_field,
)


# ---------------------------------------------------------------------------
# Default settings (quick-test friendly; bump up for headline runs)
# ---------------------------------------------------------------------------

DEFAULT_N_REP: int = 20
DEFAULT_YEARS: int = 35
DEFAULT_DAYS_PER_SEASON: int = 90
DEFAULT_ALPHA: float = 1.0
DEFAULT_RHO_NULL: float = 120.0
DEFAULT_RHO_WINTER_ALT: float = 180.0
DEFAULT_RHO_SUMMER_ALT: float = 60.0
DEFAULT_N_FACTORS: int = 50
DEFAULT_U_GRID: tuple[float, ...] = (0.90, 0.95, 0.98)
DEFAULT_DISTANCE_BIN_EDGES: tuple[float, ...] = (0.0, 50.0, 100.0, 150.0, 200.0, 1e4)
DEFAULT_DETECTION_THRESHOLDS: tuple[float, ...] = (0.0,)
DEFAULT_SEED: int = config.RANDOM_SEED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-rep", type=int, default=DEFAULT_N_REP)
    p.add_argument("--years", type=int, default=DEFAULT_YEARS)
    p.add_argument("--days-per-season", type=int, default=DEFAULT_DAYS_PER_SEASON)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--rho-null", type=float, default=DEFAULT_RHO_NULL)
    p.add_argument("--rho-winter-alt", type=float, default=DEFAULT_RHO_WINTER_ALT)
    p.add_argument("--rho-summer-alt", type=float, default=DEFAULT_RHO_SUMMER_ALT)
    p.add_argument("--n-factors", type=int, default=DEFAULT_N_FACTORS,
                   help="Truncation of Poisson points in spectral simulation.")
    p.add_argument("--n-stations", type=int, default=None,
                   help="Subset the default panel to the first N stations (for quick tests).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke-test: n_rep=5, years=10, n_factors=20.")
    return p.parse_args(argv)


def apply_quick_settings(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.n_rep = max(5, min(args.n_rep, 5))
        args.years = 10
        args.n_factors = 20
    return args


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------

def run_monte_carlo(
    panel: StationPanel,
    pair_table: pd.DataFrame,
    scenarios: dict[str, ScenarioPair],
    n_rep: int,
    n_obs_per_season: int,
    n_factors: int,
    u_grid: tuple[float, ...],
    seed: int,
    logger,
) -> pd.DataFrame:
    """Loop over replications and scenarios, returning long-format results."""
    seed_seq = np.random.SeedSequence(seed)
    children = seed_seq.spawn(n_rep)
    rows: list[pd.DataFrame] = []

    t0 = time.time()
    for rep, child in enumerate(children):
        rep_rng = np.random.default_rng(child)
        for scen_key, scen in scenarios.items():
            for season_name, regime in (("winter", scen.winter), ("summer", scen.summer)):
                X = simulate_brown_resnick_field(
                    panel, rho=regime.rho, alpha=regime.alpha,
                    n_obs=n_obs_per_season, n_factors=n_factors, rng=rep_rng,
                )
                est = estimate_all_pairs(X, pair_table, u_grid=u_grid)
                est["rep"] = rep
                est["scenario"] = scen_key
                est["season"] = season_name
                est["rho_true"] = regime.rho
                est["alpha_true"] = regime.alpha
                rows.append(est)
        if (rep + 1) % max(1, n_rep // 10) == 0 or rep == n_rep - 1:
            elapsed = time.time() - t0
            logger.info(f"Replication {rep + 1}/{n_rep} done ({elapsed:.1f}s)")

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation / summaries
# ---------------------------------------------------------------------------

def add_theoretical_columns(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["theta_true"] = br_theta(df["dist_km"].to_numpy(), df["rho_true"].to_numpy(), alpha)
    df["gamma_true"] = br_variogram(df["dist_km"].to_numpy(), df["rho_true"].to_numpy(), alpha)
    df["theta_bias"] = df["theta_hat"] - df["theta_true"]
    return df


def winter_minus_summer(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Pivot to compute winter-minus-summer differences per (rep, pair, scenario)."""
    idx_cols = ["rep", "scenario", "i", "j", "dist_km"]
    wide = df.pivot_table(index=idx_cols, columns="season", values=value_cols)
    wide.columns = [f"{val}_{season}" for val, season in wide.columns]
    wide = wide.reset_index()
    for v in value_cols:
        wide[f"{v}_diff_W_minus_S"] = wide[f"{v}_winter"] - wide[f"{v}_summer"]
    return wide


def binned_summary(
    df: pd.DataFrame, bin_edges: tuple[float, ...], value_cols: list[str]
) -> pd.DataFrame:
    """Mean / std / quantiles by (scenario, season, distance bin)."""
    df = df.copy()
    df["dist_bin"] = pd.cut(df["dist_km"], bin_edges, include_lowest=True)
    agg = {v: ["mean", "std", "count"] for v in value_cols}
    out = df.groupby(["scenario", "season", "dist_bin"], observed=True).agg(agg)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def detection_summary(
    diff_df: pd.DataFrame, bin_edges: tuple[float, ...]
) -> pd.DataFrame:
    """Probability of detecting winter > summer dependence (i.e. theta_W < theta_S)."""
    diff_df = diff_df.copy()
    diff_df["dist_bin"] = pd.cut(diff_df["dist_km"], bin_edges, include_lowest=True)
    diff_df["correct_direction_theta"] = (diff_df["theta_hat_diff_W_minus_S"] < 0).astype(int)
    chi_cols = [c for c in diff_df.columns if c.startswith("chi_u_") and c.endswith("_diff_W_minus_S")]
    for c in chi_cols:
        u_tag = c.replace("chi_u_", "").replace("_diff_W_minus_S", "")
        diff_df[f"correct_direction_chi_{u_tag}"] = (diff_df[c] > 0).astype(int)

    detect_cols = [c for c in diff_df.columns if c.startswith("correct_direction_")]
    out = (
        diff_df.groupby(["scenario", "dist_bin"], observed=True)[detect_cols]
        .mean()
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_theoretical_theta(
    scenarios: dict[str, ScenarioPair], alpha: float, max_dist_km: float, out_path: Path
) -> None:
    h = np.linspace(0.1, max_dist_km, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (key, scen) in zip(axes, scenarios.items()):
        for season_name, regime in (("winter", scen.winter), ("summer", scen.summer)):
            ax.plot(h, br_theta(h, regime.rho, alpha), label=f"{season_name} (rho={regime.rho:g})")
        ax.set_title(f"Theoretical theta(h)  -  {key}")
        ax.set_xlabel("distance (km)")
        ax.set_ylim(0.95, 2.05)
        ax.axhline(2.0, color="k", lw=0.5, ls="--")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(r"$\theta(h)$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_estimated_theta_vs_distance(
    df: pd.DataFrame, alpha: float, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    scenarios = ["null", "alternative"]
    for ax, scen in zip(axes, scenarios):
        sub = df[df["scenario"] == scen]
        for season, colour in (("winter", "tab:blue"), ("summer", "tab:red")):
            s = sub[sub["season"] == season]
            grp = s.groupby(["i", "j", "dist_km", "rho_true"])
            mean_hat = grp["theta_hat"].mean()
            sd_hat = grp["theta_hat"].std()
            mean_true = grp["theta_true"].mean()
            d = mean_hat.index.get_level_values("dist_km")
            order = np.argsort(d)
            d_sorted = np.asarray(d)[order]
            ax.scatter(d_sorted, mean_hat.values[order], s=10, color=colour, alpha=0.7,
                       label=f"{season} (MC mean)")
            ax.fill_between(d_sorted, (mean_hat - sd_hat).values[order],
                            (mean_hat + sd_hat).values[order], color=colour, alpha=0.15)
            ax.plot(d_sorted, mean_true.values[order], color=colour, lw=1.5, ls="--",
                    label=f"{season} (theoretical)")
        ax.set_title(f"Estimated theta vs distance ({scen})")
        ax.set_xlabel("distance (km)")
        ax.set_ylim(0.95, 2.05)
        ax.axhline(2.0, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\theta$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_theta_diff_vs_distance(diff_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, scen in zip(axes, ("null", "alternative")):
        sub = diff_df[diff_df["scenario"] == scen]
        grp = sub.groupby(["i", "j", "dist_km"])
        mean_diff = grp["theta_hat_diff_W_minus_S"].mean()
        sd_diff = grp["theta_hat_diff_W_minus_S"].std()
        d = mean_diff.index.get_level_values("dist_km")
        order = np.argsort(d)
        d_sorted = np.asarray(d)[order]
        ax.errorbar(d_sorted, mean_diff.values[order], yerr=sd_diff.values[order],
                    fmt="o", ms=3, color="tab:purple", elinewidth=0.5, capsize=2, alpha=0.7,
                    label="MC mean +/- sd")
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_title(f"theta_winter - theta_summer ({scen})")
        ax.set_xlabel("distance (km)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\hat\theta_W - \hat\theta_S$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_chi_diff_vs_distance(
    diff_df: pd.DataFrame, u_grid: tuple[float, ...], out_path: Path
) -> None:
    fig, axes = plt.subplots(len(u_grid), 2, figsize=(11, 3.2 * len(u_grid)), sharex=True)
    if len(u_grid) == 1:
        axes = np.array([axes])
    for i, u in enumerate(u_grid):
        col = f"chi_u_{u:.2f}_diff_W_minus_S"
        if col not in diff_df.columns:
            continue
        for j, scen in enumerate(("null", "alternative")):
            ax = axes[i, j]
            sub = diff_df[diff_df["scenario"] == scen]
            grp = sub.groupby(["i", "j", "dist_km"])
            mean_diff = grp[col].mean()
            sd_diff = grp[col].std()
            d = mean_diff.index.get_level_values("dist_km")
            order = np.argsort(d)
            d_sorted = np.asarray(d)[order]
            ax.errorbar(d_sorted, mean_diff.values[order], yerr=sd_diff.values[order],
                        fmt="o", ms=3, color="tab:green", elinewidth=0.5, capsize=2, alpha=0.7)
            ax.axhline(0.0, color="k", lw=0.7)
            ax.set_title(f"chi_u={u:.2f} W - S ({scen})")
            ax.grid(alpha=0.3)
        axes[i, 0].set_ylabel(r"$\hat\chi^u_W - \hat\chi^u_S$")
    axes[-1, 0].set_xlabel("distance (km)")
    axes[-1, 1].set_xlabel("distance (km)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal markdown renderer (avoids the tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for row in df.itertuples(index=False, name=None):
        body_lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join([header, sep, *body_lines])


def write_report(
    path: Path,
    args: argparse.Namespace,
    scenarios: dict[str, ScenarioPair],
    panel: StationPanel,
    pair_table: pd.DataFrame,
    rmse_bias: pd.DataFrame,
    detect: pd.DataFrame,
    binned: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Simulation report: Brown-Resnick benchmark for seasonal extremal dependence")
    lines.append("")
    lines.append(f"- Stations in panel: {panel.n_stations}")
    lines.append(f"- Pairs: {len(pair_table)}")
    lines.append(f"- Replications: {args.n_rep}")
    lines.append(f"- Observations per season: {args.years * args.days_per_season} "
                 f"(= {args.years} years x {args.days_per_season} days)")
    lines.append(f"- Spectral truncation (n_factors): {args.n_factors}")
    lines.append(f"- alpha: {args.alpha}")
    lines.append("")
    lines.append("## Regimes")
    for key, scen in scenarios.items():
        lines.append(f"- **{key}**: winter rho = {scen.winter.rho}, summer rho = {scen.summer.rho}")
    lines.append("")
    lines.append("## Approximation note")
    lines.append("This run uses an **approximate** Brown-Resnick simulation via the "
                 "spectral representation truncated at a finite number of Poisson points "
                 "and with the Gaussian process anchored at the first station. The "
                 "pairwise variance Var(W(s_i)-W(s_j)) = gamma(s_i, s_j) is preserved exactly, "
                 "so the theoretical pairwise extremal coefficient theta(h) = 2 Phi(sqrt(gamma)/2) "
                 "is preserved. Marginal distributions are only approximately unit Frechet; "
                 "this is absorbed by the empirical rank transform used in the estimators.")
    lines.append("")
    lines.append("## Bias and RMSE of theta")
    lines.append("")
    lines.append(_df_to_markdown(rmse_bias))
    lines.append("")
    lines.append("## Detection summary")
    lines.append("")
    lines.append("Probability that the estimator correctly recovers the sign of the "
                 "winter-summer difference, by distance bin. Under the alternative we "
                 "expect proportions close to 1 (winter stronger); under the null we "
                 "expect proportions near 0.5 (no systematic difference).")
    lines.append("")
    lines.append(_df_to_markdown(detect, floatfmt=".3f"))
    lines.append("")
    lines.append("## Distance-bin summary (theta_hat)")
    lines.append("")
    theta_view = binned[["scenario", "season", "dist_bin",
                         "theta_hat_mean", "theta_true_mean", "theta_hat_std"]].copy()
    theta_view["dist_bin"] = theta_view["dist_bin"].astype(str)
    lines.append(_df_to_markdown(theta_view))
    lines.append("")
    lines.append("## Methodological interpretation")
    lines.append("")
    lines.append("The simulation tests whether the pairwise F-madogram, theta, and "
                 "finite-level chi_u estimators can recover the true spatial extremal "
                 "dependence in finite samples comparable to the empirical KNMI panel. "
                 "Under the alternative, the simulation generates winter fields with a "
                 "larger range parameter than summer, so the true theta_winter(h) is "
                 "strictly below theta_summer(h) for relevant distances. A high "
                 "detection rate under the alternative, together with a near-0.5 "
                 "detection rate under the null, supports the use of these estimators "
                 "in the empirical analysis.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = apply_quick_settings(parse_args(argv))
    logger = config.get_logger("simulation")
    config.ensure_output_dirs()

    logger.info("Loading station metadata.")
    metadata = load_station_metadata()
    panel = select_panel(DEFAULT_PANEL_STATION_IDS, metadata=metadata, n_stations=args.n_stations)
    logger.info(f"Panel: {panel.n_stations} stations.")
    pair_table = build_pair_table(panel)
    logger.info(f"Pair table: {len(pair_table)} pairs; "
                f"distance range {pair_table['dist_km'].min():.1f}-{pair_table['dist_km'].max():.1f} km.")
    save_panel_geometry(panel, pair_table, out_dir=config.TABLES_DIR)

    scenarios = make_default_scenarios(
        alpha=args.alpha,
        rho_null=args.rho_null,
        rho_winter_alt=args.rho_winter_alt,
        rho_summer_alt=args.rho_summer_alt,
    )

    # ----- parameter dump -----
    param_rows = []
    for key, scen in scenarios.items():
        for season_name, regime in (("winter", scen.winter), ("summer", scen.summer)):
            param_rows.append({
                "scenario": key, "season": season_name,
                "rho": regime.rho, "alpha": regime.alpha,
            })
    param_df = pd.DataFrame(param_rows)
    param_df["n_rep"] = args.n_rep
    param_df["n_obs_per_season"] = args.years * args.days_per_season
    param_df["n_factors"] = args.n_factors
    param_df["n_stations"] = panel.n_stations
    param_df["seed"] = args.seed
    param_df.to_csv(config.TABLES_DIR / "sim_parameters.csv", index=False)

    n_obs_per_season = args.years * args.days_per_season
    logger.info(f"Running {args.n_rep} replications, {n_obs_per_season} obs/season, "
                f"{args.n_factors} factors.")
    raw = run_monte_carlo(
        panel, pair_table, scenarios,
        n_rep=args.n_rep,
        n_obs_per_season=n_obs_per_season,
        n_factors=args.n_factors,
        u_grid=DEFAULT_U_GRID,
        seed=args.seed,
        logger=logger,
    )
    raw = add_theoretical_columns(raw, alpha=args.alpha)

    # ----- save raw -----
    raw.to_csv(config.TABLES_DIR / "sim_pairwise_results.csv", index=False)
    try:
        raw.to_parquet(config.INTERMEDIATE_DIR / "sim_pairwise_results.parquet", index=False)
    except Exception as exc:
        logger.warning(f"Parquet save skipped ({exc}); CSV is sufficient.")

    # ----- bias / RMSE summary -----
    bias = (
        raw.assign(theta_se=lambda d: (d["theta_hat"] - d["theta_true"]) ** 2)
        .groupby(["scenario", "season"])
        .agg(theta_bias_mean=("theta_bias", "mean"),
             theta_rmse=("theta_se", lambda x: float(np.sqrt(np.mean(x)))),
             n_pairs=("theta_hat", "count"))
        .reset_index()
    )
    bias.to_csv(config.TABLES_DIR / "sim_bias_rmse_theta.csv", index=False)

    # ----- winter-minus-summer differences -----
    chi_cols = [c for c in raw.columns if c.startswith("chi_u_")]
    diff = winter_minus_summer(raw, value_cols=["theta_hat", "theta_true"] + chi_cols)
    diff.to_csv(config.TABLES_DIR / "sim_pairwise_diff.csv", index=False)

    # ----- distance-bin summaries -----
    bin_edges = DEFAULT_DISTANCE_BIN_EDGES
    binned = binned_summary(raw, bin_edges, value_cols=["theta_hat", "theta_true"] + chi_cols)
    binned["dist_bin"] = binned["dist_bin"].astype(str)
    binned.to_csv(config.TABLES_DIR / "sim_distance_bin_summary.csv", index=False)

    detect = detection_summary(diff, bin_edges)
    detect["dist_bin"] = detect["dist_bin"].astype(str)
    detect.to_csv(config.TABLES_DIR / "sim_detection_summary.csv", index=False)

    # ----- figures -----
    max_dist = float(pair_table["dist_km"].max())
    plot_theoretical_theta(scenarios, args.alpha, max_dist,
                           config.FIGURES_DIR / "sim_theoretical_theta_curves.pdf")
    plot_estimated_theta_vs_distance(raw, args.alpha,
                                     config.FIGURES_DIR / "sim_estimated_theta_vs_distance.pdf")
    plot_theta_diff_vs_distance(diff, config.FIGURES_DIR / "sim_theta_diff_vs_distance.pdf")
    plot_chi_diff_vs_distance(diff, DEFAULT_U_GRID,
                              config.FIGURES_DIR / "sim_chi_u_diff_vs_distance.pdf")

    # ----- report -----
    report_path = config.OUTPUT_DIR / "simulation_report.md"
    write_report(report_path, args, scenarios, panel, pair_table, bias, detect, binned)
    logger.info(f"Report written to {report_path}")

    # ----- console headline -----
    print("\n=== Headline detection rates (correct sign of theta_W - theta_S) ===")
    print(detect.to_string(index=False))
    print("\n=== Theta bias / RMSE by regime/season ===")
    print(bias.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())