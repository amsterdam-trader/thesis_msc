"""Monte Carlo simulation study for seasonal spatial extremal dependence.

Implements the Simulation Study of the thesis (simulation.tex,
methodology.tex). A Brown-Resnick max-stable process on the empirical
33-station KNMI geometry is the controlled data-generating process; the
study checks whether the F-madogram / extremal-coefficient and the
finite-level chi_u estimators, smoothed into dependence-distance curves
and equipped with a seasonal-block bootstrap, recover a known
winter-summer difference (power), avoid spurious differences (size),
have small bias/RMSE and near-nominal bootstrap coverage.

True (target) curves, computed in closed form (br_simulation):
    gamma(d)      = (d / rho) ** alpha
    theta_BR(d)   = 2 * Phi( sqrt(gamma(d)) / 2 )
    chi_u_BR(d)   = (1 - 2u + u ** theta_BR(d)) / (1 - u)

Designs (alpha = 1):
    null         : rho_W = rho_S = 120 km        (no seasonal difference)
    alternative  : rho_W = 180 km, rho_S = 60 km (winter more dependent)

Run from the repository root, e.g.

    python python_project/scripts/run_simulation_study.py --quick
    python python_project/scripts/run_simulation_study.py \
        --n-rep 200 --n-boot 999 --n-obs 500 1500 3150 --jobs -1

Outputs (under python_project/outputs):
    figures/sim_true_curves.pdf                 (headline; expected by simulation.tex)
    figures/sim_estimated_curves_theta.pdf
    figures/sim_estimated_curves_chi0.99.pdf
    figures/sim_difference_curves.pdf
    tables/sim_results.csv, tables/sim_results.tex   (headline tab:sim-results)
    tables/sim_results_detailed.csv                  (all designs/seasons/d0/n)
    tables/sim_bandwidths.csv, tables/sim_run_config.csv
    simulation_study_report.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Single-threaded BLAS per process: the per-replication arrays are small
# (33x33), so thread-level parallelism only oversubscribes the cores that
# joblib already uses for replication-level parallelism. Must be set before
# numpy imports BLAS. Workers (loky) inherit this environment.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

# --- project imports (src on path; also export to child processes) ----------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from joblib import Parallel, delayed                # noqa: E402

import config                                        # noqa: E402
from spatial_utils import (                          # noqa: E402
    build_pair_table, load_empirical_panel, save_panel_geometry,
)
from br_simulation import br_chi_u, br_theta, make_default_scenarios  # noqa: E402
from simulation_pipeline import (                    # noqa: E402
    aggregate, make_context, pairwise_psi, run_replication,
)

HEADLINE_U = 0.99


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-rep", type=int, default=200, help="Monte Carlo replications R.")
    p.add_argument("--n-boot", type=int, default=999, help="Bootstrap replications B (0 disables).")
    p.add_argument("--n-obs", type=int, nargs="+", default=[3150],
                   help="Sample size(s) n per regime (design factor).")
    p.add_argument("--n-years", type=int, default=35, help="Pseudo-season-years per regime.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rho-null", type=float, default=120.0)
    p.add_argument("--rho-winter-alt", type=float, default=180.0)
    p.add_argument("--rho-summer-alt", type=float, default=60.0)
    p.add_argument("--u-grid", type=float, nargs="+", default=[0.95, 0.975, 0.99])
    p.add_argument("--ref-distances", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    p.add_argument("--kernel", choices=["epanechnikov", "gaussian"], default="epanechnikov",
                   help="Smoother kernel. simulation.tex=Epanechnikov, methodology.tex=Gaussian.")
    p.add_argument("--method", choices=["exact", "approx"], default="exact",
                   help="Brown-Resnick simulator (exact DEO algorithm by default).")
    p.add_argument("--n-grid", type=int, default=60, help="Distance grid points for curves.")
    p.add_argument("--boot-alpha", type=float, default=0.05, help="Bootstrap band level.")
    p.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 = all cores).")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--headline-n", type=int, default=None,
                   help="Which n to use for headline table/figures (default: largest).")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: R=6, B=40, n=[500], n-grid=24.")
    return p.parse_args(argv)


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    if args.quick:
        args.n_rep, args.n_boot, args.n_obs, args.n_grid = 6, 40, [500], 24
    return args


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_design(args, panel, scenario, design_name, n_obs, logger):
    """Run R replications of one design at one sample size and aggregate."""
    ctx = make_context(
        panel, ref_distances=args.ref_distances, u_grid=args.u_grid, alpha=args.alpha,
        n_obs=n_obs, n_years=args.n_years, method=args.method, kernel=args.kernel,
        n_grid=args.n_grid, n_boot=args.n_boot, boot_alpha=args.boot_alpha,
    )
    # Independent, reproducible seed per (design, n, replication).
    base = np.random.SeedSequence([args.seed, abs(hash(design_name)) % (2**31), n_obs])
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_rep)]
    t0 = time.time()
    reps = Parallel(n_jobs=args.jobs, prefer="processes")(
        delayed(run_replication)(seed, ctx, scenario) for seed in seeds)
    logger.info(f"  {design_name:11s} n={n_obs:5d}: {args.n_rep} reps in {time.time()-t0:5.1f}s")
    return ctx, aggregate(reps, ctx, scenario, design_name)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_true_curves(scenarios, alpha, out_path, u=HEADLINE_U, dmax=330.0):
    """Closed-form target curves (the figure expected by simulation.tex)."""
    d = np.linspace(1.0, dmax, 300)
    null_w = scenarios["null"].winter
    alt_w, alt_s = scenarios["alternative"].winter, scenarios["alternative"].summer
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    specs = [
        ("theta", r"$\theta_{\mathrm{BR}}(d)$", lambda r: br_theta(d, r, alpha)),
        ("chi", rf"$\chi^{{\mathrm{{BR}}}}_{{{u}}}(d)$", lambda r: br_chi_u(d, r, alpha, u)),
    ]
    for ax, (_, ylab, fn) in zip(axes, specs):
        ax.plot(d, fn(null_w.rho), color="k", lw=2,
                label=fr"null ($\rho_W=\rho_S={null_w.rho:g}$)")
        ax.plot(d, fn(alt_w.rho), color="tab:blue", lw=2,
                label=fr"alt. winter ($\rho_W={alt_w.rho:g}$)")
        ax.plot(d, fn(alt_s.rho), color="tab:red", lw=2,
                label=fr"alt. summer ($\rho_S={alt_s.rho:g}$)")
        ax.set_xlabel("distance $d$ (km)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title("Extremal coefficient")
    axes[1].set_title(f"Finite-level tail dependence ($u={u}$)")
    fig.suptitle(r"Brown--Resnick true dependence--distance curves ($\alpha=1$)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _scatter_pairwise(panel, ctx, scenario, estimand, seed):
    """One representative replication's raw pairwise estimates for overlay."""
    rng = np.random.default_rng(seed)
    from br_simulation import simulate_brown_resnick_field
    out = {}
    for season, regime in (("winter", scenario.winter), ("summer", scenario.summer)):
        X = simulate_brown_resnick_field(ctx.D, regime.rho, ctx.alpha, ctx.n_obs,
                                         method=ctx.method, rng=rng)
        out[season] = pairwise_psi(X, ctx.u_grid, ctx.iu)[estimand]
    return out


def plot_estimated_curves(results, ctx, scenarios, estimand, ylab, out_path, seed):
    """MC-mean smoothed curve + envelope + true curve + raw scatter, null vs alt."""
    m = ctx.estimands.index(estimand)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, design in zip(axes, ("null", "alternative")):
        res = results[design]
        scat = _scatter_pairwise(None, ctx, scenarios[design], estimand, seed)
        for season, col, mean_c, lo_c, hi_c, true_c in (
            ("winter", "tab:blue", res.mean_curve_w[m], res.lo_curve_w[m],
             res.hi_curve_w[m], res.true_curve_w[m]),
            ("summer", "tab:red", res.mean_curve_s[m], res.lo_curve_s[m],
             res.hi_curve_s[m], res.true_curve_s[m]),
        ):
            ax.scatter(ctx.d_pairs, scat[season], s=6, color=col, alpha=0.18, lw=0)
            ax.fill_between(ctx.grid, lo_c, hi_c, color=col, alpha=0.15)
            ax.plot(ctx.grid, mean_c, color=col, lw=1.8, label=f"{season}: MC mean")
            ax.plot(ctx.grid, true_c, color=col, lw=1.3, ls="--", label=f"{season}: true")
        ax.set_title(design)
        ax.set_xlabel("distance $d$ (km)")
        ax.grid(alpha=0.3)
        for d0 in ctx.ref_distances:
            ax.axvline(d0, color="grey", lw=0.5, ls=":")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(ylab)
    fig.suptitle(f"Estimated vs true dependence-distance curves ({ylab})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_difference_curves(results, ctx, out_path):
    """Winter-minus-summer difference curves for theta and chi_0.99."""
    chi_key = f"chi_{HEADLINE_U:g}"
    estimands = [("theta", r"$\Delta_\theta(d)=\theta_W-\theta_S$"),
                 (chi_key, rf"$\Delta_{{\chi}}(d)=\chi_{{{HEADLINE_U},W}}-\chi_{{{HEADLINE_U},S}}$")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, (e, ylab) in enumerate(estimands):
        m = ctx.estimands.index(e)
        for j, design in enumerate(("null", "alternative")):
            ax = axes[i, j]
            res = results[design]
            ax.fill_between(ctx.grid, res.lo_diff[m], res.hi_diff[m],
                            color="tab:purple", alpha=0.18, label="MC 2.5-97.5%")
            ax.plot(ctx.grid, res.mean_diff[m], color="tab:purple", lw=1.8, label="MC mean")
            ax.plot(ctx.grid, res.true_diff[m], color="k", lw=1.3, ls="--", label="true")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_title(f"{e}  -  {design}")
            ax.grid(alpha=0.3)
            if i == 1:
                ax.set_xlabel("distance $d$ (km)")
            if j == 0:
                ax.set_ylabel(ylab)
            ax.legend(fontsize=7)
    fig.suptitle("Winter-summer difference curves", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Headline table (tab:sim-results shape)
# ---------------------------------------------------------------------------

def build_headline_table(detailed: pd.DataFrame, headline_n: int) -> pd.DataFrame:
    """Assemble the thesis table: Design x Measure(theta, chi_0.99) x d0.

    Bias/RMSE/Coverage are reported for the WINTER curve; Size/Power from the
    difference band (size under null, power under alternative).
    """
    chi_key = f"chi_{HEADLINE_U:g}"
    rows = []
    for design in ("null", "alternative"):
        for e, label in (("theta", "theta"), (chi_key, f"chi_{HEADLINE_U}")):
            for d0 in sorted(detailed["d0_km"].unique()):
                w = detailed.query(
                    "design==@design and estimand==@e and season=='winter' and "
                    "d0_km==@d0 and n_obs==@headline_n")
                df = detailed.query(
                    "design==@design and estimand==@e and season=='diff' and "
                    "d0_km==@d0 and n_obs==@headline_n")
                if w.empty:
                    continue
                w = w.iloc[0]; dd = df.iloc[0]
                rows.append({
                    "design": design, "measure": label, "d0_km": d0,
                    "bias": w["bias"], "rmse": w["rmse"],
                    "coverage": w.get("coverage", np.nan),
                    "size_or_power": dd.get("size_or_power", np.nan),
                })
    return pd.DataFrame(rows)


def headline_to_latex(tab: pd.DataFrame, headline_n: int, boot_alpha: float) -> str:
    """Render the headline table as a LaTeX tabular matching tab:sim-results."""
    name = {"null": "Null", "alternative": "Alt."}
    meas = {"theta": r"\(\theta\)", f"chi_{HEADLINE_U}": rf"\(\chi_{{{HEADLINE_U}}}\)"}

    def f(x, nd=4):
        return "--" if pd.isna(x) else f"{x:.{nd}f}"

    lines = [
        r"% Auto-generated by run_simulation_study.py -- do not edit by hand.",
        r"\begin{table}[t]", r"  \centering",
        rf"  \caption{{Simulation results at $n={headline_n}$ per regime "
        rf"({int(round((1-boot_alpha)*100))}\% bootstrap bands). Monte Carlo bias and RMSE of the "
        r"smoothed winter curve against the closed-form target, bootstrap coverage of the "
        r"winter curve, and the band-based detection rate (size under the null, power under the "
        r"alternative) from the winter--summer difference band, at the reference distances "
        r"\(d_0\), for the extremal coefficient \(\theta\) and the finite-level coefficient "
        rf"\(\chi_{{{HEADLINE_U}}}\).}}",
        r"  \label{tab:sim-results}",
        r"  \begin{tabular}{lllcccc}", r"    \hline",
        r"    Design & Measure & \(d_0\) (km) & Bias & RMSE & Coverage & Size/Power \\",
        r"    \hline",
    ]
    prev_design = prev_meas = None
    for _, r in tab.iterrows():
        if r["design"] != prev_design and prev_design is not None:
            lines.append(r"    \hline")
            prev_meas = None
        d_lab = name[r["design"]] if r["design"] != prev_design else ""
        m_lab = meas[r["measure"]] if r["measure"] != prev_meas else ""
        lines.append(
            f"    {d_lab:6s} & {m_lab:14s} & {int(r['d0_km']):3d} & "
            f"{f(r['bias'])} & {f(r['rmse'])} & {f(r['coverage'],3)} & {f(r['size_or_power'],3)} \\\\")
        prev_design, prev_meas = r["design"], r["measure"]
    lines += [r"    \hline", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path, args, panel, pair_table, headline_n, headline_tab,
                 bandwidth_rows, run_seconds):
    L = []
    A = L.append
    A("# Simulation study report: Brown-Resnick benchmark for seasonal extremal dependence")
    A("")
    A(f"_Generated by `scripts/run_simulation_study.py` in {run_seconds:.1f}s._")
    A("")
    A("## Configuration")
    A(f"- Stations (empirical geometry): **{panel.n_stations}** "
      f"(loaded from `data/knmi_daily_max_1991_2026.csv`); pairs: {len(pair_table)}; "
      f"distances {pair_table['dist_km'].min():.0f}-{pair_table['dist_km'].max():.0f} km.")
    A(f"- Brown-Resnick simulator: **{args.method}** "
      f"({'exact extremal-functions algorithm, Dombry-Engelke-Oesting 2016' if args.method=='exact' else 'approximate truncated spectral representation'}).")
    A(f"- alpha = {args.alpha}; null rho_W=rho_S={args.rho_null}; "
      f"alt rho_W={args.rho_winter_alt}, rho_S={args.rho_summer_alt} (km).")
    A(f"- Sample sizes n per regime: {args.n_obs} (headline n = {headline_n}); "
      f"pseudo-season-years: {args.n_years}.")
    A(f"- Tail levels u: {args.u_grid} (headline u = {HEADLINE_U}); "
      f"reference distances d0: {args.ref_distances} km.")
    A(f"- Smoother: local-linear, **{args.kernel}** kernel, LOOCV bandwidth "
      f"(one common bandwidth per estimand, both seasons).")
    A(f"- Monte Carlo replications R = {args.n_rep}; bootstrap replications B = {args.n_boot} "
      f"({int(round((1-args.boot_alpha)*100))}% bands); seed = {args.seed}.")
    A("")
    A("## Headline results (tab:sim-results shape)")
    A("")
    A("Bias/RMSE/Coverage are for the winter curve; Size/Power from the winter-summer "
      "difference band (size under null, power under alternative).")
    A("")
    A(_df_md(headline_tab.rename(columns={
        "design": "Design", "measure": "Measure", "d0_km": "d0 (km)",
        "bias": "Bias", "rmse": "RMSE", "coverage": "Coverage", "size_or_power": "Size/Power"})))
    A("")
    A("## Selected bandwidths (median over replications, km)")
    A("")
    A(_df_md(pd.DataFrame(bandwidth_rows)))
    A("")
    A("## Notes, approximations and flagged thesis mismatches")
    A("")
    A("1. **Exact simulation.** The default DGP uses the exact extremal-functions "
      "algorithm of Dombry, Engelke & Oesting (2016); it reproduces the closed-form "
      "theta_BR(d) and chi_u_BR(d) within Monte Carlo error and gives exact unit-Frechet "
      "margins (see `tests/test_br_simulation_recovers_curves.py`). The older approximate "
      "truncated-spectral simulator remains available via `--method approx` and is clearly "
      "documented; it preserves the pairwise theta exactly but only approximates the margins.")
    A("2. **Kernel inconsistency (thesis).** methodology.tex (sec:meth-distance) specifies a "
      "*Gaussian* kernel; simulation.tex (sec:sim-estimation, Algorithm 1) specifies an "
      "*Epanechnikov* kernel. Both are local-linear with LOOCV bandwidth and the fitted "
      "curves are visually identical. The code supports both (`--kernel`); this run used "
      f"**{args.kernel}**. Recommend reconciling the two sections to one kernel name.")
    A("3. **Station count.** The empirical panel has exactly **33** stations (matches N=33 in "
      "methodology.tex). The stale `DEFAULT_PANEL_STATION_IDS` constant in `spatial_utils` "
      "(25 stations, a different/older sample) is *not* used here; the geometry is read "
      "directly from the daily-max file.")
    A("4. **Tail-level grid.** This study uses u in {0.95, 0.975, 0.99} (thesis), not the "
      "`DEFAULT_TAIL_LEVELS=(0.95,0.97,0.98)` in `config` (which is for a different routine).")
    A("5. **chi_u target.** The finite-level chi_u_BR(d) = (1-2u+u^theta)/(1-u) is the target "
      "(not the limit 2-theta), per simulation.tex; both are implemented in `br_simulation`.")
    A("6. **Bootstrap bandwidth.** The LOOCV bandwidth is selected once per replication on the "
      "point estimate and held fixed across the B bootstrap resamples (standard; avoids "
      "re-selection cost and keeps bands comparable).")
    A("7. **Temporal independence.** Fields are drawn independently across 'days', so the "
      "coverage figures are the favourable (upper-bound) case, exactly as discussed in "
      "simulation.tex (sec:sim-interpretation).")
    A("")
    path.write_text("\n".join(L), encoding="utf-8")


def _df_md(df: pd.DataFrame, floatfmt=".4f") -> str:
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = apply_quick(parse_args(argv))
    logger = config.get_logger("sim_study")
    config.ensure_output_dirs()
    t_start = time.time()

    panel = load_empirical_panel()
    pair_table = build_pair_table(panel)
    save_panel_geometry(panel, pair_table, out_dir=config.TABLES_DIR)
    logger.info(f"Panel: {panel.n_stations} stations, {len(pair_table)} pairs.")

    scenarios = make_default_scenarios(
        alpha=args.alpha, rho_null=args.rho_null,
        rho_winter_alt=args.rho_winter_alt, rho_summer_alt=args.rho_summer_alt)

    headline_n = args.headline_n or max(args.n_obs)
    detailed_rows, bandwidth_rows = [], []
    headline_results, headline_ctx = {}, None

    for n_obs in args.n_obs:
        logger.info(f"--- n = {n_obs} (R={args.n_rep}, B={args.n_boot}) ---")
        for design in ("null", "alternative"):
            ctx, res = run_design(args, panel, scenarios[design], design, n_obs, logger)
            for row in res.table:
                row["n_obs"] = n_obs
                detailed_rows.append(row)
            for e, h in res.median_bw.items():
                bandwidth_rows.append({"n_obs": n_obs, "design": design,
                                       "estimand": e, "median_bw_km": h})
            if n_obs == headline_n:
                headline_results[design] = res
                headline_ctx = ctx

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(config.TABLES_DIR / "sim_results_detailed.csv", index=False)
    pd.DataFrame(bandwidth_rows).to_csv(config.TABLES_DIR / "sim_bandwidths.csv", index=False)

    headline_tab = build_headline_table(detailed, headline_n)
    headline_tab.to_csv(config.TABLES_DIR / "sim_results.csv", index=False)
    (config.TABLES_DIR / "sim_results.tex").write_text(
        headline_to_latex(headline_tab, headline_n, args.boot_alpha), encoding="utf-8")

    # run-config dump
    pd.DataFrame([{
        "n_rep": args.n_rep, "n_boot": args.n_boot, "n_obs": ",".join(map(str, args.n_obs)),
        "headline_n": headline_n, "n_years": args.n_years, "alpha": args.alpha,
        "rho_null": args.rho_null, "rho_winter_alt": args.rho_winter_alt,
        "rho_summer_alt": args.rho_summer_alt, "u_grid": ",".join(map(str, args.u_grid)),
        "ref_distances": ",".join(map(str, args.ref_distances)), "kernel": args.kernel,
        "method": args.method, "boot_alpha": args.boot_alpha,
        "n_stations": panel.n_stations, "seed": args.seed,
    }]).to_csv(config.TABLES_DIR / "sim_run_config.csv", index=False)

    # ----- figures -----
    logger.info("Rendering figures.")
    plot_true_curves(scenarios, args.alpha, config.FIGURES_DIR / "sim_true_curves.pdf")
    if headline_ctx is not None:
        chi_key = f"chi_{HEADLINE_U:g}"
        plot_estimated_curves(headline_results, headline_ctx, scenarios, "theta",
                              r"$\theta(d)$", config.FIGURES_DIR / "sim_estimated_curves_theta.pdf",
                              seed=args.seed + 1)
        plot_estimated_curves(headline_results, headline_ctx, scenarios, chi_key,
                              rf"$\chi_{{{HEADLINE_U}}}(d)$",
                              config.FIGURES_DIR / f"sim_estimated_curves_chi{HEADLINE_U}.pdf",
                              seed=args.seed + 1)
        plot_difference_curves(headline_results, headline_ctx,
                               config.FIGURES_DIR / "sim_difference_curves.pdf")

    run_seconds = time.time() - t_start
    write_report(config.OUTPUT_DIR / "simulation_study_report.md", args, panel, pair_table,
                 headline_n, headline_tab, bandwidth_rows, run_seconds)

    # ----- console summary -----
    print("\n" + "=" * 72)
    print(f"Simulation study complete in {run_seconds:.1f}s")
    print(f"  panel: {panel.n_stations} stations, {len(pair_table)} pairs | "
          f"method={args.method} | kernel={args.kernel}")
    print(f"  R={args.n_rep}, B={args.n_boot}, n={args.n_obs}, headline n={headline_n}")
    print("=" * 72)
    print(f"\nHeadline table (n={headline_n}):")
    print(headline_tab.to_string(index=False,
          formatters={c: (lambda v: f"{v:.4f}" if pd.notna(v) else "--")
                      for c in ["bias", "rmse", "coverage", "size_or_power"]}))
    print(f"\nFigures -> {config.FIGURES_DIR}")
    print(f"Tables  -> {config.TABLES_DIR}")
    print(f"Report  -> {config.OUTPUT_DIR / 'simulation_study_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
