"""Model 2 (state, age, education intercepts + sex): prior check, fit, posterior check.

Usage:
    python scripts/fit_model2.py --prior-only   # prior predictive check only
    python scripts/fit_model2.py                # prior check, NUTS fit, posterior check
    python scripts/fit_model2.py --diagonal-mass  # diagnostics only, default diagonal mass matrix

Writes (reports/ is committed, outputs/ is git-ignored):
    reports/model2_prior_predictive.png / .csv
    reports/model2_summary.csv, reports/model2_diagnostics.json
    reports/model2_diagnostics_diagonal_mass.json  (with --diagonal-mass)
    reports/model2_ppc_groups.png / .csv, reports/model2_ppc_cells.json
    outputs/model2.nc
"""

from __future__ import annotations

import numpyro

numpyro.set_host_device_count(4)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import Predictive  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brfss_mrp.data import AGE_LABELS, EDUC_LABELS, SEX_LABELS, YEAR  # noqa: E402
from brfss_mrp.fit import demographic_data, load_cells, passes, run_nuts  # noqa: E402
from brfss_mrp.models import PRIOR_SETS, demographic_intercepts  # noqa: E402
from brfss_mrp.plotting import SERIES, TEXT, TEXT_MUTED, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEED = 20252
NUM_PRIOR_DRAWS = 4000
# mu and the age/education intercepts are strongly correlated (only their sums are
# pinned down by the data), so NUTS adapts a full mass matrix instead of a diagonal one.
DENSE_MASS = True


def prior_predictive(cells, states, data) -> None:
    observed = cells["y"].sum() / cells["n"].sum()
    rows, draws = [], {}
    for k, (name, priors) in enumerate(PRIOR_SETS.items()):
        pred = Predictive(demographic_intercepts, num_samples=NUM_PRIOR_DRAWS)(
            jax.random.PRNGKey(SEED + k), **data, priors=priors)
        p = np.asarray(pred["p"])                     # draws x cells
        overall = np.asarray(pred["y"]).sum(1) / cells["n"].sum()
        # Spread between age groups within a draw: oldest minus youngest, averaged over cells.
        age = cells["age"].to_numpy()
        gap = p[:, age == len(AGE_LABELS) - 1].mean(1) - p[:, age == 0].mean(1)
        draws[name] = (p.ravel(), overall)
        rows.append({
            "priors": name,
            "cell_p_q05": np.quantile(p, 0.05), "cell_p_median": np.median(p),
            "cell_p_q95": np.quantile(p, 0.95),
            "share_cell_p_below_0.001": (p < 0.001).mean(), "share_cell_p_above_0.9": (p > 0.9).mean(),
            "overall_q05": np.quantile(overall, 0.05), "overall_median": np.median(overall),
            "overall_q95": np.quantile(overall, 0.95),
            "overall_observed": observed,
            "observed_overall_percentile": (overall < observed).mean(),
            "age_gap_q05": np.quantile(gap, 0.05), "age_gap_median": np.median(gap),
            "age_gap_q95": np.quantile(gap, 0.95),
        })
    table = pd.DataFrame(rows).set_index("priors").T
    table.to_csv(REPORTS / "model2_prior_predictive.csv")
    print("Prior predictive check:")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 51)
    for (name, (p, overall)), color in zip(draws.items(), SERIES):
        sub = np.random.default_rng(0).choice(p, 200_000, replace=False)
        axes[0].hist(sub, bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{name} priors")
        axes[1].hist(overall, bins=bins, density=True, histtype="step", lw=2, color=color, label=f"{name} priors")
    axes[1].axvline(observed, color=TEXT, lw=1.5, ls="--")
    axes[1].annotate(f"observed {observed:.3f}", (observed, axes[1].get_ylim()[1] * 0.92),
                     xytext=(6, 0), textcoords="offset points", fontsize=8, color=TEXT)
    axes[0].set(title="Prior predictive: cell prevalence", xlabel="p (one draw, one cell)", ylabel="density")
    axes[1].set(title="Prior predictive: overall sample prevalence", xlabel="sum(y_rep) / sum(n)")
    for ax in axes:
        ax.legend(loc="upper right")
    fig.text(0.01, 0.01, "Prior draws only; outcome data not used except the dashed reference line.",
             fontsize=7, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REPORTS / "model2_prior_predictive.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--diagonal-mass", action="store_true",
                        help="Fit with NUTS's default diagonal mass matrix and save only its diagnostics")
    args = parser.parse_args()

    cells, states = load_cells(ROOT / "data" / "processed", YEAR)
    data = demographic_data(cells, states, n_age=len(AGE_LABELS), n_educ=len(EDUC_LABELS))
    if args.diagonal_mass:
        fit = run_nuts(demographic_intercepts, data, cells["y"].to_numpy(), seed=SEED,
                       var_names=VAR_NAMES, dense_mass=False)
        diag = {"seed": SEED, "priors": "chosen", "dense_mass": False, **fit.diagnostics,
                "passes_thresholds": passes(fit.diagnostics)}
        (REPORTS / "model2_diagnostics_diagonal_mass.json").write_text(json.dumps(diag, indent=2) + "\n")
        print(json.dumps(diag, indent=2))
        return
    prior_predictive(cells, states, data)
    if args.prior_only:
        return

    fit = fit_model(cells, states, data)
    posterior_predictive(cells, states, data, fit)


VAR_NAMES = ["mu", "b_sex", "sigma_state", "sigma_age", "sigma_educ", "a_age", "a_educ", "a_state"]


def fit_model(cells, states, data):
    fit = run_nuts(demographic_intercepts, data, cells["y"].to_numpy(), seed=SEED,
                   var_names=VAR_NAMES,
                   coords={"state": states["state_abbr"].tolist(), "age": AGE_LABELS,
                           "educ": EDUC_LABELS},
                   dims={"a_state": ["state"], "a_age": ["age"], "a_educ": ["educ"]},
                   dense_mass=DENSE_MASS)
    (ROOT / "outputs").mkdir(exist_ok=True)
    fit.idata.to_netcdf(ROOT / "outputs" / "model2.nc")
    fit.summary.to_csv(REPORTS / "model2_summary.csv")
    diag = {"seed": SEED, "priors": "chosen", "dense_mass": DENSE_MASS, **fit.diagnostics,
            "passes_thresholds": passes(fit.diagnostics)}
    (REPORTS / "model2_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print("\nNUTS diagnostics:")
    print(json.dumps(diag, indent=2))
    print("\nPopulation-level parameters:")
    print(fit.summary.loc[[v for v in fit.summary.index if not v.startswith("a_state")]].to_string())
    return fit


def group_check(cells, y_rep, keys, label_fn):
    """Observed vs posterior predictive prevalence for each group defined by `keys`."""
    rows = []
    for key, idx in cells.groupby(keys).indices.items():
        n = cells["n"].to_numpy()[idx].sum()
        rep = y_rep[:, idx].sum(1) / n
        obs = cells["y"].to_numpy()[idx].sum() / n
        lo, hi = np.quantile(rep, [0.05, 0.95])
        rows.append({"grouping": " x ".join(keys), "group": label_fn(key), "n": int(n),
                     "observed": obs, "pred_mean": rep.mean(), "pred_q05": lo, "pred_q95": hi,
                     "inside_90": bool(lo <= obs <= hi)})
    return pd.DataFrame(rows)


def posterior_predictive(cells, states, data, fit) -> None:
    y_rep = np.asarray(Predictive(demographic_intercepts, posterior_samples=fit.mcmc.get_samples(),
                                  return_sites=["y"])(jax.random.PRNGKey(SEED + 99), **data)["y"])
    abbr = dict(zip(states["state_idx"], states["state_abbr"]))
    checks = [
        group_check(cells, y_rep, ["age"], lambda k: AGE_LABELS[k]),
        group_check(cells, y_rep, ["educ"], lambda k: EDUC_LABELS[k]),
        group_check(cells, y_rep, ["sex"], lambda k: SEX_LABELS[k]),
        group_check(cells, y_rep, ["age", "sex"], lambda k: f"{AGE_LABELS[k[0]]}, {SEX_LABELS[k[1]]}"),
        group_check(cells, y_rep, ["age", "educ"], lambda k: f"{AGE_LABELS[k[0]]}, {EDUC_LABELS[k[1]]}"),
        group_check(cells, y_rep, ["state_idx"], lambda k: abbr[k]),
    ]
    table = pd.concat(checks, ignore_index=True)
    table.to_csv(REPORTS / "model2_ppc_groups.csv", index=False)
    coverage = table.groupby("grouping", sort=False)["inside_90"].agg(["sum", "size"])
    print("\nPosterior predictive: groups whose observed prevalence is inside the 90% interval")
    print(coverage.to_string())
    print("\nGroups outside their 90% interval:")
    print(table.loc[~table["inside_90"], ["grouping", "group", "n", "observed", "pred_q05", "pred_q95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lo, hi = np.quantile(y_rep, [0.05, 0.95], axis=0)
    inside = (cells["y"].to_numpy() >= lo) & (cells["y"].to_numpy() <= hi)
    cell_stats = {"cells": int(inside.size), "cells_inside_90": int(inside.sum()),
                  "share_inside_90": float(inside.mean())}
    (REPORTS / "model2_ppc_cells.json").write_text(json.dumps(cell_stats, indent=2) + "\n")
    print("\nCell level:", cell_stats)
    plot_ppc(table)


def plot_ppc(table) -> None:
    apply_style()
    panels = ["age", "educ", "age x sex", "age x educ", "state_idx"]
    titles = ["Age", "Education", "Age x sex (not in model)",
              "Age x education (not in model)", "State"]
    fig = plt.figure(figsize=(11, 13))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 2.2, 5])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, :])]
    for ax, key, title in zip(axes, panels, titles):
        t = table[table["grouping"] == key].reset_index(drop=True)
        if key == "state_idx":
            t = t.sort_values("observed").reset_index(drop=True)
            x = np.arange(len(t))
            ax.vlines(x, t["pred_q05"], t["pred_q95"], color=SERIES[0], lw=3, label="posterior predictive 90%")
            ax.scatter(x, t["observed"], s=18, color=SERIES[1], zorder=3, label="observed")
            ax.set_xticks(x, t["group"], fontsize=7)
            ax.set_ylabel("prevalence in sample")
        else:
            y = np.arange(len(t))[::-1]
            ax.hlines(y, t["pred_q05"], t["pred_q95"], color=SERIES[0], lw=3, label="posterior predictive 90%")
            ax.scatter(t["observed"], y, s=18, color=SERIES[1], zorder=3, label="observed")
            ax.set_yticks(y, t["group"], fontsize=7)
            ax.set_xlabel("prevalence in sample")
        miss = (~t["inside_90"]).sum()
        ax.set_title(f"{title}: {miss} of {len(t)} outside", loc="left")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORTS / "model2_ppc_groups.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
