"""Fit Model 1 (state varying intercepts, centered) with NUTS and report diagnostics.

Usage:
    python scripts/fit_model1.py

Writes:
    outputs/model1.nc                    ArviZ InferenceData (git-ignored)
    reports/model1_summary.csv           posterior summary with R-hat / ESS (committed)
    reports/model1_diagnostics.json      sampler diagnostics (committed)
    reports/model1_trace.png             trace plot of mu and sigma (committed)
"""

from __future__ import annotations

import numpyro

# Must run before JAX initializes: expose 4 CPU devices so the 4 chains run in parallel.
numpyro.set_host_device_count(4)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import arviz as az  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brfss_mrp.data import YEAR  # noqa: E402
from brfss_mrp.fit import load_cells, passes, run_nuts, state_data  # noqa: E402
from brfss_mrp.models import state_intercepts_centered  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20251


def main() -> None:
    cells, states = load_cells(ROOT / "data" / "processed", YEAR)
    fit = run_nuts(state_intercepts_centered, state_data(cells, states), cells["y"].to_numpy(),
                   seed=SEED, var_names=["mu", "sigma", "alpha"],
                   coords={"state": states["state_abbr"].tolist()}, dims={"alpha": ["state"]})

    (ROOT / "outputs").mkdir(exist_ok=True)
    fit.idata.to_netcdf(ROOT / "outputs" / "model1.nc")
    reports = ROOT / "reports"
    fit.summary.to_csv(reports / "model1_summary.csv")
    diag = {"seed": SEED, **fit.diagnostics, "passes_thresholds": passes(fit.diagnostics)}
    (reports / "model1_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    print(json.dumps(diag, indent=2))
    print("\nHyperparameters:")
    print(fit.summary.loc[["mu", "sigma"]].to_string())
    print("\nState intercepts with the lowest bulk ESS:")
    print(fit.summary.drop(index=["mu", "sigma"]).sort_values("ess_bulk").head(5).to_string())

    az.plot_trace(fit.idata, var_names=["mu", "sigma"], compact=False)
    plt.tight_layout()
    plt.savefig(reports / "model1_trace.png", dpi=110)


if __name__ == "__main__":
    main()
