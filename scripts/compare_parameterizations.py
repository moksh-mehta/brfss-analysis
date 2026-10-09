"""Fit Model 1 centered and non-centered with identical settings and compare.

Documents why Model 1 is centered: with thousands of respondents per state,
the non-centered form makes mu and every z_s strongly correlated.

Usage:
    python scripts/compare_parameterizations.py

Writes:
    reports/model1_parameterization.csv
"""

from __future__ import annotations

import numpyro

numpyro.set_host_device_count(4)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brfss_mrp.data import YEAR  # noqa: E402
from brfss_mrp.fit import load_cells, passes, run_nuts  # noqa: E402
from brfss_mrp.models import state_intercepts_centered, state_intercepts_noncentered  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20251


def main() -> None:
    cells, states = load_cells(ROOT / "data" / "processed", YEAR)
    observed = states.loc[states["in_sample"], "state_idx"].to_numpy()
    rows = []
    for name, model in [("centered", state_intercepts_centered),
                        ("non-centered", state_intercepts_noncentered)]:
        fit = run_nuts(model, cells, states, seed=SEED)
        row = {"parameterization": name, **fit.diagnostics,
               "passes_thresholds": passes(fit.diagnostics)}
        for p in ["mu", "sigma"]:
            row[f"{p}_mean"] = fit.summary.loc[p, "mean"]
            row[f"{p}_ess_bulk"] = fit.summary.loc[p, "ess_bulk"]
            row[f"{p}_r_hat"] = fit.summary.loc[p, "r_hat"]
        samples = fit.mcmc.get_samples()
        if "z" in samples:
            mu, z = np.asarray(samples["mu"]), np.asarray(samples["z"])
            corr = [np.corrcoef(mu, z[:, s])[0, 1] for s in observed]
            row["median_corr_mu_z"] = float(np.median(corr))
            row["min_corr_mu_z"] = float(np.min(corr))
        rows.append(row)

    table = pd.DataFrame(rows).set_index("parameterization").T
    table.to_csv(ROOT / "reports" / "model1_parameterization.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
