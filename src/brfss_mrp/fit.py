"""Run NUTS on cell data and summarize convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import pandas as pd
from numpyro.infer import MCMC, NUTS

MAX_TREE_DEPTH = 10  # NumPyro's NUTS default


@dataclass
class FitResult:
    mcmc: MCMC
    idata: az.InferenceData
    summary: pd.DataFrame
    diagnostics: dict


def load_cells(processed_dir: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(processed_dir / f"cells_{year}.csv")
    states = pd.read_csv(processed_dir / f"states_{year}.csv")
    return cells, states


def run_nuts(model, cells: pd.DataFrame, states: pd.DataFrame, *, seed: int,
             num_chains: int = 4, num_warmup: int = 1000, num_samples: int = 1000,
             target_accept_prob: float = 0.8) -> FitResult:
    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept_prob), num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains, chain_method="parallel",
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed),
             state_idx=jnp.asarray(cells["state_idx"].to_numpy()),
             n=jnp.asarray(cells["n"].to_numpy()),
             n_states=len(states),
             y=jnp.asarray(cells["y"].to_numpy()),
             extra_fields=("diverging", "num_steps"))

    dims = {"alpha": ["state"], "z": ["state"]}
    idata = az.from_numpyro(mcmc, coords={"state": states["state_abbr"].tolist()},
                            dims={k: v for k, v in dims.items() if k in mcmc.get_samples()})
    summary = az.summary(idata, var_names=["mu", "sigma", "alpha"], round_to=4)

    # Leapfrog steps per draw; 2**MAX_TREE_DEPTH - 1 means NUTS hit the depth limit.
    num_steps = mcmc.get_extra_fields()["num_steps"]
    step_limit = 2**MAX_TREE_DEPTH - 1
    diagnostics = {
        "draws": num_chains * num_samples,
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
        "median_leapfrog_steps": int(jnp.median(num_steps)),
        "max_leapfrog_steps": int(num_steps.max()),
        "draws_at_depth_limit": int((num_steps >= step_limit).sum()),
    }
    return FitResult(mcmc, idata, summary, diagnostics)


def passes(diagnostics: dict, min_ess: float = 400.0, max_r_hat: float = 1.01) -> bool:
    """Standard thresholds: no divergences, R-hat <= 1.01, bulk and tail ESS >= 400."""
    return (diagnostics["divergences"] == 0 and diagnostics["max_r_hat"] <= max_r_hat
            and diagnostics["min_ess_bulk"] >= min_ess and diagnostics["min_ess_tail"] >= min_ess)
