"""Parameter recovery: simulate data from known mu, sigma and check the fit finds them."""

import numpy as np
import pandas as pd
import pytest

from brfss_mrp.fit import run_nuts
from brfss_mrp.models import state_intercepts_centered, state_intercepts_noncentered

MU, SIGMA, N_STATES = -1.75, 0.2, 40


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(MU, SIGMA, N_STATES)
    observed = np.arange(N_STATES - 2)  # last two states have no data
    state_idx = np.repeat(observed, 5)  # 5 cells per observed state
    n = rng.integers(50, 400, state_idx.size)
    y = rng.binomial(n, 1 / (1 + np.exp(-alpha[state_idx])))
    cells = pd.DataFrame({"state_idx": state_idx, "n": n, "y": y})
    states = pd.DataFrame({"state_abbr": [f"S{i}" for i in range(N_STATES)]})
    return cells, states, alpha


@pytest.mark.parametrize("model", [state_intercepts_centered, state_intercepts_noncentered])
def test_recovers_hyperparameters(model):
    cells, states, _ = simulate()
    fit = run_nuts(model, cells, states, seed=1, num_chains=2, num_warmup=500, num_samples=500)
    s = fit.summary
    assert s.loc["mu", "hdi_3%"] < MU < s.loc["mu", "hdi_97%"]
    assert s.loc["sigma", "hdi_3%"] < SIGMA < s.loc["sigma", "hdi_97%"]
    assert fit.diagnostics["divergences"] == 0


def test_unobserved_states_follow_population_distribution():
    cells, states, _ = simulate()
    fit = run_nuts(state_intercepts_centered, cells, states, seed=1, num_chains=2,
                   num_warmup=500, num_samples=500)
    samples = fit.mcmc.get_samples()
    # No data: alpha_s | data ~ Normal(mu, sigma) mixed over the posterior of (mu, sigma),
    # so its SD should be sqrt(E[sigma^2] + Var(mu)).
    expected_sd = float(np.sqrt((samples["sigma"] ** 2).mean() + samples["mu"].var()))
    for s in ["S38", "S39"]:
        assert fit.summary.loc[f"alpha[{s}]", "sd"] == pytest.approx(expected_sd, rel=0.1)
    assert fit.summary.loc["alpha[S0]", "sd"] < expected_sd
