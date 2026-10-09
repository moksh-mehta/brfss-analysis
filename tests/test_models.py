"""Parameter recovery: simulate data from known mu, sigma and check the fit finds them."""

import numpy as np
import pandas as pd
import pytest

from brfss_mrp.fit import run_nuts, state_data
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
    fit = run_nuts(model, state_data(cells, states), cells["y"], seed=1, var_names=["mu", "sigma", "alpha"],
                   coords={"state": states["state_abbr"].tolist()}, dims={"alpha": ["state"], "z": ["state"]},
                   num_chains=2, num_warmup=500, num_samples=500)
    s = fit.summary
    assert s.loc["mu", "hdi_3%"] < MU < s.loc["mu", "hdi_97%"]
    assert s.loc["sigma", "hdi_3%"] < SIGMA < s.loc["sigma", "hdi_97%"]
    assert fit.diagnostics["divergences"] == 0


def test_unobserved_states_follow_population_distribution():
    cells, states, _ = simulate()
    fit = run_nuts(state_intercepts_centered, state_data(cells, states), cells["y"], seed=1,
                   var_names=["mu", "sigma", "alpha"], coords={"state": states["state_abbr"].tolist()},
                   dims={"alpha": ["state"]}, num_chains=2, num_warmup=500, num_samples=500)
    samples = fit.mcmc.get_samples()
    # No data: alpha_s | data ~ Normal(mu, sigma) mixed over the posterior of (mu, sigma),
    # so its SD should be sqrt(E[sigma^2] + Var(mu)).
    expected_sd = float(np.sqrt((samples["sigma"] ** 2).mean() + samples["mu"].var()))
    for s in ["S38", "S39"]:
        assert fit.summary.loc[f"alpha[{s}]", "sd"] == pytest.approx(expected_sd, rel=0.1)
    assert fit.summary.loc["alpha[S0]", "sd"] < expected_sd


def test_demographic_model_recovers_effects():
    from brfss_mrp.fit import demographic_data
    from brfss_mrp.models import demographic_intercepts

    rng = np.random.default_rng(2)
    n_states, n_age, n_educ = 15, 5, 4
    a_state = rng.normal(0, 0.2, n_states)
    a_age = np.array([-1.8, -1.1, -0.2, 0.9, 1.3])
    a_educ = np.array([0.4, 0.0, -0.05, -0.5])
    b_sex, mu = -0.2, -2.4
    grid = np.array(np.meshgrid(range(n_states), range(n_age), [0, 1], range(n_educ),
                                indexing="ij")).reshape(4, -1).T
    cells = pd.DataFrame(grid, columns=["state_idx", "age", "sex", "educ"])
    cells["n"] = rng.integers(50, 300, len(cells))
    eta = (mu + a_state[cells.state_idx] + a_age[cells.age] + a_educ[cells.educ]
           + b_sex * (cells.sex - 0.5))
    cells["y"] = rng.binomial(cells["n"], 1 / (1 + np.exp(-eta)))
    states = pd.DataFrame({"state_abbr": [f"S{i}" for i in range(n_states)]})

    fit = run_nuts(demographic_intercepts, demographic_data(cells, states, n_age, n_educ),
                   cells["y"], seed=3, var_names=["b_sex", "a_age", "a_educ", "mu"],
                   dense_mass=True, num_chains=2, num_warmup=500, num_samples=500)
    s = fit.summary
    assert s.loc["b_sex", "hdi_3%"] < b_sex < s.loc["b_sex", "hdi_97%"]
    # Only mu + a_age and differences between levels are identified; check those.
    post = fit.mcmc.get_samples()
    age_total = np.asarray(post["mu"])[:, None] + np.asarray(post["a_age"])
    age_total += np.asarray(post["a_educ"]).mean(1, keepdims=True)
    true_total = mu + a_age + a_educ.mean()
    lo, hi = np.quantile(age_total, [0.005, 0.995], axis=0)
    assert np.all((lo < true_total) & (true_total < hi))
    assert fit.diagnostics["divergences"] == 0
