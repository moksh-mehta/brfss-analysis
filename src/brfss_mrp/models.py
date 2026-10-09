"""NumPyro models for cell-level BRFSS data."""

from __future__ import annotations

import jax
import numpyro
import numpyro.distributions as dist


def _hyperpriors():
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.5))       # population mean, logit scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))  # between-state SD, logit scale
    return mu, sigma


def _likelihood(alpha, state_idx, n, y):
    with numpyro.plate("cell", n.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=n, logits=alpha[state_idx]), obs=y)


def state_intercepts_centered(state_idx, n, n_states, y=None):
    """Binomial model with varying intercepts by state, centered.

        y_c     ~ Binomial(n_c, logistic(alpha[state_c]))
        alpha_s ~ Normal(mu, sigma)
        mu      ~ Normal(0, 1.5)
        sigma   ~ HalfNormal(1)

    Preferred here: every observed state has thousands of respondents, so each
    alpha_s is pinned down by its own data and the centered geometry is easy for
    NUTS. See state_intercepts_noncentered and scripts/compare_parameterizations.py.

    States with no respondents still get an alpha; its posterior is just the
    population distribution Normal(mu, sigma).
    """
    mu, sigma = _hyperpriors()
    with numpyro.plate("state", n_states):
        alpha = numpyro.sample("alpha", dist.Normal(mu, sigma))
    _likelihood(alpha, state_idx, n, y)


def state_intercepts_noncentered(state_idx, n, n_states, y=None):
    """Same model, non-centered: alpha_s = mu + sigma * z_s, z_s ~ Normal(0, 1).

    Non-centering removes the funnel between sigma and the alphas, which helps
    when groups have little data. When groups have a lot of data (as here), it
    instead makes mu and every z_s strongly correlated, because their combination
    mu + sigma * z_s is tightly constrained, and NUTS mixes slowly for mu.
    """
    mu, sigma = _hyperpriors()
    with numpyro.plate("state", n_states):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    alpha = numpyro.deterministic("alpha", mu + sigma * z)
    _likelihood(alpha, state_idx, n, y)


# Prior scales for the demographic model, all on the logit scale. "default" reuses
# Model 1's generic weakly informative scales. "chosen" is what Model 2 is fit with:
# mu centered at logit(0.12) ~ -2, a rough prior guess that adult diabetes is low
# but not rare; age kept wide because diabetes rises steeply with age; state, education, and
# sex narrower. Both are compared in the prior predictive check (scripts/fit_model2.py).
PRIOR_SETS = {
    "default": dict(mu_loc=0.0, mu_scale=1.5, sigma_state=1.0, sigma_age=1.0,
                    sigma_educ=1.0, b_sex_scale=1.0),
    "chosen": dict(mu_loc=-2.0, mu_scale=1.0, sigma_state=0.5, sigma_age=1.0,
                   sigma_educ=0.5, b_sex_scale=0.5),
}


def demographic_intercepts(state_idx, age_idx, educ_idx, sex, n, n_states, n_age, n_educ,
                           y=None, priors=PRIOR_SETS["chosen"]):
    """Binomial model with varying intercepts for state, age, and education and a sex effect.

        y_c       ~ Binomial(n_c, logistic(eta_c))
        eta_c     = mu + a_state[s] + a_age[a] + a_educ[e] + b_sex * sex_c
        a_state_s ~ Normal(0, sigma_state),  sigma_state ~ HalfNormal(.)
        a_age_a   ~ Normal(0, sigma_age),    sigma_age   ~ HalfNormal(.)
        a_educ_e  ~ Normal(0, sigma_educ),   sigma_educ  ~ HalfNormal(.)
        b_sex     ~ Normal(0, .)             sex_c = -0.5 (male), +0.5 (female)
        mu        ~ Normal(., .)

    Centered parameterization: every group level has thousands of respondents
    (see Model 1). Prior scales come from `priors` (see PRIOR_SETS).
    """
    mu = numpyro.sample("mu", dist.Normal(priors["mu_loc"], priors["mu_scale"]))
    b_sex = numpyro.sample("b_sex", dist.Normal(0.0, priors["b_sex_scale"]))
    effects = {}
    for name, size in [("state", n_states), ("age", n_age), ("educ", n_educ)]:
        sigma = numpyro.sample(f"sigma_{name}", dist.HalfNormal(priors[f"sigma_{name}"]))
        with numpyro.plate(name, size):
            effects[name] = numpyro.sample(f"a_{name}", dist.Normal(0.0, sigma))
    eta = (mu + effects["state"][state_idx] + effects["age"][age_idx]
           + effects["educ"][educ_idx] + b_sex * sex)
    p = numpyro.deterministic("p", jax.nn.sigmoid(eta))
    with numpyro.plate("cell", n.shape[0]):
        numpyro.sample("y", dist.Binomial(total_count=n, logits=eta), obs=y)
