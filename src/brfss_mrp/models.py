"""NumPyro models for cell-level BRFSS data."""

from __future__ import annotations

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
