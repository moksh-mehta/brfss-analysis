"""Smoke test: the probabilistic-programming stack imports and runs NUTS on CPU."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def test_jax_uses_cpu():
    assert jax.default_backend() == "cpu"


def test_nuts_recovers_normal_mean():
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)

    y = jnp.full(100, 3.0)
    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), y=y)
    mu = mcmc.get_samples()["mu"]
    assert abs(float(mu.mean()) - 3.0) < 0.1
