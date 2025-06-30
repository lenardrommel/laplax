from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsp


@partial(jax.jit, static_argnums=(2, 8, 9, 10))
def n_gaussian_log_posterior_objective(
    params, ll_rho, model, state, x, y, x_context, key, prior, n_samples, training
):
    key1, key2 = jax.random.split(key)

    # Log-likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    f_hat, new_state = model.apply_fn(params, state, key1, x, training)
    log_likelihood = (
        jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).mean() * n_samples
    )

    # Squared RKHS norm of the neural network
    f_hat, new_state = model.apply_fn(
        params, new_state, key2, x_context, training
    )  # (n_batch, n_classes)
    prior_mean, prior_cov = prior(x_context, jitter=1e-10)  # 1e-4)
    sq_rkhs_norm = (f_hat[:, 0] - prior_mean[:, 0]).T @ jnp.linalg.solve(
        prior_cov[:, :, 0], f_hat[:, 0] - prior_mean[:, 0]
    )

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood,
            "log_posterior": log_posterior,
            "sq_rkhs_norm": sq_rkhs_norm,
            "state": new_state,
        },
    )
