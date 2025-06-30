import jax

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial


@jax.jit
def compute_gaussian_log_likelihood(f_hat, y, ll_scale, n_samples):
    return jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).mean() * n_samples


@jax.jit
def compute_rkhs_norm(f_hat, prior_mean, prior_cov):
    return (f_hat[:, 0] - prior_mean).T @ jnp.linalg.solve(
        prior_cov, f_hat[:, 0] - prior_mean
    )


@partial(jax.jit, static_argnums=(2, 8, 9, 10))
def n_gaussian_log_posterior_objective(
    params, ll_rho, model, state, x, y, x_context, key, prior, n_samples, training
):
    key1, key2 = jax.random.split(key)

    # Process likelihood scale parameter
    ll_scale = jax.nn.softplus(ll_rho)

    # Compute model predictions for data points
    f_hat, new_state = model(x)

    # Compute log-likelihood
    log_likelihood = compute_gaussian_log_likelihood(f_hat, y, ll_scale, n_samples)

    # Compute model predictions for context points
    f_hat_context, new_state = model.apply_fn(
        params, new_state, key2, x_context, training
    )
    prior_mean, prior_cov = prior(x_context, jitter=1e-10)

    # Compute RKHS norm
    sq_rkhs_norm = compute_rkhs_norm(f_hat_context, prior_mean, prior_cov)

    # Log-posterior (negative of objective)
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm

    # Return negative log-posterior as objective
    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood,
            "log_posterior": log_posterior,
            "sq_rkhs_norm": sq_rkhs_norm,
            "state": new_state,
        },
    )
