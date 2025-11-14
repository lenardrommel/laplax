import jax 

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

from haiku.data_structures import partition

@partial(jax.jit, static_argnums=(2,7,8,9))
def n_gaussian_log_posterior_objective(
    params,
    ll_rho,
    model,
    state,
    x,
    y,
    key,
    rff_scale,
    n_samples, 
    training
):
    """
    Negative log-posterior objective with Gaussian likelihood.
    """
    # Log-likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    (f_hat, _), new_state = model.apply_fn(params, state, key, x, training)
    log_likelihood = jsp.stats.norm.logpdf(
        y, 
        loc=f_hat, 
        scale=ll_scale
    ).mean() * n_samples
    
    # L2 norm of the beta parameters
    beta_params, _= partition(lambda m, n, p: "beta" in n, params)
    l2_norm_beta = jax.example_libraries.optimizers.l2_norm(params)

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * l2_norm_beta**2 / rff_scale**2

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood, 
            "log_posterior": log_posterior, 
            "state": new_state
        }
    )


@partial(jax.jit, static_argnums=(1,6,7,8))
def n_categorical_log_posterior_objective(
    params,
    model,
    state,
    x,
    y,
    key,
    rff_scale,
    n_samples,
    training
):
    """
    Negative log-posterior objective with Categorical likelihood.
    """
    # Log-likelihood
    (f_hat, _), new_state = model.apply_fn(params, state, key, x, training) # (n_batch, n_classes)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f_hat.shape[-1])
    log_likelihood = n_samples * jnp.sum(
        one_hot_y * jax.nn.log_softmax(f_hat, axis=-1), # (n_batch, n_classes)
        axis=-1
    ).mean()
    
    # L2 norm of the beta parameters
    beta_params, _= partition(lambda m, n, p: "beta" in n, params)
    l2_norm_beta = jax.example_libraries.optimizers.l2_norm(beta_params)

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * l2_norm_beta**2 / rff_scale**2

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood, 
            "log_posterior": log_posterior, 
            "state": new_state
        },
    )