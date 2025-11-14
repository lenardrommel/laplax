import jax 

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial


@partial(jax.jit, static_argnums=(2,7,8))
def n_gaussian_log_posterior_objective(
    params,
    ll_rho,
    model,
    x,
    y,
    x_context,
    key,
    prior,
    n_samples
):
    """
    Negative log-posterior objective with Gaussian likelihood.
    
    params:
    - params (jax.tree_util.pytree): parameters of the mlp.
    - ll_rho (float): pre-activated scale of the likelihood model.
    - model (Model): neural network.
    - x (jax.numpy.ndarray): feature batch.
    - y (jax.numpy.ndarray): label batch.
    - x_context (jax.numpy.ndarray): context features.
    - key (jax.random.PRNGKey): random key.
    - prior (callable): prior distribution.
    - n_samples (int): number of training samples.

    returns:
    - neg_log_posterior (float): negative log-posterior.
    - other_info (dict): other information.
    """
    key1, key2 = jax.random.split(key)

    # Log-likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    f_hat = model.apply_fn(params, key1, x)
    log_likelihood = jsp.stats.norm.logpdf(
        y, 
        loc=f_hat, 
        scale=ll_scale
    ).mean() * n_samples
    
    # Squared RKHS norm of the neural network
    f_hat = model.apply_fn(params, key2, x_context) # (n_batch, n_classes)
    prior_mean, prior_cov = prior(x_context)
    sq_rkhs_norm = (f_hat[:,0] - prior_mean[:,0]).T @ jnp.linalg.solve(prior_cov[:,:,0], f_hat[:,0] - prior_mean[:,0])

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm 

    return (
        -log_posterior,
        {"log_likelihood": log_likelihood, "log_posterior": log_posterior, "sq_rkhs_norm": sq_rkhs_norm},
    )


@partial(jax.jit, static_argnums=(1,6,7))
def n_categorical_log_posterior_objective(
    params,
    model,
    x,
    y,
    x_context,
    key,
    prior,
    n_samples
):
    """
    Negative log-posterior objective with Categorical likelihood.
    
    params:
    - params (jax.tree_util.pytree): parameters of the mlp.
    - model (Model): neural network.
    - x (jax.numpy.ndarray): feature batch.
    - y (jax.numpy.ndarray): label batch.
    - x_context (jax.numpy.ndarray): context features.
    - key (jax.random.PRNGKey): random key.
    - prior (callable): prior distribution.
    - n_samples (int): number of training samples.

    returns:
    - neg_log_posterior (float): negative log-posterior.
    - other_info (dict): other information.
    """
    key1, key2 = jax.random.split(key)

    # Log-likelihood
    f_hat = model.apply_fn(params, key1, x) # (n_batch, n_classes)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f_hat.shape[-1])
    log_likelihood = n_samples * jnp.sum(
        one_hot_y * jax.nn.log_softmax(f_hat, axis=-1), # (n_batch, n_classes)
        axis=-1
    ).mean()
    
    # Squared RKHS norm of the neural network
    f_hat = model.apply_fn(params, key2, x_context) # (n_batch, n_classes)
    prior_mean, prior_cov = prior(x_context)
    sq_rkhs_norm = jax.vmap(
        lambda m_q, m_p, K_p: (m_q - m_p).T @ jnp.linalg.solve(K_p, m_q - m_p), 
        in_axes=(-1,-1,-1)
    )(f_hat, prior_mean, prior_cov).sum()

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm 

    return (
        -log_posterior,
        {"log_likelihood": log_likelihood, "log_posterior": log_posterior, "sq_rkhs_norm": sq_rkhs_norm},
    )