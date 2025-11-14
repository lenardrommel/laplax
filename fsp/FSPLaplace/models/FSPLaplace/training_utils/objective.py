import jax 

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

@jax.jit
def compute_gaussian_log_likelihood(f_hat, y, ll_scale, n_samples):
    """
    Compute log-likelihood with Gaussian likelihood.
    
    Args:
        f_hat: Model predictions
        y: Target values
        ll_scale: Likelihood scale (standard deviation)
        n_samples: Number of training samples
        
    Returns:
        Log-likelihood value
    """
    return jsp.stats.norm.logpdf(
        y, 
        loc=f_hat, 
        scale=ll_scale
    ).mean() * n_samples


@jax.jit
def compute_rkhs_norm(f_hat, prior_mean, prior_cov):
    """
    Compute squared RKHS norm of the neural network.
    
    Args:
        f_hat: Model predictions on context points
        prior_mean: Prior mean at context points
        prior_cov: Prior covariance at context points
        
    Returns:
        Squared RKHS norm
    """
    return (f_hat[:,0] - prior_mean[:,0]).T @ jnp.linalg.solve(prior_cov[:,:,0], f_hat[:,0] - prior_mean[:,0])


@partial(jax.jit, static_argnums=(2,8,9,10))
def n_gaussian_log_posterior_objective(
    params,
    ll_rho,
    model,
    state, 
    x,
    y,
    x_context,
    key,
    prior,
    n_samples, 
    training
):
    """
    Negative log-posterior objective with Gaussian likelihood.
    
    params:
    - params (jax.tree_util.pytree): parameters of the mlp.
    - ll_rho (float): pre-activated scale of the likelihood model.
    - model (Model): neural network.
    - state (dict): state of the neural network.
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

    # Process likelihood scale parameter
    ll_scale = jax.nn.softplus(ll_rho)
    
    # Compute model predictions for data points
    f_hat, new_state = model.apply_fn(params, state, key1, x, training)
    
    # Compute log-likelihood
    log_likelihood = compute_gaussian_log_likelihood(f_hat, y, ll_scale, n_samples)
    
    # Compute model predictions for context points
    f_hat_context, new_state = model.apply_fn(params, new_state, key2, x_context, training)
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
            "state": new_state
        }
    )






@partial(jax.jit, static_argnums=(1,7,8,9))
def n_categorical_log_posterior_objective(
    params,
    model,
    state,
    x,
    y,
    x_context,
    key,
    prior,
    n_samples, 
    training
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
    f_hat, new_state = model.apply_fn(params, state, key1, x, training) # (n_batch, n_classes)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f_hat.shape[-1])
    log_likelihood = n_samples * jnp.sum(
        one_hot_y * jax.nn.log_softmax(f_hat, axis=-1), # (n_batch, n_classes)
        axis=-1
    ).mean()
    
    # Squared RKHS norm of the neural network
    f_hat, new_state = model.apply_fn(params, new_state, key2, x_context, training) # (n_batch, n_classes)
    prior_mean, prior_cov = prior(x_context, jitter=1e-4)
    sq_rkhs_norm = jax.vmap(
        lambda m_q, m_p, K_p: (m_q - m_p).T @ jnp.linalg.solve(K_p, m_q - m_p), 
        in_axes=(-1,-1,-1)
    )(f_hat, prior_mean, prior_cov).sum()

    # Log-posterior
    log_posterior = -(log_likelihood - 0.5 * sq_rkhs_norm) 

    return (
        log_posterior,
        {
            "log_likelihood": log_likelihood, 
            "log_posterior": log_posterior, 
            "sq_rkhs_norm": sq_rkhs_norm, 
            "state": new_state
        },
    )