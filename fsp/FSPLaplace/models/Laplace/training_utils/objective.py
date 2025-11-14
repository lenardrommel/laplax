import jax 

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial


@partial(jax.jit, static_argnums=(2,8,9))
def n_gaussian_log_posterior_objective(
    params,
    ll_rho,
    model,
    state,
    x,
    y,
    key,
    prior_scale,
    n_samples, 
    training
):
    """
    Negative log-posterior objective.
    
    params:
    - params (jax.tree_util.pytree): parameters of the mlp.
    - ll_rho (float): pre-activated scale of the likelihood model.
    - model (Model): neural network.
    - x (jax.numpy.ndarray): feature batch.
    - y (jax.numpy.ndarray): label batch.
    - key (jax.random.PRNGKey): random key.
    - prior_scale (jax.numpy.array): prior scale.
    - n_samples (int): number of training samples.

    returns:
    - neg_log_posterior (float): negative log-posterior.
    - other_info (dict): other information.
    """
    # Compute likelihood scale
    ll_scale = jax.nn.softplus(ll_rho)

    # Compute log-likelihood
    f, new_state = model.apply_fn(params, state, key, x, training)
    log_likelihood = jsp.stats.norm.logpdf(
        y, 
        loc=f, 
        scale=ll_scale
    ).mean() * n_samples
    
    # Compute log-prior
    sto_params, _ = model.partition_inference_parameters(params)
    scaled_params = jax.tree_util.tree_map(lambda p, s: p / s, sto_params, prior_scale)
    log_prior = -0.5*jax.example_libraries.optimizers.l2_norm(scaled_params)**2
    
    # Compute log-posterior
    log_posterior = log_likelihood + log_prior

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood, 
            "log_posterior": log_posterior, 
            "state": new_state
        }
    )


@partial(jax.jit, static_argnums=(1,7,8))
def n_categorical_log_posterior_objective(
    params,
    model,
    state,
    x,
    y,
    key,
    prior_scale, 
    n_samples, 
    training
):
    """
    Negative log-posterior objective.
    
    params:
    - params (jax.tree_util.pytree): parameters of the mlp.
    - model (Model): neural network.
    - x (jax.numpy.ndarray): feature batch.
    - y (jax.numpy.ndarray): label batch.
    - key (jax.random.PRNGKey): random key.
    - prior_scale (jax.numpy.array): scale of the prior.
    - n_samples (int): number of training samples.

    returns:
    - neg_log_posterior (float): negative log-posterior.
    - other_info (dict): other information.
    """
    # Compute log-likelihood
    f, new_state = model.apply_fn(params, state, key, x, training) # (n_batch, n_classes)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
    log_likelihood = n_samples * jnp.sum(
        one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
        axis=-1
    ).mean()
    
    # Compute log-prior
    sto_params, _ = model.partition_inference_parameters(params)
    scaled_params = jax.tree_util.tree_map(lambda p, s: p / s, sto_params, prior_scale)
    log_prior = -0.5*jax.example_libraries.optimizers.l2_norm(scaled_params)**2
    
    # Compute log-posterior
    log_posterior = log_likelihood + log_prior

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood, 
            "log_posterior": log_posterior,
            "state": new_state
        }
    )