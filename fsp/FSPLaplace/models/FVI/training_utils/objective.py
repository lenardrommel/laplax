import jax

import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial

from models.FVI.training_utils.utils import SpectralScoreEstimator


@partial(jax.jit, static_argnums=(3,4,9,10))
def n_felbo_gaussian_objective(
    mean_params,
    rho_params,
    ll_rho,
    model,
    prior,
    x,
    y,
    x_sampled_context,
    key,
    mc_samples,
    n_context_points
):
    """
    Functional ELBO objective.
    
    params:
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
    - ll_rho (float): pre-activated scale parameter of the likelihood.
    - model (Model): stochastic neural networks.
    - prior (Prior): prior distribution on context points
    - x (jnp.ndarray): input data used to calculate the expected log likelihood in the ELBO.
    - y (jnp.ndarray): targets used to calculate the expected log likelihoodin the ELBO.
    - x_sampled_context (jnp.ndarray): context points sampled from feature distribution.
    - key (jax.random.PRNGKey): random key.
    - mc_samples (int): number of Monte Carlo samples.
    - n_context_points (int): total number of context points.

    returns:
    - neg_felbo_objective (float): functional ELBO.
    - expected_ll (float): expected log likelihood.
    - kl_div (float): KL divergence.
    """
    # Split keys
    key1, key2, key3 = jax.random.split(key, 3)

    # Default mc samples to compute the KL divergence
    mc_samples_kl = 100

    # Sample from approximate posterior
    f_context = model.predict_f(
        mean_params, 
        rho_params,
        x_sampled_context, 
        key1, 
        mc_samples_kl
    ) # (mc_samples, n_batch, n_classes)
    
    # Compute prior mean and covariance
    prior_mean, prior_cov = prior(x_sampled_context)

    # Compute KL divergence
    kl_div = 0
    for k in range(f_context.shape[-1]):
        kl_div += kl_divergence(
            f_context[:,:,k], 
            prior_mean[:,k], 
            prior_cov[:,:,k], 
            key2
        ) / x.shape[0]

    # Compute expected log likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    f = model.predict_f(
        mean_params, 
        rho_params, 
        x, 
        key3, 
        mc_samples
    ) # (mc_samples, n_batch, 1)
    expected_ll = jnp.mean(
        jsp.stats.norm.logpdf(y, loc=f, scale=ll_scale),
        axis=0
    ).mean()

    # Compute FELBO
    felbo = expected_ll - kl_div

    return (
        -felbo,
        {"expected_ll": expected_ll, "kl_div": kl_div, "felbo": felbo}
    )


@partial(jax.jit, static_argnums=(2,3,8,9))
def n_felbo_categorical_objective(
    mean_params,
    rho_params,
    model,
    prior,
    x,
    y,
    x_sampled_context,
    key,
    mc_samples, 
    n_context_points
):
    """
    Functional ELBO objective.
    
    params:
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
    - model (Model): stochastic neural networks.
    - prior (Prior): prior distribution on context points
    - x (jnp.ndarray): input data used to calculate the expected log likelihood in the ELBO.
    - y (jnp.ndarray): targets used to calculate the expected log likelihoodin the ELBO.
    - x_sampled_context (jnp.ndarray): context points sampled from feature distribution.
    - key (jax.random.PRNGKey): random key.
    - mc_samples (int): number of Monte Carlo samples.
    - n_context_points (int): total number of context points.

    returns:
    - neg_felbo_objective (float): functional ELBO.
    - expected_ll (float): expected log likelihood.
    - kl_div (float): KL divergence.
    """
    # Split keys
    key1, key2, key3 = jax.random.split(key, 3)

    # Default mc samples to compute the KL divergence
    mc_samples_kl = 100

    # Sample from approximate posterior
    f_context = model.predict_f(
        mean_params, 
        rho_params, 
        x_sampled_context, 
        key1, 
        mc_samples_kl
    ) # (mc_samples, n_batch, n_classes)
    
    # Compute prior mean and covariance
    prior_mean, prior_cov = prior(x_sampled_context)

    # Compute KL divergence
    kl_div = 0
    for k in range(f_context.shape[-1]):
        kl_div += kl_divergence(
            f_context[:,:,k], 
            prior_mean[:,k], 
            prior_cov[:,:,k], 
            key2
        ) / x.shape[0]

    # Compute expected log likelihood
    f = model.predict_f(
        mean_params,
        rho_params, 
        x,
        key3, 
        mc_samples
    ) # (mc_samples, n_batch, 1)
    one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
    expected_ll = jnp.mean(
        jnp.sum(
            one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_samples, n_batch, n_classes)
            axis=-1
        ), # (n_samples, n_batch)
        axis=0
    ).mean()

    # Compute FELBO
    felbo = expected_ll - kl_div

    return (
        -felbo,
        {"expected_ll": expected_ll, "kl_div": kl_div, "felbo": felbo}
    )


@partial(jax.jit, static_argnums=(2,3))
def kl_divergence_objective(
    mean_params,
    rho_params,
    model,
    prior,
    x_sampled_context,
    key
):
    """
    KL between approx posterior and prior objective.
    
    params:
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
    - model (Model): stochastic neural networks.
    - prior (Prior): prior distribution on context points
    - x_sampled_context (jnp.ndarray): context points sampled from feature distribution.
    - key (jax.random.PRNGKey): random key.

    returns:
    - kl_div (float): functional KL divergence.
    """
    # Default mc samples to compute the KL divergence
    mc_samples_kl = 100

    # Sample from approximate posterior
    f_context = model.predict_f(
        mean_params, 
        rho_params, 
        x_sampled_context, 
        key, 
        mc_samples_kl
    )

    # Compute prior mean and covariance
    prior_mean, prior_cov = prior(x_sampled_context)

    # Compute KL divergence
    kl_div = 0
    for k in range(f_context.shape[-1]):
        kl_div += kl_divergence(
            f_context[:,:,k], 
            prior_mean[:,k], 
            prior_cov[:,:,k], 
            key
        )

    return kl_div, {"kl_div": kl_div}


@partial(jax.jit, static_argnums=(4,5))
def kl_divergence(
    f_context,
    prior_mean, 
    prior_cov,
    key, 
    eta=0.,
    n_eigen_threshold=0.99
):
    # Add jitter for numerical stability
    jitter=0.01
    f_context += jitter * jax.random.normal(key, f_context.shape)
    prior_cov += jitter**2 * jnp.eye(prior_cov.shape[0])
    
    # Estimate entropy surrogate
    estimator = SpectralScoreEstimator(eta=eta, n_eigen_threshold=n_eigen_threshold)
    dlog_q = estimator.compute_gradients(f_context)
    entropy_sur = jnp.mean(
        jnp.sum(jax.lax.stop_gradient(-dlog_q) * f_context, -1)
    )

    # Compute cross entropy
    cross_entropy = -jax.scipy.stats.multivariate_normal.logpdf(
        f_context, 
        prior_mean.reshape(1, -1), 
        prior_cov.reshape(1, prior_mean.shape[-1], -1)
    ).mean()

    # Compute KL 
    KL_div = -entropy_sur + cross_entropy

    return KL_div