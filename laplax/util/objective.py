from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from laplax.types import Array, ModelFn, Params


@jax.jit
def compute_gaussian_log_likelihood(
    f_hat: Array, y: Array, ll_scale: Array | float, n_samples: int
) -> Array:
    """Compute log-likelihood with Gaussian likelihood.

    Args:
        f_hat: Model predictions (batch_size, output_dim)
        y: Target values (batch_size, output_dim)
        ll_scale: Likelihood scale (standard deviation)
        n_samples: Total number of training samples (for scaling)

    Returns:
        Log-likelihood value (scalar)
    """
    # Sum over batch/output dims, then scale by N/batch_size
    batch_ll = jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).sum()

    batch_size = f_hat.shape[0]
    return batch_ll * (n_samples / batch_size)


@jax.jit
def compute_rkhs_norm(
    f_hat: Array, prior_mean: Array, prior_cov: Array, jitter: float = 1e-6
) -> Array:
    """Compute squared RKHS norm of the neural network function.

    ||f||_H^2 = (f - m)^T K^{-1} (f - m)

    Args:
        f_hat: Model predictions on context points (n_context, output_dim)
        prior_mean: Prior mean at context points (n_context, output_dim)
        prior_cov: Prior covariance at context points (n_context, n_context)
                   (Assumes same covariance for all outputs, or single output)
        jitter: Jitter for stability.

    Returns:
        Squared RKHS norm (scalar)
    """
    # Assuming f_hat is (N, 1) or outputs are independent with same kernel
    diff = f_hat - prior_mean
    # Flatten outputs if necessary, or treat as sum over outputs
    # Case 1: Single output (N, 1) -> (N,)
    # Case 2: Multi output (N, D) -> Sum over D norms?

    # If prior_cov is (N, N), we likely reuse it for all D outputs.
    # norm = sum_d (diff[:, d].T @ K^{-1} @ diff[:, d])

    n_context = prior_cov.shape[0]
    L = jnp.linalg.cholesky(prior_cov + jitter * jnp.eye(n_context))

    def single_dim_norm(d_diff):
        alpha = jsp.linalg.solve_triangular(L, d_diff, lower=True)
        return jnp.dot(alpha, alpha)

    if diff.ndim == 1:
        return single_dim_norm(diff)
    # Sum over output dimensions
    return jax.vmap(single_dim_norm, in_axes=1)(diff).sum()


@partial(jax.jit, static_argnames=("model_fn", "prior_fn", "n_samples"))
def n_gaussian_log_posterior_objective(
    params: Params,
    model_fn: ModelFn,
    x_batch: Array,
    y_batch: Array,
    x_context: Array,
    prior_fn: Callable[[Array], tuple[Array, Array]],
    n_samples: int,
    ll_scale: float | Array = 1.0,
) -> tuple[Array, dict]:
    """Negative log-posterior objective with Gaussian likelihood.

    Args:
        params: Model parameters.
        model_fn: Function (input, params) -> output.
        x_batch: Feature batch.
        y_batch: Label batch.
        x_context: Context features for RKHS norm.
        prior_fn: Function (x) -> (mean, cov).
        n_samples: Total number of training samples.
        ll_scale: Likelihood scale/std dev.

    Returns:
        (neg_log_posterior, metrics_dict)
    """
    # 1. Likelihood term
    f_hat = model_fn(input=x_batch, params=params)
    log_likelihood = compute_gaussian_log_likelihood(
        f_hat, y_batch, ll_scale, n_samples
    )

    # 2. Prior term (RKHS norm)
    # Usually we don't need rng for deterministic models, but keeping for compatibility
    f_hat_context = model_fn(input=x_context, params=params)
    prior_mean, prior_cov = prior_fn(x_context)

    sq_rkhs_norm = compute_rkhs_norm(f_hat_context, prior_mean, prior_cov)

    # Log-posterior = LL - 0.5 * ||f||^2
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm

    return -log_posterior, {
        "log_likelihood": log_likelihood,
        "log_posterior": log_posterior,
        "sq_rkhs_norm": sq_rkhs_norm,
    }
