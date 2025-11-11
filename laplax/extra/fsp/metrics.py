"""FSP-specific evaluation metrics.

This module provides metrics for evaluating FSP posterior predictions,
including negative log predictive density (NLPD) and Mahalanobis distance.
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from laplax.curv.utils import LowRankTerms
from laplax.types import Array, Float


# =============================================================================
# MAHALANOBIS DISTANCE
# =============================================================================


def mahalanobis_distance(
    pred_mean: Array,
    target: Array,
    pred_cov_low_rank: LowRankTerms,
    noise_variance: Float,
) -> Float:
    """Compute Mahalanobis distance using low-rank covariance.

    For Σ = U diag(S²) U^T + σ² I, computes:
        d² = (μ - y)^T Σ^{-1} (μ - y)

    using the Woodbury identity:
        Σ^{-1} = (1/σ²)[I - U diag(S²/(S²+σ²)) U^T]

    Parameters
    ----------
    pred_mean : Array
        Predicted mean of shape (N,)
    target : Array
        Target values of shape (N,)
    pred_cov_low_rank : LowRankTerms
        Low-rank covariance terms (U, S, scalar)
    noise_variance : Float
        Observation noise variance

    Returns
    -------
    Float
        Mahalanobis distance

    Examples
    --------
    >>> from laplax.curv.utils import LowRankTerms
    >>> low_rank = LowRankTerms(U=U, S=S, scalar=0.0)
    >>> dist = mahalanobis_distance(pred_mean, target, low_rank, noise_var)
    """
    U = pred_cov_low_rank.U
    S = pred_cov_low_rank.S

    # Ensure noise variance is positive
    sigma2 = jnp.maximum(noise_variance, 1e-8)

    # Residual vector
    v = (pred_mean - target).reshape(-1)

    # Woodbury formula: Σ^{-1} v = (1/σ²)[v - U @ diag(S²/(S²+σ²)) @ U^T @ v]
    Ut_v = U.T @ v  # (rank,)
    w = (S**2) / (S**2 + sigma2)  # (rank,)
    inv_Sigma_v = (v - U @ (w * Ut_v)) / sigma2

    # Quadratic form
    quad = jnp.dot(v, inv_Sigma_v)

    # Return distance (ensure non-negative)
    return jnp.sqrt(jnp.maximum(quad, 0.0))


# =============================================================================
# LOG DETERMINANT
# =============================================================================


def log_determinant(
    pred_cov_low_rank: LowRankTerms, noise_variance: Float
) -> Float:
    """Compute log determinant of covariance matrix.

    For Σ = U diag(S²) U^T + σ² I, computes:
        log|Σ| = Σ_i log(S_i² + σ²) + (n - k) log(σ²)

    where k is the rank and n is the dimension.

    Parameters
    ----------
    pred_cov_low_rank : LowRankTerms
        Low-rank covariance terms
    noise_variance : Float
        Observation noise variance

    Returns
    -------
    Float
        Log determinant
    """
    U = pred_cov_low_rank.U
    S = pred_cov_low_rank.S

    # Ensure positive
    sigma2 = jnp.maximum(noise_variance, 1e-8)

    n, k = U.shape

    # Log det in eigenspace
    eigenspace_logdet = jnp.sum(jnp.log(S**2 + sigma2))

    # Log det in complement space
    complement_logdet = (n - k) * jnp.log(sigma2)

    return eigenspace_logdet + complement_logdet


# =============================================================================
# NEGATIVE LOG PREDICTIVE DENSITY (NLPD)
# =============================================================================


def negative_log_predictive_density(
    pred_mean: Array,
    target: Array,
    pred_cov_low_rank: LowRankTerms,
    noise_variance: Float,
) -> Float:
    """Compute negative log predictive density (NLPD).

    For Gaussian predictive distribution N(μ, Σ), computes:
        -log p(y|μ,Σ) = 0.5 [log|Σ| + (μ-y)^T Σ^{-1} (μ-y) + n log(2π)]

    Parameters
    ----------
    pred_mean : Array
        Predicted mean
    target : Array
        Target values
    pred_cov_low_rank : LowRankTerms
        Low-rank covariance
    noise_variance : Float
        Observation noise variance

    Returns
    -------
    Float
        NLPD value

    Examples
    --------
    >>> nlpd = negative_log_predictive_density(
    ...     pred_mean, target, low_rank, noise_var
    ... )
    """
    n = pred_mean.size

    # Log determinant term
    log_det = log_determinant(pred_cov_low_rank, noise_variance)

    # Mahalanobis distance squared
    md = mahalanobis_distance(pred_mean, target, pred_cov_low_rank, noise_variance)
    md_squared = md**2

    # NLPD = 0.5 * (log|Σ| + d² + n*log(2π))
    nlpd = 0.5 * (log_det + md_squared + n * jnp.log(2 * jnp.pi))

    return nlpd


def marginal_nlpd(
    pred_mean: Array, pred_var: Array, target: Array, noise_variance: Float
) -> Float:
    """Compute marginal NLPD (per-dimension independent).

    Assumes diagonal covariance structure: Σ = diag(σ_1², ..., σ_n²) + σ²I

    Parameters
    ----------
    pred_mean : Array
        Predicted mean
    pred_var : Array
        Predicted variance (diagonal)
    target : Array
        Target values
    noise_variance : Float
        Observation noise variance

    Returns
    -------
    Float
        Marginal NLPD
    """
    sigma2 = jnp.maximum(noise_variance, 1e-8)

    # Total variance
    v_tot = pred_var + sigma2

    # Squared residuals
    resid2 = (pred_mean - target) ** 2

    # Per-dimension NLPD
    per_dim = 0.5 * (jnp.log(2 * jnp.pi * v_tot) + resid2 / v_tot)

    return jnp.mean(per_dim)


# =============================================================================
# CALIBRATION METRICS
# =============================================================================


def expected_calibration_error(
    pred_std: Array, errors: Array, n_bins: int = 10
) -> Float:
    """Compute expected calibration error (ECE).

    Measures whether predicted uncertainties match empirical errors.

    Parameters
    ----------
    pred_std : Array
        Predicted standard deviations
    errors : Array
        Absolute errors |pred - target|
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    Float
        Expected calibration error
    """
    # Create bins based on predicted uncertainty
    bin_boundaries = jnp.linspace(0, jnp.max(pred_std), n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        # Find points in this bin
        mask = (pred_std >= bin_boundaries[i]) & (pred_std < bin_boundaries[i + 1])
        n_in_bin = jnp.sum(mask)

        if n_in_bin > 0:
            # Average predicted std in bin
            avg_pred_std = jnp.mean(pred_std[mask])

            # Average actual error in bin
            avg_error = jnp.mean(errors[mask])

            # Add weighted difference
            ece += (n_in_bin / len(pred_std)) * jnp.abs(avg_pred_std - avg_error)

    return ece


# =============================================================================
# COMPOSITE METRICS
# =============================================================================


def compute_fsp_metrics(
    pred_mean: Array,
    target: Array,
    pred_cov_low_rank: LowRankTerms,
    noise_variance: Float,
    pred_var: Optional[Array] = None,
) -> Dict[str, Float]:
    """Compute all FSP metrics.

    Parameters
    ----------
    pred_mean : Array
        Predicted mean
    target : Array
        Target values
    pred_cov_low_rank : LowRankTerms
        Low-rank covariance
    noise_variance : Float
        Observation noise variance
    pred_var : Array, optional
        Predicted variance (for marginal NLPD)

    Returns
    -------
    Dict[str, Float]
        Dictionary of metrics

    Examples
    --------
    >>> metrics = compute_fsp_metrics(
    ...     pred_mean, target, low_rank, noise_var
    ... )
    >>> print(f"NLPD: {metrics['nlpd']:.4f}")
    >>> print(f"Mahalanobis: {metrics['mahalanobis']:.4f}")
    """
    metrics = {}

    # RMSE
    metrics["rmse"] = jnp.sqrt(jnp.mean((pred_mean - target) ** 2))

    # Mahalanobis distance
    metrics["mahalanobis"] = mahalanobis_distance(
        pred_mean, target, pred_cov_low_rank, noise_variance
    )

    # Log determinant
    metrics["log_det"] = log_determinant(pred_cov_low_rank, noise_variance)

    # NLPD
    metrics["nlpd"] = negative_log_predictive_density(
        pred_mean, target, pred_cov_low_rank, noise_variance
    )

    # Marginal NLPD (if variance provided)
    if pred_var is not None:
        metrics["marginal_nlpd"] = marginal_nlpd(
            pred_mean, pred_var, target, noise_variance
        )

    return metrics


# =============================================================================
# CONVENIENCE WRAPPERS FOR LAPLAX EVALUATION
# =============================================================================


def create_fsp_metric_fn(metric_name: str):
    """Create a metric function compatible with laplax evaluation.

    Parameters
    ----------
    metric_name : str
        Name of metric: "nlpd", "mahalanobis", "log_det"

    Returns
    -------
    Callable
        Metric function

    Examples
    --------
    >>> nlpd_fn = create_fsp_metric_fn("nlpd")
    >>> nlpd = nlpd_fn(
    ...     pred_mean=pred_mean,
    ...     target=target,
    ...     low_rank_terms=low_rank,
    ...     observation_noise=noise_var
    ... )
    """

    def metric_fn(
        pred_mean, target, low_rank_terms, observation_noise, pred_var=None, **kwargs
    ):
        if metric_name == "nlpd":
            return negative_log_predictive_density(
                pred_mean, target, low_rank_terms, observation_noise
            )
        elif metric_name == "mahalanobis":
            return mahalanobis_distance(
                pred_mean, target, low_rank_terms, observation_noise
            )
        elif metric_name == "log_det":
            return log_determinant(low_rank_terms, observation_noise)
        elif metric_name == "marginal_nlpd":
            if pred_var is None:
                raise ValueError("pred_var required for marginal_nlpd")
            return marginal_nlpd(pred_mean, pred_var, target, observation_noise)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    return metric_fn
