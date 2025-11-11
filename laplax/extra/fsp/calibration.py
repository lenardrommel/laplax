"""Prior calibration utilities for FSP inference.

This module provides tools for calibrating GP priors for FSP inference,
including hyperparameter optimization with simple callable kernels.

Note: For full-featured GP kernels, use GPJax or GPyTorch and wrap with adapters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from laplax.extra.fsp.kernels import KernelProtocol, build_gram_matrix
from laplax.types import Array, Callable, Float


# =============================================================================
# HYPERPARAMETER MANAGEMENT
# =============================================================================


@dataclass
class PriorHyperparameters:
    """Container for GP prior hyperparameters.

    All parameters are stored in log space for unconstrained optimization.
    """

    lengthscale: float = -3.0  # log lengthscale
    variance: float = 0.0  # log variance
    noise_variance: float = -1.0  # log noise variance

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "lengthscale": self.lengthscale,
            "variance": self.variance,
            "noise_variance": self.noise_variance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PriorHyperparameters":
        """Load from dictionary."""
        return cls(
            lengthscale=float(d.get("lengthscale", -3.0)),
            variance=float(d.get("variance", 0.0)),
            noise_variance=float(d.get("noise_variance", -1.0)),
        )

    def transform(self) -> Dict[str, float]:
        """Transform to constrained space (exponential)."""
        return {
            "lengthscale": jnp.exp(self.lengthscale),
            "variance": jnp.exp(self.variance),
            "noise_variance": jnp.exp(self.noise_variance),
        }


def save_hyperparameters(hparams: PriorHyperparameters, path: Path) -> None:
    """Save hyperparameters to JSON file."""
    import json

    path = Path(path).with_suffix(".json")
    with open(path, "w") as f:
        json.dump(hparams.to_dict(), f, indent=2)


def load_hyperparameters(path: Path) -> PriorHyperparameters:
    """Load hyperparameters from JSON file."""
    import json

    path = Path(path).with_suffix(".json")
    with open(path) as f:
        d = json.load(f)
    return PriorHyperparameters.from_dict(d)


# =============================================================================
# SIMPLE GP PRIOR
# =============================================================================


class SimpleGPPrior:
    """Simple GP prior with callable kernel.

    This provides basic GP functionality for calibration and prediction.
    For advanced features, use GPJax or GPyTorch.

    Examples
    --------
    >>> def rbf_kernel(x1, x2=None):
    ...     if x2 is None: x2 = x1
    ...     r2 = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    ...     return jnp.exp(-r2 / 2.0)
    >>> prior = SimpleGPPrior(rbf_kernel, noise_variance=0.01)
    >>> nll = prior.log_likelihood(x_train, y_train)
    """

    def __init__(
        self,
        kernel: Callable | KernelProtocol,
        noise_variance: float = 0.01,
    ):
        """Initialize GP prior.

        Parameters
        ----------
        kernel : Callable or KernelProtocol
            Prior covariance kernel
        noise_variance : float
            Observation noise variance (in linear space)
        """
        self.kernel = kernel
        self.noise_variance = jnp.log(noise_variance)  # Store in log space

    def log_likelihood(
        self, x: Array, y: Array, kernel_params: Optional[Dict] = None
    ) -> Float:
        """Compute log marginal likelihood.

        Parameters
        ----------
        x : Array
            Input locations of shape (N, D)
        y : Array
            Observations of shape (N,) or (N, 1)
        kernel_params : Dict, optional
            Kernel hyperparameters (not used for callable kernels)

        Returns
        -------
        Float
            Negative log marginal likelihood
        """
        # Build Gram matrix with noise
        K = build_gram_matrix(self.kernel, x, jitter=1e-8)

        # Add observation noise
        noise_var = jnp.exp(self.noise_variance)
        K_noisy = K + noise_var * jnp.eye(K.shape[0])

        # Flatten observations
        y_flat = y.reshape(-1, 1)

        # Compute log likelihood
        try:
            L = jnp.linalg.cholesky(K_noisy)
            alpha = jax.scipy.linalg.cho_solve((L, True), y_flat)

            # Quadratic term
            quad = 0.5 * jnp.dot(y_flat.ravel(), alpha.ravel())

            # Log determinant
            logdet = jnp.sum(jnp.log(jnp.diag(L)))

            # Constant
            n = K.shape[0]
            const = 0.5 * n * jnp.log(2 * jnp.pi)

            return quad + logdet + const

        except Exception:
            # If Cholesky fails, use eigendecomposition (more stable but slower)
            eigvals, eigvecs = jnp.linalg.eigh(K_noisy)
            eigvals = jnp.maximum(eigvals, 1e-8)  # Ensure positive

            # Compute alpha using eigendecomposition
            alpha = eigvecs @ (eigvecs.T @ y_flat / eigvals[:, None])

            # Quadratic term
            quad = 0.5 * jnp.dot(y_flat.ravel(), alpha.ravel())

            # Log determinant
            logdet = 0.5 * jnp.sum(jnp.log(eigvals))

            # Constant
            n = K.shape[0]
            const = 0.5 * n * jnp.log(2 * jnp.pi)

            return quad + logdet + const

    def predict(
        self, x_train: Array, y_train: Array, x_test: Array
    ) -> Tuple[Array, Array]:
        """Predict mean and variance at test points.

        Parameters
        ----------
        x_train : Array
            Training inputs of shape (N, D)
        y_train : Array
            Training outputs of shape (N,) or (N, 1)
        x_test : Array
            Test inputs of shape (M, D)

        Returns
        -------
        Tuple[Array, Array]
            Predicted mean (M,) and variance (M,)
        """
        # Build covariance matrices
        K_train = build_gram_matrix(self.kernel, x_train, jitter=1e-8)
        noise_var = jnp.exp(self.noise_variance)
        K_train_noisy = K_train + noise_var * jnp.eye(K_train.shape[0])

        K_test = self.kernel(x_test, x_test)
        K_cross = self.kernel(x_test, x_train)

        # Solve for alpha
        y_flat = y_train.reshape(-1, 1)
        L = jnp.linalg.cholesky(K_train_noisy)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_flat)

        # Predictive mean
        pred_mean = (K_cross @ alpha).ravel()

        # Predictive variance
        v = jax.scipy.linalg.solve_triangular(L, K_cross.T, lower=True)
        pred_var = jnp.diag(K_test) - jnp.sum(v**2, axis=0)

        # Add noise variance
        pred_var = pred_var + noise_var

        return pred_mean, pred_var


# =============================================================================
# CALIBRATION
# =============================================================================


def calibrate_gp_prior(
    kernel_factory: Callable,
    x: Array,
    y: Array,
    init_hparams: Optional[PriorHyperparameters] = None,
    max_iter: int = 100,
    tol: float = 1e-3,
) -> Tuple[PriorHyperparameters, Callable, SimpleGPPrior]:
    """Calibrate GP prior hyperparameters using L-BFGS.

    Parameters
    ----------
    kernel_factory : Callable
        Function that creates kernel given hyperparameters.
        Signature: kernel_factory(lengthscale, variance) -> kernel_fn
    x : Array
        Training inputs of shape (N, D)
    y : Array
        Training outputs of shape (N,) or (N, 1)
    init_hparams : PriorHyperparameters, optional
        Initial hyperparameters. If None, use defaults.
    max_iter : int
        Maximum optimization iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    Tuple[PriorHyperparameters, Callable, SimpleGPPrior]
        Calibrated hyperparameters, calibrated kernel, and GP prior

    Examples
    --------
    >>> def rbf_factory(lengthscale, variance):
    ...     def kernel(x1, x2=None):
    ...         if x2 is None: x2 = x1
    ...         r2 = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    ...         return variance * jnp.exp(-r2 / (2 * lengthscale**2))
    ...     return kernel
    >>> hparams, kernel, prior = calibrate_gp_prior(rbf_factory, x_train, y_train)
    """
    if init_hparams is None:
        init_hparams = PriorHyperparameters()

    # Define loss function
    def loss_fn(params_dict):
        lengthscale = jnp.exp(params_dict["lengthscale"])
        variance = jnp.exp(params_dict["variance"])
        kernel = kernel_factory(lengthscale, variance)

        prior = SimpleGPPrior(kernel)
        prior.noise_variance = params_dict["noise_variance"]

        return prior.log_likelihood(x, y)

    # Initial parameters
    init_params = init_hparams.to_dict()

    # Optimize
    optimizer = optax.lbfgs()
    value_and_grad_fun = jax.value_and_grad(loss_fn)

    params = init_params
    state = optimizer.init(params)

    def cond_fun(val):
        _, state = val
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    def body_fun(val):
        params, state = val
        loss, grad = value_and_grad_fun(params)
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    print(f"Initial NLL: {loss_fn(init_params):.4e}")

    final_params, final_state = jax.lax.while_loop(
        cond_fun, body_fun, (params, state)
    )

    print(f"Final NLL: {loss_fn(final_params):.4e}")

    # Create calibrated hyperparameters and kernel
    final_hparams = PriorHyperparameters.from_dict(final_params)
    transformed = final_hparams.transform()

    calibrated_kernel = kernel_factory(transformed["lengthscale"], transformed["variance"])
    prior = SimpleGPPrior(calibrated_kernel)
    prior.noise_variance = final_hparams.noise_variance

    print("\nCalibrated hyperparameters:")
    for key, value in transformed.items():
        print(f"  {key}: {value:.6f}")

    return final_hparams, calibrated_kernel, prior
