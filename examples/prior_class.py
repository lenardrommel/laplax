import jax
from jax import numpy as jnp
from typing import Callable, Dict, Any
from abc import ABC, abstractmethod

# =============================================================================
# BASE KERNEL INTERFACE
# =============================================================================


class Kernel(ABC):
    """Base class for all kernels."""

    @abstractmethod
    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute kernel between x and y.

        Args:
            x: Input points of shape [..., D]
            y: Input points of shape [..., D]
            params: Kernel parameters

        Returns:
            Kernel values of shape [...]
        """
        pass


# =============================================================================
# BUILT-IN KERNEL IMPLEMENTATIONS
# =============================================================================


class PeriodicKernel(Kernel):
    """Periodic kernel: variance * exp[-2 * sum(sin^2(pi*(x-y)/period)) / lengthscale^2]"""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        lengthscale = params["lengthscale"]
        period = params["period"]
        variance = params.get("variance", 1.0)

        arg = jnp.pi * (x - y) / period
        sin2 = jnp.sin(arg) ** 2
        return variance * jnp.exp(-2.0 * jnp.sum(sin2, axis=-1) / lengthscale**2)


class Matern52Kernel(Kernel):
    """Matérn 5/2 kernel: variance * (1 + √5r/ℓ + 5r²/(3ℓ²)) * exp(-√5r/ℓ)"""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        lengthscale = params["lengthscale"]
        variance = params.get("variance", 1.0)

        r = jnp.linalg.norm(x - y, axis=-1)
        sqrt5 = jnp.sqrt(5.0)
        sr = sqrt5 * r / lengthscale
        return variance * (1.0 + sr + sr**2 / 3.0) * jnp.exp(-sr)


class Matern12Kernel(Kernel):
    """Matérn 1/2 (Exponential) kernel: variance * exp(-r/ℓ)"""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        lengthscale = params["lengthscale"]
        variance = params.get("variance", 1.0)

        r = jnp.linalg.norm(x - y, axis=-1)
        return variance * jnp.exp(-r / lengthscale)


class RBFKernel(Kernel):
    """RBF/Squared Exponential kernel: variance * exp(-r²/(2ℓ²))"""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        lengthscale = params["lengthscale"]
        variance = params.get("variance", 1.0)

        r = jnp.linalg.norm(x - y, axis=-1)
        return variance * jnp.exp(-(r**2) / (2 * lengthscale**2))


# =============================================================================
# KERNEL COMPOSITION UTILITIES
# =============================================================================


class SumKernel(Kernel):
    """Sum of multiple kernels."""

    def __init__(self, kernels: list[Kernel], param_keys: list[str]):
        """
        Args:
            kernels: List of kernel objects
            param_keys: List of parameter prefixes for each kernel
        """
        self.kernels = kernels
        self.param_keys = param_keys

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        result = 0.0
        for kernel, prefix in zip(self.kernels, self.param_keys):
            # Extract parameters for this kernel
            kernel_params = {
                k.replace(f"{prefix}_", ""): v
                for k, v in params.items()
                if k.startswith(f"{prefix}_")
            }
            result += kernel(x, y, kernel_params)
        return result


class ProductKernel(Kernel):
    """Product of multiple kernels."""

    def __init__(self, kernels: list[Kernel], param_keys: list[str]):
        self.kernels = kernels
        self.param_keys = param_keys

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        result = 1.0
        for kernel, prefix in zip(self.kernels, self.param_keys):
            kernel_params = {
                k.replace(f"{prefix}_", ""): v
                for k, v in params.items()
                if k.startswith(f"{prefix}_")
            }
            result *= kernel(x, y, kernel_params)
        return result


# =============================================================================
# FUNCTIONAL INTERFACE (for backward compatibility)
# =============================================================================


def make_kernel_function(kernel: Kernel) -> Callable:
    """Convert a Kernel object to a function that can be vmapped."""

    def kernel_fn(
        x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
    ) -> jnp.ndarray:
        # Handle broadcasting for matrices
        if x.ndim == 2 and y.ndim == 2:
            # x: [N, D], y: [M, D] → [N, M]
            X = x[:, None, :]  # [N, 1, D]
            Y = y[None, :, :]  # [1, M, D]
            return jax.vmap(
                jax.vmap(kernel, in_axes=(1, 1, None)), in_axes=(0, 0, None)
            )(X, Y, params)
        else:
            # Element-wise
            return kernel(x, y, params)

    return kernel_fn


# =============================================================================
# GRAM MATRIX UTILITIES
# =============================================================================


def gram_matrix(
    x: jnp.ndarray, kernel: Kernel, params: Dict[str, Any], jitter: float = 1e-6
) -> jnp.ndarray:
    """Compute Gram matrix K + jitter * I.

    Args:
        x: Input points of shape [N, D]
        kernel: Kernel object
        params: Kernel parameters
        jitter: Diagonal noise for numerical stability

    Returns:
        Gram matrix of shape [N, N]
    """
    kernel_fn = make_kernel_function(kernel)
    K = kernel_fn(x, x, params)
    return K + jitter * jnp.eye(x.shape[0])


# Backward compatible function
def gram(
    x: jnp.ndarray, params: Dict[str, Any], kernel_fn: Callable, jitter: float = 1e-6
) -> jnp.ndarray:
    """Backward compatible gram function."""
    K = kernel_fn(x, x, params)
    return K + jitter * jnp.eye(x.shape[0])


# =============================================================================
# EXAMPLES AND PRESETS
# =============================================================================


# Recreate your original composite kernel
def create_composite_kernel():
    """Recreate the original composite kernel."""
    return SumKernel(
        kernels=[PeriodicKernel(), Matern52Kernel(), Matern12Kernel()],
        param_keys=["per", "m52", "m12"],
    )


# Example: Create your original kernel as a function
def composite_kernel_function(
    x: jnp.ndarray, y: jnp.ndarray, params: Dict[str, Any]
) -> jnp.ndarray:
    """Your original composite kernel as a function."""
    composite = create_composite_kernel()
    return make_kernel_function(composite)(x, y, params)
