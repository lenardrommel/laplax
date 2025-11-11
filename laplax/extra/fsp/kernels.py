"""Minimal kernel interface for FSP inference.

This module provides only the minimal interface needed for FSP to work with
kernels from external libraries (GPJax, GPyTorch) or custom callables.

laplax does NOT implement its own kernel library - users should use:
- GPJax for GP kernels in JAX
- GPyTorch for GP kernels in PyTorch
- Custom callables for simple cases
"""

from typing import Any, Protocol

import jax
import jax.numpy as jnp

from laplax.types import Array, Callable


# =============================================================================
# KERNEL PROTOCOL
# =============================================================================


class KernelProtocol(Protocol):
    """Minimal protocol for kernel functions.

    Any object implementing __call__(x1, x2=None) -> Array can be used
    as a kernel in FSP inference.
    """

    def __call__(self, x1: Array, x2: Array | None = None, **kwargs) -> Array:
        """Compute kernel matrix.

        Parameters
        ----------
        x1 : Array
            First input array of shape (N, D)
        x2 : Array, optional
            Second input array of shape (M, D). If None, computes K(x1, x1)
        **kwargs
            Additional kernel parameters

        Returns
        -------
        Array
            Kernel matrix of shape (N, M) or (N, N) if x2 is None
        """
        ...


# =============================================================================
# WRAPPER FOR CALLABLES
# =============================================================================


def wrap_kernel_fn(kernel_fn: Callable) -> "WrappedKernel":
    """Wrap a kernel function into a kernel object with diagonal method.

    This allows simple callables to be used with FSP inference that
    expects a diagonal() method for variance computation.

    Parameters
    ----------
    kernel_fn : Callable
        A function that computes kernel: kernel_fn(x1, x2) -> K

    Returns
    -------
    WrappedKernel
        Wrapped kernel object

    Examples
    --------
    >>> def my_rbf(x1, x2=None):
    ...     if x2 is None:
    ...         x2 = x1
    ...     sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    ...     return jnp.exp(-sq_dist / 2.0)
    >>> kernel = wrap_kernel_fn(my_rbf)
    >>> K = kernel(x, x)
    >>> diag = kernel.diagonal(x)
    """

    class WrappedKernel:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, x1: Array, x2: Array | None = None) -> Array:
            if x2 is None:
                x2 = x1
            return self.fn(x1, x2)

        def diagonal(self, x: Array) -> Array:
            """Compute kernel diagonal K(x, x)."""
            return jax.vmap(lambda xi: self.fn(xi[None, :], xi[None, :])[0, 0])(x)

    return WrappedKernel(kernel_fn)


# =============================================================================
# ADAPTERS FOR EXTERNAL LIBRARIES
# =============================================================================


class GPJaxKernelAdapter:
    """Adapter for GPJax kernels.

    Wraps a GPJax kernel to provide the minimal interface expected by FSP.

    Examples
    --------
    >>> import gpjax as gpx
    >>> gpjax_kernel = gpx.kernels.RBF()
    >>> kernel = GPJaxKernelAdapter(gpjax_kernel, params={"lengthscale": 1.0})
    >>> K = kernel(x, x)
    """

    def __init__(self, gpjax_kernel: Any, params: dict | None = None):
        """Initialize adapter.

        Parameters
        ----------
        gpjax_kernel : GPJax kernel
            A GPJax kernel object
        params : dict, optional
            Kernel parameters
        """
        self.gpjax_kernel = gpjax_kernel
        self.params = params or {}

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        if x2 is None:
            x2 = x1

        # GPJax kernels typically have a gram method
        if hasattr(self.gpjax_kernel, "gram"):
            return self.gpjax_kernel.gram(self.params, x1, x2)
        else:
            # Fallback to computing pairwise
            return jax.vmap(
                lambda x: jax.vmap(lambda y: self.gpjax_kernel(x, y, self.params))(x2)
            )(x1)

    def diagonal(self, x: Array) -> Array:
        # Most GP kernels have variance on diagonal
        if hasattr(self.gpjax_kernel, "variance"):
            return self.gpjax_kernel.variance * jnp.ones(x.shape[0])
        return jax.vmap(lambda xi: self.gpjax_kernel(xi, xi, self.params))(x)


class GPyTorchKernelAdapter:
    """Adapter for GPyTorch kernels.

    Note: This requires converting between JAX and PyTorch tensors.

    Examples
    --------
    >>> import gpytorch
    >>> gpytorch_kernel = gpytorch.kernels.RBFKernel()
    >>> kernel = GPyTorchKernelAdapter(gpytorch_kernel)
    >>> K = kernel(x, x)  # x must be JAX array
    """

    def __init__(self, gpytorch_kernel: Any):
        """Initialize adapter.

        Parameters
        ----------
        gpytorch_kernel : GPyTorch kernel
            A GPyTorch kernel object
        """
        self.gpytorch_kernel = gpytorch_kernel

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for GPyTorchKernelAdapter")

        if x2 is None:
            x2 = x1

        # Convert to PyTorch
        x1_torch = torch.from_numpy(jnp.asarray(x1))
        x2_torch = torch.from_numpy(jnp.asarray(x2))

        # Compute kernel
        K_torch = self.gpytorch_kernel(x1_torch, x2_torch).evaluate()

        # Convert back to JAX
        return jnp.array(K_torch.detach().numpy())

    def diagonal(self, x: Array) -> Array:
        K = self(x, x)
        return jnp.diag(K)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def build_gram_matrix(
    kernel: Callable | KernelProtocol, x: Array, jitter: float = 1e-6
) -> Array:
    """Build Gram matrix with jitter.

    Parameters
    ----------
    kernel : Callable or KernelProtocol
        Kernel function or object
    x : Array
        Input points of shape (N, D)
    jitter : float
        Diagonal jitter for numerical stability

    Returns
    -------
    Array
        Gram matrix K + jitter * I of shape (N, N)
    """
    K = kernel(x, x)
    return K + jitter * jnp.eye(x.shape[0])


def kernel_variance(kernel: Callable | KernelProtocol, x: Array) -> float:
    """Compute total kernel variance (trace).

    Parameters
    ----------
    kernel : Callable or KernelProtocol
        Kernel function or object
    x : Array
        Input points

    Returns
    -------
    float
        Sum of diagonal elements
    """
    if hasattr(kernel, "diagonal"):
        return jnp.sum(kernel.diagonal(x))
    else:
        # Fallback: compute diagonal from full kernel
        K = kernel(x, x)
        return jnp.sum(jnp.diag(K))
