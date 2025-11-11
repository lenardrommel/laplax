"""Kernel abstractions for FSP inference.

This module provides kernel interfaces compatible with various libraries
(GPJax, GPyTorch, custom kernels) and supports structured kernels like
Kronecker products for operator learning.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from laplax.types import Array, Callable, Float, PredArray


# =============================================================================
# KERNEL PROTOCOL
# =============================================================================


class KernelProtocol(Protocol):
    """Protocol for kernel functions.

    Compatible with most kernel implementations from GPJax, GPyTorch, etc.
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
# BASE KERNEL CLASSES
# =============================================================================


class Kernel(ABC):
    """Base class for all kernels in laplax FSP."""

    @abstractmethod
    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        """Compute kernel matrix between x1 and x2.

        Parameters
        ----------
        x1 : Array
            Input points of shape (N, D)
        x2 : Array, optional
            Input points of shape (M, D). If None, use x1.

        Returns
        -------
        Array
            Kernel matrix of shape (N, M) or (N, N)
        """
        pass

    @abstractmethod
    def diagonal(self, x: Array) -> Array:
        """Compute kernel diagonal K(x, x).

        Parameters
        ----------
        x : Array
            Input points of shape (N, D)

        Returns
        -------
        Array
            Diagonal values of shape (N,)
        """
        pass


class RBFKernel(Kernel):
    """RBF (Squared Exponential) kernel.

    K(x, y) = σ² exp(-||x - y||² / (2ℓ²))
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        """Initialize RBF kernel.

        Parameters
        ----------
        lengthscale : float
            Length scale parameter
        variance : float
            Output variance
        """
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        """Compute RBF kernel matrix."""
        if x2 is None:
            x2 = x1

        # Compute pairwise squared distances
        sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
        return self.variance * jnp.exp(-sq_dist / (2 * self.lengthscale**2))

    def diagonal(self, x: Array) -> Array:
        """Compute kernel diagonal."""
        return self.variance * jnp.ones(x.shape[0])


class Matern52Kernel(Kernel):
    """Matérn 5/2 kernel.

    K(x, y) = σ² (1 + √5r/ℓ + 5r²/(3ℓ²)) exp(-√5r/ℓ)
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        if x2 is None:
            x2 = x1

        r = jnp.sqrt(jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1))
        sqrt5 = jnp.sqrt(5.0)
        sqrt5_r_l = sqrt5 * r / self.lengthscale
        return (
            self.variance
            * (1.0 + sqrt5_r_l + sqrt5_r_l**2 / 3.0)
            * jnp.exp(-sqrt5_r_l)
        )

    def diagonal(self, x: Array) -> Array:
        return self.variance * jnp.ones(x.shape[0])


# =============================================================================
# STRUCTURED KERNELS FOR OPERATOR LEARNING
# =============================================================================


class KroneckerKernel(Kernel):
    """Kronecker product of multiple kernels.

    For operator learning with spatial/temporal structure:
    K = K_spatial ⊗ K_function

    This is memory-efficient for structured data.
    """

    def __init__(self, kernels: list[Kernel]):
        """Initialize Kronecker kernel.

        Parameters
        ----------
        kernels : list[Kernel]
            List of kernels to combine via Kronecker product
        """
        self.kernels = kernels

    def __call__(self, x1: list[Array], x2: list[Array] | None = None) -> Array:
        """Compute Kronecker product kernel.

        Parameters
        ----------
        x1 : list[Array]
            List of input arrays, one per kernel
        x2 : list[Array], optional
            List of second input arrays. If None, use x1.

        Returns
        -------
        Array
            Kronecker product kernel matrix
        """
        if x2 is None:
            x2 = x1

        if len(x1) != len(self.kernels):
            raise ValueError(
                f"Expected {len(self.kernels)} inputs, got {len(x1)}"
            )

        # Compute individual kernel matrices
        kernel_matrices = [
            kernel(x1_i, x2_i)
            for kernel, x1_i, x2_i in zip(self.kernels, x1, x2)
        ]

        # Compute Kronecker product
        result = kernel_matrices[0]
        for k_mat in kernel_matrices[1:]:
            result = jnp.kron(result, k_mat)

        return result

    def diagonal(self, x: list[Array]) -> Array:
        """Compute diagonal of Kronecker product."""
        diagonals = [kernel.diagonal(x_i) for kernel, x_i in zip(self.kernels, x)]

        # Kronecker product of diagonals
        result = diagonals[0]
        for diag in diagonals[1:]:
            result = jnp.kron(result, diag)

        return result

    def low_rank_approximation(
        self, x: list[Array], rank: int | None = None
    ) -> tuple[Array, Array]:
        """Compute low-rank approximation using Lanczos per kernel.

        Parameters
        ----------
        x : list[Array]
            List of input arrays
        rank : int, optional
            Rank for approximation. If None, determined automatically.

        Returns
        -------
        tuple[Array, Array]
            (U, S) where K ≈ U @ S @ U.T in the Kronecker structure
        """
        # For each kernel, compute its low-rank factors
        factors = []
        for kernel, x_i in zip(self.kernels, x):
            K_i = kernel(x_i)
            # Simple eigendecomposition (can be replaced with Lanczos)
            eigvals, eigvecs = jnp.linalg.eigh(K_i)

            # Sort in descending order
            idx = jnp.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            if rank is not None:
                eigvals = eigvals[:rank]
                eigvecs = eigvecs[:, :rank]

            factors.append((eigvecs, jnp.sqrt(jnp.maximum(eigvals, 0.0))))

        return factors


class AdditiveKernel(Kernel):
    """Sum of multiple kernels.

    K = K₁ + K₂ + ... + Kₙ
    """

    def __init__(self, kernels: list[Kernel]):
        self.kernels = kernels

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        result = self.kernels[0](x1, x2)
        for kernel in self.kernels[1:]:
            result += kernel(x1, x2)
        return result

    def diagonal(self, x: Array) -> Array:
        result = self.kernels[0].diagonal(x)
        for kernel in self.kernels[1:]:
            result += kernel.diagonal(x)
        return result


class ProductKernel(Kernel):
    """Product of multiple kernels.

    K = K₁ * K₂ * ... * Kₙ
    """

    def __init__(self, kernels: list[Kernel]):
        self.kernels = kernels

    def __call__(self, x1: Array, x2: Array | None = None) -> Array:
        result = self.kernels[0](x1, x2)
        for kernel in self.kernels[1:]:
            result *= kernel(x1, x2)
        return result

    def diagonal(self, x: Array) -> Array:
        result = self.kernels[0].diagonal(x)
        for kernel in self.kernels[1:]:
            result *= kernel.diagonal(x)
        return result


# =============================================================================
# KERNEL ADAPTERS FOR EXTERNAL LIBRARIES
# =============================================================================


class GPJaxKernelAdapter(Kernel):
    """Adapter for GPJax kernels.

    Wraps a GPJax kernel to be compatible with laplax FSP.
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

        # GPJax kernels typically have a Gram method
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


def wrap_kernel_fn(kernel_fn: Callable) -> Kernel:
    """Wrap a kernel function into a Kernel object.

    Parameters
    ----------
    kernel_fn : Callable
        A function that computes kernel: kernel_fn(x1, x2) -> K

    Returns
    -------
    Kernel
        Wrapped kernel object
    """

    class WrappedKernel(Kernel):
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, x1: Array, x2: Array | None = None) -> Array:
            if x2 is None:
                x2 = x1
            return self.fn(x1, x2)

        def diagonal(self, x: Array) -> Array:
            return jax.vmap(lambda xi: self.fn(xi[None, :], xi[None, :])[0, 0])(x)

    return WrappedKernel(kernel_fn)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def build_gram_matrix(
    kernel: Kernel, x: Array, jitter: float = 1e-6
) -> Array:
    """Build Gram matrix with jitter.

    Parameters
    ----------
    kernel : Kernel
        Kernel object
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


def kernel_variance(kernel: Kernel, x: Array) -> Float:
    """Compute total kernel variance.

    Parameters
    ----------
    kernel : Kernel
        Kernel object
    x : Array
        Input points

    Returns
    -------
    Float
        Sum of diagonal elements (trace)
    """
    return jnp.sum(kernel.diagonal(x))
