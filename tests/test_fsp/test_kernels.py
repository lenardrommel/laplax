"""Tests for FSP kernel abstractions and compatibility."""

import jax
import jax.numpy as jnp
import pytest

from laplax.extra.fsp.kernels import (
    AdditiveKernel,
    KroneckerKernel,
    Matern52Kernel,
    ProductKernel,
    RBFKernel,
    build_gram_matrix,
    kernel_variance,
    wrap_kernel_fn,
)

jax.config.update("jax_enable_x64", True)


def test_rbf_kernel():
    """Test RBF kernel implementation."""
    kernel = RBFKernel(lengthscale=1.0, variance=2.0)

    # Test with simple inputs
    x = jnp.array([[0.0], [1.0], [2.0]])

    # Compute Gram matrix
    K = kernel(x, x)

    # Check properties
    assert K.shape == (3, 3)
    assert jnp.isfinite(K).all()

    # Diagonal should be variance
    assert jnp.allclose(jnp.diag(K), 2.0)

    # Kernel should be symmetric
    assert jnp.allclose(K, K.T)

    # Test diagonal method
    diag = kernel.diagonal(x)
    assert jnp.allclose(diag, jnp.diag(K))


def test_matern52_kernel():
    """Test Matérn 5/2 kernel implementation."""
    kernel = Matern52Kernel(lengthscale=0.5, variance=1.5)

    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    K = kernel(x, x)

    # Check properties
    assert K.shape == (3, 3)
    assert jnp.isfinite(K).all()
    assert jnp.allclose(K, K.T)
    assert jnp.allclose(jnp.diag(K), 1.5)


def test_additive_kernel():
    """Test additive kernel composition."""
    k1 = RBFKernel(lengthscale=1.0, variance=1.0)
    k2 = Matern52Kernel(lengthscale=0.5, variance=0.5)
    kernel = AdditiveKernel([k1, k2])

    x = jnp.array([[0.0], [1.0], [2.0]])

    K = kernel(x, x)
    K1 = k1(x, x)
    K2 = k2(x, x)

    # Should be sum of individual kernels
    assert jnp.allclose(K, K1 + K2)

    # Test diagonal
    diag = kernel.diagonal(x)
    assert jnp.allclose(diag, k1.diagonal(x) + k2.diagonal(x))


def test_product_kernel():
    """Test product kernel composition."""
    k1 = RBFKernel(lengthscale=1.0, variance=2.0)
    k2 = RBFKernel(lengthscale=0.5, variance=1.5)
    kernel = ProductKernel([k1, k2])

    x = jnp.array([[0.0], [1.0]])

    K = kernel(x, x)
    K1 = k1(x, x)
    K2 = k2(x, x)

    # Should be product of individual kernels
    assert jnp.allclose(K, K1 * K2)


def test_kronecker_kernel():
    """Test Kronecker product kernel."""
    # Create simple kernels for each dimension
    k_spatial = RBFKernel(lengthscale=1.0, variance=1.0)
    k_function = RBFKernel(lengthscale=0.5, variance=2.0)

    kernel = KroneckerKernel([k_spatial, k_function])

    # Create input data
    x_spatial = jnp.array([[0.0], [1.0]])  # 2 spatial points
    x_function = jnp.array([[0.0], [1.0], [2.0]])  # 3 functions

    # Compute Kronecker product
    K = kernel([x_spatial, x_function])

    # Expected shape: (2*3, 2*3) = (6, 6)
    assert K.shape == (6, 6)
    assert jnp.isfinite(K).all()
    assert jnp.allclose(K, K.T)  # Should be symmetric

    # Verify Kronecker structure
    K_spatial = k_spatial(x_spatial, x_spatial)
    K_function = k_function(x_function, x_function)
    K_expected = jnp.kron(K_spatial, K_function)
    assert jnp.allclose(K, K_expected)


def test_wrap_kernel_fn():
    """Test wrapping a callable kernel function."""

    # Define a simple RBF kernel function
    def my_rbf(x1, x2):
        sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-sq_dist / 2.0)

    # Wrap it
    kernel = wrap_kernel_fn(my_rbf)

    # Test
    x = jnp.array([[0.0], [1.0], [2.0]])
    K = kernel(x, x)

    assert K.shape == (3, 3)
    assert jnp.isfinite(K).all()
    assert jnp.allclose(K, K.T)

    # Test diagonal
    diag = kernel.diagonal(x)
    assert diag.shape == (3,)
    assert jnp.isfinite(diag).all()


def test_build_gram_matrix():
    """Test Gram matrix builder with jitter."""
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)
    x = jnp.array([[0.0], [1.0], [2.0]])

    jitter = 1e-5
    K = build_gram_matrix(kernel, x, jitter=jitter)

    # Check that jitter was added
    K_no_jitter = kernel(x, x)
    expected = K_no_jitter + jitter * jnp.eye(3)
    assert jnp.allclose(K, expected)


def test_kernel_variance():
    """Test kernel variance computation."""
    kernel = RBFKernel(lengthscale=1.0, variance=2.0)
    x = jnp.array([[0.0], [1.0], [2.0]])

    # Variance is sum of diagonal
    var = kernel_variance(kernel, x)
    expected = jnp.sum(kernel.diagonal(x))

    assert jnp.allclose(var, expected)
    assert jnp.allclose(var, 6.0)  # 3 points * variance 2.0


def test_kronecker_low_rank():
    """Test Kronecker low-rank approximation."""
    k_spatial = RBFKernel(lengthscale=1.0, variance=1.0)
    k_function = RBFKernel(lengthscale=0.5, variance=1.0)
    kernel = KroneckerKernel([k_spatial, k_function])

    x_spatial = jnp.linspace(-2, 2, 5).reshape(-1, 1)
    x_function = jnp.linspace(-1, 1, 4).reshape(-1, 1)

    # Compute low-rank approximation
    factors = kernel.low_rank_approximation([x_spatial, x_function], rank=3)

    # Check structure
    assert len(factors) == 2  # One per kernel
    for eigvecs, eigvals_sqrt in factors:
        assert eigvecs.shape[1] == 3  # Rank 3
        assert eigvals_sqrt.shape[0] == 3
        assert jnp.isfinite(eigvecs).all()
        assert jnp.isfinite(eigvals_sqrt).all()
        assert jnp.all(eigvals_sqrt >= 0)  # Eigenvalues should be non-negative


def test_kernel_cross_covariance():
    """Test kernel evaluation on different inputs."""
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)

    x1 = jnp.array([[0.0], [1.0]])
    x2 = jnp.array([[0.5], [1.5], [2.0]])

    K = kernel(x1, x2)

    # Check shape
    assert K.shape == (2, 3)
    assert jnp.isfinite(K).all()

    # Verify some values
    # K[0, 0] should be kernel between x1[0] and x2[0]
    expected_00 = jnp.exp(-0.5**2 / 2.0)
    assert jnp.allclose(K[0, 0], expected_00)


@pytest.mark.parametrize(
    "lengthscale,variance",
    [
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 0.5),
    ],
)
def test_kernel_parameters(lengthscale, variance):
    """Test kernels with different hyperparameters."""
    kernel = RBFKernel(lengthscale=lengthscale, variance=variance)

    x = jnp.array([[0.0], [1.0]])

    K = kernel(x, x)

    # Diagonal should match variance
    assert jnp.allclose(jnp.diag(K), variance)

    # Off-diagonal elements should be less than variance
    assert K[0, 1] < variance
    assert K[0, 1] > 0  # Should be positive


if __name__ == "__main__":
    # Run tests without pytest
    test_rbf_kernel()
    test_matern52_kernel()
    test_additive_kernel()
    test_product_kernel()
    test_kronecker_kernel()
    test_wrap_kernel_fn()
    test_build_gram_matrix()
    test_kernel_variance()
    test_kronecker_low_rank()
    test_kernel_cross_covariance()

    print("✓ All kernel tests passed!")
