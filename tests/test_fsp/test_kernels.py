"""Tests for FSP kernel interface and adapters.

This test file demonstrates how to use simple callable kernels with FSP,
as laplax does NOT implement its own kernel library. Users should use:
- GPJax for GP kernels in JAX
- GPyTorch for GP kernels in PyTorch
- Custom callables for simple cases
"""

import jax
import jax.numpy as jnp
import pytest

from laplax.extra.fsp.kernels import (
    GPJaxKernelAdapter,
    KernelProtocol,
    build_gram_matrix,
    kernel_variance,
    wrap_kernel_fn,
)

jax.config.update("jax_enable_x64", True)


# =============================================================================
# SIMPLE CALLABLE KERNELS (for testing)
# =============================================================================


def rbf_kernel(lengthscale=1.0, variance=1.0):
    """Create RBF kernel as a simple callable."""

    def kernel(x1, x2=None):
        if x2 is None:
            x2 = x1
        sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
        return variance * jnp.exp(-sq_dist / (2 * lengthscale**2))

    return kernel


def matern52_kernel(lengthscale=1.0, variance=1.0):
    """Create Matérn 5/2 kernel as a simple callable."""

    def kernel(x1, x2=None):
        if x2 is None:
            x2 = x1
        r = jnp.sqrt(jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1))
        scaled_r = jnp.sqrt(5.0) * r / lengthscale
        return variance * (1 + scaled_r + scaled_r**2 / 3) * jnp.exp(-scaled_r)

    return kernel


# =============================================================================
# TESTS FOR CALLABLE KERNELS
# =============================================================================


def test_rbf_kernel_callable():
    """Test RBF kernel as a simple callable."""
    kernel = rbf_kernel(lengthscale=1.0, variance=2.0)

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


def test_matern52_kernel_callable():
    """Test Matérn 5/2 kernel as a simple callable."""
    kernel = matern52_kernel(lengthscale=0.5, variance=1.5)

    x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    K = kernel(x, x)

    # Check properties
    assert K.shape == (3, 3)
    assert jnp.isfinite(K).all()
    assert jnp.allclose(K, K.T)
    assert jnp.allclose(jnp.diag(K), 1.5)


def test_additive_kernel_callable():
    """Test additive kernel composition with callables."""
    k1 = rbf_kernel(lengthscale=1.0, variance=1.0)
    k2 = matern52_kernel(lengthscale=0.5, variance=0.5)

    def additive_kernel(x1, x2=None):
        return k1(x1, x2) + k2(x1, x2)

    x = jnp.array([[0.0], [1.0], [2.0]])

    K = additive_kernel(x, x)
    K1 = k1(x, x)
    K2 = k2(x, x)

    # Should be sum of individual kernels
    assert jnp.allclose(K, K1 + K2)


def test_product_kernel_callable():
    """Test product kernel composition with callables."""
    k1 = rbf_kernel(lengthscale=1.0, variance=2.0)
    k2 = rbf_kernel(lengthscale=0.5, variance=1.5)

    def product_kernel(x1, x2=None):
        return k1(x1, x2) * k2(x1, x2)

    x = jnp.array([[0.0], [1.0]])

    K = product_kernel(x, x)
    K1 = k1(x, x)
    K2 = k2(x, x)

    # Should be product of individual kernels
    assert jnp.allclose(K, K1 * K2)


def test_kronecker_kernel_callable():
    """Test Kronecker product kernel with callables."""
    k_spatial = rbf_kernel(lengthscale=1.0, variance=1.0)
    k_function = rbf_kernel(lengthscale=0.5, variance=2.0)

    def kronecker_kernel(xs):
        """xs is a list [x_spatial, x_function]"""
        x_spatial, x_function = xs
        K_spatial = k_spatial(x_spatial, x_spatial)
        K_function = k_function(x_function, x_function)
        return jnp.kron(K_spatial, K_function)

    # Create input data
    x_spatial = jnp.array([[0.0], [1.0]])  # 2 spatial points
    x_function = jnp.array([[0.0], [1.0], [2.0]])  # 3 functions

    # Compute Kronecker product
    K = kronecker_kernel([x_spatial, x_function])

    # Expected shape: (2*3, 2*3) = (6, 6)
    assert K.shape == (6, 6)
    assert jnp.isfinite(K).all()
    assert jnp.allclose(K, K.T)  # Should be symmetric

    # Verify Kronecker structure
    K_spatial = k_spatial(x_spatial, x_spatial)
    K_function = k_function(x_function, x_function)
    K_expected = jnp.kron(K_spatial, K_function)
    assert jnp.allclose(K, K_expected)


# =============================================================================
# TESTS FOR KERNEL WRAPPERS
# =============================================================================


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

    # Test diagonal method
    diag = kernel.diagonal(x)
    assert diag.shape == (3,)
    assert jnp.isfinite(diag).all()


def test_build_gram_matrix():
    """Test Gram matrix builder with jitter."""
    kernel = rbf_kernel(lengthscale=1.0, variance=1.0)
    x = jnp.array([[0.0], [1.0], [2.0]])

    jitter = 1e-5
    K = build_gram_matrix(kernel, x, jitter=jitter)

    # Check that jitter was added
    K_no_jitter = kernel(x, x)
    expected = K_no_jitter + jitter * jnp.eye(3)
    assert jnp.allclose(K, expected)


def test_kernel_variance():
    """Test kernel variance computation with wrapped callable."""
    kernel_fn = rbf_kernel(lengthscale=1.0, variance=2.0)
    kernel = wrap_kernel_fn(kernel_fn)
    x = jnp.array([[0.0], [1.0], [2.0]])

    # Variance is sum of diagonal
    var = kernel_variance(kernel, x)
    expected = jnp.sum(kernel.diagonal(x))

    assert jnp.allclose(var, expected)
    assert jnp.allclose(var, 6.0)  # 3 points * variance 2.0


def test_kernel_cross_covariance():
    """Test kernel evaluation on different inputs."""
    kernel = rbf_kernel(lengthscale=1.0, variance=1.0)

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
    kernel = rbf_kernel(lengthscale=lengthscale, variance=variance)

    x = jnp.array([[0.0], [1.0]])

    K = kernel(x, x)

    # Diagonal should match variance
    assert jnp.allclose(jnp.diag(K), variance)

    # Off-diagonal elements should be less than variance
    assert K[0, 1] < variance
    assert K[0, 1] > 0  # Should be positive


# =============================================================================
# TESTS FOR GPJAX ADAPTER (if GPJax is available)
# =============================================================================


@pytest.mark.skipif(
    not hasattr(GPJaxKernelAdapter, "__module__"), reason="GPJax not available"
)
def test_gpjax_adapter():
    """Test GPJax kernel adapter (requires GPJax)."""
    try:
        import gpjax as gpx

        # Create a GPJax kernel
        gpjax_kernel = gpx.kernels.RBF()
        params = {"lengthscale": jnp.array([1.0]), "variance": jnp.array([2.0])}

        # Wrap it
        kernel = GPJaxKernelAdapter(gpjax_kernel, params)

        # Test
        x = jnp.array([[0.0], [1.0], [2.0]])
        K = kernel(x, x)

        assert K.shape == (3, 3)
        assert jnp.isfinite(K).all()
        assert jnp.allclose(K, K.T)

    except ImportError:
        pytest.skip("GPJax not installed")


# =============================================================================
# PROTOCOL COMPLIANCE
# =============================================================================


def test_kernel_protocol_compliance():
    """Test that wrapped callables comply with KernelProtocol."""
    kernel_fn = rbf_kernel(lengthscale=1.0, variance=1.0)
    wrapped = wrap_kernel_fn(kernel_fn)

    # Check that wrapped kernel has required methods
    assert callable(wrapped)
    assert hasattr(wrapped, "diagonal")

    # Test usage
    x = jnp.array([[0.0], [1.0]])
    K = wrapped(x, x)
    diag = wrapped.diagonal(x)

    assert K.shape == (2, 2)
    assert diag.shape == (2,)
    assert jnp.allclose(diag, jnp.diag(K))


if __name__ == "__main__":
    # Run tests without pytest
    test_rbf_kernel_callable()
    test_matern52_kernel_callable()
    test_additive_kernel_callable()
    test_product_kernel_callable()
    test_kronecker_kernel_callable()
    test_wrap_kernel_fn()
    test_build_gram_matrix()
    test_kernel_variance()
    test_kernel_cross_covariance()
    test_kernel_protocol_compliance()

    print("✓ All kernel interface tests passed!")
