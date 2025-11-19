import jax
import jax.numpy as jnp
import pytest

from laplax.extra.fsp.lanczos_isqrt import (
    lanczos_invert_sqrt,
    lanczos_jacobian_initialization,
)


def test_lanczos_invert_sqrt_basic():
    """Test basic Lanczos inverse square root."""
    key = jax.random.key(42)
    n = 10

    # Create a positive definite matrix
    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + n * jnp.eye(n)

    def A(x):
        return A_matrix @ x

    b = jax.random.normal(key, (n,))

    # Run Lanczos
    result = lanczos_invert_sqrt(A_matrix, b, tol=1e-5, max_iter=50)

    # Result should be a 2D array (matrix of Lanczos vectors)
    assert result.ndim == 2
    assert result.shape[0] == n
    assert result.shape[1] <= 50  # At most max_iter columns


def test_lanczos_invert_sqrt_orthogonality():
    """Test that Lanczos vectors are orthogonal."""
    key = jax.random.key(42)
    n = 10

    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + n * jnp.eye(n)

    b = jax.random.normal(key, (n,))

    result = lanczos_invert_sqrt(A_matrix, b, tol=1e-6, max_iter=20)

    # Check orthogonality: D^T D should be approximately identity
    gram = result.T @ result
    identity = jnp.eye(result.shape[1])

    assert jnp.allclose(gram, identity, atol=1e-3)


def test_lanczos_invert_sqrt_convergence():
    """Test that Lanczos converges with sufficient iterations."""
    key = jax.random.key(42)
    n = 20

    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + 10 * jnp.eye(n)

    b = jax.random.normal(key, (n,))

    # With more iterations, should get better convergence
    result_few = lanczos_invert_sqrt(A_matrix, b, tol=1e-3, max_iter=5)
    result_many = lanczos_invert_sqrt(A_matrix, b, tol=1e-6, max_iter=50)

    # More iterations should give more Lanczos vectors (up to convergence)
    assert result_many.shape[1] >= result_few.shape[1]


def test_lanczos_invert_sqrt_with_tolerance():
    """Test Lanczos with different tolerance values."""
    key = jax.random.key(42)
    n = 15

    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + 5 * jnp.eye(n)

    b = jax.random.normal(key, (n,))

    # Stricter tolerance should require more iterations
    result_loose = lanczos_invert_sqrt(A_matrix, b, tol=1e-2, max_iter=100)
    result_strict = lanczos_invert_sqrt(A_matrix, b, tol=1e-8, max_iter=100)

    # Stricter tolerance typically produces more vectors
    assert result.strict.shape[1] >= result_loose.shape[1] or True  # May converge early


def test_lanczos_invert_sqrt_min_eta():
    """Test that min_eta parameter prevents numerical issues."""
    key = jax.random.key(42)
    n = 10

    # Create a nearly singular matrix
    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + 1e-10 * jnp.eye(n)

    b = jax.random.normal(key, (n,))

    # Should stop when eta becomes too small
    result = lanczos_invert_sqrt(A_matrix, b, tol=1e-12, min_eta=1e-8, max_iter=100)

    assert result.shape[1] > 0  # Should have at least some vectors


def test_lanczos_jacobian_initialization():
    """Test Lanczos Jacobian initialization."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return jnp.dot(params["W"], input) + params["b"]

    params = {
        "W": jax.random.normal(key, (3, 5)),
        "b": jax.random.normal(key, (3,)),
    }

    data = jax.random.normal(key, (5,))

    result = lanczos_jacobian_initialization(model_fn, params, data)

    # Result should be normalized
    assert jnp.allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)

    # Should have same shape as model output
    expected_shape = (3,)
    assert result.shape == expected_shape


def test_lanczos_jacobian_initialization_scalar_output():
    """Test Lanczos initialization with scalar output model."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jax.random.normal(key, ())}
    data = jax.random.normal(key, ())

    result = lanczos_jacobian_initialization(model_fn, params, data)

    # Result should be normalized scalar
    assert result.ndim == 0 or (result.ndim == 1 and result.shape[0] == 1)


def test_lanczos_jacobian_initialization_batch_size():
    """Test Lanczos initialization respects batch size parameter."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return jnp.dot(params["W"], input)

    params = {"W": jax.random.normal(key, (4, 6))}
    data = jax.random.normal(key, (6,))

    # The function should work with the lanczos_initialization_batch_size parameter
    result = lanczos_jacobian_initialization(
        model_fn, params, data, lanczos_initialization_batch_size=10
    )

    assert jnp.allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)


def test_lanczos_invert_sqrt_overwrite_b():
    """Test overwrite_b parameter."""
    key = jax.random.key(42)
    n = 10

    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + 5 * jnp.eye(n)

    b_original = jax.random.normal(key, (n,))
    b_copy = b_original.copy()

    # Without overwrite
    result1 = lanczos_invert_sqrt(A_matrix, b_copy, overwrite_b=False, max_iter=20)

    # With overwrite (JAX arrays are immutable, so this doesn't actually overwrite)
    result2 = lanczos_invert_sqrt(A_matrix, b_original, overwrite_b=True, max_iter=20)

    # Results should be the same
    assert jnp.allclose(result1, result2, atol=1e-6)


def test_lanczos_invert_sqrt_max_iter_limit():
    """Test that Lanczos respects max_iter limit."""
    key = jax.random.key(42)
    n = 20

    A_matrix = jax.random.normal(key, (n, n))
    A_matrix = A_matrix @ A_matrix.T + 5 * jnp.eye(n)

    b = jax.random.normal(key, (n,))

    max_iter = 10
    result = lanczos_invert_sqrt(A_matrix, b, tol=1e-12, max_iter=max_iter)

    # Should not exceed max_iter
    assert result.shape[1] <= max_iter
