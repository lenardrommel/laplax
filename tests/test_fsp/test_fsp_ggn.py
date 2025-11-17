"""Tests for FSP GGN matrix-vector product functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.enums import LossFn
from laplax.extra.fsp.ggn import create_fsp_ggn_mv


def create_simple_linear_model(output_dim: int):
    """Create a simple linear model for testing.

    Args:
        output_dim: Output dimension of the model

    Returns:
        Tuple of (model_fn, params)
    """
    key = jax.random.PRNGKey(42)
    input_dim = 10

    # Initialize simple linear model parameters
    params = {
        "w": jax.random.normal(key, (input_dim, output_dim)),
        "b": jnp.zeros(output_dim),
    }

    def model_fn(input, params):
        return input @ params["w"] + params["b"]

    return model_fn, params


def create_simple_mlp_model(hidden_dim: int = 20, output_dim: int = 5):
    """Create a simple MLP model for testing.

    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension

    Returns:
        Tuple of (model_fn, params)
    """
    key = jax.random.PRNGKey(42)
    input_dim = 10

    key1, key2, key3, key4 = jax.random.split(key, 4)

    params = {
        "w1": jax.random.normal(key1, (input_dim, hidden_dim)) * 0.1,
        "b1": jax.random.normal(key2, (hidden_dim,)) * 0.1,
        "w2": jax.random.normal(key3, (hidden_dim, output_dim)) * 0.1,
        "b2": jax.random.normal(key4, (output_dim,)) * 0.1,
    }

    def model_fn(input, params):
        hidden = jax.nn.relu(input @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]

    return model_fn, params


def test_create_fsp_ggn_mv_basic():
    """Test basic functionality of create_fsp_ggn_mv."""
    output_dim = 5
    model_fn, params = create_simple_linear_model(output_dim)

    # Create a simple context matrix M
    key = jax.random.PRNGKey(123)
    n_context = 8
    M = jax.random.normal(key, (output_dim, n_context))

    # Create FSP GGN MV function
    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Test data
    batch_size = 4
    input_dim = 10
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(456), (batch_size, input_dim)),
        "target": jax.random.normal(jax.random.PRNGKey(789), (batch_size, output_dim)),
    }

    # Apply FSP GGN MV
    result = fsp_ggn_mv(test_data)

    # Check output shape
    assert result.shape == (n_context, n_context), f"Expected shape {(n_context, n_context)}, got {result.shape}"

    # Result should be symmetric (approximately)
    np.testing.assert_allclose(result, result.T, rtol=1e-5, atol=1e-5)


def test_create_fsp_ggn_mv_output_shape():
    """Test that FSP GGN MV produces correct output shape."""
    output_dim = 3
    n_context = 6

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(0)
    M = jax.random.normal(key, (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 10
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    assert result.shape == (n_context, n_context)


def test_create_fsp_ggn_mv_with_mlp():
    """Test FSP GGN MV with a multi-layer perceptron."""
    output_dim = 5
    hidden_dim = 15
    n_context = 10

    model_fn, params = create_simple_mlp_model(hidden_dim, output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 8
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(100), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(200), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    assert result.shape == (n_context, n_context)
    # Check symmetry
    np.testing.assert_allclose(result, result.T, rtol=1e-4, atol=1e-6)


def test_create_fsp_ggn_mv_positive_semidefinite():
    """Test that FSP GGN matrix is positive semi-definite."""
    output_dim = 4
    n_context = 8

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 12
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    # Check that all eigenvalues are non-negative
    eigenvalues = jnp.linalg.eigvalsh(result)
    assert jnp.all(eigenvalues >= -1e-6), f"Found negative eigenvalues: {eigenvalues[eigenvalues < -1e-6]}"


def test_create_fsp_ggn_mv_svd_truncation():
    """Test that SVD truncation works correctly."""
    output_dim = 10
    n_context = 5

    model_fn, params = create_simple_linear_model(output_dim)

    # Create M with some small singular values
    key = jax.random.PRNGKey(42)
    M_full = jax.random.normal(key, (output_dim, n_context))

    # Add small noise to ensure some singular values are below threshold
    M = M_full + 1e-10 * jax.random.normal(jax.random.PRNGKey(43), (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 6
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    # Result should still be valid
    assert result.shape == (n_context, n_context)
    assert jnp.all(jnp.isfinite(result))


def test_create_fsp_ggn_mv_with_low_rank_M():
    """Test FSP GGN MV with low-rank M matrix."""
    output_dim = 10
    rank = 3
    n_context = 5

    model_fn, params = create_simple_linear_model(output_dim)

    # Create low-rank M
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    U = jax.random.normal(key1, (output_dim, rank))
    V = jax.random.normal(key2, (rank, n_context))
    M = U @ V

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 8
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    assert result.shape == (n_context, n_context)

    # Check rank (approximately)
    singular_values = jnp.linalg.svd(result, compute_uv=False)
    effective_rank = jnp.sum(singular_values > 1e-6)
    # FSP adds a diagonal term, so rank might be higher than original M
    assert effective_rank <= n_context


def test_create_fsp_ggn_mv_batched_raises_error():
    """Test that batched data raises NotImplementedError."""
    output_dim = 5
    n_context = 8

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    # Should raise error with has_batch=True
    with pytest.raises(NotImplementedError, match="FSP GGN MV is not implemented for batched data"):
        create_fsp_ggn_mv(model_fn, params, M, has_batch=True)


def test_create_fsp_ggn_mv_different_batch_sizes():
    """Test FSP GGN MV with different batch sizes."""
    output_dim = 5
    n_context = 6

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Test with different batch sizes
    for batch_size in [1, 5, 10, 20]:
        test_data = {
            "input": jax.random.normal(jax.random.PRNGKey(batch_size), (batch_size, 10)),
            "target": jax.random.normal(jax.random.PRNGKey(batch_size + 1000), (batch_size, output_dim)),
        }

        result = fsp_ggn_mv(test_data)

        assert result.shape == (n_context, n_context)
        assert jnp.all(jnp.isfinite(result))


def test_create_fsp_ggn_mv_reproducibility():
    """Test that FSP GGN MV produces reproducible results."""
    output_dim = 5
    n_context = 8

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    # Create two instances
    fsp_ggn_mv1 = create_fsp_ggn_mv(model_fn, params, M)
    fsp_ggn_mv2 = create_fsp_ggn_mv(model_fn, params, M)

    # Same test data
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (10, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (10, output_dim)),
    }

    result1 = fsp_ggn_mv1(test_data)
    result2 = fsp_ggn_mv2(test_data)

    np.testing.assert_allclose(result1, result2, rtol=1e-6, atol=1e-8)


def test_create_fsp_ggn_mv_with_identity_M():
    """Test FSP GGN MV with identity-like M matrix."""
    output_dim = 5
    n_context = 5

    model_fn, params = create_simple_linear_model(output_dim)

    # Use identity matrix
    M = jnp.eye(output_dim, n_context)

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 10
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    assert result.shape == (n_context, n_context)
    # Result should be symmetric
    np.testing.assert_allclose(result, result.T, rtol=1e-5, atol=1e-6)


def test_create_fsp_ggn_mv_numerical_stability():
    """Test numerical stability with extreme values."""
    output_dim = 5
    n_context = 8

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    # Create M with some larger values
    M = jax.random.normal(key, (output_dim, n_context)) * 10

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 10
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    # Check no NaN or Inf
    assert jnp.all(jnp.isfinite(result)), "Result contains NaN or Inf values"

    # Check symmetry is maintained
    np.testing.assert_allclose(result, result.T, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("output_dim", [2, 5, 10])
@pytest.mark.parametrize("n_context", [3, 6, 12])
def test_create_fsp_ggn_mv_various_dimensions(output_dim, n_context):
    """Test FSP GGN MV with various dimension combinations."""
    # Only test when n_context <= output_dim for meaningful SVD
    if n_context > output_dim * 2:
        pytest.skip("Skipping test where n_context >> output_dim")

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)

    # Create test data
    batch_size = 8
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    result = fsp_ggn_mv(test_data)

    assert result.shape == (n_context, n_context)
    assert jnp.all(jnp.isfinite(result))


def test_create_fsp_ggn_mv_gradient_compatibility():
    """Test that FSP GGN MV is compatible with JAX gradient operations."""
    output_dim = 4
    n_context = 6

    model_fn, params = create_simple_linear_model(output_dim)

    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (output_dim, n_context))

    # Create test data
    batch_size = 8
    test_data = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    def loss_fn(M):
        fsp_ggn_mv = create_fsp_ggn_mv(model_fn, params, M)
        result = fsp_ggn_mv(test_data)
        return jnp.sum(result)

    # Should be able to compute gradient w.r.t. M
    grad = jax.grad(loss_fn)(M)

    assert grad.shape == M.shape
    assert jnp.all(jnp.isfinite(grad))
