import jax
import jax.numpy as jnp
import pytest

from laplax.extra.fsp.ggn import create_fsp_ggn_mv


def test_create_fsp_ggn_mv_basic():
    """Test basic FSP GGN matrix-vector product creation."""
    key = jax.random.key(42)

    # Simple linear model
    def model_fn(input, params):
        return jnp.dot(input, params["w"])

    params = {"w": jax.random.normal(key, (5,))}

    # Create a random matrix M
    M = jax.random.normal(key, (10, 8))

    fsp_ggn_mv_fn = create_fsp_ggn_mv(model_fn, params, M, has_batch=False)

    # Create test data
    data = {"input": jax.random.normal(key, (5,)), "target": jax.random.normal(key, (1,))}

    # Compute FSP GGN
    result = fsp_ggn_mv_fn(data)

    # Result should be a matrix
    assert isinstance(result, jax.Array)
    assert result.ndim == 2


def test_fsp_ggn_mv_svd_filtering():
    """Test that FSP GGN properly filters small singular values."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}

    # Create matrix with some very small singular values
    M = jnp.array([[1.0, 0.0], [0.0, 1e-20]])

    fsp_ggn_mv_fn = create_fsp_ggn_mv(model_fn, params, M, has_batch=False)

    data = {"input": jnp.array(1.0), "target": jnp.array(1.0)}

    result = fsp_ggn_mv_fn(data)

    # Should work without errors despite small singular values
    assert isinstance(result, jax.Array)


def test_fsp_ggn_mv_batched_raises_error():
    """Test that has_batch=True raises NotImplementedError."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}
    M = jax.random.normal(key, (5, 3))

    with pytest.raises(NotImplementedError, match="FSP GGN MV is not implemented for batched data"):
        create_fsp_ggn_mv(model_fn, params, M, has_batch=True)


def test_fsp_ggn_mv_with_different_shapes():
    """Test FSP GGN with different input/output shapes."""
    key = jax.random.key(42)

    # Model with vector parameters
    def model_fn(input, params):
        return jnp.dot(params["W"], input) + params["b"]

    params = {
        "W": jax.random.normal(key, (3, 5)),
        "b": jax.random.normal(key, (3,)),
    }

    M = jax.random.normal(key, (10, 7))

    fsp_ggn_mv_fn = create_fsp_ggn_mv(model_fn, params, M, has_batch=False)

    data = {
        "input": jax.random.normal(key, (5,)),
        "target": jax.random.normal(key, (3,)),
    }

    result = fsp_ggn_mv_fn(data)

    assert isinstance(result, jax.Array)
    assert result.shape[0] == result.shape[1]  # Should be square


def test_fsp_ggn_mv_with_custom_loss_hessian():
    """Test FSP GGN with custom loss hessian."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    def custom_hess_mv(jv, pred=None, target=None):
        return 2.0 * jv

    params = {"w": jnp.array(1.0)}
    M = jax.random.normal(key, (5, 3))

    fsp_ggn_mv_fn = create_fsp_ggn_mv(
        model_fn, params, M, has_batch=False, loss_hessian_mv=custom_hess_mv
    )

    data = {"input": jnp.array(1.5), "target": jnp.array(2.0)}

    result = fsp_ggn_mv_fn(data)

    assert isinstance(result, jax.Array)


def test_fsp_ggn_mv_deterministic():
    """Test that FSP GGN is deterministic."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return jnp.dot(input, params["w"])

    params = {"w": jax.random.normal(key, (5,))}
    M = jax.random.normal(key, (10, 8))

    data = {"input": jax.random.normal(key, (5,)), "target": jax.random.normal(key, (1,))}

    fsp_ggn_mv_fn = create_fsp_ggn_mv(model_fn, params, M, has_batch=False)

    result1 = fsp_ggn_mv_fn(data)
    result2 = fsp_ggn_mv_fn(data)

    assert jnp.allclose(result1, result2)


def test_fsp_ggn_mv_positive_semidefinite():
    """Test that FSP GGN matrix is positive semidefinite."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}
    M = jax.random.normal(key, (5, 3))

    fsp_ggn_mv_fn = create_fsp_ggn_mv(model_fn, params, M, has_batch=False)

    data = {"input": jnp.array(2.0), "target": jnp.array(1.0)}

    result = fsp_ggn_mv_fn(data)

    # Check that eigenvalues are non-negative
    eigvals = jnp.linalg.eigvalsh(result)
    assert jnp.all(eigvals >= -1e-6)  # Allow small numerical errors
