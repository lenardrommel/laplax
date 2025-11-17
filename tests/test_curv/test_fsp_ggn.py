"""Tests for FSP-specific GGN functionality."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.curv.ggn import create_ggn_mv_without_data, create_jmp
from laplax.enums import LossFn


def create_simple_linear_model(output_dim: int):
    """Create a simple linear model for testing."""
    key = jax.random.PRNGKey(42)
    input_dim = 10

    params = {
        "w": jax.random.normal(key, (input_dim, output_dim)),
        "b": jnp.zeros(output_dim),
    }

    def model_fn(x, params):
        return x @ params["w"] + params["b"]

    return model_fn, params


def create_simple_mlp_model(hidden_dim: int = 20, output_dim: int = 5):
    """Create a simple MLP model for testing."""
    key = jax.random.PRNGKey(42)
    input_dim = 10

    key1, key2, key3, key4 = jax.random.split(key, 4)

    params = {
        "w1": jax.random.normal(key1, (input_dim, hidden_dim)) * 0.1,
        "b1": jax.random.normal(key2, (hidden_dim,)) * 0.1,
        "w2": jax.random.normal(key3, (hidden_dim, output_dim)) * 0.1,
        "b2": jax.random.normal(key4, (output_dim,)) * 0.1,
    }

    def model_fn(x, params):
        hidden = jax.nn.relu(x @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]

    return model_fn, params


def test_create_jmp_basic():
    """Test basic JMP (Jacobian-matrix product) functionality."""
    output_dim = 5
    model_fn, params = create_simple_linear_model(output_dim)

    # Create JMP function
    jmp = create_jmp(model_fn, vmap_over_data=True)

    # Test data
    batch_size = 4
    input_dim = 10
    rank = 3
    x_context = jax.random.normal(jax.random.PRNGKey(456), (batch_size, input_dim))

    # Create random tangent vectors (rank directions in parameter space)
    u = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(789), p.shape + (rank,)),
        params,
    )

    # Apply JMP
    result = jmp(params=params, x_context=x_context, u=u)

    # Check output shape: (batch_size, output_dim, rank)
    assert result.shape == (batch_size, output_dim, rank)


def test_create_jmp_output_consistency():
    """Test that JMP produces consistent results with manual JVP."""
    output_dim = 3
    model_fn, params = create_simple_linear_model(output_dim)

    jmp = create_jmp(model_fn, vmap_over_data=True)

    # Single input and single tangent vector
    x = jax.random.normal(jax.random.PRNGKey(1), (10,))
    tangent = jax.tree.map(lambda p: jnp.ones_like(p), params)
    u_single = jax.tree.map(lambda t: t[..., jnp.newaxis], tangent)

    # JMP result
    result = jmp(params=params, x_context=x[jnp.newaxis, :], u=u_single)

    # Manual JVP
    _, jvp_result = jax.jvp(lambda p: model_fn(x, p), (params,), (tangent,))

    # Compare
    np.testing.assert_allclose(result[0, :, 0], jvp_result, rtol=1e-5, atol=1e-6)


def test_create_jmp_multiple_ranks():
    """Test JMP with multiple rank directions."""
    output_dim = 4
    model_fn, params = create_simple_linear_model(output_dim)

    jmp = create_jmp(model_fn, vmap_over_data=True)

    batch_size = 6
    rank = 8
    x_context = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10))
    u = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(2), p.shape + (rank,)),
        params,
    )

    result = jmp(params=params, x_context=x_context, u=u)

    assert result.shape == (batch_size, output_dim, rank)
    assert jnp.all(jnp.isfinite(result))


def test_create_ggn_mv_fsp_mode_basic():
    """Test FSP mode of GGN MV."""
    output_dim = 5
    model_fn, params = create_simple_linear_model(output_dim)

    # Create FSP GGN MV
    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    # Test data with context points
    batch_size = 4
    rank = 3
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    # Create parameter tangent vectors with rank dimension
    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    # Apply FSP GGN MV
    result = ggn_fsp_mv(vec, test_data)

    # Check output shape: should be (rank, rank)
    assert result.shape == (rank, rank)

    # Result should be symmetric (approximately)
    np.testing.assert_allclose(result, result.T, rtol=1e-4, atol=1e-5)


def test_create_ggn_mv_fsp_mode_positive_semidefinite():
    """Test that FSP GGN matrix is positive semi-definite."""
    output_dim = 4
    model_fn, params = create_simple_linear_model(output_dim)

    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.CROSS_ENTROPY,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 8
    rank = 5
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.randint(
            jax.random.PRNGKey(2), (batch_size,), 0, output_dim
        ),
    }

    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    result = ggn_fsp_mv(vec, test_data)

    # Check eigenvalues are non-negative
    eigenvalues = jnp.linalg.eigvalsh(result)
    assert jnp.all(
        eigenvalues >= -1e-5
    ), f"Found negative eigenvalues: {eigenvalues[eigenvalues < -1e-5]}"


def test_create_ggn_mv_fsp_vs_standard_mode():
    """Test that FSP and standard modes produce different results (as expected)."""
    output_dim = 3
    model_fn, params = create_simple_linear_model(output_dim)

    # Standard GGN MV
    ggn_standard = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=False,
    )

    # FSP GGN MV
    ggn_fsp = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 4
    test_data_standard = {
        "input": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    test_data_fsp = {
        "context": test_data_standard["input"],
        "target": test_data_standard["target"],
    }

    vec = jax.tree.map(lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape), params)

    result_standard = ggn_standard(vec, test_data_standard)

    # For FSP mode, need to add rank dimension
    rank = 3
    vec_fsp = jax.tree.map(
        lambda p: jnp.broadcast_to(p[..., jnp.newaxis], p.shape + (rank,)),
        vec,
    )
    result_fsp = ggn_fsp(vec_fsp, test_data_fsp)

    # Results should have different types/shapes
    assert isinstance(result_standard, dict)  # PyTree result
    assert isinstance(result_fsp, jnp.ndarray)  # Gram matrix result
    assert result_fsp.shape == (rank, rank)


def test_create_ggn_mv_fsp_mode_with_classification():
    """Test FSP GGN MV with classification loss."""
    output_dim = 5
    model_fn, params = create_simple_linear_model(output_dim)

    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.CROSS_ENTROPY,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 10
    rank = 6
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.randint(
            jax.random.PRNGKey(2), (batch_size,), 0, output_dim
        ),
    }

    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    result = ggn_fsp_mv(vec, test_data)

    assert result.shape == (rank, rank)
    assert jnp.all(jnp.isfinite(result))

    # Should be symmetric
    np.testing.assert_allclose(result, result.T, rtol=1e-4, atol=1e-5)


def test_create_ggn_mv_fsp_mode_reproducibility():
    """Test that FSP GGN MV produces reproducible results."""
    output_dim = 4
    model_fn, params = create_simple_linear_model(output_dim)

    # Create two instances
    ggn_fsp_mv1 = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    ggn_fsp_mv2 = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 6
    rank = 4
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    result1 = ggn_fsp_mv1(vec, test_data)
    result2 = ggn_fsp_mv2(vec, test_data)

    np.testing.assert_allclose(result1, result2, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("output_dim", [2, 5, 10])
@pytest.mark.parametrize("rank", [2, 4, 8])
def test_create_ggn_mv_fsp_mode_various_dimensions(output_dim, rank):
    """Test FSP GGN MV with various dimension combinations."""
    model_fn, params = create_simple_linear_model(output_dim)

    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 8
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    result = ggn_fsp_mv(vec, test_data)

    assert result.shape == (rank, rank)
    assert jnp.all(jnp.isfinite(result))


def test_create_ggn_mv_fsp_mode_with_mlp():
    """Test FSP GGN MV with a multi-layer perceptron."""
    output_dim = 5
    hidden_dim = 15
    rank = 6

    model_fn, params = create_simple_mlp_model(hidden_dim, output_dim)

    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    batch_size = 8
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    vec = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
        params,
    )

    result = ggn_fsp_mv(vec, test_data)

    assert result.shape == (rank, rank)

    # Check symmetry
    np.testing.assert_allclose(result, result.T, rtol=1e-4, atol=1e-5)


def test_create_jmp_with_mlp():
    """Test JMP with MLP model."""
    output_dim = 5
    hidden_dim = 20
    model_fn, params = create_simple_mlp_model(hidden_dim, output_dim)

    jmp = create_jmp(model_fn, vmap_over_data=True)

    batch_size = 4
    rank = 3
    x_context = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10))
    u = jax.tree.map(
        lambda p: jax.random.normal(jax.random.PRNGKey(2), p.shape + (rank,)),
        params,
    )

    result = jmp(params=params, x_context=x_context, u=u)

    assert result.shape == (batch_size, output_dim, rank)
    assert jnp.all(jnp.isfinite(result))


def test_create_ggn_mv_fsp_mode_gradient_compatibility():
    """Test that FSP GGN MV is compatible with JAX gradient operations."""
    output_dim = 3
    model_fn, params = create_simple_linear_model(output_dim)

    batch_size = 6
    rank = 4
    test_data = {
        "context": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 10)),
        "target": jax.random.normal(jax.random.PRNGKey(2), (batch_size, output_dim)),
    }

    def loss_fn(params_inner):
        ggn_fsp_mv = create_ggn_mv_without_data(
            model_fn=model_fn,
            params=params_inner,
            loss_fn=LossFn.NONE,
            factor=1.0,
            vmap_over_data=True,
            fsp=True,
        )

        vec = jax.tree.map(
            lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape + (rank,)),
            params_inner,
        )

        result = ggn_fsp_mv(vec, test_data)
        return jnp.sum(result)

    # Should be able to compute gradient w.r.t. params
    grad = jax.grad(loss_fn)(params)

    # Check that gradients have the same structure as params
    assert jax.tree.structure(grad) == jax.tree.structure(params)

    # Check that gradients are finite
    grad_flat, _ = jax.tree.flatten(grad)
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_flat)
