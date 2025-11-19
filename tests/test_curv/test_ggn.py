from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest
import pytest_cases

from laplax.curv.ggn import (
    create_ggn_mv,
    create_ggn_mv_without_data,
    create_loss_hessian_mv,
)
from laplax.enums import LossFn

from .cases.rosenbrock import RosenbrockCase

# ---------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------


def test_binary_cross_entropy_loss_hessian():
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (1,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        optax.sigmoid_binary_cross_entropy,
    )(logits, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv("binary_cross_entropy")
    hess_laplax = jax.vmap(partial(hess_mv, pred=logits))(jnp.eye(1))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_cross_entropy_loss_hessian():
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        optax.softmax_cross_entropy_with_integer_labels,
    )(logits, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv("cross_entropy")
    hess_laplax = jax.vmap(partial(hess_mv, pred=logits))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_mse_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        lambda pred, target: jnp.sum((pred - target) ** 2),
    )(pred, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv(LossFn.MSE)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_callable_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set random loss function
    random_arr = jax.random.normal(keys[2], (10,))

    def loss_func(pred, target):
        return jnp.sum(random_arr @ (pred - target) ** 3)

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(loss_func)(pred, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv(loss_func)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred, target=target))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_none_loss_hessian():
    """Test that LossFn.NONE returns identity."""
    key = jax.random.key(0)
    pred = jax.random.normal(key, (10,))
    target = jnp.zeros(10)

    hess_mv = create_loss_hessian_mv(LossFn.NONE)
    vec = jax.random.normal(key, (10,))
    result = hess_mv(vec, pred=pred, target=target)

    assert jnp.allclose(result, vec)


def test_loss_hessian_none_raises():
    """Test that passing None as loss_fn raises ValueError."""
    with pytest.raises(ValueError, match="loss_fn cannot be None"):
        create_loss_hessian_mv(None)


# ---------------------------------------------------------------
# GGN - Rosenbrock
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def test_ggn_rosenbrock(rosenbrock):
    # Setup ggn_mv
    ggn_mv = create_ggn_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        num_curv_samples=1,
        num_total_samples=1,
    )

    # Compute the GGN
    ggn_calc = jax.lax.map(ggn_mv, jnp.eye(2))

    # Compare with the manual GGN
    ggn_manual = rosenbrock.ggn_manual
    assert jnp.allclose(ggn_calc, ggn_manual)


# ---------------------------------------------------------------
# GGN - Additional tests
# ---------------------------------------------------------------


def test_ggn_mv_without_data():
    """Test create_ggn_mv_without_data function."""
    key = jax.random.key(42)

    # Simple linear model
    def model_fn(input, params):
        return params["w"] * input + params["b"]

    params = {"w": jnp.array(2.0), "b": jnp.array(1.0)}
    data = {"input": jax.random.normal(key, (5,)), "target": jnp.zeros(5)}

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=True,
    )

    # Test with a vector
    vec = {"w": jnp.array(1.0), "b": jnp.array(1.0)}
    result = ggn_mv(vec, data)

    # Result should have same structure as params
    assert "w" in result
    assert "b" in result


def test_ggn_mv_with_batched_data():
    """Test GGN matrix-vector product with batched data."""
    key = jax.random.key(42)
    batch_size = 10
    input_dim = 5

    # Simple linear model
    def model_fn(input, params):
        return jnp.dot(input, params["w"])

    params = {"w": jax.random.normal(key, (input_dim,))}
    data = {
        "input": jax.random.normal(key, (batch_size, input_dim)),
        "target": jax.random.normal(key, (batch_size,)),
    }

    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
    )

    # Test with a vector
    vec = {"w": jax.random.normal(key, (input_dim,))}
    result = ggn_mv(vec)

    assert result["w"].shape == (input_dim,)


def test_ggn_mv_custom_num_samples():
    """Test GGN with custom num_curv_samples and num_total_samples."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}
    data = {"input": jax.random.normal(key, (100,)), "target": jnp.zeros(100)}

    # Use only subset of data for curvature
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=LossFn.MSE,
        num_curv_samples=50,
        num_total_samples=100,
    )

    vec = {"w": jnp.array(1.0)}
    result = ggn_mv(vec)

    # Should apply scaling factor of 100/50 = 2.0
    assert result["w"].shape == ()


def test_ggn_mv_error_no_loss():
    """Test that providing neither loss_fn nor loss_hessian_mv raises error."""
    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}
    data = {"input": jnp.array([1.0]), "target": jnp.array([1.0])}

    with pytest.raises(ValueError, match="Either loss_fn or loss_hessian_mv"):
        create_ggn_mv(
            model_fn=model_fn,
            params=params,
            data=data,
            loss_fn=None,
            loss_hessian_mv=None,
        )


def test_ggn_mv_error_both_loss():
    """Test that providing both loss_fn and loss_hessian_mv raises error."""
    def model_fn(input, params):
        return params["w"] * input

    def custom_hess_mv(jv, pred, **kwargs):
        return 2.0 * jv

    params = {"w": jnp.array(1.0)}
    data = {"input": jnp.array([1.0]), "target": jnp.array([1.0])}

    with pytest.raises(ValueError, match="Only one of loss_fn or loss_hessian_mv"):
        create_ggn_mv(
            model_fn=model_fn,
            params=params,
            data=data,
            loss_fn=LossFn.MSE,
            loss_hessian_mv=custom_hess_mv,
        )


def test_ggn_mv_with_custom_loss_hessian():
    """Test GGN with custom loss hessian mv."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    def custom_hess_mv(jv, pred, **kwargs):
        return 3.0 * jv

    params = {"w": jnp.array(1.0)}
    data = {"input": jax.random.normal(key, (5,)), "target": jnp.zeros(5)}

    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=None,
        loss_hessian_mv=custom_hess_mv,
    )

    vec = {"w": jnp.array(1.0)}
    result = ggn_mv(vec)

    assert result["w"].shape == ()


def test_ggn_mv_no_vmap():
    """Test GGN without vmapping over data."""
    key = jax.random.key(42)

    def model_fn(input, params):
        return params["w"] * input

    params = {"w": jnp.array(1.0)}
    # Single data point, no batch dimension
    data = {"input": jnp.array(2.0), "target": jnp.array(1.0)}

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.MSE,
        factor=1.0,
        vmap_over_data=False,
    )

    vec = {"w": jnp.array(1.0)}
    result = ggn_mv(vec, data)

    assert result["w"].shape == ()


def test_binary_cross_entropy_with_enum():
    """Test using LossFn.BINARY_CROSS_ENTROPY enum."""
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (1,))

    hess_mv = create_loss_hessian_mv(LossFn.BINARY_CROSS_ENTROPY)
    result = hess_mv(jnp.ones(1), pred=logits, target=target)

    assert result.shape == (1,)


def test_cross_entropy_with_enum():
    """Test using LossFn.CROSS_ENTROPY enum."""
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (10,))

    hess_mv = create_loss_hessian_mv(LossFn.CROSS_ENTROPY)
    result = hess_mv(jnp.ones(10), pred=logits, target=target)

    assert result.shape == (10,)
