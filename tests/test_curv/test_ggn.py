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
# GGN - FSP
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
@pytest.mark.parametrize("rank", [2, 3])
def case_rosenbrock_fsp(x, alpha, rank):
    return RosenbrockCase(x, alpha), rank


@pytest_cases.parametrize_with_cases("rosenbrock_fsp", cases=[case_rosenbrock_fsp])
def test_ggn_fsp_rosenbrock(rosenbrock_fsp):
    """Test FSP GGN computation with Rosenbrock function."""
    rosenbrock, rank = rosenbrock_fsp

    # Setup context points (single point for simplicity)
    n_context = 1
    x_context = jnp.zeros((n_context, 1))

    # Create U matrix: for Rosenbrock, params are 2D, so U should be (2, rank)
    key = jax.random.key(42)
    U_flat = jax.random.normal(key, (2, rank))
    U = U_flat  # For Rosenbrock, params are just a 1D array, so U is (2, rank)

    # Setup FSP GGN
    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        loss_fn=rosenbrock.loss_fn,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    # Compute FSP GGN Gram matrix: U^T G U
    # Note: In FSP mode, the function expects U as a Params tree with last dim
    # as rank. For Rosenbrock, params are just a 1D array, so we need to
    # structure U correctly. The FSP implementation expects U to have the same
    # structure as params but with an extra dimension at the end for the rank
    data = {"context": x_context, "target": jnp.zeros((n_context, 2))}
    gram_fsp = ggn_fsp_mv(U, data)

    # Manual computation: U^T G U where G is the GGN matrix
    ggn_manual = rosenbrock.ggn_manual
    gram_manual = U.T @ ggn_manual @ U

    # Compare results
    assert gram_fsp.shape == (rank, rank), (
        f"Expected shape ({rank}, {rank}), got {gram_fsp.shape}"
    )
    max_diff = jnp.max(jnp.abs(gram_fsp - gram_manual))
    assert jnp.allclose(gram_fsp, gram_manual, atol=1e-5), (
        f"FSP GGN Gram matrix doesn't match manual computation. Max diff: {max_diff}"
    )


@pytest.mark.parametrize("loss_fn", [LossFn.NONE, LossFn.MSE])
def test_ggn_fsp_loss_functions(loss_fn):
    """Test FSP GGN with different loss functions."""

    # Simple linear model for testing
    def model_fn(x, params):
        return params @ x

    # Setup
    key = jax.random.key(123)
    params = jax.random.normal(key, (3, 5))  # 3 outputs, 5 inputs
    n_context = 4
    rank = 3

    x_context = jax.random.normal(key, (n_context, 5))

    # Create U matrix: params are (3, 5), so U should be (3, 5, rank)
    U = jax.random.normal(key, (3, 5, rank))

    # Setup FSP GGN
    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    # Compute FSP GGN Gram matrix
    data = {"context": x_context, "target": jnp.zeros((n_context, 3))}
    gram_fsp = ggn_fsp_mv(U, data)

    # Verify output shape
    assert gram_fsp.shape == (rank, rank), (
        f"Expected shape ({rank}, {rank}), got {gram_fsp.shape}"
    )

    # Verify symmetry (Gram matrix should be symmetric)
    assert jnp.allclose(gram_fsp, gram_fsp.T, atol=1e-6), (
        "FSP GGN Gram matrix should be symmetric"
    )


def test_ggn_fsp_vs_standard_consistency():
    """Test that FSP GGN with identity U matches standard GGN for single vector."""
    rosenbrock = RosenbrockCase(jnp.array([1.0, 1.0]), alpha=1.0)

    # Setup context points
    n_context = 1
    x_context = jnp.zeros((n_context, 1))

    # For standard GGN, compute GGN matrix-vector product for a single vector
    ggn_mv = create_ggn_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
        num_curv_samples=1,
        num_total_samples=1,
    )

    # Test vector
    v = jnp.array([0.5, 0.3])
    ggn_v_standard = ggn_mv(v)

    # For FSP GGN, use U as a single column (rank=1)
    U = v[:, jnp.newaxis]  # Shape: (2, 1)

    ggn_fsp_mv = create_ggn_mv_without_data(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        loss_fn=rosenbrock.loss_fn,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
    )

    data = {"context": x_context, "target": jnp.zeros((n_context, 2))}
    gram_fsp = ggn_fsp_mv(U, data)  # Shape: (1, 1)

    # For rank=1, U^T G U should equal v^T G v
    # We can compute v^T G v using the standard GGN
    vT_G_v = v @ ggn_v_standard

    # Compare
    assert jnp.allclose(gram_fsp[0, 0], vT_G_v, atol=1e-5), (
        f"FSP GGN with rank=1 should match v^T G v. "
        f"Got {gram_fsp[0, 0]}, expected {vT_G_v}"
    )
