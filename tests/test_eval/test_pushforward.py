"""Tests for `laplax.eval.pushforward`."""

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_fn
from laplax.curv.fsp import create_fsp_posterior
from laplax.curv.ggn import create_ggn_mv
from laplax.enums import CovarianceStructure, CurvApprox
from laplax.eval.pushforward import (
    fsp_samples,
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_setup,
    nonlin_pred_var,
    set_lin_pushforward,
    set_nonlin_pushforward,
)

from .cases.regression import case_regression

DEFAULT_CASE_LIST = [case_regression]
# DEFAULT_CASE_LIST = [case_regression, case_classification]
# # case_classifciation is slow.


@pytest_cases.parametrize(
    "curv_op",
    [CurvApprox.FULL, CurvApprox.DIAGONAL, CurvApprox.LANCZOS],
)
@pytest_cases.parametrize_with_cases("task", cases=DEFAULT_CASE_LIST)
def test_nonlin_pushforward(curv_op, task):
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    num_training_samples = 100
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # Set get posterior function
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=task.loss_fn_type,
        num_total_samples=num_training_samples,
    )
    posterior_fn = create_posterior_fn(
        curv_op,
        mv=ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_nonlin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        prior_arguments={"prior_prec": 99999999999.0},
        num_samples=100000,
    )

    # Compute pushforwards
    # pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # Check results
    pred = jax.vmap(lambda x: model_fn(input=x, params=params))(data["input"])
    assert (5, task.out_channels) == results["samples"].shape[1:]  # Check shape
    assert jnp.all(results["pred_std"] >= 0)
    assert jnp.allclose(pred, results["map"])
    assert (
        jnp.squeeze(results["pred_var"], axis=-1).shape
        == jnp.squeeze(results["pred_mean"], axis=-1).shape
    )

    if results["pred_cov"].ndim >= results["pred_mean"].ndim + 1:
        diag = jnp.diagonal(results["pred_cov"], axis1=-2, axis2=-1)
        expected_var = diag.reshape(jnp.squeeze(results["pred_mean"], axis=-1).shape)
        assert jnp.allclose(jnp.squeeze(results["pred_var"], axis=-1), expected_var)

    single_results = {"pred_mean": results["pred_mean"][0]}
    aux_no_cov = {"pred_ensemble": results["samples"][0]}
    single_results, _ = nonlin_pred_var(single_results, aux_no_cov)
    assert (
        jnp.squeeze(single_results["pred_var"], axis=-1).shape
        == jnp.squeeze(results["pred_mean"], axis=-1)[0].shape
    )
    assert jnp.allclose(
        single_results["pred_var"], jnp.var(results["samples"][0], axis=0)
    )

    # Test nonlin_pred_lsqrt_low_rank_cov
    assert "pred_cov_low_rank_terms" in results
    assert "low_rank_terms" in results
    lr_terms = results["pred_cov_low_rank_terms"]
    pred_mean_flat_size = jnp.prod(jnp.array(results["pred_mean"].shape[1:]))
    assert lr_terms.U.shape[0] == pred_mean_flat_size
    assert lr_terms.S.shape[0] == lr_terms.U.shape[1]


@pytest_cases.parametrize(
    "curv_op",
    [CurvApprox.FULL, CurvApprox.DIAGONAL, CurvApprox.LANCZOS],
)
@pytest_cases.parametrize_with_cases("task", cases=DEFAULT_CASE_LIST)
def test_lin_pushforward(curv_op, task):
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    num_training_samples = 100
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # Set get posterior function
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=task.loss_fn_type,
        num_total_samples=num_training_samples,
    )
    get_posterior = create_posterior_fn(
        curv_op,
        ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_lin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=get_posterior,
        prior_arguments={"prior_prec": 99999999999.0},
        num_samples=5,  # TODO(2bys): Find a better way of setting this.
    )

    # Compute pushforward
    pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # Check results
    pred = jax.vmap(lambda x: model_fn(input=x, params=params))(data["input"])
    assert (5, task.out_channels) == results["samples"].shape[
        1:
    ]  # (batch, samples, out)
    jnp.allclose(pred, results["map"])
    jnp.allclose(pred, results["pred_mean"], rtol=1e-2)
    assert results["pred_var"].shape == results["pred_mean"].shape

    if results["pred_cov"].ndim >= results["pred_mean"].ndim + 1:
        diag = jnp.diagonal(results["pred_cov"], axis1=-2, axis2=-1)
        expected_var = diag.reshape(results["pred_mean"].shape)
        assert jnp.allclose(results["pred_var"], expected_var)

    toy_map = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)
    toy_results, _ = lin_pred_var({"map": toy_map}, {"cov_mv": lambda vec: vec})
    assert toy_results["pred_var"].shape == toy_map.shape
    assert jnp.allclose(toy_results["pred_var"], jnp.ones_like(toy_map))

    # Test lin_pred_lsqrt_low_rank_cov
    if "low_rank_terms" in results and "observation_noise" in results:
        assert "pred_var" in results
        assert results["pred_var"].shape == results["pred_mean"].shape


@pytest_cases.parametrize_with_cases("task", cases=DEFAULT_CASE_LIST)
def test_lin_pushforward_fsp_samples(task):
    """Test linear pushforward with FSP samples."""
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # For FSP, we need context points (use data input as context)
    x_context = data["input"]

    # Create a simple kernel function (identity-like for testing)
    def kernel_fn(x1, x2=None):
        if x2 is None:
            x2 = x1
        # Simple RBF-like kernel for testing
        dist_sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-0.5 * dist_sq)

    # Create kernel matrix and prior variance
    kernel_matrix = kernel_fn(x_context)
    prior_variance = jnp.diag(kernel_matrix)

    # Create FSP posterior
    posterior = create_fsp_posterior(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        kernel_structure=CovarianceStructure.NONE,
        kernel=kernel_fn,
        prior_variance=prior_variance,
        n_chunks=2,
        max_iter=20,
        is_classification=(task.loss_fn_type == "cross_entropy"),
    )

    # Wrap posterior in a function matching the expected signature
    def get_posterior(prior_arguments, loss_scaling_factor):
        del prior_arguments, loss_scaling_factor  # FSP posterior is pre-computed
        return posterior

    # Set pushforward with fsp_samples instead of lin_samples
    pushforward_fns = [
        lin_setup,
        lin_pred_mean,
        lin_pred_std,
        fsp_samples,
    ]

    pushforward = set_lin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=get_posterior,
        prior_arguments={"prior_prec": 99999999999.0},
        pushforward_fns=pushforward_fns,
        num_samples=5,
    )

    # Compute pushforward
    pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # Check results
    pred = jax.vmap(lambda x: model_fn(input=x, params=params))(data["input"])
    assert (5, task.out_channels) == results["samples"].shape[
        1:
    ]  # (batch, samples, out)
    assert jnp.allclose(pred, results["map"])
    assert jnp.allclose(pred, results["pred_mean"], rtol=1e-2)
    assert results["samples"].shape[0] == batch_size
    assert results["samples"].shape[1] == 5  # num_samples
    assert results["samples"].shape[2:] == results["pred_mean"].shape[1:]


@pytest_cases.parametrize(
    "curv_op",
    [CurvApprox.FULL, CurvApprox.DIAGONAL, CurvApprox.LANCZOS],
)
@pytest_cases.parametrize_with_cases("task", cases=DEFAULT_CASE_LIST)
def test_nonlin_pushforward_lsqrt(curv_op, task):
    """Test nonlinear pushforward with pred_lsqrt_low_rank_cov."""
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    num_training_samples = 100
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # Set get posterior function
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=task.loss_fn_type,
        num_total_samples=num_training_samples,
    )
    posterior_fn = create_posterior_fn(
        curv_op,
        mv=ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_nonlin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        prior_arguments={"prior_prec": 99999999999.0},
        num_samples=1000,  # Reduced for faster test
    )

    # Compute pushforwards
    results = jax.vmap(pushforward)(data["input"])

    # Check lsqrt results
    assert "pred_cov_low_rank_terms" in results
    assert "low_rank_terms" in results
    lr_terms = results["pred_cov_low_rank_terms"]
    pred_mean_flat_size = jnp.prod(jnp.array(results["pred_mean"].shape[1:]))
    assert lr_terms.U.shape[0] == pred_mean_flat_size
    assert lr_terms.S.shape[0] == lr_terms.U.shape[1]
    assert lr_terms.scalar == 0.0

    # Check that pred_var was computed from lsqrt if not already present
    if "pred_var" in results:
        assert results["pred_var"].shape == results["pred_mean"].shape


@pytest_cases.parametrize(
    "curv_op",
    [CurvApprox.FULL, CurvApprox.DIAGONAL, CurvApprox.LANCZOS],
)
@pytest_cases.parametrize_with_cases("task", cases=DEFAULT_CASE_LIST)
def test_lin_pushforward_lsqrt(curv_op, task):
    """Test linear pushforward with pred_lsqrt_low_rank_cov."""
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    num_training_samples = 100
    batch_size = 20
    data = task.get_data_batch(batch_size=batch_size)

    # Set get posterior function
    ggn_mv = create_ggn_mv(
        model_fn=model_fn,
        params=params,
        data=data,
        loss_fn=task.loss_fn_type,
        num_total_samples=num_training_samples,
    )
    get_posterior = create_posterior_fn(
        curv_op,
        ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_lin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=get_posterior,
        prior_arguments={"prior_prec": 99999999999.0},
        num_samples=5,
    )

    # Compute pushforward
    pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # Check lsqrt results - lin_pred_lsqrt_low_rank_cov requires
    # low_rank_terms and observation_noise to compute pred_var
    if "low_rank_terms" in results and "observation_noise" in results:
        assert "pred_var" in results
        assert results["pred_var"].shape == results["pred_mean"].shape
        lr = results["low_rank_terms"]
        assert lr.U.shape[0] == jnp.prod(jnp.array(results["pred_mean"].shape[1:]))
        assert lr.S.shape[0] == lr.U.shape[1]
