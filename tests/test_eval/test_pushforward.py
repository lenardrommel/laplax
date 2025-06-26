"""Tests for `laplax.eval.pushforward`."""

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_fn
from laplax.curv.ggn import create_ggn_mv
from laplax.enums import CurvApprox
from laplax.eval.pushforward import set_lin_pushforward, set_nonlin_pushforward

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
