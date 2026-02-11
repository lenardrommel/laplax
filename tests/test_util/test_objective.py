# /tests/test_util/test_objective.py

"""Tests for FSP objective builders."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn

from laplax.types import Array, Params
from laplax.util.objective import (
    add_ll_rho,
    compute_gaussian_log_likelihood,
    fsp_wrapper,
    n_gaussian_log_posterior_objective,
)

# ---------------------------------------------------------------------------
# Helpers for periodic objective test
# ---------------------------------------------------------------------------


def kernel_periodic_1d(x, y, variance=1.0, lengthscale=1.0, period=1.0):
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    d = jnp.abs(x[:, None, 0] - y[None, :, 0])
    s = jnp.sin(jnp.pi * d / period)
    return variance * jnp.exp(-(2.0 * s**2) / (lengthscale**2))


def prior_fn_periodic(x):
    # Mean zero, Periodic covariance
    mean = jnp.zeros((x.shape[0], 1))
    cov = kernel_periodic_1d(
        x, x, period=2.0
    )  # Period 2.0 matches [-1, 1] range length
    return mean, cov


class SmallMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(32)(x)
        x = jax.nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def test_fsp_objective_periodicity():
    """Test that training with FSP objective enforces periodicity on extrapolation."""
    key = jr.PRNGKey(42)
    key_init, key_data, key_train = jr.split(key, 3)

    # Data: sin(pi * x) on [-1, 1]. Period is 2.
    x_train = jr.uniform(key_data, (50, 1), minval=-1.0, maxval=1.0)
    y_train = jnp.sin(jnp.pi * x_train)

    model = SmallMLP()
    params = model.init(key_init, jnp.zeros((1, 1)))

    # Context points for regularization
    x_context = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p, opt_st):
        loss, _aux = n_gaussian_log_posterior_objective(
            params=p,
            model_fn=lambda input, params: model.apply(params, input),
            x_batch=x_train,
            y_batch=y_train,
            x_context=x_context,
            prior_fn=prior_fn_periodic,
            n_samples=50,
            ll_scale=0.1,
        )
        grads = jax.grad(
            lambda p_: n_gaussian_log_posterior_objective(
                params=p_,
                model_fn=lambda input, params: model.apply(params, input),
                x_batch=x_train,
                y_batch=y_train,
                x_context=x_context,
                prior_fn=prior_fn_periodic,
                n_samples=50,
                ll_scale=0.1,
            )[0]
        )(p)

        updates, opt_st = optimizer.update(grads, opt_st)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    # Train
    for _ in range(500):
        key_train, _k = jr.split(key_train)
        params, opt_state, _loss = step(params, opt_state)

    # Check periodicity
    # Check if f(1.5) approx f(-0.5) (since period is 2)
    x_test = jnp.array([[1.5]])
    x_target = jnp.array([[-0.5]])

    pred_test = model.apply(params, x_test)
    pred_target = model.apply(params, x_target)

    diff = jnp.abs(pred_test - pred_target)
    # print(f"Periodicity diff: {diff}")

    assert diff < 0.5, "FSP Regularizer failed to enforce approximate periodicity"


def linear_model_fn(*, input: Array, params: Params) -> Array:
    """Simple linear regression model: f(x) = x @ w + b."""
    w = params["w"]
    b = params["b"]
    return input @ w + b


def rbf_prior_fn(x_context: Array, *, jitter: float = 1e-4):
    """Zero-mean RBF GP prior for 1D inputs (sufficient for tests)."""
    x = x_context.reshape(-1, x_context.shape[-1])
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    d2 = x2 - 2.0 * (x @ x.T) + x2.T
    K = jnp.exp(-0.5 * d2 / (0.5**2))
    K = K + jitter * jnp.eye(K.shape[0])
    m = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
    return m, K


def test_add_ll_rho_inserts_scalar_leaf():
    """`add_ll_rho` should insert a scalar `ll_rho` into a
    dict-like params structure.
    """
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    assert "ll_rho" in params
    assert params["ll_rho"].shape == ()


def test_wrapper_forwards_model_fn_and_sigma_positive():
    """FSP wrapper should forward predictions and produce a positive sigma."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)

    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.array([[2.0], [3.0]])
    y_wrapped = fsp_model(input=x, params=params)
    y_base = linear_model_fn(input=x, params=base_params)

    assert jnp.allclose(y_wrapped, y_base)
    assert float(fsp_model.sigma(params)) > 0.0


def test_ll_rho_receives_gradient_and_updates():
    """`ll_rho` should receive gradients and update under an optimizer step."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = jnp.zeros_like(x)

    def loss_fn(p):
        f = fsp_model(input=x, params=p)
        sigma = fsp_model.sigma(p)
        # use your log-likelihood scaling (N/batch_size) via the provided helper
        return -compute_gaussian_log_likelihood(f, y, sigma, n_samples=x.shape[0])

    ll_before = params["ll_rho"]
    grads = jax.grad(loss_fn)(params)
    assert "ll_rho" in grads
    assert jnp.isfinite(grads["ll_rho"])

    opt = optax.adam(1e-2)
    opt_state = opt.init(params)
    updates, opt_state = opt.update(grads, opt_state)
    params2 = optax.apply_updates(params, updates)

    assert not jnp.allclose(ll_before, params2["ll_rho"])


def test_objective_runs_with_learned_sigma():
    """The full negative log-posterior objective should run using `sigma(params)`."""
    base_params = {"w": jnp.ones((1, 1)), "b": jnp.zeros((1,))}
    params = add_ll_rho(base_params, init_ll_rho=0.0)
    fsp_model = fsp_wrapper(linear_model_fn)

    x = jnp.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    y = jnp.sin(2 * jnp.pi * x)
    xc = jnp.linspace(-2.0, 2.0, 20).reshape(-1, 1)

    loss, metrics = n_gaussian_log_posterior_objective(
        params=params,
        model_fn=fsp_model,
        x_batch=x,
        y_batch=y,
        x_context=xc,
        prior_fn=lambda z: rbf_prior_fn(z, jitter=1e-4),
        n_samples=x.shape[0],
        ll_scale=fsp_model.sigma(params),
    )

    assert jnp.isfinite(loss)
    assert "log_likelihood" in metrics
    assert "sq_rkhs_norm" in metrics
    assert jnp.isfinite(metrics["sq_rkhs_norm"])
