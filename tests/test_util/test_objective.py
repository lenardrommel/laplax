"""Tests for FSP objective builders."""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flax import linen as nn

from laplax.util.objective import n_gaussian_log_posterior_objective

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
