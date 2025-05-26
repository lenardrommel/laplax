import jax
from jax import numpy as jnp


def periodic_kernel(x, y, lengthscale, period, variance=1.0):
    arg = jnp.pi * jnp.abs(x - y) / period
    sin2 = jnp.sin(arg) ** 2
    return variance * jnp.exp(jnp.sum(sin2, axis=-1) / (-2 * lengthscale**2))


def matern12_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    return variance * jnp.exp(-r / lengthscale)


def matern52_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    sqrt5 = jnp.sqrt(5.0)
    sr = sqrt5 * r / lengthscale
    return variance * (1.0 + sr + sr**2 / 3.0) * jnp.exp(-sr)


def rbf_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    return variance * jnp.exp(-(r**2) / (2 * lengthscale**2))


def composite_kernel(x, y, params):
    X = x[:, None, :]
    Y = y[None, :, :]
    k_per = periodic_kernel(
        X, Y, params["per_ls"], params["per_p"], params.get("per_var", 1.0)
    )
    k_matern52 = matern52_kernel(X, Y, params.get("matern52_ls", 1.0), 1.0)
    k_matern12 = matern12_kernel(
        X, Y, params.get("matern12_ls", 1.0), params.get("matern12_var", 0.0)
    )
    return k_per * k_matern52 + k_matern12


def gram(x: jnp.ndarray, params: dict, kernel_fn, jitter: float = 1e-6) -> jnp.ndarray:
    K = kernel_fn(x, x, params)
    return K + jitter * jnp.eye(x.shape[0])
