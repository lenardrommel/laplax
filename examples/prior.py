import jax
from jax import numpy as jnp


def periodic_kernel(x, y, lengthscale, period, variance=1.0):
    """x, y: arrays of shape [..., D]
    returns: variance * exp[-2 * sum(sin^2(pi*(x-y)/period)) / lengthscale^2].
    """  # noqa: D205
    # compute pairwise sin^2 term
    arg = jnp.pi * (x - y) / period
    sin2 = jnp.sin(arg) ** 2  # [..., D]
    return variance * jnp.exp(-2.0 * jnp.sum(sin2, axis=-1) / lengthscale**2)


def matern52_kernel(x, y, lengthscale, variance=1.0):
    """k(r) = variance * (1 + sqrt(5)r/ℓ + 5r^2/(3ℓ^2)) * exp(-sqrt(5)r/ℓ)
    where r = ||x-y||_2.
    """  # noqa: D205
    r = jnp.linalg.norm(x - y, axis=-1)
    sqrt5 = jnp.sqrt(5.0)
    sr = sqrt5 * r / lengthscale
    return variance * (1.0 + sr + sr**2 / 3.0) * jnp.exp(-sr)


def matern12_kernel(x, y, lengthscale, variance=1.0):
    """Exponential kernel: k(r) = variance * exp(-r/ℓ)."""
    r = jnp.linalg.norm(x - y, axis=-1)
    return variance * jnp.exp(-r / lengthscale)


def composite_kernel(x, y, params):
    """params: {
      "per_ls": float,
      "per_p":   float,
      "per_var": float,
      "m52_ls":  float,
      "m52_var": float,
      "m12_ls":  float,
      "m12_var": float,
    }
    x: [N, D], y: [M, D] → returns [N, M].
    """  # noqa: D205
    # broadcast to [N, M, D]
    X = x[:, None, :]
    Y = y[None, :, :]

    k_per = periodic_kernel(
        X, Y, params["per_ls"], params["per_p"], params.get("per_var", 1.0)
    )
    k_m52 = matern52_kernel(X, Y, params["m52_ls"], params.get("m52_var", 1.0))
    k_m12 = matern12_kernel(X, Y, params["m12_ls"], params.get("m12_var", 1.0))

    return k_per  # + k_m52 + k_m12


def gram(x: jnp.ndarray, params: dict, kernel_fn, jitter: float = 1e-6) -> jnp.ndarray:
    K = kernel_fn(x, x, params)  # [N, N]
    return K + jitter * jnp.eye(x.shape[0])
