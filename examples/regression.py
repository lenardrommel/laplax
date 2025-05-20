#!/usr/bin/env python3
"""
Standalone script for Gaussian Process regression on noisy sinusoid data.
Optimizes marginal negative log-likelihood (MNLL) hyperparameters using Optax L-BFGS,
then plots prior and posterior predictive distributions relative to data.
Uses object-oriented kernels: RBF, Periodic, ScaleKernel, ProductKernel.
Supports multiplicative combination of RBF and scaled periodic kernels, with extended evaluation domain.
"""

import jax
import jax.numpy as jnp
import optax
from jax import random
from matplotlib import pyplot as plt

# Enable double precision
jax.config.update("jax_enable_x64", True)


# -------------------------------
# Data generation
# -------------------------------


# -------------------------------
# Kernel classes
# -------------------------------
class RBFKernel:
    def __init__(self, lengthscale: float):
        self.lengthscale = lengthscale

    def covariance(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sq = jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-0.5 * sq / (self.lengthscale**2))


class PeriodicKernel:
    def __init__(self, lengthscale: float, period: float):
        self.lengthscale = lengthscale
        self.period = period

    def covariance(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        sine = jnp.sin(jnp.pi * diff / self.period)
        sq = jnp.sum((sine / self.lengthscale) ** 2, axis=-1)
        return jnp.exp(-0.5 * sq)


class ScaleKernel:
    """Scale a base kernel by an amplitude factor."""

    def __init__(self, base_kernel, scale: float):
        self.base = base_kernel
        self.scale = scale

    def covariance(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return (self.scale**2) * self.base.covariance(X1, X2)


class ProductKernel:
    """Multiply two kernels elementwise."""

    def __init__(self, kernel1, kernel2):
        self.k1 = kernel1
        self.k2 = kernel2

    def covariance(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return self.k1.covariance(X1, X2) * self.k2.covariance(X1, X2)


class SumKernel:
    def __init__(self, kernel1, kernel2):
        self.k1 = kernel1
        self.k2 = kernel2

    def covariance(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        return self.k1.covariance(X1, X2) + self.k2.covariance(X1, X2)


# -------------------------------
# Utility to build a predictive kernel_fn from tuned hyperparameters
# -------------------------------
def make_tuned_kernel_fn(params):
    ls_rbf = float(jnp.exp(params["log_ls_rbf"]))
    amp_rbf = float(jnp.exp(params["log_amp_rbf"]))
    ls_per = float(jnp.exp(params["log_ls_per"]))
    amp_per = float(jnp.exp(params["log_amp_per"]))
    period = float(jnp.exp(params["log_period"]))
    noise = float(jnp.exp(params["log_noise"]))
    # instantiate kernels
    rbf = ScaleKernel(RBFKernel(ls_rbf), scale=amp_rbf)
    per = ScaleKernel(PeriodicKernel(ls_per, period), scale=amp_per)
    kernel = RBFKernel(ls_rbf)  # SumKernel(rbf, per)

    def kernel_fn(x, y=None):
        y = x if y is None else y
        K = kernel.covariance(x, y)
        if x is y:
            K = K + (noise**2 + 1e-6) * jnp.eye(K.shape[0])
        # K = (K + K.T) / 2.0
        return K

    prior_args = {"prior_prec": 1.0, "noise_variance": noise}
    return kernel_fn, prior_args


# -------------------------------
# MNLL loss
# -------------------------------
def loss_fn(params, X, y):
    # build kernels from params
    ls_rbf = jnp.exp(params["log_ls_rbf"])
    amp_rbf = jnp.exp(params["log_amp_rbf"])
    ls_per = jnp.exp(params["log_ls_per"])
    amp_per = jnp.exp(params["log_amp_per"])
    period = jnp.exp(params["log_period"])
    noise = jnp.exp(params["log_noise"])
    N = X.shape[0]

    # instantiate object-oriented kernels
    rbf_kernel = RBFKernel(lengthscale=ls_rbf)
    per_kernel = PeriodicKernel(lengthscale=ls_per, period=period)
    # scaled_per = ScaleKernel(per_kernel, scale=amp_per)
    prod_kernel = SumKernel(rbf_kernel, per_kernel)

    # covariance
    K = prod_kernel.covariance(X, X)
    K = amp_rbf**2 * K + (noise**2 + 1e-6) * jnp.eye(N)

    # MNLL
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y.reshape(-1))
    data_fit = 0.5 * (y.reshape(-1) @ alpha)
    logdet = jnp.sum(jnp.log(jnp.diag(L))) * 2.0
    const = 0.5 * N * jnp.log(2 * jnp.pi)
    return data_fit + 0.5 * logdet + const


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

# -------------------------------
# Optimization utility
# -------------------------------
from optax import tree_utils as otu


def run_optimization(init_params, loss_wrapper, opt, max_iter=1000, tol=1e-8):
    value_and_grad_fn = optax.value_and_grad_from_state(loss_wrapper)

    def step(carry):
        params, state = carry
        loss, grad = value_and_grad_fn(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=loss, grad=grad, value_fn=loss_wrapper
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def cond(carry):
        _, state = carry
        it = otu.tree_get(state, "count")
        gnorm = otu.tree_l2_norm(otu.tree_get(state, "grad"))
        return (it == 0) | ((it < max_iter) & (gnorm >= tol))

    init_state = opt.init(init_params)
    final_params, _ = jax.lax.while_loop(cond, step, (init_params, init_state))
    return final_params


def optimize(data):
    init_params = {
        "log_ls_rbf": jnp.log(2.0),
        "log_amp_rbf": jnp.log(2.0),
        "log_ls_per": jnp.log(2.0),
        "log_amp_per": jnp.log(0.3),
        "log_period": jnp.log(6.0),
        "log_noise": jnp.log(0.3),
    }
    X_train = data["input"]
    y_train = data["target"]
    final_params = run_optimization(
        init_params,
        loss_wrapper=lambda p: loss_fn(p, X_train, y_train),
        opt=optax.lbfgs(),
    )

    return final_params


# -------------------------------
# Main script
# -------------------------------
# def main():
#     # generate data
#     key = random.PRNGKey(0)
#     X_train, y_train, X_valid, y_valid, _, _ = get_sinusoid_example(
#         num_train_data=150,
#         num_valid_data=50,
#         num_test_data=150,
#         sigma_noise=0.0,
#         intervals=[(0, 8)],
#         rng_key=key,
#     )

#     # initial parameters
#     init_params = {
#         "log_ls_rbf": jnp.log(2.0),
#         "log_amp_rbf": jnp.log(2.0),
#         "log_ls_per": jnp.log(2.0),
#         "log_amp_per": jnp.log(0.3),
#         "log_period": jnp.log(6.0),
#         "log_noise": jnp.log(0.3),
#     }

#     def transform_params(params):
#         return {k: jnp.exp(v) for k, v in params.items()}

#     print("Initial params:", {k: float(v) for k, v in init_params.items()})
#     init_nll = loss_fn(init_params, X_train, y_train)
#     print(f"Initial NLL: {init_nll:.4f}")
#     print("Initial params (transformed):", transform_params(init_params))

#     # optimize
#     optimizer = optax.lbfgs()
#     final_params = run_optimization(
#         init_params, loss_wrapper=lambda p: loss_fn(p, X_train, y_train), opt=optimizer
#     )

#     final_nll = loss_fn(final_params, X_train, y_train)
#     print(f"Final NLL:   {final_nll:.4f}")
#     print("Optimized params:", {k: float(v) for k, v in final_params.items()})
#     print("Optimized params (transformed):", transform_params(final_params))
#     # extended prediction domain
#     x_min, x_max = float(jnp.min(X_train)), float(jnp.max(X_train))
#     X_ext = jnp.linspace(-x_max * 2, x_max * 3, 300, dtype=X_train.dtype).reshape(-1, 1)

#     # rebuild optimized kernels
#     rbf_kernel = RBFKernel(lengthscale=jnp.exp(final_params["log_ls_rbf"]))
#     per_kernel = PeriodicKernel(
#         lengthscale=jnp.exp(final_params["log_ls_per"]),
#         period=jnp.exp(final_params["log_period"]),
#     )
#     scaled_per = ScaleKernel(per_kernel, scale=jnp.exp(final_params["log_amp_per"]))
#     prod_kernel = ProductKernel(rbf_kernel, scaled_per)
#     noise = jnp.exp(final_params["log_noise"])

#     # covariance matrices
#     K_rr = prod_kernel.covariance(X_train, X_train)
#     K_rr = jnp.exp(final_params["log_amp_rbf"]) ** 2 * K_rr + (
#         noise**2 + 1e-6
#     ) * jnp.eye(X_train.shape[0])
#     K_tr = jnp.exp(final_params["log_amp_rbf"]) ** 2 * prod_kernel.covariance(
#         X_ext, X_train
#     )
#     K_tt = jnp.exp(final_params["log_amp_rbf"]) ** 2 * prod_kernel.covariance(
#         X_ext, X_ext
#     )

#     # prior
#     prior_mean = jnp.zeros(X_ext.shape[0])
#     prior_std = jnp.sqrt(jnp.clip(jnp.diag(K_tt), 0, None))

#     # posterior
#     L = jnp.linalg.cholesky(K_rr)
#     alpha = jax.scipy.linalg.cho_solve((L, True), y_train.reshape(-1))
#     post_mean = K_tr @ alpha
#     v = jax.scipy.linalg.solve_triangular(L, K_tr.T, lower=True)
#     post_cov = K_tt - v.T @ v
#     post_std = jnp.sqrt(jnp.clip(jnp.diag(post_cov), 0, None))

#     # plot
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.fill_between(
#         X_ext.flatten(),
#         prior_mean - 2 * prior_std,
#         prior_mean + 2 * prior_std,
#         alpha=0.3,
#         label="Prior ±2σ",
#     )
#     plt.scatter(X_train, y_train, c="k", s=8)
#     plt.title("Prior Predictive (Extended)")

#     plt.subplot(1, 2, 2)
#     plt.fill_between(
#         X_ext.flatten(),
#         post_mean - 2 * post_std,
#         post_mean + 2 * post_std,
#         alpha=0.3,
#         label="Posterior ±2σ",
#     )
#     plt.plot(X_ext, post_mean, label="Mean")
#     plt.scatter(X_train, y_train, c="k", s=8)
#     plt.title("Posterior Predictive (Extended)")

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()
"""
Optimized params (transformed): {'log_amp_per': Array(0.3, dtype=float64), 
'log_amp_rbf': Array(2.36416025, dtype=float64), 'log_ls_per': Array(165012.08363295, dtype=float64), 
'log_ls_rbf': Array(2.77848507, dtype=float64), 'log_noise': Array(7.48190397e-20, dtype=float64), 
'log_period': Array(24440.87371623, dtype=float64)}

"""
