import pickle

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from matplotlib import pyplot as plt
from plotting import plot_sinusoid_task  # , plot_gp_prediction

from laplax.curv.cov import Posterior
from laplax.curv.fsp import create_fsp_objective
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from jax.flatten_util import ravel_pytree

from laplax.util.flatten import create_partial_pytree_flattener

jax.config.update("jax_enable_x64", True)

n_epochs = 1000
key = jax.random.key(0)

# Sample toy data example
num_training_samples = 150
num_calibration_samples = 50
num_test_samples = 150

batch_size = 20
X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
    num_train_data=num_training_samples,
    num_valid_data=num_calibration_samples,
    num_test_data=num_test_samples,
    sigma_noise=0.3,
    intervals=[(0, 8)],
    rng_key=jax.random.key(0),
    dtype=jnp.float64,
)
train_loader = DataLoader(X_train, y_train, batch_size)


class RBFKernel:
    def __init__(self, lengthscale=2.60):
        self.lengthscale = lengthscale

    def __call__(self, x, y: jax.Array | None = None) -> jax.Array:
        """Compute RBF kernel between individual points"""
        if y is None:
            y = x

        sq_dist = jnp.sum((x - y) ** 2)

        return jnp.exp(-0.5 * sq_dist / self.lengthscale**2)


class L2InnerProductKernel:
    def __init__(self, bias=1e-4):
        self.bias = bias

    def __call__(self, x1: jax.Array, x2: jax.Array | None = None) -> jax.Array:
        """Compute L² inner product kernel between x1 and x2."""
        if x2 is None:
            x2 = x1

        return jnp.sum(x1 * x2) + self.bias


def build_covariance_matrix(kernel, X1, X2):
    return jnp.array([[kernel(x1, x2) for x2 in X2] for x1 in X1])


def gp_regression(x_train, y_train, x_test, kernel, noise_variance=1e-2):
    K = build_covariance_matrix(kernel, x_train, x_train)

    K_noise = K + noise_variance * jnp.eye(K.shape[0])

    alpha = jnp.linalg.solve(K_noise, y_train)

    K_star = build_covariance_matrix(kernel, x_test, x_train)

    mu_star = K_star @ alpha

    K_ss = build_covariance_matrix(kernel, x_test, x_test)
    v = jnp.linalg.solve(K_noise, K_star.T)
    cov_star = K_ss - K_star @ v

    return jnp.array(mu_star), jnp.array(cov_star)


noise_std = 0.3
noise_variance = noise_std**2
lengthscale = 8 / jnp.pi

kernel = RBFKernel(lengthscale=lengthscale)
# kernel = L2InnerProductKernel(bias=1e-4)

# pred_mean, pred_cov = gp_regression(X_train, y_train, X_test, kernel, noise_variance)

# std_dev = jnp.sqrt(jnp.maximum(jnp.diag(pred_cov), 0))


# fig = plot_gp_prediction(
#     X_train, y_train, X_test, pred_mean, std_dev, noise_std=noise_std
# )

kernel = RBFKernel(lengthscale=8 / jnp.pi)


def kernel_fn(x, y=None, noise_variance=noise_variance):
    if y is None:
        y = x
    K = build_covariance_matrix(kernel, x, y)
    return K + noise_variance * jnp.eye(K.shape[0])


# data = {
#     "inputs": X_train,
#     "targets": y_train,
# }

# zero_mean = jnp.zeros((X_train.shape[0], 1))


# def mse_loss(y_pred, y):
#     return jnp.mean((y_pred - y) ** 2)


# def reg_loss(y_pred, prior):
#     return y_pred.T @ jnp.linalg.solve(prior, y_pred)


# def _loss_fn(y_pred, y):
#     return mse_loss(y_pred, y) + reg_loss(y_pred, kernel_fn)


# Create and train MAP model
class Model(nnx.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, rngs, dtype=jnp.float64
    ):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs, dtype=dtype)
        self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        x = self.linear2(nnx.tanh(self.linear1(x)))
        return x


model = Model(in_channels=1, hidden_channels=64, out_channels=1, rngs=nnx.Rngs(0))

graph_def, params = nnx.split(model)


def model_fn(input, params):
    return nnx.call((graph_def, params))(input)[0]


# r_fsp = create_fsp_objective(model_fn, mse_loss, zero_mean, kernel_fn)
# r_fsp(data, X_train, params)
zero_mean = jnp.zeros((X_train.shape[0], 1))


def mse_loss(model_fn, x, y):
    N = x.shape[0]
    y_pred = model_fn(x)
    se = jnp.sum((y_pred - y) ** 2)

    return (
        0.5
        * N
        / batch_size
        * (se / noise_variance + N * jnp.log(2 * jnp.pi * noise_variance))
    )


def reg_loss(model_fn, prior_fn, x):
    y_pred = model_fn(x)
    prior = prior_fn(x)
    left = jnp.linalg.solve(prior, y_pred)
    # return 0.5 * jax.numpy.einsum("ij,ij->", y_pred, left)
    return 0.0


# fsp_loss = create_fsp_objective(
#     model_fn,
#     loss_fn=mse_loss,
#     prior_mean=zero_mean,
#     prior_cov_kernel=kernel_fn,
# )


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        return mse_loss(m, x, y) + reg_loss(m, kernel_fn, x)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


def train_model(model, n_epochs, lr=1e-3):
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr))  # Reference sharing

    # Train epoch
    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            loss = train_step(model, optimizer, x_tr, y_tr)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


model = train_model(model, n_epochs=10)


X_pred = jnp.linspace(0.0, 8.0, 200).reshape(200, 1)
y_pred = jax.vmap(model)(X_pred)

_ = plot_sinusoid_task(X_train, y_train, X_test, y_test, X_pred, y_pred)
# plt.show()

v = jnp.ones_like(X_train.squeeze(-1))
L = lanczos_isqrt(kernel_fn(X_train, X_train), v)
# M = J(X_train, X_train)^T @ kernel_matrix
# compute vjp of model and L
graph, params = nnx.split(model)
print(L.shape)  # 150, rank
print(f"Number of params: {sum(p.size for p in jax.tree.leaves(params))}")  # 193


def f_params(params):
    y = model_fn(X_train, params)
    return jnp.reshape(y, (150,))


_, pullback = jax.vjp(f_params, params)
param_vjp_tree = jax.vmap(lambda seed: pullback(seed)[0], in_axes=1, out_axes=1)(L)
flat_M, unravel_fn = ravel_pytree(param_vjp_tree)
M = flat_M.reshape((-1, 7))
flatten, unflatten = create_partial_pytree_flattener(flat_M)
print(M.shape)
_u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
tol = jnp.finfo(M.dtype).eps ** 2
s = _s[_s > tol]  # (80,)
_u = _u[:, : s.size]  # (p, rank) (161, 80)  # shape: (P, C_2)
u = flatten(_u)

flat_params, unravel_params = ravel_pytree(params)
flat_params = flat_params.astype(jnp.float64)
f_params_flat = lambda p: model_fn(X_train, p)
ju = jnp.transpose(
    jax.vmap(
        lambda u_flat: jax.jvp(f_params_flat, (params,), (unravel_params(u_flat),))[1],
        in_axes=1,
    )(_u),
    (1, 0, 2),
)
ju = ju.squeeze(-1)
ggn_matrix = jnp.einsum("ji,jk->ik", ju, ju)
eigvals, eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + ggn_matrix)
eigvals = jnp.flip(eigvals, axis=0)
eigvecs = jnp.flip(eigvecs, axis=1)


cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(eigvals[::-1]))
_, unravel_fn = jax.flatten_util.ravel_pytree(params)

params = jax.tree.map(lambda x: x.astype(jnp.float64), params)


def jvp(x, v):
    return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]


def scan_fn(carry, i):
    running_sum, truncation_idx = carry
    lr_fac = unravel_fn(cov_sqrt[:, i])
    sqrt_jvp = jax.vmap(lambda xc: jvp(xc, lr_fac) ** 2)(X_train)
    pv = jnp.sum(sqrt_jvp)
    new_running_sum = running_sum + pv
    new_truncation_idx = jax.lax.cond(
        (new_running_sum >= 150) & (truncation_idx == -1),
        lambda _: i + 1,  # We found our index
        lambda _: truncation_idx,  # Keep current value
        operand=None,
    )

    return (new_running_sum, new_truncation_idx), sqrt_jvp


init_carry = (0.0, -1)
indices = jnp.arange(eigvals.shape[0])
(_, truncation_idx), post_var = jax.lax.scan(scan_fn, init_carry, indices)

truncation_idx = jax.lax.cond(
    truncation_idx == -1,
    lambda _: eigvals.shape[0],
    lambda _: truncation_idx,
    operand=None,
)
cov_sqrt = cov_sqrt[:, :truncation_idx]
