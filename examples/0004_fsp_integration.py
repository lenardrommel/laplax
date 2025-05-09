import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from matplotlib import pyplot as plt
from plotting import plot_sinusoid_task  # , plot_gp_prediction

from laplax.curv.cov import Posterior
from laplax.curv.fsp import create_fsp_objective
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from jax.flatten_util import ravel_pytree

from laplax.util.flatten import create_partial_pytree_flattener
import laplax
from laplax.curv import estimate_curvature
from laplax.curv.fsp import create_fsp_objective

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
kernel = RBFKernel(lengthscale=8 / jnp.pi)


def kernel_fn(x, y=None, noise_variance=noise_variance):
    if y is None:
        y = x
    K = build_covariance_matrix(kernel, x, y)
    return K + noise_variance * jnp.eye(K.shape[0])

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


def mse_loss(model_fn, data):
    N = data["inputs"].shape[0]
    y_pred = model_fn(data["inputs"])
    se = jnp.sum((y_pred - data['targets']) ** 2)

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


def fsp_loss(model_fn, prior_fn, data):
    """FSP loss function."""
    # Compute the NLL loss
    nll_loss = mse_loss(model_fn, data)

    # Compute the regularization loss
    reg_loss_value = reg_loss(model_fn, prior_fn, data["inputs"])

    return nll_loss + reg_loss_value


rfsp = create_fsp_objective(model_fn, mse_loss, None, kernel_fn)

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        data = {"inputs": x, "targets": y}
        return fsp_loss(m, kernel_fn, data)

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