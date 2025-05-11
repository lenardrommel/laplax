import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from matplotlib import pyplot as plt
from plotting import plot_sinusoid_task, plot_gp_prediction

from laplax.curv.cov import Posterior
from laplax.curv.fsp import create_fsp_objective
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from jax.flatten_util import ravel_pytree

from laplax.util.flatten import create_partial_pytree_flattener
import laplax
from laplax.curv import estimate_curvature
from laplax.curv.fsp import (
    compute_matrix_jacobian_product,
    create_fsp_objective,
    lanczos_jacobian_initialization,
)
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from helper import (
    DataLoader,
    get_sinusoid_example,
    gp_regression,
    RBFKernel,
    build_covariance_matrix,
)

from laplax.enums import LossFn
from laplax.types import Callable, Data, Float, ModelFn, Params, PredArray

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


from laplax.curv.fsp import create_fsp_objective


# Create and train MAP model
class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = self.linear2(nnx.tanh(self.linear1(x)))
        return x


model = Model(in_channels=1, hidden_channels=64, out_channels=1, rngs=nnx.Rngs(0))
graph_def, params = nnx.split(model)


def model_fn(input, params):
    return nnx.call((graph_def, params))(input)[0]


def mse_loss(x, y):
    return 0.5 * jnp.sum((x - y) ** 2)


kernel = RBFKernel(lengthscale=2.6)
kernel_fn = lambda x, y, sigma=1e-4: build_covariance_matrix(
    kernel, x, y
) + sigma * jnp.eye(x.shape[0])

fsp_loss_fn = create_fsp_objective(
    model_fn,
    loss_fn=mse_loss,
    prior_mean=jnp.zeros((150)),
    prior_cov_kernel=kernel_fn,
)


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(m):
        data = {"inputs": x, "targets": y}
        return fsp_loss_fn(
            data, data["inputs"], params
        )  # mse_loss(m, x, y) + reg_loss(m, kernel_fn, x)

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
data = {"inputs": X_train, "targets": y_train}


def create_fsp_ggn_mv(
    model_fn,
    params,
    M,
    has_batch,
    loss_hessian_mv=None,
):
    flat_params, unravel_fn = ravel_pytree(params)

    _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
    tol = jnp.finfo(M.dtype).eps ** 2
    s = _s[_s > tol]  # (80,)
    u = _u[:, : s.size]

    def jac_mv(model_fn, params, x):
        return jax.vmap(
            lambda u_flat: jax.jvp(
                lambda p: model_fn(x, p), (params,), (unravel_fn(u_flat),)
            )[1],  # noqa: E501
            in_axes=1,
        )(u)

    if has_batch:
        msg = (
            "FSP GGN MV is not implemented for batched data. "
            "Please set has_batch=False."
        )
        raise NotImplementedError(msg)

    def ggn_mv(data):
        ju = jac_mv(
            model_fn,
            params,
            data["inputs"],
        )
        ju = jnp.transpose(ju, (1, 0, 2)).squeeze(-1)
        return jnp.diag(s**2) + jnp.einsum("ji,jk->ik", ju, ju)

    return ggn_mv


def fsp_laplace(
    model_fn,
    params,
    data,
    prior_mean,
    prior_cov_kernel,
    context_points,
    **kwargs,
):
    # Initial vector
    v = lanczos_jacobian_initialization(model_fn, params, data, **kwargs)
    # Define cov operator
    op = prior_cov_kernel(context_points, context_points)
    op = op if isinstance(op, Callable) else lambda x: op @ x

    L = lanczos_isqrt(kernel_fn(X_train, X_train), v)

    M, unravel_fn = compute_matrix_jacobian_product(
        model_fn,
        params,
        data,
        L,
        has_batch_dim=False,
    )

    M = M.reshape((-1, L.shape[-1]))

    _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
    tol = jnp.finfo(M.dtype).eps ** 2
    s = _s[_s > tol]  # (80,)
    _u = _u[:, : s.size]

    ggn_matrix = create_fsp_ggn_mv(model_fn, params, M, False)(data)
    eigvals, eigvecs = jnp.linalg.eigh(ggn_matrix)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)


fsp_laplace(
    model_fn,
    params,
    data,
    prior_mean=jnp.zeros((150)),
    prior_cov_kernel=kernel_fn,
    context_points=X_train,
    has_batch_dim=False,
)
