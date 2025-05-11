from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from matplotlib import pyplot as plt
from plotting import (
    plot_sinusoid_task,
    plot_gp_prediction,
    plot_regression_with_uncertainty,
)
from laplax.curv.utils import LowRankTerms
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
from laplax.curv.cov import set_posterior_fn
from laplax.eval.pushforward import set_lin_pushforward
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from helper import (
    DataLoader,
    get_sinusoid_example,
    gp_regression,
    RBFKernel,
    build_covariance_matrix,
)
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_setup,
    set_lin_pushforward,
)
from laplax.enums import LossFn
from laplax.types import (
    Callable,
    Data,
    Float,
    ModelFn,
    Params,
    PredArray,
    PosteriorState,
)
from functools import partial

from plotting import plot_regression_with_uncertainty

from laplax.eval.pushforward import (
    nonlin_pred_mean,
    nonlin_pred_std,
    nonlin_pred_var,
    nonlin_setup,
    set_nonlin_pushforward,
)
from laplax.curv.cov import Posterior

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


def to_float64(model):
    graph_def, params = nnx.split(model)
    params = laplax.util.tree.to_dtype(params, jnp.float64)
    return nnx.merge(graph_def, params)


model = to_float64(
    Model(
        in_channels=1,
        hidden_channels=64,
        out_channels=1,
        rngs=nnx.Rngs(0),
    )
)

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


@nnx.jit
def _train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)  # Call methods directly
        return jnp.sum((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


def train_model(model, n_epochs, lr=1e-3):
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr))  # Reference sharing

    # Train epoch
    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            loss = _train_step(model, optimizer, x_tr, y_tr)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


model = train_model(model, n_epochs=1000)
data = {"inputs": X_train, "targets": y_train}
X_pred = jnp.linspace(0.0, 8.0, 200).reshape(200, 1)
y_pred = jax.vmap(model)(X_pred)

_ = plot_sinusoid_task(X_train, y_train, X_test, y_test, X_pred, y_pred)
plt.show()


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


def compute_curvature_fn(model_fn, params, ggn):
    eigvals, eigvecs = jnp.linalg.eigh(ggn)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    _, unravel_fn = ravel_pytree(params)

    def compute_cov_sqrt(_u):
        cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(eigvals[::-1]))

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

        return cov_sqrt[:, :truncation_idx]

    return compute_cov_sqrt


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

    # ==========
    params = laplax.util.tree.to_dtype(params, jnp.float64)
    cov_sqrt = compute_curvature_fn(model_fn, params, ggn_matrix)(_u)

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt}
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: state["scale_sqrt"] @ state["scale_sqrt"].T @ x,
        scale_mv=lambda state: lambda x: state["scale_sqrt"] @ x,
    )
    curv_estimate = LowRankTerms(cov_sqrt, s, 1)
    posterior_fn = set_posterior_fn("lanczos", curv_estimate, layout=params)

    set_nonlin_prob_predictive = partial(
        set_nonlin_pushforward,
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        pushforward_fns=[
            nonlin_setup,
            nonlin_pred_mean,
            nonlin_pred_var,
            nonlin_pred_std,
        ],
        key=jax.random.key(42),
        num_samples=10000,
    )
    prior_arguments = {"prior_prec": 1.0}  # Choose any prior precision.
    prob_predictive = set_nonlin_prob_predictive(
        prior_arguments=prior_arguments,
    )

    X_pred = jnp.linspace(0, 8, 200).reshape(200, 1)
    pred = jax.vmap(prob_predictive)(X_pred)

    _ = plot_regression_with_uncertainty(
        X_train=data["inputs"],
        y_train=data["targets"],
        X_pred=X_pred,
        y_pred=pred["pred_mean"][:, 0],
        y_std=jnp.sqrt(pred["pred_var"][:, 0]),
    )

    plt.show()


fsp_laplace(
    model_fn,
    params,
    data,
    prior_mean=jnp.zeros((150)),
    prior_cov_kernel=kernel_fn,
    context_points=X_train,
    has_batch_dim=False,
)
