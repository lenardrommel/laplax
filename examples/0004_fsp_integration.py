from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import (
    DataLoader,
    L2InnerProductKernel,
    RBFKernel,
    _create_kernel_fn,
    get_sinusoid_example,
    gp_regression,
    to_float64,
)
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from plotting import (
    plot_regression_with_uncertainty,
    plot_sinusoid_task,
)
from pathlib import Path
import pickle

# from regression import *
import gpjax as gpx
import laplax
from laplax.curv import estimate_curvature
from laplax.curv.cov import Posterior, set_posterior_fn
from laplax.curv.fsp import (
    compute_curvature_fn,
    compute_matrix_jacobian_product,
    create_fsp_objective,
    create_loss_mse,
    create_loss_nll,
    create_loss_reg,
    lanczos_jacobian_initialization,
)
from laplax.curv.ggn import create_fsp_ggn_mv, create_ggn_mv_without_data
from laplax.extra.fsp.lanczos_isqrt import lanczos_isqrt
from laplax.enums import LossFn
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_samples,
    lin_setup,
    set_lin_pushforward,
)
from laplax.types import (
    Callable,
    Data,
    Float,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
)
from laplax.util.flatten import create_partial_pytree_flattener

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = self.linear2(nnx.tanh(self.linear1(x)))
        return x


class ModelWrapper(nnx.Module):
    class ModelWrapper:
        """A wrapper class for a neural network model that integrates parameters and data size.

        Attributes:
            model (Callable): The neural network model to be wrapped.
            param (nnx.Param): A parameter object initialized with the given parameter values.
            N (int): The size of the dataset.

        Methods:
            __call__(x):
                Invokes the wrapped model with the input `x`.

            to_float64():
                Converts the model parameters to float64 precision.

        Args:
            model (Callable): The neural network model to be wrapped.
            param (array-like): The initial parameter values for the model.
            data_size (int): The size of the dataset.
        """

    def __init__(self, model, param, data_size, dtype):
        self.model = model
        self.param = nnx.Param(jnp.asarray(param))
        self.N = data_size
        if dtype is jnp.float64:
            self.model = self.to_float64()

    def __call__(self, x):
        return self.model(x)

    def to_float64(self):  # noqa: F811
        graph_def, params = nnx.split(self.model)
        params = laplax.util.tree.to_dtype(params, jnp.float64)
        return nnx.merge(graph_def, params)


def mse_loss(x, y):
    return 0.5 * jnp.sum((x - y) ** 2)


def nll_loss(y_hat, y, sigma, N=150):
    sigma = sigma.value
    sq_diff = jnp.square(y_hat - y)
    log_term = 0.5 * jnp.log(2 * jnp.pi * sigma**2)
    precision_term = 0.5 * sq_diff / (sigma**2)
    nll = log_term + precision_term
    return jnp.mean(nll) * N


def create_train_step(loss_fn):
    @nnx.jit
    def train_step(model, optimizer, data):
        _, params = nnx.split(model)

        def _loss_fn(p):
            return loss_fn(data, p)

        loss, grads = nnx.value_and_grad(_loss_fn)(params)
        optimizer.update(grads)  # Only update model part
        return loss

    return train_step


def train_model(model, n_epochs, train_loader, loss_fn, lr=1e-3):
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    train_step_fn = create_train_step(loss_fn)

    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            data = {"input": x_tr, "target": y_tr}
            loss = train_step_fn(model, optimizer, data)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


def create_model(config, loss_type):
    in_channels = config.get("in_channels", 1)
    hidden_channels = config.get("hidden_channels", 64)
    out_channels = config.get("out_channels", 1)
    rngs = config.get("rngs", nnx.Rngs(0))
    param = config.get("param", None)
    data_size = config.get("data_size", None)
    dtype = config.get("dtype", jnp.float64)

    model = Model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        rngs=rngs,
    )

    graph_def, _ = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    if loss_type.lower() != "fsp":
        return model, model_fn, graph_def

    if param is not None and data_size is not None:
        model = ModelWrapper(model, param, data_size, dtype)

    return model, model_fn, graph_def


def create_loss_fn(
    loss_type,
    model_fn,
    prior_mean,
    prior_cov_kernel,
    num_training_samples,
    batch_size,
    context_points,
):
    if loss_type.lower() == "mse":
        return create_loss_mse(model_fn)
    elif loss_type.lower() == "nll":
        return create_loss_nll(model_fn, mse_loss)
    elif loss_type.lower() == "fsp":
        if prior_mean is None or prior_cov_kernel is None or context_points is None:
            msg = "prior_mean, prior_cov_kernel, and context_points must be provided for FSP loss"
            raise ValueError(msg)

        fsp_obj = create_fsp_objective(
            model_fn,
            nll_loss,
            prior_mean,
            prior_cov_kernel,
        )

        # Return a function that includes the context_points
        context_points = jnp.linspace(-10, 10, 150).reshape(150, 1)
        random_vector = jax.random.normal(jax.random.key(42), context_points.shape)
        context_points += random_vector

        return lambda data, params: fsp_obj(
            data,
            context_points,
            params["model"],
            other_params=params["param"],
        )
    else:
        msg = f"Unknown loss type: {loss_type}. Supported types are: mse, nll, fsp"
        raise ValueError(msg)


def fsp_laplace(
    model_fn,
    params,
    data,
    prior_arguments,
    prior_mean,
    prior_cov_kernel,
    context_points,
    **kwargs,
):
    params_correct = deepcopy(params)

    # Initial vector
    v = lanczos_jacobian_initialization(model_fn, params, data, **kwargs)
    # Define cov operator
    op = prior_cov_kernel(context_points, context_points)
    # op = op if isinstance(op, Callable) else lambda x: op @ x
    cov_matrix = prior_cov_kernel(data["input"], data["input"])

    L = lanczos_isqrt(cov_matrix, v)
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

    ggn_matrix = create_fsp_ggn_mv(model_fn, params, M)(data)

    prior_var = jnp.diag(prior_cov_kernel(data["test_input"], data["test_input"]))

    S = compute_curvature_fn(model_fn, params, data, ggn_matrix, prior_var, _u)

    posterior_state: PosteriorState = {"scale_sqrt": S}
    flatten, unflatten = laplax.util.flatten.create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten(
            state["scale_sqrt"] @ state["scale_sqrt"].T @ flatten(x)
        ),
        scale_mv=lambda state: lambda x: unflatten(state["scale_sqrt"] @ x),
        rank=s.shape[0],
    )

    posterior_fn = lambda *args, **kwargs: posterior  # noqa: E731

    set_prob_predictive = partial(
        set_lin_pushforward,
        model_fn=model_fn,
        mean_params=params,
        posterior_fn=posterior_fn,
        pushforward_fns=[
            lin_setup,
            lin_pred_mean,
            lin_pred_var,
            lin_pred_std,
            lin_samples,
        ],
        key=jax.random.key(35892),
        num_samples=100,
    )

    prob_predictive = set_prob_predictive(
        prior_arguments=prior_arguments,
    )

    return prob_predictive, posterior

    # X_pred = jnp.linspace(-4, 12, 200).reshape(200, 1)
    # pred = jax.vmap(prob_predictive)(X_pred)
    # _ = plot_regression_with_uncertainty(
    #     X_train=data["input"],
    #     y_train=data["target"],
    #     X_pred=X_pred,
    #     y_pred=pred["pred_mean"][:, 0],
    #     y_std=jnp.sqrt(pred["pred_var"][:, 0]),
    #     y_samples=pred["samples"],
    # )


def main():
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
        sigma_noise=0.0,
        intervals=[(0, 2), (4, 5), (6, 8)],
        rng_key=jax.random.key(0),
    )
    train_loader = DataLoader(X_train, y_train, batch_size)
    data = {"input": X_train, "target": y_train, "test_input": X_test}

    # final_params = optimize(data)
    # print("Tuned params:", {k: float(v) for k, v in final_params.items()})
    # make kernel_fn and prior_args
    # kernel_fn, prior_args = make_tuned_kernel_fn(final_params)
    # kernel_fn, prior_args = _create_kernel_fn(lengthscale=2.0)
    from gpjax.kernels.computations import DenseKernelComputation
    from gpjax.kernels import Periodic, Matern12

    kernel = Periodic(
        lengthscale=2.5,
        period=6.2,
        variance=10.0,
    )
    # kernel = Matern12(lengthscale=2.5)

    def kernel_fn(x, y):
        if y is None:
            y = x
        K = DenseKernelComputation().cross_covariance(kernel, x, y)
        return (K + K.T) / 2.0 + 1e-4 * jnp.eye(x.shape[0])

    prior_args = {"prior_prec": 1, "noise_variance": 1}

    initial_log_scale = jnp.log(0.01)
    config = {
        "in_channels": 1,
        "hidden_channels": 64,
        "out_channels": 1,
        "rngs": nnx.Rngs(key),
        "param": initial_log_scale,
        "data_size": num_training_samples,
        "dtype": jnp.float64,
    }

    loss_type = "fsp"
    model, model_fn, _ = create_model(config, loss_type)
    _, params = nnx.split(model)

    # Choose loss function type: 'mse', 'nll', or 'fsp'

    loss_fn = create_loss_fn(
        loss_type,
        model_fn,
        prior_mean=jnp.zeros((num_training_samples)),
        prior_cov_kernel=kernel_fn,
        num_training_samples=num_training_samples,
        batch_size=batch_size,
        context_points=X_train,
    )

    model = train_model(
        model, n_epochs=n_epochs, train_loader=train_loader, loss_fn=loss_fn
    )

    X_pred = jnp.linspace(0.0, 8.0, 200).reshape(200, 1)
    y_pred = jax.vmap(model)(X_pred)

    graph_def, params = nnx.split(model)

    # _ = plot_sinusoid_task(X_train, y_train, X_test, y_test, X_pred, y_pred)

    prob_predictive, posterior = fsp_laplace(
        model_fn,
        params["model"],
        data,
        prior_arguments=prior_args,
        prior_mean=jnp.zeros((150)),
        prior_cov_kernel=kernel_fn,
        context_points=X_train,
        has_batch_dim=False,
    )

    X_pred = jnp.linspace(-4, 12, 200).reshape(200, 1)
    pred = jax.vmap(prob_predictive)(X_pred)
    with Path("prob_predictive.pkl").open("wb") as f:
        pickle.dump({"X_pred": X_pred, "pred": pred}, f)

    _ = plot_regression_with_uncertainty(
        X_train=data["input"],
        y_train=data["target"],
        X_pred=X_pred,
        y_pred=pred["pred_mean"][:, 0],
        y_std=jnp.sqrt(pred["pred_var"][:, 0]),
        y_samples=pred["samples"],
    )
    plt.show()


if __name__ == "__main__":
    main()
