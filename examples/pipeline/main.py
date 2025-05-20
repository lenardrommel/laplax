import pickle
from functools import partial
from pathlib import Path

import gpjax as gpx
import jax
import matplotlib.pyplot as plt
import optax
from flax import nnx
from helper import DataLoader, get_standard_data
from jax import numpy as jnp
import jax.scipy as jsp
from model import create_model
from plot import plot_regression_with_uncertainty
from prior import Prior

import laplax
from laplax.curv.cov import Posterior
from laplax.curv.fsp import (
    compute_curvature_fn,
    compute_matrix_jacobian_product,
    create_fsp_objective,
    lanczos_jacobian_initialization,
)
from laplax.curv.ggn import create_fsp_ggn_mv
from laplax.curv.lanczos_isqrt import lanczos_isqrt
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
from laplax.util.flatten import create_partial_pytree_flattener, create_pytree_flattener

jax.config.update("jax_enable_x64", True)


def select_context_points(
    n_context_points,
    context_selection,
    context_points_maxval,
    context_points_minval,
    datapoint_shape,
    key=jax.random.PRNGKey(0),  # noqa: B008
):
    if context_selection == "random":
        context_points = jax.random.uniform(
            key=key,
            shape=(n_context_points,) + datapoint_shape,
            minval=context_points_minval,
            maxval=context_points_maxval,
        )
    elif context_selection == "grid":
        feature_dim = datapoint_shape[-1]
        if feature_dim == 1:
            context_points = jnp.linspace(
                context_points_minval[0], context_points_maxval[0], n_context_points
            ).reshape(-1, 1)

    return context_points


def nll_loss(y_hat, y, sigma, N=150):
    sigma = jax.nn.softplus(sigma.value)
    sigma = jnp.clip(sigma, a_min=jnp.log(jnp.exp(1e-2) - 1))
    # sq_diff = jnp.square(y_hat - y)
    # log_term = 0.5 * jnp.log(2 * jnp.pi * sigma**2)
    # precision_term = 0.5 * sq_diff / (sigma**2)
    # nll = log_term + precision_term
    return -jsp.stats.norm.logpdf(y, loc=y_hat, scale=sigma).mean() * 150
    # return jnp.mean(nll) * N


def compute_gaussian_log_likelihood(f_hat, y, ll_scale, n_samples):
    return (
        100 * jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).mean()  # n_samples
    )  # * n_samples


def compute_rkhs_norm(f_hat, prior_cov):
    return ((f_hat).T @ jnp.linalg.solve(prior_cov, f_hat)).reshape(-1).squeeze(-1)


def create_train_step(
    model_fn,
    prior_cov_kernel,
    ll_scale,
    n_context_points,
    context_selection,
    context_points_maxval,
    context_points_minval,
):
    @nnx.jit
    def train_step(model, optimizer, data, key):
        _, params = nnx.split(model)
        x_tr, y_tr = data["input"], data["target"]

        key, key1 = jax.random.split(key)

        x_context = select_context_points(
            n_context_points,
            context_selection,
            context_points_maxval,
            context_points_minval,
            x_tr.shape[1:],
        )

        def _loss_fn(p):
            f_hat = jax.vmap(lambda x: model_fn(x, p))(x_tr)

            prior_cov = prior_cov_kernel(x_context, x_context, 0)

            f_hat_context = jax.vmap(lambda x: model_fn(x, p))(x_context)

            ll = compute_gaussian_log_likelihood(
                f_hat, y_tr, ll_scale, n_samples=x_tr.shape[0]
            )

            rkhs_norm = compute_rkhs_norm(f_hat_context, prior_cov)

            log_posterior = ll - 0.5 * rkhs_norm
            return -log_posterior, (ll, rkhs_norm, log_posterior)

        (loss, (ll, rkhs_norm, log_posterior)), grads = nnx.value_and_grad(
            _loss_fn, has_aux=True
        )(params)
        optimizer.update(grads)
        return loss, key, ll, rkhs_norm, log_posterior

    return train_step


def train_model(
    model,
    n_epochs,
    train_loader,
    model_fn,
    prior_cov_kernel,
    ll_scale,
    n_context_points,
    context_selection,
    context_points_maxval,
    context_points_minval,
    lr=1e-3,
    key=jax.random.PRNGKey(0),
):
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    train_step_fn = create_train_step(
        model_fn,
        prior_cov_kernel,
        ll_scale,
        n_context_points,
        context_selection,
        context_points_maxval,
        context_points_minval,
    )

    for epoch in range(n_epochs):
        train_ll, train_rkhs, train_lpost = 0.0, 0.0, 0.0
        batch_count = 0

        for x_tr, y_tr in train_loader:
            data = {"input": x_tr, "target": y_tr}
            loss, key, ll, rkhs_norm, log_posterior = train_step_fn(
                model, optimizer, data, key
            )

            # Accumulate metrics
            train_ll += ll
            train_rkhs += rkhs_norm
            train_lpost += log_posterior
            batch_count += 1

        # Average metrics over batches
        train_ll /= batch_count
        train_rkhs /= batch_count
        train_lpost /= batch_count

        if epoch % 100 == 0:
            print(
                f"[epoch {epoch}]: loss: {loss:.4f}, log_likelihood: {train_ll:.4f}, "
                f"rkhs_norm: {train_rkhs:.4f}, log_posterior: {train_lpost:.4f}"
            )

    print(f"Final loss: {loss:.4f}")
    return model


def create_loss_fn(
    model_fn,
    prior_mean,
    prior_cov_kernel,
    nll_loss,
    context_points,
    num_training_samples,
    batch_size=20,
):
    fsp_obj = create_fsp_objective(
        model_fn,
        nll_loss,
        prior_mean,
        prior_cov_kernel,
    )

    # Return a function that includes the context_points
    #

    return lambda data, params: fsp_obj(
        data,
        (context_points),
        params["model"],
        other_params=params["param"],
    )


def _compute_curvature_fn(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    ggn: PredArray,
    prior_var: PredArray,
    u: PredArray,
):
    """Original function to compute the curvature function of FSP Laplace.
    Can be used to test the compute_curvature_fn in fsp.
    """  # noqa: D205
    _eigvals, _eigvecs = jnp.linalg.eigh(ggn)
    eps = jnp.finfo(ggn.dtype).eps
    tol = eps * (_eigvals.max() ** 0.5) * _eigvals.shape[0]
    eigvals = _eigvals[_eigvals > tol]
    eigvecs = _eigvecs[:, _eigvals > tol]
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    def normalize_eigvecs(evals, u, evecs):
        return u @ (evecs * (1 / jnp.sqrt(evals)))

    def jvp(x, v):
        return jax.jvp(lambda p: model_fn(x, params=p), (params,), (v,))[1]

    i = 0
    post_var = jnp.zeros((prior_var.shape[0],))
    prior_var_sum = jnp.sum(prior_var)
    cov_sqrt = []
    _, unflatten = create_pytree_flattener(params)
    while jnp.all(post_var < prior_var) and i < eigvals.shape[0]:
        cov_sqrt += [jax.jit(normalize_eigvecs)(eigvals[i], u, eigvecs[:, i])]
        lr_fac_i = jax.jit(unflatten)(cov_sqrt[-1])
        post_var += jnp.concatenate(
            [jax.jit(jvp)(x_c, lr_fac_i) ** 2 for x_c in data["test_input"]], axis=0
        )
        print(f"{i} - post_tr={post_var.sum()} - prior_tr={prior_var_sum}")
        i += 1

    truncation_idx = i if i == eigvals.shape[0] else i - 1
    print(f"Truncation index: {truncation_idx}")
    return jnp.stack(cov_sqrt[:truncation_idx], axis=-1)


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
    # Initial vector
    v = lanczos_jacobian_initialization(model_fn, params, data, **kwargs)
    # Define cov operator
    context_points = select_context_points(
        100,
        "grid",
        [2],
        [-2],
        data["input"].shape[1:],
    )

    cov_matrix = prior_cov_kernel(context_points, context_points, 0.0)
    eps = jnp.finfo(context_points.dtype).eps
    L = lanczos_isqrt(cov_matrix, v, tol=eps**2)
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
    s = _s[_s > tol]
    _u = _u[:, : s.size]

    ggn_matrix = create_fsp_ggn_mv(model_fn, params, M)(data)

    prior_var = jnp.diag(prior_cov_kernel(context_points, context_points, 0.0))

    S = _compute_curvature_fn(model_fn, params, data, ggn_matrix, prior_var, _u)

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


def _truncated_sine(key, n_samples, feature_dim):
    key1, key2, key3 = jax.random.split(key, num=3)

    # Features
    X1 = jax.random.uniform(
        key1, minval=-1, maxval=-0.5, shape=(n_samples // 2, feature_dim)
    )
    X2 = jax.random.uniform(
        key2, minval=0.5, maxval=1, shape=(n_samples // 2, feature_dim)
    )
    X = jnp.concatenate([X1, X2], axis=0)

    # Targets
    eps = 0.1 * jax.random.normal(key3, shape=(n_samples,))
    y = jnp.sin(2 * jnp.pi * X.mean(axis=-1)) + eps

    # Format
    X = X.reshape(-1, feature_dim)
    y = y.reshape(-1, 1)

    return X, y


def run_experiment():
    from sklearn.model_selection import train_test_split
    from helper import DataLoader

    n_epochs = 2000
    X, y = _truncated_sine(jax.random.PRNGKey(0), 300, 1)
    dataset_size = X.shape[0] // 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dataset_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=dataset_size, random_state=42
    )
    train_loader = DataLoader(X_train, y_train, batch_size=20, shuffle=True)

    print(X_train.min(), X_train.max(), X_train.shape, flush=True)
    data = {
        "input": X_train,
        "target": y_train,
        "test_input": X_test,
        "test_target": y_test,
    }

    kernel_per = gpx.kernels.Periodic(lengthscale=1.0, period=1)
    kernel_mat52 = gpx.kernels.Matern52(lengthscale=1.0, variance=0.25)
    kernel_mat12 = gpx.kernels.Matern12(lengthscale=0.25, variance=0.001)

    kernel_time = gpx.kernels.ProductKernel([kernel_per, kernel_mat52])

    kernel = gpx.kernels.SumKernel([kernel_time, kernel_mat12])

    ll_scale = jnp.log(jnp.exp(0.1) - 1)

    def kernel_fn(kernel, x, y=None, ll_scale=ll_scale):
        if y is None:
            y = x
        K = gpx.kernels.computations.DenseKernelComputation().cross_covariance(
            kernel, x, y
        )
        K = K + (1e-10 + ll_scale**2) * jnp.eye(x.shape[0])
        return K

    model = create_model()

    def to_float64(model):  # noqa: F811
        graph_def, params = nnx.split(model)
        params = laplax.util.tree.to_dtype(params, jnp.float64)
        return nnx.merge(graph_def, params)

    model = to_float64(model)
    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    # Define context selection parameters
    n_context_points = 100
    context_selection = "grid"  # or "grid" or "subset"
    context_points_minval = jnp.array([-1.9])
    context_points_maxval = jnp.array([1.9])

    # Initialize a random key
    key = jax.random.PRNGKey(42)

    model = train_model(
        model,
        n_epochs=n_epochs,
        train_loader=train_loader,
        model_fn=model_fn,
        prior_cov_kernel=lambda x, y, l: kernel_fn(kernel, x, y, l),
        ll_scale=ll_scale,
        n_context_points=n_context_points,
        context_selection=context_selection,
        context_points_maxval=context_points_maxval,
        context_points_minval=context_points_minval,
        lr=1e-3,
        key=key,
    )

    _, params = nnx.split(model)

    prob_predictive, _ = fsp_laplace(
        model_fn,
        params,
        data,
        prior_arguments={},
        prior_mean=jnp.zeros((X_train.shape[0],)),
        prior_cov_kernel=lambda x, y, l: kernel_fn(kernel, x, y, l),
        context_points=X_train,
        has_batch_dim=False,
    )

    X_pred = jnp.linspace(-3, 3, 200).reshape(-1, 1)
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


def main():
    run_experiment()


if __name__ == "__main__":
    main()
