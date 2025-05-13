from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import (
    DataLoader,
    RBFKernel,
    build_covariance_matrix,
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

import laplax
from laplax.curv import estimate_curvature
from laplax.curv.cov import Posterior, set_posterior_fn
from laplax.curv.fsp import (
    compute_matrix_jacobian_product,
    create_fsp_objective,
    create_loss_mse,
    create_loss_nll,
    create_loss_reg,
    lanczos_jacobian_initialization,
)
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from laplax.curv.utils import LowRankTerms
from laplax.enums import LossFn
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
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
    def __init__(self, model, param, data_size):
        self.model = model
        self.param = nnx.Param(jnp.asarray(param))
        self.N = data_size

    def __call__(self, x):
        return self.model(x)


def mse_loss(x, y):
    return 0.5 * jnp.sum((x - y) ** 2)


def train_step(model, optimizer, loss_fn, data):
    _, params = nnx.split(model)

    def _loss_fn(p):
        return loss_fn(data, p)

    loss, grads = nnx.value_and_grad(_loss_fn)(params)
    optimizer.update(grads)

    return loss


def create_train_step(loss_fn):
    @nnx.jit
    def train_step(model, optimizer, data):
        _, params = nnx.split(model)

        def _loss_fn(p):
            return loss_fn(data, p)

        loss, grads = nnx.value_and_grad(_loss_fn)(params)
        optimizer.update(grads)

        return loss

    return train_step


def train_model(model, n_epochs, train_loader, loss_fn, lr=1e-3):
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    train_step_fn = create_train_step(loss_fn)

    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            data = {"inputs": x_tr, "targets": y_tr}
            loss = train_step_fn(model, optimizer, data)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


def create_model(config):
    in_channels = config.get("in_channels", 1)
    hidden_channels = config.get("hidden_channels", 64)
    out_channels = config.get("out_channels", 1)
    rngs = config.get("rngs", nnx.Rngs(0))
    param = config.get("param", None)
    data_size = config.get("data_size", None)

    model = to_float64(
        Model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            rngs=rngs,
        )
    )

    graph_def, _ = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    if param is not None and data_size is not None:
        model = ModelWrapper(model, param, data_size)

    # graph_def, _ = nnx.split(model)

    # def model_fn(input, params):
    #     return nnx.call((graph_def, params))(input)[0]

    return model, model_fn, graph_def


def create_kernel_fn(lengthscale=8 / jnp.pi, output_scale=0.5, noise_variance=0.001):
    kernel = RBFKernel(lengthscale=lengthscale)

    prior_arguments = {"prior_prec": output_scale, "noise_variance": noise_variance}

    def kernel_fn(x, y=None, output_scale=output_scale, noise_variance=noise_variance):
        if y is None:
            y = x
        K = build_covariance_matrix(kernel, x, y)
        return jnp.exp(output_scale) * (K + noise_variance**2 * jnp.eye(K.shape[0]))

    return kernel_fn, prior_arguments


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
            mse_loss,
            prior_mean,
            prior_cov_kernel,
            num_training_samples,
            batch_size,
        )

        # Return a function that includes the context_points
        return lambda data, params: fsp_obj(data, context_points, params)
    else:
        msg = f"Unknown loss type: {loss_type}. Supported types are: mse, nll, fsp"
        raise ValueError(msg)


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

    # def ggn_mv(data):
    #     ju = jac_mv(
    #         model_fn,
    #         params,
    #         data["inputs"],
    #     )
    #     ju = jnp.transpose(ju, (1, 0, 2)).squeeze(-1)
    #     return jnp.diag(s**2) + jnp.einsum(
    #         "ji,jk->ik", ju, ju
    #     )  # jnp.diag(s**2) + jnp.einsum("ji,jk->ik", ju, ju)

    def ggn_mv(vec, data):
        # Step 1: Single jvp for entire batch, if has_batch is True
        def fwd(p):
            if has_batch:
                return jax.vmap(lambda x: model_fn(input=x, params=p))(data["inputs"])
            return model_fn(input=data["inputs"], params=p)

        # Step 2: Linearize the forward pass
        z, jvp = jax.linearize(fwd, params)

        # Step 3: Compute J^T H J v
        HJv = jvp(vec)
        # HJv = loss_hessian_mv(jvp(vec), pred=z, target=data["targets"])

        # Step 4: Compute the GGN vector
        arr = jax.linear_transpose(jvp, vec)(HJv)[0]

        factor = 1.0
        return laplax.util.tree.mul(factor, arr)

    ggn_mv_wrapped = laplax.util.flatten.flatten_function(ggn_mv, layout=params)

    def fsp_ggn_mv(data):
        return jnp.diag(s**2) + u.T @ jax.vmap(
            ggn_mv_wrapped, in_axes=(-1, None), out_axes=-1
        )(u, data)

    return fsp_ggn_mv


def compute_curvature_fn(model_fn, params, ggn):
    eigvals, eigvecs = jnp.linalg.eigh(ggn)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    _, unravel_fn = ravel_pytree(params)

    def _compute_cov_sqrt(_u):
        cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(eigvals[::-1]))

        def jvp(x, v):
            return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

        def scan_fn(
            carry, i
        ):  # TODO: implement scan to not compare the sum, but all elements of vectors
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

    return _compute_cov_sqrt


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
    op = op if isinstance(op, Callable) else lambda x: op @ x
    cov_matrix = prior_cov_kernel(data["inputs"], data["inputs"])

    # L = lanczos_isqrt(cov_matrix, v)
    L = laplax.curv.cov.full_prec_to_scale(cov_matrix)
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
    _eigvals, _eigvecs = jnp.linalg.eigh(ggn_matrix)
    eps = jnp.finfo(M.dtype).eps
    tol = eps * (_eigvals.max() ** 0.5) * s.shape[0]
    eigvals = _eigvals[_eigvals > tol]
    eigvecs = _eigvecs[:, _eigvals > tol]
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # ==========
    params = laplax.util.tree.to_dtype(params, jnp.float64)

    # Tristan ==========
    i = 0
    post_var = jnp.zeros((150, 1))
    prior_var = jnp.diag(prior_cov_kernel(data["inputs"], data["inputs"]))
    cov_sqrt = []
    _, unflatten = laplax.util.flatten.create_pytree_flattener(params)

    def _f1(_e, _u, _v):
        return _u @ (_v * (1 / _e**0.5))

    def _f2(_x, _v):
        return jax.jvp(lambda _p: model_fn(_x, _p), (params,), (_v,))[1]

    while jnp.all(post_var < prior_var) and i < eigvals.shape[0]:
        cov_sqrt += [jax.jit(_f1)(eigvals[i], _u, eigvecs[:, i])]
        lr_fac_i = jax.jit(unflatten)(cov_sqrt[-1])
        post_var += jnp.concatenate(
            [jax.jit(_f2)(x_c, lr_fac_i) ** 2 for x_c in data["inputs"]], axis=0
        )
        print(f"{i} - post_tr={post_var.sum()} - prior_tr={prior_var.sum()}")
        i += 1

    truncation_idx = i if i == eigvals.shape[0] else i - 1
    print(f"Truncation index: {truncation_idx}")
    S = jnp.stack(cov_sqrt[:truncation_idx], axis=-1)
    # Tristan ==========

    # cov_sqrt = compute_curvature_fn(model_fn, params, ggn_matrix)(_u)

    posterior_state: PosteriorState = {"scale_sqrt": S}
    flatten, unflatten = laplax.util.flatten.create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten(
            state["scale_sqrt"] @ state["scale_sqrt"].T @ flatten(x)
        ),
        scale_mv=lambda state: lambda x: state["scale_sqrt"] @ x,
    )
    # curv_estimate = LowRankTerms(S, s, 1)
    posterior_fn = lambda *args, **kwargs: posterior

    # posterior_fn = set_posterior_fn("lanczos", curv_estimate, layout=params_correct)

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
        ],
    )

    prob_predictive = set_prob_predictive(
        prior_arguments=prior_arguments,
    )

    X_pred = jnp.linspace(0, 8, 200).reshape(200, 1)
    pred = jax.vmap(prob_predictive)(X_pred)
    pred_model = jax.vmap(model_fn, in_axes=(0, None))(X_pred, params)
    _ = plot_regression_with_uncertainty(
        X_train=data["inputs"],
        y_train=data["targets"],
        X_pred=X_pred,
        y_pred=pred_model[
            :, 0
        ],  # pred["pred_mean"][:, 0],  # y_pred=pred["pred_mean"][:, 0],
        y_std=jnp.sqrt(pred["pred_var"][:, 0]),
        y_max=10,
    )

    plt.show()
    print("Posterior mean:", pred["pred_mean"][:, 0])


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
        sigma_noise=0.01,
        intervals=[(3, 3.1), (4.0, 4.1), (5, 5.1)],
        rng_key=jax.random.key(0),
        dtype=jnp.float64,
    )
    train_loader = DataLoader(X_train, y_train, batch_size)
    data = {"inputs": X_train, "targets": y_train}
    initial_log_scale = jnp.log(0.1)
    config = {
        "in_channels": 1,
        "hidden_channels": 64,
        "out_channels": 1,
        "rngs": nnx.Rngs(key),
        "param": initial_log_scale,
        "data_size": num_training_samples,
    }
    model, model_fn, _ = create_model(config)

    kernel_fn, prior_arguments = create_kernel_fn(
        lengthscale=8 / jnp.pi, output_scale=0.5, noise_variance=0.001
    )

    # Choose loss function type: 'mse', 'nll', or 'fsp'
    loss_type = "fsp"

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

    _ = plot_sinusoid_task(X_train, y_train, X_test, y_test, X_pred, y_pred)

    fsp_laplace(
        model_fn,
        params["model"],
        data,
        prior_arguments=prior_arguments,
        prior_mean=jnp.zeros((150)),
        prior_cov_kernel=kernel_fn,
        context_points=X_train,
        has_batch_dim=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
