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


def mse_loss(x, y):
    return 0.5 * jnp.sum((x - y) ** 2)


# fsp_loss_fn = create_fsp_objective(
#     model_fn,
#     loss_fn=mse_loss,
#     prior_mean=jnp.zeros((150)),
#     prior_cov_kernel=kernel_fn,
# )


@nnx.jit
def train_step(model, params, optimizer, loss_fn, data):
    def loss_fn(m):
        return loss_fn(
            data, data["inputs"], params
        )  # mse_loss(m, x, y) + reg_loss(m, kernel_fn, x)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


@nnx.jit
def _train_step(model, optimizer, data):
    def loss_fn(model):
        y_pred = model(data["inputs"])  # Call methods directly
        return jnp.sum((y_pred - data["targets"]) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


def train_model(model, n_epochs, train_loader, loss_fn=None, lr=1e-3):
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(lr))  # Reference sharing

    # Train epoch
    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            data = {"inputs": x_tr, "targets": y_tr}
            loss = _train_step(model, optimizer, data)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


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

    L = lanczos_isqrt(
        prior_cov_kernel(data["inputs"], data["inputs"]), v
    )  # Multiply by prior variance

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
        sigma_noise=0.001,
        intervals=[(2, 3), (4, 5), (6, 7)],
        rng_key=jax.random.key(0),
        dtype=jnp.float64,
    )
    train_loader = DataLoader(X_train, y_train, batch_size)
    data = {"inputs": X_train, "targets": y_train}

    model = to_float64(
        Model(
            in_channels=1,
            hidden_channels=64,
            out_channels=1,
            rngs=nnx.Rngs(0),
        )
    )

    graph_def, _ = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    model = train_model(model, n_epochs=1000, train_loader=train_loader)

    X_pred = jnp.linspace(0.0, 8.0, 200).reshape(200, 1)
    y_pred = jax.vmap(model)(X_pred)

    graph_def, params = nnx.split(model)

    _ = plot_sinusoid_task(X_train, y_train, X_test, y_test, X_pred, y_pred)

    kernel = RBFKernel(lengthscale=8 / jnp.pi)

    def kernel_fn(x, y=None, noise_std=0.001):
        if y is None:
            y = x
        K = build_covariance_matrix(kernel, x, y)
        return K + noise_std**2 * jnp.eye(K.shape[0])

    prior_arguments = {"prior_prec": 0.5, "noise_variance": 0.001}

    def kernel_fn(
        x,
        y=None,
        output_scale=prior_arguments["prior_prec"],
        noise_variance=prior_arguments["noise_variance"],
    ):
        if y is None:
            y = x
        K = build_covariance_matrix(kernel, x, y)
        return jnp.exp(output_scale) * (K + noise_variance**2 * jnp.eye(K.shape[0]))

    fsp_laplace(
        model_fn,
        params,
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
