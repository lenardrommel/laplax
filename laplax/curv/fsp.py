from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx

import laplax
from laplax.curv.cov import Posterior
from laplax.curv.ggn import create_fsp_ggn_mv, create_ggn_mv_without_data
from laplax.extra.fsp.objective import select_context_points
from laplax.enums import LossFn
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_samples,
    lin_setup,
    set_lin_pushforward,
)
from laplax.extra.fsp.lanczos_isqrt import lanczos_invert_sqrt
from laplax.types import (
    Callable,
    Data,
    Float,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
)
from laplax.util import tree

# --------------------------------------------------------------------------------------
# FSP training utilities
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# FSP training objective
# --------------------------------------------------------------------------------------


def create_loss_mse(model_fn: ModelFn):
    """Create the MSE loss function for FSP training."""

    def loss_mse(data: Data, params: Params) -> Float:
        pred = model_fn(data["input"], params)
        return jnp.mean(jax.numpy.square(pred - data["target"]))

    return loss_mse


def create_loss_nll(
    model_fn: ModelFn,
    loss_fn: LossFn,
):
    r"""Create the NLL loss function for FSP training.
    $$
    - \log p(f(X) + \epsilon = y) = \frac{1}{2}(y - \mu_{\theta}(X))^{T} (\Sigma_{\theta}(X, X) + \sigma^2I)^\dagger (y - \mu_{\theta}(X)) - \frac{1}{2} \log |(\Sigma_{\theta}(X, X) + \sigma^2I)| + \frac{d}{2} \log(2\pi)
    $$.
    """  # noqa: D205

    def loss_nll(
        data: Data, params: Params, other_params: Params | None = None
    ) -> Float:
        preds = jax.vmap(model_fn, in_axes=(0, None))(data["input"], params)
        return loss_fn(preds, data["target"], other_params)

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
):
    def loss_reg(context_points, params: Params) -> Float:
        r"""FSP regularization loss.

        $$1/2 (f(c^{i}) - m)^{T} K^{-1}(c^{i}, c^{i}) (f(c^{i}) - m)$$
        """
        f_c = jax.vmap(model_fn, in_axes=(0, None))(context_points, params) - prior_mean
        K_c_c = prior_cov_kernel(context_points, context_points)
        left = jax.numpy.linalg.solve(K_c_c, f_c)
        return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    loss_fn: LossFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable,
):
    """Create FSP objective using the wrapped model with learnable scale."""
    # Create loss functions
    loss_nll = create_loss_nll(model_fn, loss_fn)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    # Create objective
    def fsp_objective(
        data: Data, context_points: PredArray, params: Params, other_params: Params
    ) -> Float:
        return loss_nll(data, params, other_params) + loss_reg(context_points, params)

    return fsp_objective


# --------------------------------------------------------------------------------------
# FSP Laplace approximation
# --------------------------------------------------------------------------------------


def lanczos_jacobian_initialization(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    has_batch_dim: bool = True,
    *,
    lanczos_initialization_batch_size: int = 32,
):
    # Define model Jacobian vector product
    if has_batch_dim:
        model_jvp = jax.vmap(
            lambda x: jax.jvp(
                lambda w: model_fn(x, params=w),
                (params,),
                (tree.ones_like(params),),
            )[1],
            in_axes=0,
            out_axes=0,
        )

        initial_vec = jax.lax.map(
            model_jvp,
            data["input"],
            batch_size=lanczos_initialization_batch_size,
        ).reshape(-1)
    else:
        initial_vec = jax.jvp(
            lambda w: model_fn(data["input"], params=w),
            (params,),
            (tree.ones_like(params),),
        )[1]

    # Normalize
    initial_vec = initial_vec / jnp.linalg.norm(initial_vec, 2)

    return initial_vec.squeeze(-1)


def compute_matrix_jacobian_product(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    matrix: PredArray,
    has_batch_dim: bool = True,
):
    """Compute the matrix Jacobian product."""
    if has_batch_dim:
        flatten, unflatten = laplax.util.flatten.create_partial_pytree_flattener(params)

        raise NotImplementedError(
            "Batch dimension not implemented for matrix Jacobian product."
        )

    M_tree = jax.vmap(
        jax.vjp(
            lambda p: jnp.reshape(
                model_fn(data["input"], params=p), (matrix.shape[0],)
            ),
            params,
        )[1],
        in_axes=1,  # 0
        out_axes=1,  # 0
    )(matrix)

    flat_M, unravel_fn = jax.flatten_util.ravel_pytree(
        M_tree,
    )
    return flat_M.reshape((-1, matrix.shape[-1])), unravel_fn


def compute_curvature_fn(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    ggn: PredArray,
    prior_var: PredArray,
    u: PredArray,
) -> PredArray:
    _eigvals, _eigvecs = jnp.linalg.eigh(ggn)
    eps = jnp.finfo(ggn.dtype).eps
    tol = eps * (_eigvals.max() ** 0.5) * _eigvals.shape[0]
    eigvals = _eigvals[_eigvals > tol]
    eigvecs = _eigvecs[:, _eigvals > tol]
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)
    x_context = select_context_points(
        int(data["input"].shape[0]),
        "grid",
        data["input"].max(axis=0),
        data["input"].min(axis=0),
        data["input"].shape,
        key=jax.random.key(0),
    )

    def create_scan_fn(
        unflatten_fn, _u, eigvecs, eigvals, params, model_fn, data, prior_var
    ):
        def scan_fn(carry, i):
            post_var, valid_indices = carry

            new_cov = _u @ (eigvecs[:, i] * (1 / eigvals[i] ** 0.5))
            lr_fac_i = unflatten_fn(new_cov)  # Use captured unflatten

            all_inputs = jnp.array(x_context)

            def compute_jvp_squared(x):
                jvp_result = jax.jvp(lambda p: model_fn(x, p), (params,), (lr_fac_i,))[
                    1
                ]
                return jnp.square(jvp_result)

            jvp_squared_results = jax.vmap(compute_jvp_squared)(all_inputs)
            jvp_squared_concat = jnp.reshape(jvp_squared_results, post_var.shape)

            new_post_var = post_var + jvp_squared_concat
            is_valid = jnp.all(new_post_var < prior_var)

            new_valid_indices = jax.lax.cond(
                is_valid,
                lambda _: valid_indices.at[i].set(True),
                lambda _: valid_indices,
                None,
            )

            return (new_post_var, new_valid_indices), (new_cov, is_valid)

        return scan_fn

    _, unflatten = laplax.util.flatten.create_pytree_flattener(params)

    scan_fn = create_scan_fn(
        unflatten, u, eigvecs, eigvals, params, model_fn, data, prior_var
    )

    init_post_var = jnp.zeros((prior_var.shape[0],))
    init_valid_indices = jnp.zeros(eigvals.shape[0], dtype=jnp.bool_)

    (final_post_var, final_valid_indices), (covs, validity) = jax.lax.scan(
        scan_fn, (init_post_var, init_valid_indices), jnp.arange(eigvals.shape[0])
    )
    cumulative_validity = jnp.cumprod(validity)

    n_valid = jnp.sum(cumulative_validity)

    n_valid_int = jnp.minimum(
        jnp.array(eigvals.shape[0], dtype=jnp.int32),
        jnp.array(n_valid, dtype=jnp.int32),
    )
    valid_covs = jax.lax.dynamic_slice(covs, (0, 0), (n_valid_int, covs.shape[1]))
    return jnp.transpose(valid_covs)


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
    # op = op if isinstance(op, Callable) else lambda x: op @ x
    cov_matrix = prior_cov_kernel(data["input"])

    L = lanczos_invert_sqrt(cov_matrix, v)
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
    x_context = select_context_points(
        int(data["input"].shape[0]),
        "grid",
        data["input"].max(axis=0),
        data["input"].min(axis=0),
        data["input"].shape,
        key=jax.random.key(0),
    )
    x_context = select_context_points(
        150,
        "grid",
        [3],
        [-3],
        data["input"].shape[1:],
    )

    prior_var = jnp.diag(prior_cov_kernel(x_context))

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
