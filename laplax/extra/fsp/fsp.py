from functools import partial

import jax
from jax import numpy as jnp

from laplax.curv.cov import Posterior, PosteriorState
from laplax.eval.pushforward import (
    lin_pred_mean,
    lin_pred_std,
    lin_pred_var,
    lin_samples,
    lin_setup,
    set_lin_pushforward,
)
from laplax.extra.fsp.curv import compute_curvature_fn
from laplax.extra.fsp.ggn import create_fsp_ggn_mv
from laplax.extra.fsp.lanczos_isqrt import (
    lanczos_invert_sqrt,
    lanczos_jacobian_initialization,
)
from laplax.types import Data, ModelFn, Params, PosteriorState, PredArray
from laplax.util.flatten import create_partial_pytree_flattener, create_pytree_flattener


def compute_matrix_jacobian_product(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    matrix: PredArray,
    has_batch_dim: bool = True,
):
    """Compute the matrix Jacobian product."""
    if has_batch_dim:
        flatten, unflatten = create_partial_pytree_flattener(params)

        msg = "Batch dimension not implemented for matrix Jacobian product."
        raise NotImplementedError(msg)

    M_tree = jax.vmap(
        jax.vjp(
            lambda p: jnp.reshape(model_fn(data, params=p), (matrix.shape[0],)),
            params,
        )[1],
        in_axes=1,  # 1
        out_axes=1,  # 1
    )(matrix)

    flat_M, unravel_fn = jax.flatten_util.ravel_pytree(
        M_tree,
    )
    M = flat_M.reshape((-1, matrix.shape[-1]))

    return M, unravel_fn


def fsp_laplace(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    prior_cov_kernel,
    context_points,
    *,
    prior_arguments: dict = {},
    **kwargs,
):
    v = lanczos_jacobian_initialization(model_fn, params, context_points, **kwargs)
    cov_matrix = prior_cov_kernel(context_points)
    prior_var = jnp.diag(cov_matrix)
    L = lanczos_invert_sqrt(cov_matrix, v, tol=jnp.finfo(v.dtype).eps)
    M, unravel_fn = compute_matrix_jacobian_product(
        model_fn, params, context_points, L, has_batch_dim=False
    )
    _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
    tol = jnp.finfo(M.dtype).eps ** 2
    s = _s[_s > tol]
    u = _u[:, : s.size]
    ggn_matrix = create_fsp_ggn_mv(model_fn, params, M)(data)
    cov_sqrt = compute_curvature_fn(
        model_fn, params, context_points, ggn_matrix, prior_var, u
    )
    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt}
    flatten, unflatten = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten(
            state["scale_sqrt"] @ state["scale_sqrt"].T @ flatten(x)
        ),
        scale_mv=lambda state: lambda x: unflatten(state["scale_sqrt"] @ x),
        rank=cov_sqrt.shape[1],
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
        key=jax.random.key(6548),
        num_samples=100,
    )
    prob_predictive = set_prob_predictive(
        prior_arguments=prior_arguments,
    )
    return prob_predictive
