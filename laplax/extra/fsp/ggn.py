import jax
from jax import numpy as jnp

from laplax.curv.ggn import create_ggn_mv_without_data
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Callable,
    Data,
    Float,
    Int,
    ModelFn,
    Num,
    Params,
    PredArray,
    TargetArray,
)
from laplax.util.flatten import flatten_function
from laplax.util.tree import mul


def create_fsp_ggn_mv(
    model_fn: ModelFn,
    params: Params,
    M: PredArray,
    *,
    has_batch: bool = False,
    loss_hessian_mv: Callable[[PredArray, PredArray], Num[Array, "..."]] | None = None,
) -> Callable[[Params, Data], Params]:
    r"""Implements the FSP-Laplace equation (3.8) from the paper:

    Λ = Σ† w⋆ − Σ_{i=1}^n Jw⋆(x(i))⊤ L(i) w⋆ Jw⋆(x(i)).

    Where:
    - Λ is the FSP-Laplace matrix,
    - Σ† is the pseudo-inverse of the covariance matrix Σ,
    - w⋆ represents the optimal weights,
    - Jw⋆(x(i)) is the Jacobian of the model with respect to the weights at input x(i),
    - L(i) w⋆ is the loss Hessian at the optimal weights for the i-th data point.

    This equation describes the FSP-Laplace approximation for the Generalized Gauss-Newton
    matrix in the context of Bayesian deep learning.
    """  # noqa: D415, DOC201
    _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
    tol = jnp.finfo(M.dtype).eps ** 2
    s = _s[_s > tol]
    u = _u[:, : s.size]

    if has_batch:
        msg = (
            "FSP GGN MV is not implemented for batched data. "
            "Please set has_batch=False."
        )
        raise NotImplementedError(msg)

    def identity_loss_hessian_mv(v, pred=None, target=None):
        return v

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=LossFn.NONE,  # Placeholder - we're providing custom loss_hessian_mv
        factor=1.0,
        has_batch=has_batch,
        loss_hessian_mv=identity_loss_hessian_mv,
    )

    ggn_mv_wrapped = flatten_function(ggn_mv, layout=params)

    def fsp_ggn_mv(data):
        return jnp.diag(s**2) + u.T @ jax.vmap(
            ggn_mv_wrapped, in_axes=(-1, None), out_axes=-1
        )(u, data)

    return fsp_ggn_mv
