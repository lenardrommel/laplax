import jax
from jax import numpy as jnp

from laplax.curv.ggn import create_ggn_mv_without_data
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Callable,
    Data,
    Float,
    InputArray,
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
        vmap_over_data=has_batch,
        loss_hessian_mv=identity_loss_hessian_mv,
    )

    ggn_mv_wrapped = flatten_function(ggn_mv, layout=params)

    def fsp_ggn_mv(data):
        return jnp.diag(s**2) + u.T @ jax.vmap(
            ggn_mv_wrapped, in_axes=(-1, None), out_axes=-1
        )(u, data)

    return fsp_ggn_mv


def create_ggn_pytree_mv(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    hessian_diag: bool = True,
) -> Callable[[Params], Array]:
    """Create a GGN matrix-vector product function that works with pytrees.

    This function creates a Generalized Gauss-Newton (GGN) matrix-vector product
    operator that works directly with pytree parameters without requiring linear
    operators or dense matrices.

    Args:
        model_fn: Model function taking input and params
        params: Model parameters as pytree
        x_context: Context points for GGN computation
        hessian_diag: If True, assumes diagonal Hessian (identity for regression)

    Returns:
        Function that computes GGN @ u for pytree u
    """

    def _jacobian_matrix_product(u):
        """Calculates the product of the Jacobian and matrix u (pytree).

        Parameters
        ----------
        u : pytree
            Parameter pytree with same structure as params

        Returns
        -------
        Array with shape (B,) + output_shape + (R,)
            Batch of Jacobian-matrix products
        """
        return jax.vmap(
            lambda x_c: jax.vmap(
                lambda u_c: jax.jvp(lambda p: model_fn(x_c, p), (params,), (u_c,))[1],
                in_axes=-1,
                out_axes=-1,
            )(u)
        )(x_context)

    def ggn_vector_product(u):
        """Compute u^T @ GGN @ u.

        Args:
            u: pytree with same structure as params

        Returns:
            GGN matrix-vector product
        """
        if hessian_diag:
            ju = _jacobian_matrix_product(u)
            batch_size = ju.shape[0]
            rank = ju.shape[-1]
            ju_flat = ju.reshape(batch_size, -1, rank)
            return jnp.einsum("bji,bjk->ik", ju_flat, ju_flat)
        else:
            msg = "Full Hessian not implemented yet."
            raise NotImplementedError(msg)

    return ggn_vector_product
