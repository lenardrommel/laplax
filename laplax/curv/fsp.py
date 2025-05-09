import jax
import jax.flatten_util
import jax.numpy as jnp

import laplax
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from laplax.enums import LossFn
from laplax.types import Callable, Data, Float, ModelFn, Params, PredArray
from laplax.util import tree

# --------------------------------------------------------------------------------------
# FSP training objective
# --------------------------------------------------------------------------------------


def create_loss_nll(
    model_fn: ModelFn,
    loss_fn: LossFn,
):
    """Create the NLL loss function for FSP training."""

    def loss_nll(data: Data, params: Params) -> Float:
        pred = model_fn(data["inputs"], params)
        N = data["inputs"].shape[0]
        return loss_fn(pred, data["targets"])

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
        f_c = jax.vmap(model_fn, in_axes=(0, None))(context_points, params)
        K_c_c = prior_cov_kernel(context_points, context_points)
        left = jax.numpy.linalg.solve(K_c_c, f_c)
        return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    loss_fn: LossFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
    num_training_samples: int = 150,
    batch_size: int = 20,
):
    # Create loss functions
    loss_nll = create_loss_nll(model_fn, loss_fn)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    # Create objective
    def fsp_objective(data: Data, context_points: PredArray, params: Params) -> Float:
        return num_training_samples / batch_size * loss_nll(data, params) + loss_reg(
            context_points, params
        )

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
            data["inputs"],
            batch_size=lanczos_initialization_batch_size,
        ).reshape(-1)
    else:
        initial_vec = jax.jvp(
            lambda w: model_fn(data["inputs"], params=w),
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
        flatten, unflatten = laplax.util.create_partial_pytree_flattener(params)

        raise NotImplementedError(
            "Batch dimension not implemented for matrix Jacobian product."
        )

    M_tree = jax.vmap(
        jax.vjp(
            lambda p: jnp.reshape(
                model_fn(data["inputs"], params=p), (matrix.shape[0],)
            ),
            params,
        )[1],
        in_axes=1,  # 0
        out_axes=1,  # 0
    )(matrix)

    M, unravel_fn = jax.flatten_util.ravel_pytree(
        M_tree,
    )
    return M, unravel_fn


def fsp_laplace(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
    context_points: PredArray,
    **kwargs,
):
    """FSP Laplace approximation."""
    # Initial vector

    # Define cov operator
    op = prior_cov_kernel(context_points, context_points)
    op = op if isinstance(op, Callable) else lambda x: op @ x

    # Compute low rank terms
    # USE INVERSE OF COV MATRIX
    initial_vec = lanczos_jacobian_initialization(
        model_fn,
        params,
        data,
        **kwargs,
    )
    L = lanczos_isqrt(op, initial_vec, **kwargs)

    # Compute
    model_vjp = jax.vmap(
        lambda x, v: jax.vjp(lambda w: model_fn(x, w), params)[1](v),
        in_axes=(0, 0),
        out_axes=0,
    )(context_points, L)
    S = S
    # Posterior state
    posterior_state: PosteriorState = {"scale_sqrt": S}

    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: state["scale_sqrt"] @ state["scale_sqrt"].T @ x,
        scale_mv=lambda state: lambda x: state["scale_sqrt"] @ x,
    )

    def posterior_fn(*args, **kwargs):
        del args, kwargs
        return posterior

    return posterior_fn


# lambda prior_args: posteiror_fn(prior_args)
