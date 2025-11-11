"""This is a local file for development, which should not be included in the pull request."""

import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from laplax import util
from laplax.curv.cov import Posterior
from laplax.curv.utils import LowRankTerms
from laplax.eval.predictives import (
    laplace_bridge,
    mean_field_0_predictive,
    mean_field_1_predictive,
    mean_field_2_predictive,
)
from laplax.eval.utils import finalize_fns
from laplax.types import (
    Any,
    Array,
    Callable,
    DistState,
    Float,
    InputArray,
    Int,
    KeyType,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
    PriorArguments,
)
from laplax.util.flatten import create_pytree_flattener
from laplax.util.ops import precompute_list
from laplax.util.tree import add


def set_output_low_rank_mv(
    posterior_state: Posterior,
    input: InputArray,
    jvp: Callable[[InputArray, Params], PredArray],
    vjp: Callable[[InputArray, PredArray], Params],
    model_fn: ModelFn,
    params: Params,
):
    flatten, unflatten = create_pytree_flattener(params)
    cov_mv = posterior_state.cov_mv(posterior_state.state)

    def output_cov_mv(vec: PredArray) -> PredArray:
        return jvp(input, cov_mv(vjp(input, vec)[0]))

    def output_jac_mv(vec: PredArray) -> PredArray:
        return jvp(input, vec)


# def set_output_low_rank_mv(
#     posterior_state: Posterior,
#     input: InputArray,
#     jvp: Callable[[InputArray, Params], PredArray],
#     vjp: Callable[[InputArray, PredArray], Params],
#     mean_params: Params,
#     *,
#     model_fn: Callable | None = None,
# ):
#     low_rank_terms = posterior_state.low_rank_terms
#     posterior_mean_full = mean_params
#     sigma_sq = posterior_mean_full["param"]
#     model_params = posterior_mean_full["model"]
#     flatten, unflatten = create_pytree_flattener(model_params)

#     def cov_mv(x):
#         flatten, unflatten = create_pytree_flattener(x)
#         x_flat = flatten(x)
#         range_proj = low_rank_terms.U.T @ x_flat
#         range_inv = range_proj / (sigma_sq.value + low_rank_terms.S**2)
#         range_result = low_rank_terms.U @ range_inv
#         null_proj = x_flat - low_rank_terms.U @ (low_rank_terms.U.T @ x_flat)
#         null_result = null_proj / sigma_sq.value
#         result = range_result + null_result

#         return unflatten(result)

#     def output_cov_mv(vec: PredArray) -> PredArray:
#         return jvp(input, cov_mv(vjp(input, vec)[0]))

#     def output_jac_mv(vec: PredArray) -> PredArray:
#         return jvp(input, vec)

#     U_param = low_rank_terms.U  # (P, k)

#     if model_fn is not None:
#         _, lin = jax.linearize(lambda p: model_fn(input=input, params=p), model_params)

#         def apply_col(u_flat):
#             return jnp.ravel(lin(unflatten(u_flat)))

#         U_output = jax.vmap(apply_col)(jnp.moveaxis(U_param, 1, 0))  # (k, D)
#         U_output = jnp.moveaxis(U_output, 0, 1)  # (D, k)
#     else:

#         def apply_col(u_flat):
#             return jnp.ravel(jvp(input, unflatten(u_flat)))

#         U_output = jax.vmap(apply_col)(jnp.moveaxis(U_param, 1, 0))  # (k, D)
#         U_output = jnp.moveaxis(U_output, 0, 1)  # (D, k)

#     S_output = low_rank_terms.S

#     output_low_rank_terms = LowRankTerms(
#         U=U_output, S=S_output, scalar=low_rank_terms.scalar
#     )

#     return {
#         "cov_mv": output_cov_mv,
#         "jac_mv": output_jac_mv,
#         "low_rank_terms": output_low_rank_terms,
#     }


def lin_setup(
    results: dict[str, Array],
    aux: dict[str, Any],
    input: InputArray,
    dist_state: DistState,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Prepare linearized pushforward functions for uncertainty propagation.

    This function sets up matrix-vector product functions for the output covariance
    and scale matrices in a linearized pushforward framework. It verifies the
    validity of input components (posterior state, JVP, and VJP) and stores the
    resulting functions in the auxiliary dictionary.

    Args:
        results: Dictionary to store computed results.
        aux: Auxiliary data to store matrix-vector product functions.
        input: Input data for the model.
        dist_state: Distribution state containing posterior state, JVP, and VJP
            functions.
        **kwargs: Additional arguments (ignored).

    Returns:
        tuple: Updated `results` and `aux`.
    """  # noqa: DOC501
    low_rank = kwargs.get("low_rank", False)
    del kwargs

    jvp = dist_state["jvp"]
    vjp = dist_state["vjp"]
    posterior_state = dist_state["posterior_state"]

    # Check types (mainly needed for type checker)
    if not isinstance(posterior_state, Posterior):
        msg = "posterior state is not a Posterior type"
        raise TypeError(msg)

    if not isinstance(jvp, Callable):
        msg = "JVP is not a JVPType"
        raise TypeError(msg)

    if not isinstance(vjp, Callable):
        msg = "VJP is not a VJPType"
        raise TypeError(msg)

    # if low_rank:
    #     mv_lr = set_output_low_rank_mv(
    #         posterior_state,
    #         input,
    #         jvp,
    #         vjp,
    #         mean_params=aux["mean_params"],
    #         model_fn=aux["model_fn"],
    #     )
    #     aux["cov_lr_mv"] = mv_lr["cov_mv"]
    #     aux["jac_lr_mv"] = mv_lr["jac_mv"]
    #     results["low_rank_terms"] = mv_lr["low_rank_terms"]
    #     results["observation_noise"] = aux["mean_params"]["param"]

    mv = set_output_mv(posterior_state, input, jvp, vjp)
    aux["cov_mv"] = mv["cov_mv"]
    aux["jac_mv"] = mv["jac_mv"]
    results["low_rank_terms"] = mv["low_rank_terms"]

    return results, aux
