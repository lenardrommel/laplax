# fsp.py

"""FSP (Function-Space Prior) inference module with support for different kernel structures."""

import functools
import operator
import time
from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger

from laplax.curv.cov import Posterior, PosteriorState
from laplax.curv.utils import LowRankTerms
from laplax.enums import CovarianceStructure
from laplax.types import (
    Callable,
    InputArray,
    Kernel,
    Kwargs,
    ModelFn,
    Params,
    PredArray,
)
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
)
from laplax.util.lanczos import lanczos_invert_sqrt

KernelStructure = CovarianceStructure


# ==============================================================================
# Helper functions
# ==============================================================================


@partial(jax.jit, static_argnames=("model_fn",))
def _model_jvp(
    model_fn: ModelFn, params: Params, xs: InputArray, vs: Params
) -> PredArray:
    """Compute multiple Jacobian-vector products of the model.

    res[b] == jvp(f(xs[b], :), vs)

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : jnp.array with shape `(B,) + input_shape`
        Primals
    vs : Params
        Tangent vectors (pytree matching params structure)

    Returns
    -------
    jnp.array with shape `(B,) + output_shape`
        Batch of Jacobian-vector products
    """
    return jax.vmap(
        lambda x: jax.jvp(lambda w: model_fn(x, w), (params,), (vs,))[1],
        in_axes=0,
        out_axes=0,
    )(xs)


def _model_vjp(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    vs: PredArray,
    *,
    batch_axis: int = 0,
    output_batch_axis: int = 0,
) -> jax.Array:
    return jax.vmap(
        lambda x, v: jax.vjp(lambda w: model_fn(x, w), params)[1](v)[0],
        in_axes=(batch_axis, output_batch_axis),
        out_axes=output_batch_axis,
    )(xs, vs)


@partial(jax.jit, static_argnames=("model_fn",))
def _M_batch(model_fn: ModelFn, params: Params, xs: InputArray, L: PredArray):
    """Compute batched matrix-Jacobian product.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : InputArray
        Input data
    L : PredArray
        Matrix to multiply with Jacobian

    Returns
    -------
    Pytree
        Batched matrix-Jacobian product
    """

    def process_single_vs(vs):
        vjp_result = _model_vjp(model_fn, params, xs, vs)
        return jax.tree.map(lambda param: jnp.sum(param, axis=0), vjp_result)

    L_transposed = jnp.moveaxis(L, -1, 0)
    result = jax.lax.map(process_single_vs, L_transposed)

    return jax.tree.map(lambda x: jnp.moveaxis(x, 0, -1), result)


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def _lanczos_init(model_fn: ModelFn, params: Params, xs, num_chunks):
    """Initialize Lanczos vectors using HOSVD.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    xs : Array
        Input data with shape (B, S1, S2, ..., C)
    num_chunks : int
        Number of chunks for processing

    Returns
    -------
    tuple
        (initial_vectors_function, initial_vectors_spatial)
    """  # noqa: DOC501
    if xs.ndim < 4:
        msg = f"Input must have shape (B, S1, S2, ..., C), but got {xs.shape}"
        raise ValueError(msg)

    ones_pytree = jax.tree.map(jnp.ones_like, params)

    model_jvp = jax.vmap(
        lambda x: jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1],
        in_axes=0,
        out_axes=0,
    )

    b = jnp.concatenate(
        [model_jvp(xs_batch) for xs_batch in jnp.split(xs, num_chunks, axis=0)],
        axis=0,
    )

    spatial_dims = tuple(s for s in xs.shape[1:-2] if s > 1)
    n_function = xs.shape[0]
    b = b.reshape((n_function,) + spatial_dims)

    initial_vectors = []

    # HOSVD: compute SVD along each mode
    for mode in range(len(b.shape)):
        # Unfold along mode
        n_mode = b.shape[mode]
        b_unfolded = jnp.moveaxis(b, mode, 0).reshape(n_mode, -1)

        # SVD
        u, _s, _v = jnp.linalg.svd(b_unfolded, full_matrices=False)

        # Take dominant singular vector
        vec = u[:, 0] / jnp.linalg.norm(u[:, 0])
        initial_vectors.append(vec)

    # Split into function and spatial
    initial_vectors_function = [initial_vectors[0]]
    initial_vectors_spatial = initial_vectors[1:]

    return initial_vectors_function, initial_vectors_spatial


def create_ggn_pytree_mv(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    hessian_diag: bool = True,
) -> Callable[[Params], jnp.ndarray]:
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

    def _unwrap(u_like):
        return (
            u_like[0]
            if isinstance(u_like, (tuple, list)) and len(u_like) == 1
            else u_like
        )

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
        u = _unwrap(u)

        return jax.vmap(
            lambda x_c: jax.vmap(
                lambda u_c: jax.jvp(
                    lambda p: model_fn(x_c, p), (params,), (_unwrap(u_c),)
                )[1],
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


# ==============================================================================
# Kernel structure specific implementations
# ==============================================================================


def _lanczos_kronecker_structure(
    kernels_list: list[Callable],
    initial_vectors: list[jnp.ndarray],
    max_iters: list[int] | None = None,
):
    """Compute Lanczos inverse sqrt factor for Kronecker structured kernel.

    Parameters
    ----------
    kernels_list : list[Callable]
        List of kernel functions for each dimension
    initial_vectors : list[jnp.ndarray]
        Initial vectors for Lanczos for each dimension
    max_iters : list[int] | None
        Maximum iterations for each dimension

    Returns
    -------
    list
        List of Lanczos inverse sqrt factors for each dimension
    """
    if max_iters is None:
        max_iters = [None] * len(kernels_list)

    lanczos_results = []
    for kernel, init_vec, max_iter in zip(kernels_list, initial_vectors, max_iters):
        kwargs = {}
        if max_iter is not None:
            kwargs["max_iter"] = max_iter

        result = lanczos_invert_sqrt(kernel, init_vec, **kwargs)
        lanczos_results.append(result)

    return lanczos_results


def _lanczos_none_structure(
    kernel: Callable,
    initial_vector: jnp.ndarray,
    max_iter: int | None = None,
):
    """Compute Lanczos inverse sqrt factor for unstructured kernel.

    Parameters
    ----------
    kernel : Callable
        Kernel function or matrix
    initial_vector : jnp.ndarray
        Initial vector for Lanczos
    max_iter : int | None
        Maximum iterations

    Returns
    -------
    Array
        Lanczos inverse sqrt factor
    """
    kwargs = {}
    if max_iter is not None:
        kwargs["max_iter"] = max_iter

    return lanczos_invert_sqrt(kernel, initial_vector, **kwargs)


def _kronecker_product(factors: list) -> jnp.ndarray:
    """Compute Kronecker product of a list of matrices.

    Parameters
    ----------
    factors : list
        List of matrices to compute Kronecker product

    Returns
    -------
    jnp.ndarray
        Kronecker product of all factors
    """
    result = factors[0]
    for factor in factors[1:]:
        result = jnp.kron(result, factor)
    return result


def _compute_kronecker_diagonal(factors: list[jnp.ndarray]) -> jnp.ndarray:
    """Compute diagonal of Kronecker product efficiently.

    Parameters
    ----------
    factors : list[jnp.ndarray]
        List of matrices in Kronecker product

    Returns
    -------
    jnp.ndarray
        Diagonal of Kronecker product
    """
    # For Kronecker product A ⊗ B, diag(A ⊗ B) = diag(A) ⊗ diag(B)
    diagonals = [jnp.diag(factor) for factor in factors]
    result = diagonals[0]
    for diag in diagonals[1:]:
        result = jnp.kron(result, diag)
    return result


def _compute_none_diagonal(matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute diagonal of unstructured matrix.

    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix

    Returns
    -------
    jnp.ndarray
        Diagonal of matrix
    """
    return jnp.diag(matrix)


# ==============================================================================
# Main inference functions
# ==============================================================================


def create_fsp_posterior_kronecker(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    spatial_kernels: list[Callable],
    function_kernels: list[Callable],
    prior_variance: jnp.ndarray,
    n_chunks: int,
    *,
    spatial_max_iters: list[int] | None = None,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with Kronecker structured prior.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    spatial_kernels : list[Callable]
        List of spatial kernel functions
    function_kernels : list[Callable]
        List of function kernel functions
    prior_variance : jnp.ndarray
        Prior variance
    n_chunks : int
        Number of chunks for processing
    spatial_max_iters : list[int] | None
        Maximum Lanczos iterations for each spatial dimension

    Returns
    -------
    Posterior
        FSP posterior approximation
    """
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count to evenly divide number of functions
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    dim = sum(x.size for x in jax.tree_util.tree_leaves(params))

    initial_vectors_function, initial_vectors_spatial = _lanczos_init(
        model_fn, params, x_context, num_chunks=n_chunks_eff
    )

    # Lanczos inverse sqrt factors for spatial dimensions
    if spatial_max_iters is None:
        spatial_max_iters = [8] * len(spatial_kernels)

    spatial_lanczos_results = _lanczos_kronecker_structure(
        spatial_kernels, initial_vectors_spatial, spatial_max_iters
    )

    function_lanczos_results = _lanczos_kronecker_structure(
        function_kernels, initial_vectors_function, max_iters=None
    )

    # Combine Lanczos results using Kronecker product
    k_spatial_inv_sqrt = _kronecker_product(spatial_lanczos_results)
    k_function_inv_sqrt = _kronecker_product(function_lanczos_results)

    k_inv_sqrt = jnp.kron(k_spatial_inv_sqrt, k_function_inv_sqrt)

    n_function = x_context.shape[0]
    rank = k_inv_sqrt.shape[-1]

    k_inv_sqrt_dense = k_inv_sqrt.reshape(
        n_function,
        *output_shape[1:],
        rank,
    )

    start = time.time()
    M = functools.reduce(
        lambda params1, params2: jax.tree.map(operator.add, params1, params2),
        (
            _M_batch(model_fn, params, x_c, k_c)
            for x_c, k_c in zip(
                jnp.split(x_context, n_chunks_eff, axis=0),
                jnp.split(k_inv_sqrt_dense, n_chunks_eff, axis=0),
                strict=False,
            )
        ),
    )
    logger.info(f"Time for M_batch: {time.time() - start:.2f} seconds")

    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Calculate the SVD of M
    _u, _s, _ = jnp.linalg.svd(M_flat, full_matrices=False)
    tol = jnp.finfo(M_flat.dtype).eps ** 2
    s = _s[_s > tol]
    _u = _u[:, : s.size]

    u = unflatten(_u)
    ggn_mv = create_ggn_pytree_mv(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        hessian_diag=True,
    )

    uTggnu = ggn_mv(u)

    # Compute U_A, D_A
    eigvals, eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + uTggnu)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    prior_var_sum = jnp.sum(prior_variance)

    # Compute S: $S = U_M U_A D_A^\dagger$
    cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def jvp(x, v):
        return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

    def scan_fn(carry, i):
        running_sum, truncation_idx = carry
        lr_fac = unravel_fn(cov_sqrt[:, i])
        sqrt_jvp = jax.vmap(lambda xc: jvp(xc, lr_fac) ** 2)(x_context)
        pv = jnp.sum(sqrt_jvp)
        new_running_sum = running_sum + pv
        new_truncation_idx = jax.lax.cond(
            (new_running_sum >= prior_var_sum) & (truncation_idx == -1),
            lambda _: i + 1,
            lambda _: truncation_idx,
            operand=None,
        )

        return (new_running_sum, new_truncation_idx), sqrt_jvp

    init_carry = (0.0, -1)
    indices = jnp.arange(eigvals.shape[0])
    (_, truncation_idx), _post_var = jax.lax.scan(scan_fn, init_carry, indices)

    truncation_idx = jax.lax.cond(
        truncation_idx == -1,
        lambda _: eigvals.shape[0],
        lambda _: truncation_idx,
        operand=None,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    parameter_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"FSP posterior using {truncation_idx} / {parameter_count} components.")
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            (state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x)))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=cov_sqrt.shape[-1],
        low_rank_terms=low_rank_terms,
    )

    logger.info(
        f"Created FSP posterior with rank {posterior.rank} and shape {M_flat.shape}"
    )

    return posterior


def create_fsp_posterior_none(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel: Callable,
    prior_variance: jnp.ndarray,
    n_chunks: int,
    *,
    max_iter: int | None = None,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with unstructured prior (full covariance Lanczos).

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    kernel : Callable
        Full kernel function
    prior_variance : jnp.ndarray
        Prior variance
    n_chunks : int
        Number of chunks for processing
    max_iter : int | None
        Maximum Lanczos iterations

    Returns
    -------
    Posterior
        FSP posterior approximation
    """
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    dim = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # Initialize with ones (simple initialization for unstructured case)
    ones_pytree = jax.tree.map(lambda x: jnp.ones_like(x), params)
    model_jvp = jax.vmap(
        lambda x: jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1],
        in_axes=0,
        out_axes=0,
    )
    b = model_jvp(x_context)
    initial_vector = b.flatten() / jnp.linalg.norm(b.flatten())

    # Compute Lanczos inverse sqrt for full covariance
    k_inv_sqrt = _lanczos_none_structure(kernel, initial_vector, max_iter)

    rank = k_inv_sqrt.shape[-1]
    k_inv_sqrt_dense = k_inv_sqrt.reshape(*output_shape, rank)

    start = time.time()
    M = functools.reduce(
        lambda params1, params2: jax.tree.map(operator.add, params1, params2),
        (
            _M_batch(model_fn, params, x_c, k_c)
            for x_c, k_c in zip(
                jnp.split(x_context, n_chunks_eff, axis=0),
                jnp.split(k_inv_sqrt_dense, n_chunks_eff, axis=0),
            )
        ),
    )
    logger.info(f"Time for M_batch: {time.time() - start:.2f} seconds")

    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Calculate the SVD of M
    _u, _s, _ = jnp.linalg.svd(M_flat, full_matrices=False)
    tol = jnp.finfo(M_flat.dtype).eps ** 2
    s = _s[_s > tol]
    _u = _u[:, : s.size]

    u = unflatten(_u)
    ggn_mv = create_ggn_pytree_mv(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        hessian_diag=True,
    )

    uTggnu = ggn_mv(u)

    # Compute U_A, D_A
    eigvals, eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + uTggnu)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    prior_var_sum = jnp.sum(prior_variance)

    # Compute S
    cov_sqrt = _u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def jvp(x, v):
        return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

    def scan_fn(carry, i):
        running_sum, truncation_idx = carry
        lr_fac = unravel_fn(cov_sqrt[:, i])
        sqrt_jvp = jax.vmap(lambda xc: jvp(xc, lr_fac) ** 2)(x_context)
        pv = jnp.sum(sqrt_jvp)
        new_running_sum = running_sum + pv
        new_truncation_idx = jax.lax.cond(
            (new_running_sum >= prior_var_sum) & (truncation_idx == -1),
            lambda _: i + 1,
            lambda _: truncation_idx,
            operand=None,
        )

        return (new_running_sum, new_truncation_idx), sqrt_jvp

    init_carry = (0.0, -1)
    indices = jnp.arange(eigvals.shape[0])
    (_, truncation_idx), _post_var = jax.lax.scan(scan_fn, init_carry, indices)

    truncation_idx = jax.lax.cond(
        truncation_idx == -1,
        lambda _: eigvals.shape[0],
        lambda _: truncation_idx,
        operand=None,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    parameter_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(f"FSP posterior using {truncation_idx} / {parameter_count} components.")
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            (state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x)))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=cov_sqrt.shape[-1],
        low_rank_terms=low_rank_terms,
    )

    logger.info(
        f"Created FSP posterior with rank {posterior.rank} and shape {M_flat.shape}"
    )

    return posterior


def create_fsp_posterior(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel_structure: CovarianceStructure | str,
    n_chunks: int,
    *,
    kernel: Callable | None = None,
    spatial_kernels: list[Callable] | None = None,
    function_kernels: list[Callable] | None = None,
    prior_variance: jnp.ndarray | None = None,
    spatial_max_iters: list[int] | None = None,
    max_iter: int | None = None,
    **kwargs,
) -> Posterior:
    """Create FSP posterior with specified kernel structure.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    x_context : InputArray
        Context points
    kernel_structure : KernelStructure
        Type of kernel structure ('kronecker' or 'none')
    n_chunks : int
        Number of chunks for processing
    kernel : Callable | None
        Full kernel for 'none' structure
    spatial_kernels : list[Callable] | None
        Spatial kernels for 'kronecker' structure
    function_kernels : list[Callable] | None
        Function kernels for 'kronecker' structure
    prior_variance : jnp.ndarray | None
        Prior variance (computed if not provided)
    spatial_max_iters : list[int] | None
        Max iterations for spatial Lanczos
    max_iter : int | None
        Max iterations for full Lanczos

    Returns
    -------
    Posterior
        FSP posterior approximation

    Raises
    ------
    ValueError
        If required arguments for the specified kernel structure are missing
    """
    if (
        kernel_structure == CovarianceStructure.KRONECKER
        or str(kernel_structure) == "kronecker"
    ):
        if spatial_kernels is None or function_kernels is None:
            msg = (
                "spatial_kernels and function_kernels must be provided "
                "for Kronecker structure"
            )
            raise ValueError(msg)

        return create_fsp_posterior_kronecker(
            model_fn=model_fn,
            params=params,
            x_context=x_context,
            spatial_kernels=spatial_kernels,
            function_kernels=function_kernels,
            prior_variance=prior_variance,
            n_chunks=n_chunks,
            spatial_max_iters=spatial_max_iters,
            **kwargs,
        )

    elif (
        kernel_structure == CovarianceStructure.NONE or str(kernel_structure) == "none"
    ):
        if kernel is None:
            msg = "kernel must be provided for None structure"
            raise ValueError(msg)

        return create_fsp_posterior_none(
            model_fn=model_fn,
            params=params,
            x_context=x_context,
            kernel=kernel,
            prior_variance=prior_variance,
            n_chunks=n_chunks,
            max_iter=max_iter,
            **kwargs,
        )

    else:
        msg = f"Unknown kernel structure: {kernel_structure}"
        raise ValueError(msg)


# Public mapping similar to CURVATURE_PRECISION_METHODS
COVARIANCE_STRUCTURE_METHODS: dict[CovarianceStructure | str, Callable] = {
    CovarianceStructure.KRONECKER: create_fsp_posterior_kronecker,
    CovarianceStructure.NONE: create_fsp_posterior_none,
}
