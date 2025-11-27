# fsp.py

"""FSP (Function-Space Prior) inference module.

Supports different kernel structures.
"""

from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp

from laplax.curv.cov import Posterior, PosteriorState
from laplax.curv.ggn import create_ggn_mv_without_data
from laplax.curv.lanczos import lanczos_invert_sqrt
from laplax.curv.utils import (
    LowRankTerms,
    create_model_jvp,
)
from laplax.enums import CovarianceStructure, LossFn
from laplax.types import (
    Callable,
    InputArray,
    Int,
    ModelFn,
    Params,
    PredArray,
)
from laplax.util import mv as util_mv
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
)
from laplax.util.tree import ones_like

KernelStructure = CovarianceStructure


def compute_inverse_sqrt_covariance(
    *,
    kernel_mv: Callable,
    structure: KernelStructure,
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    output_shape: tuple[int, ...],
    n_chunks: int,
    lanczos_max_iter: int | None = None,
) -> tuple[jnp.ndarray, int]:
    """Compute dense inverse sqrt factors for a prior covariance operator.

    Currently only the unstructured (``KernelStructure.NONE``) path is implemented.
    For that case, the provided ``kernel_mv`` is expected to either be a callable
    implementing a matrix-vector product ``v -> K v`` or the dense covariance
    matrix itself. The result is reshaped back to match the model output layout
    so it can be consumed by the existing FSP routines.

    Returns:
    tuple[jnp.ndarray, int]
        Dense inverse sqrt factor reshaped to match the model outputs as well as
        the effective rank returned by the Lanczos routine.

    Raises:
        ValueError: If a kernel structure other than ``KernelStructure.NONE`` is
            requested.
    """
    if structure != KernelStructure.NONE:
        msg = (
            "compute_inverse_sqrt_covariance currently supports only the "
            "'none' kernel structure."
        )
        raise ValueError(msg)

    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    init_vec = _lanczos_init_full(model_fn, params, x_context, num_chunks=n_chunks_eff)
    norm = jnp.linalg.norm(init_vec)
    init_vec = jnp.where(norm > 0, init_vec / norm, init_vec)

    inverse_sqrt = create_lanczos_factor_unstructured(
        kernel_mv, init_vec, max_iter=lanczos_max_iter
    )
    rank = int(inverse_sqrt.shape[-1])

    reshaped = inverse_sqrt.reshape(
        n_functions,
        *output_shape[1:],
        rank,
    )
    return reshaped, rank


# ==============================================================================
# Helper functions
# ==============================================================================


def _truncated_left_svd(M_flat: jnp.ndarray):
    """Compute truncated left SVD (U, s) of a matrix efficiently.

    For speed, uses an eigen decomposition of the smaller Gram matrix
    and reconstructs the left singular vectors when advantageous.

    Returns:
    -------
    tuple (U, s)
        U: left singular vectors with only columns above tolerance
        s: corresponding singular values (descending)
    """
    d, r = M_flat.shape
    tol = jnp.finfo(M_flat.dtype).eps ** 2

    if d > r:
        gram = M_flat.T @ M_flat  # (r, r)
        eigvals, V = jnp.linalg.eigh(gram)
        order = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V = V[:, order]
        s_all = jnp.sqrt(jnp.clip(eigvals, 0.0))
        mask = s_all > tol
        s = s_all[mask]
        if s.size == 0:
            return jnp.zeros((d, 0), dtype=M_flat.dtype), s
        V = V[:, : s.size]
        U = M_flat @ V
        U = U / s  # column-wise divide via broadcasting
        return U, s
    gram = M_flat @ M_flat.T  # (d, d)
    eigvals, U = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = U[:, order]
    s_all = jnp.sqrt(jnp.clip(eigvals, 0.0))
    mask = s_all > tol
    s = s_all[mask]
    if s.size == 0:
        return jnp.zeros((d, 0), dtype=M_flat.dtype), s
    U = U[:, : s.size]
    return U, s


def _compute_fsp_ggn_gram_with_dataloader(
    model_fn: ModelFn,
    params: Params,
    dataloader: Iterable,
    U: Params,
    *,
    loss_fn: LossFn | str | None = None,
    is_classification: bool = False,
    batch_size: int = 1,
) -> jax.Array:
    """Compute U^T G U over the FULL training dataloader.

    This is the correct implementation matching inference.py where GGN
    is accumulated over all training data, not just context points.

    Args:
        model_fn: Model function
        params: Model parameters
        dataloader: Full training dataloader (yields {"input": x, "target": y})
        U: Low-rank factors (pytree with trailing rank dimension)
        loss_fn: Loss function (defaults based on is_classification)
        is_classification: Whether this is a classification task
        batch_size: Batch size for lax.map operations

    Returns:
        Gram matrix U^T G U of shape (rank, rank)
    """
    if loss_fn is None:
        loss_fn = LossFn.CROSS_ENTROPY if is_classification else LossFn.NONE

    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn,
        params=params,
        loss_fn=loss_fn,
        factor=1.0,
        vmap_over_data=True,
        fsp=True,
        batch_size=batch_size,
    )
    ggn_mv = jax.jit(ggn_mv)

    # Get rank from U
    flatten, _ = create_partial_pytree_flattener(U)
    U_flat = flatten(U)
    rank = U_flat.shape[-1]

    # Accumulate Gram matrix over all batches
    gram = jnp.zeros((rank, rank), dtype=U_flat.dtype)

    for batch in dataloader:
        # Handle both dict and tuple batch formats
        if isinstance(batch, dict):
            x_batch = batch["input"]
            y_batch = batch["target"]
        else:
            x_batch, y_batch = batch[0], batch[1]

        batch_gram = ggn_mv(U, {"input": x_batch, "target": y_batch})
        gram = gram + batch_gram

    return gram


def compute_posterior_truncation_index_correct(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    cov_sqrt: jnp.ndarray,
    prior_variance: jnp.ndarray,
    batch_size: int = 1,
) -> int:
    """Compute truncation index using correct element-wise comparison.

    The posterior variance must satisfy:
        diag(J S_{:,:k} S_{:,:k}^T J^T) <= diag(Σ_prior) for all context points

    We find the largest k such that this holds for ALL elements.

    Args:
        model_fn: Model function
        params: Model parameters
        x_context: Context points
        cov_sqrt: Posterior covariance sqrt factor (P x rank)
        prior_variance: Prior variance at context points
        batch_size: Batch size for lax.map

    Returns:
        Truncation index (number of columns to keep)
    """
    n_functions = x_context.shape[0]
    n_outputs = int(jnp.prod(jnp.array(prior_variance.shape[1:])))
    prior_var_flat = prior_variance.reshape(n_functions, n_outputs)

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def jvp_fn(x, v):
        return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

    def compute_cov_sqrt_col(i):
        return jax.lax.dynamic_slice(cov_sqrt, (0, i), (cov_sqrt.shape[0], 1))

    def scan_fn(carry, i):
        post_var, truncation_idx = carry

        # Get column i of covariance sqrt
        cov_sqrt_col = compute_cov_sqrt_col(i)
        cov_sqrt_col_flat = cov_sqrt_col.reshape(-1)
        lr_fac = unravel_fn(cov_sqrt_col_flat)

        # Compute J @ col and square it (variance contribution)
        # Use lax.map for memory efficiency
        def compute_jvp_squared(xc):
            return jvp_fn(xc, lr_fac) ** 2

        sqrt_jvp = jax.lax.map(compute_jvp_squared, x_context, batch_size=batch_size)
        sqrt_jvp_flat = sqrt_jvp.reshape(n_functions, n_outputs)

        # Accumulate posterior variance
        new_post_var = post_var + sqrt_jvp_flat

        # Check if ALL posterior variances are still <= prior variances
        all_less = jnp.all(new_post_var <= prior_var_flat)

        # Record first index where constraint is violated
        new_truncation_idx = jax.lax.cond(
            (~all_less) & (truncation_idx == -1),
            lambda _: i,
            lambda _: truncation_idx,
            operand=None,
        )

        return (new_post_var, new_truncation_idx), None

    # Initialize with zero variance and -1 (no truncation yet)
    init_carry = (jnp.zeros((n_functions, n_outputs), dtype=cov_sqrt.dtype), -1)
    indices = jnp.arange(cov_sqrt.shape[1])

    (_, truncation_idx), _ = jax.lax.scan(scan_fn, init_carry, indices)

    # If no truncation needed, use all columns
    truncation_idx = jax.lax.cond(
        truncation_idx == -1,
        lambda _: cov_sqrt.shape[1],
        lambda _: truncation_idx,
        operand=None,
    )

    return int(truncation_idx)


def _accumulate_M_over_chunks(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    k_inv_sqrt_dense: jnp.ndarray,
    n_chunks: int,
    *,
    mode: str = "map",
    batch_size: int = 1,
) -> Params:
    r"""Accumulate _M_batch over context chunks using a chosen reduction mode.

    Modes:
        - "vmap": vectorized map over stacked chunks, then sum the results
        - "map": lax.map over chunk pairs, then sum the results
        - "scan": lax.scan accumulating the sum (default, typically fastest/memory-lean)

    Returns:
        Params: Pytree representing the accumulated ``M`` over all chunks.

    Raises:
        ValueError: If ``mode`` is not one of ``\"vmap\"``, ``\"map\"``,
            or ``\"scan\"``.
    """
    x_chunks = jnp.split(x_context, n_chunks, axis=0)
    k_chunks = jnp.split(k_inv_sqrt_dense, n_chunks, axis=0)

    if mode == "vmap":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        M_chunks = jax.vmap(
            partial(_M_batch, model_fn, params, batch_size=batch_size),
            in_axes=(0, 0),
        )(x_stacked, k_stacked)
        return jax.tree.map(lambda x: x.sum(axis=0), M_chunks)

    if mode == "map":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        pairs = (x_stacked, k_stacked)
        M_chunks = jax.lax.map(
            lambda pair: _M_batch(
                model_fn, params, pair[0], pair[1], batch_size=batch_size
            ),
            pairs,
        )
        return jax.tree.map(lambda x: x.sum(axis=0), M_chunks)

    if mode == "scan":
        x_stacked = jnp.stack(x_chunks)
        k_stacked = jnp.stack(k_chunks)
        init_M = _M_batch(
            model_fn, params, x_stacked[0], k_stacked[0], batch_size=batch_size
        )

        def scan_fn(carry, pair):
            x_c, k_c = pair
            M_chunk = _M_batch(model_fn, params, x_c, k_c, batch_size=batch_size)
            return jax.tree.map(jnp.add, carry, M_chunk), None

        M, _ = jax.lax.scan(scan_fn, init_M, (x_stacked[1:], k_stacked[1:]))
        return M

    msg = f"Unknown chunk accumulation mode: {mode}"
    raise ValueError(msg)


def _accumulate_M_over_kron_streaming(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    *,
    mv_kron: Callable[[jnp.ndarray], jnp.ndarray],
    rank: int,
    out_shape: tuple[int, ...],
    n_chunks: int,
    batch_size: int = 1,
) -> Params:
    """Accumulate M streaming columns from a Kronecker MVP without densifying.

    For each column j in 0..rank-1, compute ``vs = reshape(mv_kron(e_j),
    out_shape)``, then sum VJPs over function chunks. Stacks results across the
    last axis.

    Returns:
        Params: Pytree whose leaves contain stacked columns of ``M`` along the
            last axis.
    """
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1
    chunk_size = n_functions // n_chunks_eff

    def grad_for_chunk(xs_chunk, vs_chunk):
        vjp_res = _model_vjp(
            model_fn, params, xs_chunk, vs_chunk, batch_size=batch_size
        )
        return jax.tree.map(lambda p: jnp.sum(p, axis=0), vjp_res)

    cols: list[Params] = []
    for j in range(int(rank)):
        e_j = jnp.zeros((int(rank),), dtype=x_context.dtype).at[j].set(1.0)
        col_full = mv_kron(e_j)
        vs_full = col_full.reshape(out_shape)

        acc: Params | None = None
        for c in range(n_chunks_eff):
            s = c * chunk_size
            e = (c + 1) * chunk_size
            g = grad_for_chunk(x_context[s:e], vs_full[s:e])
            acc = g if acc is None else jax.tree.map(jnp.add, acc, g)
        if acc is not None:
            cols.append(acc)

    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=-1), *cols)


@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def _model_jvp(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    tangent: Params,
    batch_size: int = 1,
) -> PredArray:
    """Memory-efficient JVP using lax.map instead of vmap.

    Args:
        model_fn: Model function
        params: Model parameters
        xs: Input data of shape (B, ...)
        tangent: Tangent vector (pytree matching params structure)
        batch_size: Batch size for lax.map (use 1 for minimal memory)

    Returns:
        Batch of Jacobian-vector products of shape (B, ...)
    """

    def single_jvp(x):
        return jax.jvp(lambda w: model_fn(x, w), (params,), (tangent,))[1]

    return jax.lax.map(single_jvp, xs, batch_size=batch_size)


@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def _model_vjp(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    vs: PredArray,
    batch_size: int = 1,
) -> Params:
    """Memory-efficient VJP using lax.map instead of vmap.

    Args:
        model_fn: Model function
        params: Model parameters
        xs: Input data of shape (B, ...)
        vs: Output vectors of shape (B, *output_shape)
        batch_size: Batch size for lax.map (use 1 for minimal memory)

    Returns:
        VJP result as pytree matching params structure
    """

    def single_vjp(args):
        x, v = args
        return jax.vjp(lambda w: model_fn(x, w), params)[1](v)[0]

    return jax.lax.map(single_vjp, (xs, vs), batch_size=batch_size)


@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def _M_batch(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    L: PredArray,
    batch_size: int = 1,
) -> Params:
    """Compute M = sum_i J(x_i)^T L_i using lax.map for memory efficiency.

    L has shape (n_batch, *output_shape, rank)

    Args:
        model_fn: Model function
        params: Model parameters
        xs: Input data
        L: Matrix to multiply with Jacobian, shape (n_batch, *output_shape, rank)
        batch_size: Batch size for lax.map (use 1 for minimal memory)

    Returns:
        Pytree with batched matrix-Jacobian product
    """

    def process_single_column(L_col):
        # L_col has shape (n_batch, *output_shape)
        vjp_result = _model_vjp(model_fn, params, xs, L_col, batch_size=batch_size)
        return jax.tree.map(lambda p: jnp.sum(p, axis=0), vjp_result)

    L_transposed = jnp.moveaxis(L, -1, 0)
    result = jax.lax.map(process_single_column, L_transposed, batch_size=batch_size)

    return jax.tree.map(lambda x: jnp.moveaxis(x, 0, -1), result)


def compute_posterior_components(
    M_flat: jnp.ndarray,
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    prior_variance: jnp.ndarray,
    dataloader: Iterable,
    *,
    is_classification: bool = False,
    regression_noise_scale: float | None = None,
    batch_size: int = 1,
) -> tuple[jnp.ndarray, int]:
    """Compute posterior covariance sqrt and truncation index.

    Returns:
        tuple[jnp.ndarray, int]: A pair ``(cov_sqrt, truncation_idx)`` where
            ``cov_sqrt`` is the posterior covariance square root and
            ``truncation_idx`` is the chosen truncation index.
    """
    del regression_noise_scale
    # Truncated SVD
    u_, s = _truncated_left_svd(M_flat)

    # Unflatten for GGN computation
    _flatten, unflatten = create_partial_pytree_flattener(params)
    u = unflatten(u_)

    # Efficient GGN quadratic form U^T G U over full dataloader
    uTggnu = _compute_fsp_ggn_gram_with_dataloader(
        model_fn=model_fn,
        params=params,
        dataloader=dataloader,
        U=u,
        is_classification=is_classification,
        batch_size=batch_size,
    )

    # Eigendecomposition of A = M^T M + GGN
    A_eigh = jnp.diag(s**2) + uTggnu
    eigvals, eigvecs = jnp.linalg.eigh(A_eigh)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Compute posterior covariance sqrt
    cov_sqrt = u_ @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    # Compute truncation index with correct element-wise comparison
    truncation_idx = compute_posterior_truncation_index_correct(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        cov_sqrt=cov_sqrt,
        prior_variance=prior_variance,
        batch_size=batch_size,
    )

    return cov_sqrt, truncation_idx


def create_lanczos_factors_kronecker(
    spatial_kernels: list[Callable],
    function_kernels: list[Callable],
    initial_vectors_spatial: list[jnp.ndarray],
    initial_vectors_function: list[jnp.ndarray],
    spatial_max_iters: list[int] | None = None,
    function_max_iters: list[int] | None = None,
) -> list[jnp.ndarray]:
    """Create Lanczos inverse sqrt factors for Kronecker structure.

    Returns:
        list[jnp.ndarray]: List of inverse square-root factors corresponding to
            the spatial and function kernels.
    """
    if spatial_max_iters is None:
        spatial_max_iters = [None] * len(spatial_kernels)
    if function_max_iters is None:
        function_max_iters = [None] * len(function_kernels)

    factors = []

    # Spatial factors
    for kernel, init_vec, max_iter in zip(
        spatial_kernels, initial_vectors_spatial, spatial_max_iters, strict=False
    ):
        kwargs = {"max_iter": max_iter} if max_iter else {}
        factors.append(lanczos_invert_sqrt(kernel, init_vec, **kwargs))

    # Function factors
    for kernel, init_vec, max_iter in zip(
        function_kernels, initial_vectors_function, function_max_iters, strict=False
    ):
        kwargs = {"max_iter": max_iter} if max_iter else {}
        factors.append(lanczos_invert_sqrt(kernel, init_vec, **kwargs))

    return factors


def create_lanczos_factor_unstructured(
    kernel: Callable,
    initial_vector: jnp.ndarray,
    max_iter: int | None = None,
) -> jnp.ndarray:
    """Create Lanczos inverse sqrt for an unstructured kernel.

    Returns:
        jnp.ndarray: Inverse square-root factor for the given kernel.
    """
    kwargs = {"max_iter": max_iter} if max_iter else {}
    return lanczos_invert_sqrt(kernel, initial_vector, **kwargs)


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def _lanczos_init_kron(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    num_chunks: Int,
):
    """Initialize Lanczos vectors for Kronecker structure using HOSVD.

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

    Returns:
    -------
    tuple
        (initial_vectors_function, initial_vectors_spatial)
    """  # noqa: DOC501
    if xs.ndim < 4:
        msg = f"Input must have shape (B, S1, S2, ..., C), but got {xs.shape}"
        raise ValueError(msg)

    ones_pytree = jax.tree.map(jnp.ones_like, params)

    def model_jvp_single(x):
        return jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1]

    batches = jnp.split(xs, num_chunks, axis=0)
    b = jnp.concatenate(
        [jax.lax.map(model_jvp_single, batch, batch_size=1) for batch in batches],
        axis=0,
    )

    spatial_dims = tuple(s for s in xs.shape[1:-2] if s > 1)
    n_function = xs.shape[0]

    # Collapse any non-spatial output axes (e.g., time/channel) by averaging,
    # so we retain only (n_function, *spatial_dims) for HOSVD initialization.
    if b.ndim > 1 + len(spatial_dims):
        reduce_axes = tuple(range(1 + len(spatial_dims), b.ndim))
        b = jnp.mean(b, axis=reduce_axes)

    b = b.reshape((n_function, *spatial_dims))

    initial_vectors = []

    for mode in range(len(b.shape)):
        n_mode = b.shape[mode]
        b_unfolded = jnp.moveaxis(b, mode, 0).reshape(n_mode, -1)

        u, _s, _v = jnp.linalg.svd(b_unfolded, full_matrices=False)
        vec = u[:, 0] / jnp.linalg.norm(u[:, 0])
        initial_vectors.append(vec)

    initial_vectors_function = [initial_vectors[0]]
    initial_vectors_spatial = initial_vectors[1:]

    return initial_vectors_function, initial_vectors_spatial


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def _lanczos_init_full(
    model_fn: ModelFn,
    params: Params,
    xs: InputArray,
    num_chunks: Int,
) -> jnp.ndarray:
    """Compute a dense initialization vector for unstructured kernels.

    Returns:
    jnp.ndarray
        Flattened Jacobian-vector product used as the starting vector.
    """
    ones_pytree = jax.tree.map(jnp.ones_like, params)

    def model_jvp_single(x):
        return jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1]

    # Use lax.map for memory efficiency
    batches = jnp.split(xs, num_chunks, axis=0)
    b = jnp.concatenate(
        [jax.lax.map(model_jvp_single, batch, batch_size=1) for batch in batches],
        axis=0,
    )
    return b.reshape(-1)


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

    Returns:
    -------
    list
        List of Lanczos inverse sqrt factors for each dimension
    """
    if max_iters is None:
        max_iters = [None] * len(kernels_list)

    lanczos_results = []
    for kernel, init_vec, max_iter in zip(
        kernels_list, initial_vectors, max_iters, strict=False
    ):
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

    Returns:
    -------
    Array
        Lanczos inverse sqrt factor
    """
    kwargs = {}
    if max_iter is not None:
        kwargs["max_iter"] = max_iter

    return lanczos_invert_sqrt(kernel, initial_vector, **kwargs)


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
    dataloader: Iterable,
    *,
    spatial_max_iters: list[int] | None = None,
    is_classification: bool = False,
    chunk_mode: str = "scan",
    kron_mode: str = "dense",  # 'dense' or 'streaming'
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    batch_size: int = 1,
    **kwargs,
) -> Posterior:
    del regression_noise_scale, ggn_col_chunk_size
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

    Returns:
    -------
    Posterior
        FSP posterior approximation
    """
    del kwargs
    if spatial_max_iters is None:
        spatial_max_iters = [8, 3]
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count to evenly divide number of functions
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))

    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    sum(x.size for x in jax.tree_util.tree_leaves(params))

    initial_vectors_function, initial_vectors_spatial = _lanczos_init_kron(
        model_fn, params, x_context, num_chunks=n_chunks_eff
    )

    spatial_lanczos_results = _lanczos_kronecker_structure(
        spatial_kernels, initial_vectors_spatial, spatial_max_iters
    )

    function_lanczos_results = _lanczos_kronecker_structure(
        function_kernels, initial_vectors_function, max_iters=None
    )

    if kron_mode == "dense":
        all_factors = spatial_lanczos_results + function_lanczos_results
        k_inv_sqrt_dense = all_factors[0]
        for factor in all_factors[1:]:
            k_inv_sqrt_dense = jnp.kron(k_inv_sqrt_dense, factor)

        rank = k_inv_sqrt_dense.shape[-1]
        k_inv_sqrt_dense = k_inv_sqrt_dense.reshape(
            n_functions,
            *output_shape[1:],
            rank,
        )

        M = _accumulate_M_over_chunks(
            model_fn,
            params,
            x_context,
            k_inv_sqrt_dense,
            n_chunks_eff,
            mode=chunk_mode,
            batch_size=batch_size,
        )
    else:

        def make_mv(matrix):
            return lambda v: matrix @ v

        all_factors = spatial_lanczos_results + function_lanczos_results
        all_mvs = [make_mv(factor) for factor in all_factors]
        all_layouts = [factor.shape[1] for factor in all_factors]

        k_inv_sqrt_mv = util_mv.kronecker_product_factors(all_mvs, all_layouts)
        total_rank = int(jnp.prod(jnp.array(all_layouts)))
        out_shape = (n_functions, *tuple(int(s) for s in output_shape[1:]))

        M = _accumulate_M_over_kron_streaming(
            model_fn,
            params,
            x_context,
            mv_kron=k_inv_sqrt_mv,
            rank=total_rank,
            out_shape=out_shape,
            n_chunks=n_chunks_eff,
            batch_size=batch_size,
        )

    # Flatten M
    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Truncated left SVD
    u_, s = _truncated_left_svd(M_flat)

    # Unflatten U to pytree (with trailing rank dim k)
    u = unflatten(u_)
    uTggnu = _compute_fsp_ggn_gram_with_dataloader(
        model_fn=model_fn,
        params=params,
        dataloader=dataloader,
        U=u,
        is_classification=is_classification,
        batch_size=batch_size,
    )

    # Compute U_A, D_A
    A_eigh = jnp.diag(s**2) + uTggnu
    eigvals, eigvecs = jnp.linalg.eigh(A_eigh)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Compute S: $S = U_M U_A D_A^\dagger$
    cov_sqrt = u_ @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    truncation_idx = compute_posterior_truncation_index_correct(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        cov_sqrt=cov_sqrt,
        prior_variance=prior_variance,
        batch_size=batch_size,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    # Create flatten/unflatten for posterior
    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=posterior_state["scale_sqrt"].shape[-1],
        low_rank_terms=low_rank_terms,
    )

    return posterior


def create_fsp_posterior_none(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel: Callable,
    prior_variance: jnp.ndarray,
    n_chunks: int,
    dataloader: Iterable,
    *,
    max_iter: int | None = None,
    is_classification: bool = False,
    independent_outputs: bool = False,
    kernels_per_output: list[Callable] | None = None,
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    batch_size: int = 1,
    **kwargs,
) -> Posterior:
    del regression_noise_scale, ggn_col_chunk_size
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

    Returns:
    -------
    Posterior
        FSP posterior approximation
    """
    del kwargs
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape

    # Adjust chunk count
    n_functions = int(x_context.shape[0])
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    sum(x.size for x in jax.tree_util.tree_leaves(params))

    # Compute Lanczos inverse sqrt for the prior kernel
    if independent_outputs or kernels_per_output is not None:
        ones_pytree = ones_like(params)
        model_jvp = create_model_jvp(
            params, ones_pytree, model_fn, in_axes=0, out_axes=0
        )
        b = model_jvp(x_context)
        # Build block-diagonal K^{-1/2} across output channels using per-output kernels
        output_dim = 1 if b.ndim == 1 else int(b.shape[-1])
        per_output_cols = []
        for k in range(output_dim):
            b_k = b if output_dim == 1 else b[:, k]
            b_k = b_k.reshape(-1)
            init_k = b_k / (jnp.linalg.norm(b_k) + 1e-12)
            kernel_k = (
                kernels_per_output[k]
                if kernels_per_output is not None and k < len(kernels_per_output)
                else kernel
            )
            k_inv_sqrt_k = _lanczos_none_structure(kernel_k, init_k, max_iter)
            per_output_cols.append(k_inv_sqrt_k)

        ranks = [c.shape[-1] for c in per_output_cols]
        total_rank = int(sum(ranks))
        B = int(x_context.shape[0])
        out_dim = 1 if b.ndim == 1 else int(b.shape[-1])
        k_inv_sqrt_dense = jnp.zeros((B, out_dim, total_rank), dtype=b.dtype)
        offset = 0
        for k, cols in enumerate(per_output_cols):
            r = int(cols.shape[-1])
            k_inv_sqrt_dense = k_inv_sqrt_dense.at[:, k, offset : offset + r].set(
                cols.reshape(B, r)
            )
            offset += r
    else:
        k_inv_sqrt_dense, _ = compute_inverse_sqrt_covariance(
            kernel_mv=kernel,
            structure=KernelStructure.NONE,
            model_fn=model_fn,
            params=params,
            x_context=x_context,
            output_shape=output_shape,
            n_chunks=n_chunks_eff,
            lanczos_max_iter=max_iter,
        )

    M = _accumulate_M_over_chunks(
        model_fn,
        params,
        x_context,
        k_inv_sqrt_dense,
        n_chunks_eff,
        mode="map",  # Use map for memory efficiency (lax.map over chunks)
        batch_size=batch_size,
    )

    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # Truncated left SVD
    u_, s = _truncated_left_svd(M_flat)

    u = unflatten(u_)
    uTggnu = _compute_fsp_ggn_gram_with_dataloader(
        model_fn=model_fn,
        params=params,
        dataloader=dataloader,
        U=u,
        is_classification=is_classification,
        batch_size=batch_size,
    )

    # Compute U_A, D_A
    A_eigh = jnp.diag(s**2) + uTggnu
    eigvals, eigvecs = jnp.linalg.eigh(A_eigh)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Compute S
    cov_sqrt = u_ @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    truncation_idx = compute_posterior_truncation_index_correct(
        model_fn=model_fn,
        params=params,
        x_context=x_context,
        cov_sqrt=cov_sqrt,
        prior_variance=prior_variance,
        batch_size=batch_size,
    )

    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt[:, :truncation_idx]}
    U, S, _ = jnp.linalg.svd(posterior_state["scale_sqrt"], full_matrices=False)
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten_params, unflatten_params = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_params(
            state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_params(x))
        ),
        scale_mv=lambda state: lambda x: unflatten_params(state["scale_sqrt"] @ x),
        rank=posterior_state["scale_sqrt"].shape[-1],
        low_rank_terms=low_rank_terms,
    )

    return posterior


def create_fsp_posterior(
    model_fn: ModelFn,
    params: Params,
    x_context: InputArray,
    kernel_structure: CovarianceStructure | str,
    n_chunks: int,
    dataloader: Iterable,
    *,
    kernel: Callable | None = None,
    spatial_kernels: list[Callable] | None = None,
    function_kernels: list[Callable] | None = None,
    prior_variance: jnp.ndarray | None = None,
    spatial_max_iters: list[int] | None = None,
    max_iter: int | None = None,
    is_classification: bool = False,
    independent_outputs: bool = False,
    kernels_per_output: list[Callable] | None = None,
    regression_noise_scale: float | None = None,
    ggn_col_chunk_size: int = 64,
    batch_size: int = 1,
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

    Returns:
    -------
    Posterior
        FSP posterior approximation

    Raises:
    ------
    ValueError
        If required arguments for the specified kernel structure are missing
    """  # noqa: DOC501
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
            dataloader=dataloader,
            spatial_max_iters=spatial_max_iters,
            is_classification=is_classification,
            regression_noise_scale=regression_noise_scale,
            ggn_col_chunk_size=ggn_col_chunk_size,
            batch_size=batch_size,
            **kwargs,
        )

    if kernel_structure == CovarianceStructure.NONE or str(kernel_structure) == "none":
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
            dataloader=dataloader,
            max_iter=max_iter,
            is_classification=is_classification,
            independent_outputs=independent_outputs,
            kernels_per_output=kernels_per_output,
            regression_noise_scale=regression_noise_scale,
            ggn_col_chunk_size=ggn_col_chunk_size,
            batch_size=batch_size,
            **kwargs,
        )

    msg = f"Unknown kernel structure: {kernel_structure}"
    raise ValueError(msg)


# Public mapping similar to CURVATURE_PRECISION_METHODS
COVARIANCE_STRUCTURE_METHODS: dict[CovarianceStructure | str, Callable] = {
    CovarianceStructure.KRONECKER: create_fsp_posterior_kronecker,
    CovarianceStructure.NONE: create_fsp_posterior_none,
}
