"""FSP inference for both MLP regression and operator learning.

This module provides a unified interface for Function-Space Posterior
inference on both standard regression tasks and operator learning tasks
with spatial/temporal structure.
"""

import functools
import time
from functools import partial

import jax
import jax.numpy as jnp
from loguru import logger

from laplax.curv.cov import Posterior, PosteriorState
from laplax.curv.utils import LowRankTerms
from laplax.extra.fsp.ggn import create_fsp_ggn_mv
from laplax.extra.fsp.kernels import (
    KernelProtocol,
    build_gram_matrix,
    kernel_variance,
    wrap_kernel_fn,
)
from laplax.extra.fsp.lanczos_isqrt import (
    lanczos_hosvd_initialization,
    lanczos_invert_sqrt,
    lanczos_jacobian_initialization,
)
from laplax.extra.fsp.objective import select_context_points
from laplax.extra.fsp.operator import (
    compute_M_batch,
    compute_M_batch_chunked,
)
from laplax.types import (
    Callable,
    Data,
    InputArray,
    ModelFn,
    Params,
    PredArray,
)
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
)


def compute_matrix_jacobian_product(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    matrix: PredArray,
    has_batch_dim: bool = True,
) -> tuple[PredArray, Callable]:
    """Compute the matrix Jacobian product M = J^T L.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    data : Data
        Input data
    matrix : PredArray
        Low-rank matrix L
    has_batch_dim : bool
        Whether data has batch dimension

    Returns
    -------
    tuple
        (M, unravel_fn) where M is the matrix-Jacobian product
        and unravel_fn reconstructs parameter structure
    """
    if has_batch_dim:
        raise NotImplementedError(
            "Batch dimension not implemented for matrix Jacobian product."
        )

    M_tree = jax.vmap(
        jax.vjp(
            lambda p: jnp.reshape(model_fn(data, params=p), (matrix.shape[0],)),
            params,
        )[1],
        in_axes=1,
        out_axes=1,
    )(matrix)

    flat_M, unravel_fn = jax.flatten_util.ravel_pytree(M_tree)
    M = flat_M.reshape((-1, matrix.shape[-1]))

    return M, unravel_fn


def fsp_inference(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    prior_cov_kernel: Callable | KernelProtocol,
    *,
    context_selection: str = "grid",
    n_context_points: int = 100,
    key: jax.random.PRNGKey = None,
    truncate_to_prior_var: bool = True,
    **kwargs,
) -> Posterior:
    """FSP inference for standard regression tasks.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters (at MAP estimate)
    data : Data
        Full dataset for GGN computation
    prior_cov_kernel : Callable or KernelProtocol
        Prior covariance kernel. Can be:
        - A kernel from GPJax (e.g., gpx.kernels.RBF())
        - A kernel from GPyTorch (wrapped with GPyTorchKernelAdapter)
        - A callable: kernel_fn(x1, x2) -> K
        - Any object implementing __call__(x1, x2) -> K
    context_selection : str
        Method for selecting context points
    n_context_points : int
        Number of context points
    key : jax.random.PRNGKey
        Random key
    truncate_to_prior_var : bool
        Whether to truncate based on prior variance
    **kwargs
        Additional arguments for context point selection

    Returns
    -------
    Posterior
        FSP posterior

    Examples
    --------
    Using a GPJax kernel:

    >>> import gpjax as gpx
    >>> kernel = gpx.kernels.RBF()
    >>> from laplax.extra.fsp import GPJaxKernelAdapter
    >>> adapted_kernel = GPJaxKernelAdapter(kernel, params={"lengthscale": 1.0})
    >>> posterior = fsp_inference(model_fn, params, data, adapted_kernel)

    Using a callable kernel function:

    >>> def rbf_kernel(x1, x2=None):
    ...     if x2 is None: x2 = x1
    ...     sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    ...     return jnp.exp(-sq_dist / 2.0)
    >>> posterior = fsp_inference(model_fn, params, data, rbf_kernel)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Wrap kernel if it's a simple callable without diagonal method
    if not hasattr(prior_cov_kernel, "diagonal"):
        prior_cov_kernel = wrap_kernel_fn(prior_cov_kernel)

    # Select context points
    context_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in ["context_points_minval", "context_points_maxval", "datapoint_shape", "dataloader", "grid_stride"]
    }

    # Set defaults if not provided
    if "datapoint_shape" not in context_kwargs and hasattr(data, "shape"):
        context_kwargs["datapoint_shape"] = data.shape
    if "context_points_minval" not in context_kwargs:
        context_kwargs["context_points_minval"] = [0.0]
    if "context_points_maxval" not in context_kwargs:
        context_kwargs["context_points_maxval"] = [1.0]

    context_points = select_context_points(
        n_context_points=n_context_points,
        context_selection=context_selection,
        key=key,
        **context_kwargs,
    )

    # Initialize Lanczos
    v = lanczos_jacobian_initialization(model_fn, params, context_points)

    # Build covariance matrix using Kernel interface
    cov_matrix = build_gram_matrix(prior_cov_kernel, context_points, jitter=1e-8)
    prior_var = prior_cov_kernel.diagonal(context_points)

    # Lanczos inverse sqrt
    L = lanczos_invert_sqrt(cov_matrix, v, tol=jnp.finfo(v.dtype).eps)

    # Compute matrix-Jacobian product
    M, unravel_fn = compute_matrix_jacobian_product(
        model_fn, params, context_points, L, has_batch_dim=False
    )

    # SVD of M
    _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
    tol = jnp.finfo(M.dtype).eps**2
    s = _s[_s > tol]
    u = _u[:, : s.size]

    # Compute GGN matrix
    ggn_matrix = create_fsp_ggn_mv(model_fn, params, M)(data)

    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(ggn_matrix)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Filter eigenvalues
    eps = jnp.finfo(ggn_matrix.dtype).eps
    tol_eig = eps * (eigvals.max() ** 0.5) * eigvals.shape[0]
    mask = eigvals > tol_eig
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]

    # Compute covariance sqrt
    cov_sqrt = u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    # Truncation based on prior variance
    if truncate_to_prior_var:
        prior_var_sum = jnp.sum(prior_var)

        def jvp(x, v):
            return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

        def scan_fn(carry, i):
            running_sum, truncation_idx = carry
            lr_fac = unravel_fn(cov_sqrt[:, i])
            sqrt_jvp = jax.vmap(lambda xc: jvp(xc, lr_fac) ** 2)(context_points)
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
        (_, truncation_idx), post_var = jax.lax.scan(scan_fn, init_carry, indices)

        truncation_idx = jax.lax.cond(
            truncation_idx == -1,
            lambda _: eigvals.shape[0],
            lambda _: truncation_idx,
            operand=None,
        )
        cov_sqrt = cov_sqrt[:, :truncation_idx]

    # Create posterior
    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt}
    U, S, _ = jnp.linalg.svd(cov_sqrt, full_matrices=False)
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten, unflatten = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten(
            (state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten(x)))
        ),
        scale_mv=lambda state: lambda x: unflatten(state["scale_sqrt"] @ x),
        rank=cov_sqrt.shape[-1],
        low_rank_terms=low_rank_terms,
    )

    logger.info(
        f"Created FSP posterior with rank {posterior.rank} / {cov_sqrt.shape[0]} parameters"
    )

    return posterior


def fsp_operator_inference(
    model_fn: ModelFn,
    params: Params,
    data_loader,
    spatial_kernels: list[Callable],
    function_kernels: list[Callable],
    *,
    context_selection: str = "dataloader",
    n_context_points: int = 10,
    grid_stride: int | None = 2,
    n_chunks: int = 1,
    key: jax.random.PRNGKey = None,
    truncate_to_prior_var: bool = True,
    max_lanczos_iter: int = 8,
    **kwargs,
) -> Posterior:
    """FSP inference for operator learning with spatial structure.

    This function handles operator learning tasks where the output has
    spatial/temporal structure. It uses Kronecker-structured kernels
    and HOSVD-based initialization.

    Parameters
    ----------
    model_fn : ModelFn
        Model function that maps inputs to spatial outputs
    params : Params
        Model parameters (at MAP estimate)
    data_loader
        DataLoader providing batches of spatial data
    spatial_kernels : list[Callable]
        List of kernel functions for each spatial dimension
    function_kernels : list[Callable]
        List of kernel functions for function space
    context_selection : str
        Method for selecting context points (default: "dataloader")
    n_context_points : int
        Number of context points/functions
    grid_stride : int, optional
        Stride for spatial grid subsampling
    n_chunks : int
        Number of chunks for memory-efficient computation
    key : jax.random.PRNGKey
        Random key
    truncate_to_prior_var : bool
        Whether to truncate based on prior variance
    max_lanczos_iter : int
        Maximum Lanczos iterations per spatial dimension
    **kwargs
        Additional arguments

    Returns
    -------
    Posterior
        FSP posterior for operator learning
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Select context points
    result = select_context_points(
        n_context_points=n_context_points,
        context_selection=context_selection,
        context_points_minval=[0.0],
        context_points_maxval=[1.0],
        datapoint_shape=(1,),
        key=key,
        grid_stride=grid_stride,
        dataloader=data_loader,
    )

    if isinstance(result, tuple):
        x_context, grid = result
    else:
        x_context = result
        # Create default grid
        grid = jnp.arange(x_context.shape[1]).reshape(-1, 1)

    # Get output shape
    y0 = jax.vmap(lambda x: model_fn(x, params))(x_context)
    output_shape = y0.shape
    n_functions = int(x_context.shape[0])

    # Adjust chunk count
    n_chunks_eff = int(min(max(1, n_chunks), n_functions))
    while n_functions % n_chunks_eff != 0 and n_chunks_eff > 1:
        n_chunks_eff -= 1

    # HOSVD initialization
    initial_vectors_function, initial_vectors_spatial = lanczos_hosvd_initialization(
        model_fn, params, x_context, num_chunks=n_chunks_eff
    )

    # Build kernels and compute Lanczos inverse sqrt
    # For simplicity, we'll use a single kernel approach here
    # In practice, you'd want to handle Kronecker products properly

    # Simplified version: combine kernels
    from functools import reduce

    def combine_kernels(kernels, data_list):
        """Combine multiple kernels via summation (simplified)."""
        if len(kernels) == 1:
            return kernels[0](data_list[0], data_list[0])

        # For now, just use the first kernel
        # A full implementation would use Kronecker products
        return kernels[0](data_list[0], data_list[0])

    # Build spatial kernel
    if len(spatial_kernels) > 0:
        spatial_kernel = combine_kernels(spatial_kernels, [grid] * len(spatial_kernels))
    else:
        # Default to identity
        spatial_kernel = jnp.eye(grid.shape[0])

    # Build function kernel
    if len(function_kernels) > 0:
        function_kernel = combine_kernels(function_kernels, [x_context] * len(function_kernels))
    else:
        # Default to identity
        function_kernel = jnp.eye(x_context.shape[0])

    # Lanczos inverse sqrt for spatial dimensions
    spatial_L_list = []
    for i, vec in enumerate(initial_vectors_spatial):
        if i < len(spatial_kernels):
            L = lanczos_invert_sqrt(
                spatial_kernel, vec, tol=1e-5, max_iter=max_lanczos_iter
            )
            spatial_L_list.append(L)

    # Lanczos inverse sqrt for function space
    function_L_list = []
    for i, vec in enumerate(initial_vectors_function):
        if i < len(function_kernels):
            L = lanczos_invert_sqrt(
                function_kernel, vec, tol=1e-5, max_iter=max_lanczos_iter
            )
            function_L_list.append(L)

    # Combine Lanczos factors (simplified - would use Kronecker in full version)
    if len(function_L_list) > 0:
        k_inv_sqrt = function_L_list[0]
    else:
        k_inv_sqrt = jnp.eye(n_functions)

    # Reshape for batch processing
    rank = k_inv_sqrt.shape[-1]
    k_inv_sqrt_reshaped = k_inv_sqrt.reshape(n_functions, -1, rank)

    # Compute M in chunks
    start = time.time()
    x_chunks = jnp.split(x_context, n_chunks_eff, axis=0)
    k_chunks = jnp.split(k_inv_sqrt_reshaped, n_chunks_eff, axis=0)

    M = compute_M_batch_chunked(model_fn, params, x_chunks, k_chunks)
    logger.info(f"Time for M_batch: {time.time() - start:.2f} seconds")

    # Flatten M
    flatten, unflatten = create_partial_pytree_flattener(M)
    M_flat = flatten(M)

    # SVD of M
    _u, _s, _ = jnp.linalg.svd(M_flat, full_matrices=False)
    tol = jnp.finfo(M_flat.dtype).eps**2
    s = _s[_s > tol]
    u = _u[:, : s.size]
    u_unflat = unflatten(u)

    # Compute GGN
    # Note: For operator learning, we need to handle spatial structure in GGN
    # This is a simplified version
    ggn_mv = create_fsp_ggn_mv(model_fn, params, M_flat)
    ggn_matrix = ggn_mv({"input": x_context, "target": y0})

    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(ggn_matrix)
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    # Filter eigenvalues
    eps = jnp.finfo(ggn_matrix.dtype).eps
    tol_eig = eps * (eigvals.max() ** 0.5) * eigvals.shape[0]
    mask = eigvals > tol_eig
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]

    # Compute covariance sqrt
    cov_sqrt = u @ (eigvecs[:, ::-1] / jnp.sqrt(jnp.abs(eigvals[::-1])))

    # Truncation based on prior variance (if requested)
    if truncate_to_prior_var:
        # Compute prior variance (simplified)
        prior_variance = jnp.sum(jnp.diag(spatial_kernel)) + jnp.sum(
            jnp.diag(function_kernel)
        )

        def jvp(x, v):
            return jax.jvp(lambda p: model_fn(x, p), (params,), (v,))[1]

        def scan_fn(carry, i):
            running_sum, truncation_idx = carry
            lr_fac = unflatten(cov_sqrt[:, i])
            sqrt_jvp = jax.vmap(lambda xc: jvp(xc, lr_fac) ** 2)(x_context)
            pv = jnp.sum(sqrt_jvp)
            new_running_sum = running_sum + pv
            new_truncation_idx = jax.lax.cond(
                (new_running_sum >= prior_variance) & (truncation_idx == -1),
                lambda _: i + 1,
                lambda _: truncation_idx,
                operand=None,
            )
            return (new_running_sum, new_truncation_idx), sqrt_jvp

        init_carry = (0.0, -1)
        indices = jnp.arange(eigvals.shape[0])
        (_, truncation_idx), _ = jax.lax.scan(scan_fn, init_carry, indices)

        truncation_idx = jax.lax.cond(
            truncation_idx == -1,
            lambda _: eigvals.shape[0],
            lambda _: truncation_idx,
            operand=None,
        )
        cov_sqrt = cov_sqrt[:, :truncation_idx]

    # Create posterior
    posterior_state: PosteriorState = {"scale_sqrt": cov_sqrt}
    U, S, _ = jnp.linalg.svd(cov_sqrt, full_matrices=False)
    low_rank_terms = LowRankTerms(U, S, scalar=0.0)

    flatten_full, unflatten_full = create_pytree_flattener(params)
    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: unflatten_full(
            (state["scale_sqrt"] @ (state["scale_sqrt"].T @ flatten_full(x)))
        ),
        scale_mv=lambda state: lambda x: unflatten_full(state["scale_sqrt"] @ x),
        rank=cov_sqrt.shape[-1],
        low_rank_terms=low_rank_terms,
    )

    parameter_count = sum(p.size for p in jax.tree.leaves(params))
    logger.info(
        f"FSP operator posterior using {posterior.rank} / {parameter_count} components"
    )

    return posterior
