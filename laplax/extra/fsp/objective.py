import jax
from jax import numpy as jnp
from laplax.enums import LossFn
from laplax.types import Callable, Data, Float, Int, ModelFn, Params, PredArray


def create_loss_nll(
    model_fn: ModelFn,
    dataset_size: int | None = None,
):
    r"""Create the NLL loss function for FSP training.

    Computes the negative log-likelihood:
    $$
    -\log p(y | f(X)) = -\sum_i \log \mathcal{N}(y_i | f(x_i), \sigma^2)
    $$
    """

    def loss_nll(
        data: Data, params: Params, scale: Float | Params | None = None
    ) -> Float:
        preds = jax.vmap(model_fn, in_axes=(0, None))(data["input"], params)
        nll = -jax.scipy.stats.norm.logpdf(
            data["target"], loc=preds, scale=scale
        ).mean()
        return nll * dataset_size

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
    has_batch_dim: bool = True,
):
    r"""Create the FSP regularization loss function.

    Computes the RKHS regularization:
    $$
    \frac{1}{2} (f(c) - m)^T K^{-1}(c, c) (f(c) - m)
    $$
    """
    if not has_batch_dim:

        def loss_reg(context_points: PredArray, params: Params) -> Float:
            f_c = (
                jax.vmap(model_fn, in_axes=(0, None))(context_points, params)
                - prior_mean
            )
            K_c_c = prior_cov_kernel(*context_points)
            left = jax.scipy.linalg.solve(K_c_c, f_c, assume_a="sym")
            return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    if has_batch_dim:

        def loss_reg(context_points: PredArray, params: Params) -> Float:
            f_c = (
                jax.vmap(model_fn, in_axes=(0, None))(context_points[0], params)
                - prior_mean[None, :]
            )
            K_c_c = prior_cov_kernel(*context_points)
            left = jax.scipy.linalg.solve(K_c_c, f_c, assume_a="sym")
            return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    dataset_size: Int,
    prior_mean: PredArray,
    prior_cov_kernel: Callable,
):
    """Create FSP objective combining NLL and regularization losses."""
    loss_nll = create_loss_nll(model_fn, dataset_size)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    def fsp_objective(
        data: Data,
        context_points: PredArray,
        params: Params,
        scale: Float | Params | None = None,
    ) -> Float:
        nll_term = loss_nll(data, params, scale)
        reg_term = loss_reg(context_points, params)
        return nll_term + reg_term

    return fsp_objective


def select_context_points(
    n_context_points: int,
    context_selection: str,
    context_points_maxval: list[float],
    context_points_minval: list[float],
    datapoint_shape: tuple[int, ...],
    key: jax.random.PRNGKey,
    *,
    grid_stride: int | None = None,
    dataloader = None,
):
    """Select context points for FSP inference.

    Parameters
    ----------
    n_context_points : int
        Number of context points to select
    context_selection : str
        Method for selecting context points:
        - "random": Random sampling in a window
        - "grid": Regular grid
        - "sobol": Sobol sequence (quasi-random)
        - "dataloader": Sample from dataloader
    context_points_maxval : list[float]
        Maximum values for each dimension
    context_points_minval : list[float]
        Minimum values for each dimension
    datapoint_shape : tuple[int, ...]
        Shape of data points
    key : jax.random.PRNGKey
        Random key
    grid_stride : int, optional
        Stride for grid sampling (for dataloader method)
    dataloader : optional
        Dataloader to sample from (for dataloader method)

    Returns
    -------
    jnp.ndarray or tuple
        Context points, and optionally grid for operator learning
    """
    D = datapoint_shape[-1]

    scaled_max = jnp.array(context_points_maxval)
    scaled_min = jnp.array(context_points_minval)

    if context_selection == "random":
        window_len = (scaled_max - scaled_min) / 4.0

        start = jax.random.uniform(
            key,
            shape=(D,),
            minval=scaled_min,
            maxval=scaled_max - window_len,
        )

        w_min = start
        w_max = start + window_len

        context_points = jax.random.uniform(
            key,
            shape=(n_context_points, D),
            minval=w_min,
            maxval=w_max,
        )

    elif context_selection == "grid":
        if D == 1:
            context_points = jnp.linspace(
                context_points_minval[0], context_points_maxval[0], n_context_points
            ).reshape(-1, 1)
        else:
            # Multi-dimensional grid
            # Create grid points per dimension
            points_per_dim = int(jnp.ceil(n_context_points ** (1 / D)))
            grids = [
                jnp.linspace(context_points_minval[i], context_points_maxval[i], points_per_dim)
                for i in range(D)
            ]
            mesh = jnp.meshgrid(*grids, indexing="ij")
            context_points = jnp.stack([g.flatten() for g in mesh], axis=-1)
            # Take first n_context_points
            context_points = context_points[:n_context_points]

    elif context_selection == "sobol":
        # Sobol sequence (quasi-random low-discrepancy sequence)
        # Use jax.random to generate quasi-random Sobol sequence
        # Note: JAX doesn't have native Sobol, so we use stratified sampling
        # as an approximation
        import warnings
        warnings.warn(
            "Sobol sequence not natively available in JAX. "
            "Using stratified random sampling as approximation."
        )

        # Stratified sampling: divide space into strata and sample one point per stratum
        n_per_dim = int(jnp.ceil(n_context_points ** (1 / D)))
        strata_size = (scaled_max - scaled_min) / n_per_dim

        # Generate stratified samples
        samples = []
        for i in range(n_context_points):
            # Compute stratum indices
            stratum_idx = jnp.array(
                [(i // (n_per_dim ** d)) % n_per_dim for d in range(D)]
            )
            # Random sample within stratum
            stratum_min = scaled_min + stratum_idx * strata_size
            stratum_max = stratum_min + strata_size

            sample = jax.random.uniform(
                jax.random.fold_in(key, i),
                shape=(D,),
                minval=stratum_min,
                maxval=stratum_max,
            )
            samples.append(sample)

        context_points = jnp.stack(samples, axis=0)

    elif context_selection == "dataloader":
        # Sample from dataloader with optional grid stride
        if dataloader is None:
            raise ValueError("dataloader must be provided for 'dataloader' context_selection")

        # Get first batch from dataloader
        x_full, _ = next(iter(dataloader))

        if grid_stride is not None and grid_stride > 1:
            # Apply stride to spatial dimensions
            # Assuming shape is (B, S1, S2, ..., C)
            slices = [slice(None)]  # Keep batch dimension
            for i in range(1, x_full.ndim - 1):
                slices.append(slice(None, None, grid_stride))
            slices.append(slice(None))  # Keep channel dimension
            x_context = x_full[tuple(slices)]
        else:
            x_context = x_full

        # Subsample to n_context_points if needed
        if x_context.shape[0] > n_context_points:
            indices = jax.random.choice(
                key, x_context.shape[0], shape=(n_context_points,), replace=False
            )
            x_context = x_context[indices]

        # For operator learning, also return the spatial grid
        if x_full.ndim >= 4:
            # Extract grid from spatial dimensions
            spatial_shape = x_full.shape[1:-1]
            grid = jnp.meshgrid(
                *[jnp.arange(s) for s in spatial_shape],
                indexing="ij"
            )
            grid = jnp.stack([g.flatten() for g in grid], axis=-1)
            if grid_stride is not None and grid_stride > 1:
                slices = [slice(None, None, grid_stride) for _ in range(len(spatial_shape))]
                grid_mesh = jnp.meshgrid(
                    *[jnp.arange(0, s, grid_stride) for s in spatial_shape],
                    indexing="ij"
                )
                grid = jnp.stack([g.flatten() for g in grid_mesh], axis=-1)
            return x_context, grid

        return x_context

    else:
        raise ValueError(f"Unknown context_selection={context_selection!r}")

    return context_points
