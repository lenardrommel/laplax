import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from laplax.types import Array


def _load_all_data_from_dataloader(
    dataloader: DataLoader,
) -> tuple[Array, Array]:
    """Load all batches from a DataLoader into JAX arrays.

    Args:
        dataloader: PyTorch DataLoader containing (x, y) batches

    Returns:
        Tuple of (all_x, all_y) as JAX arrays
    """
    x_list = []
    y_list = []

    for batch_x, batch_y in dataloader:
        x_list.append(jnp.array(batch_x))
        y_list.append(jnp.array(batch_y))

    all_x = jnp.array(jnp.concatenate(x_list, axis=0))
    all_y = jnp.array(jnp.concatenate(y_list, axis=0))

    return all_x, all_y


def _flatten_spatial_dims(data: Array) -> tuple[Array, tuple]:
    """Flatten all axes except batch and last channel axis.

    Avoids using jax.numpy on Python tuples by computing the product
    with numpy to obtain a plain integer.

    Args:
        data: Input array with shape (batch, *spatial_dims, channels)

    Returns:
        Tuple of (flattened_data, original_shape)
    """
    original_shape = data.shape
    batch_size = int(original_shape[0])
    middle = original_shape[1:-1]
    n_spatial = int(np.prod(middle)) if len(middle) > 0 else 1
    flattened = data.reshape(batch_size, n_spatial)

    return flattened, original_shape


def _pca_transform_jax(
    y_data: Array,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[Array, PCA]:
    """Standardize features and run PCA transformation.

    Centers and scales each feature to unit variance prior to PCA.

    Args:
        y_data: Input data array
        n_components: Number of PCA components (if None, use variance_threshold)
        variance_threshold: Explained variance threshold for automatic component selection

    Returns:
        Tuple of (transformed_data, fitted_pca_model)
    """
    y_np = np.array(y_data)
    feat_mean = y_np.mean(axis=0, keepdims=True)
    feat_std = y_np.std(axis=0, keepdims=True) + 1e-8
    y_np_std = (y_np - feat_mean) / feat_std

    if n_components is None:
        pca = PCA(n_components=variance_threshold, svd_solver="full")
    else:
        pca = PCA(n_components=n_components)

    pca.fit(y_np_std)
    transformed = pca.transform(y_np_std)
    logger.info(
        f"PCA reduced output dimension from {y_np_std.shape[1]} to {transformed.shape[1]}"
    )

    return jax.device_put(transformed), pca


def _generate_low_discrepancy_sequence(
    n_dims: int,
    n_points: int,
    sequence_type: str = "sobol",
    seed: int | None = None,
) -> np.ndarray:
    """Generate low-discrepancy quasi-random sequences.

    Args:
        n_dims: Dimensionality of the sequence
        n_points: Number of points to generate
        sequence_type: Type of sequence ('sobol', 'halton', or 'latin_hypercube')
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_points, n_dims) with values in [0, 1]
    """
    if sequence_type.lower() == "sobol":
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        points = sampler.random_base2(n_points)
    elif sequence_type.lower() == "halton":
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=seed)
        points = sampler.random(n_points)
    elif sequence_type.lower() == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        points = sampler.random(n_points)
    else:
        raise ValueError(
            f"Unknown sequence type: {sequence_type}. "
            "Choose from 'sobol', 'halton', 'latin_hypercube'"
        )

    return points


def _normalize_to_unit_cube(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data to unit hypercube [0, 1]^d.

    Args:
        data: Input data array

    Returns:
        Tuple of (normalized_data, data_min, data_max)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    normalized = (data - data_min) / (data_max - data_min + 1e-10)
    return normalized, data_min, data_max


def _find_nearest_neighbors(
    query_points: np.ndarray,
    data_points: np.ndarray,
) -> np.ndarray:
    """Find nearest neighbors for each query point in data points.

    Args:
        query_points: Query points array
        data_points: Data points to search in

    Returns:
        Array of indices of nearest neighbors
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(data_points)

    distances, indices = nn.kneighbors(query_points)
    return indices.flatten()


def _pca_context_points(
    dataloader: DataLoader,
    n_context_points: int,
    sequence_type: str = "sobol",
    n_pca_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: int | None = None,
    return_pca: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, PCA]:
    """Select context points using PCA-based low-discrepancy sampling.

    Projects output space to PCA coordinates, generates low-discrepancy
    points in PCA space, and finds nearest neighbors in the data.

    Args:
        dataloader: DataLoader containing the dataset
        n_context_points: Number of context points to select
        sequence_type: Type of low-discrepancy sequence
        n_pca_components: Number of PCA components
        pca_variance_threshold: Variance threshold for PCA
        seed: Random seed
        return_pca: Whether to return the fitted PCA model

    Returns:
        Tuple of (context_x, context_y) or (context_x, context_y, pca)
    """
    all_x, all_y = _load_all_data_from_dataloader(dataloader)
    y_flat, original_shape = _flatten_spatial_dims(all_y)
    x_flat, _ = _flatten_spatial_dims(all_x)

    y_pca, pca = _pca_transform_jax(
        y_flat,
        n_components=n_pca_components,
        variance_threshold=pca_variance_threshold,
    )
    y_pca_norm, pca_min, pca_max = _normalize_to_unit_cube(np.array(y_pca))

    ld_points = _generate_low_discrepancy_sequence(
        n_dims=pca.n_components_,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    if sequence_type.lower() == "sobol":
        centered = 2.0 * (ld_points - 0.5)
        variances = pca.explained_variance_
        scales = 2.0 * variances
        ld_scaled = centered * scales
        indices = _find_nearest_neighbors(ld_scaled, np.array(y_pca))
    else:
        indices = _find_nearest_neighbors(ld_points, y_pca_norm)

    unique_indices = np.unique(indices)
    if len(unique_indices) < n_context_points:
        remaining = n_context_points - len(unique_indices)
        available = np.setdiff1d(np.arange(len(all_y)), unique_indices)

        if len(available) < remaining:
            logger.warning(
                f"Cannot reach {n_context_points} context points. "
                f"Only {len(unique_indices)} unique matches found and "
                f"{len(available)} additional points available. "
                f"Using all {len(unique_indices) + len(available)} points."
            )
            indices = np.concatenate([unique_indices, available])
        else:
            logger.info(
                f"Warning: Only {len(unique_indices)} unique points found. "
                f"Adding {remaining} random points to reach {n_context_points}."
            )
            rng = np.random.default_rng(seed)
            additional = rng.choice(available, size=remaining, replace=False)
            indices = np.concatenate([unique_indices, additional])
    else:
        indices = unique_indices[:n_context_points]

    context_x = all_x[indices]
    context_y = all_y[indices]

    if return_pca:
        return context_x, context_y, pca
    else:
        return context_x, context_y


def _sobol_context_points(
    dataloader: DataLoader,
    n_context_points: int,
    n_pca_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Select context points using Sobol sequences in PCA space."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="sobol",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _latin_hypercube_context_points(
    dataloader: DataLoader,
    n_context_points: int,
    n_pca_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Select context points using Latin Hypercube sampling in PCA space."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="latin_hypercube",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _halton_context_points(
    dataloader: DataLoader,
    n_context_points: int,
    n_pca_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Select context points using Halton sequences in PCA space."""
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="halton",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _random_context_points(
    dataloader: DataLoader,
    n_context_points: int,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Select context points randomly from the dataset.

    Args:
        dataloader: DataLoader containing the dataset
        n_context_points: Number of context points to select
        seed: Random seed for reproducibility

    Returns:
        Tuple of (context_x, context_y)
    """
    all_x, all_y = _load_all_data_from_dataloader(dataloader)

    effective_seed = seed if seed is not None else 0
    key = jax.random.PRNGKey(effective_seed)
    n_total = len(all_y)

    if n_context_points >= n_total:
        return all_x, all_y

    indices = jax.random.choice(key, n_total, shape=(n_context_points,), replace=False)

    context_x = all_x[indices]
    context_y = all_y[indices]

    return context_x, context_y


def _make_grid_from_data_shape(
    data_shape, min_domain=0.0, max_domain=2 * np.pi
) -> tuple[jnp.ndarray, float]:
    """Create a spatial grid based on data shape.

    Args:
        data_shape: Shape of the data array
        min_domain: Minimum domain value
        max_domain: Maximum domain value

    Returns:
        Tuple of (grid, dx) where dx is the grid spacing
    """
    spatial_dims = tuple(dim for dim in data_shape[1:] if dim > 1)

    if len(spatial_dims) > 3:
        spatial_dims = spatial_dims[1:]

    num_spatial_dims = len(spatial_dims)
    domain_extent = max_domain - min_domain

    if num_spatial_dims == 1:
        num_points = spatial_dims[0]
        dx = domain_extent / num_points
        grid = jnp.linspace(min_domain, max_domain - dx, num_points)

    elif num_spatial_dims == 2:
        num_points_x, num_points_y = spatial_dims
        dx = domain_extent / num_points_x
        dy = domain_extent / num_points_y

        x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
        y = jnp.linspace(min_domain, max_domain - dy, num_points_y)

        X, Y = jnp.meshgrid(x, y, indexing="xy")
        grid = jnp.stack([X, Y], axis=-1)

    elif num_spatial_dims == 3:
        num_points_x, num_points_y, num_points_z = spatial_dims
        dx = domain_extent / num_points_x
        dy = domain_extent / num_points_y
        dz = domain_extent / num_points_z

        x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
        y = jnp.linspace(min_domain, max_domain - dy, num_points_y)
        z = jnp.linspace(min_domain, max_domain - dz, num_points_z)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        grid = jnp.stack([X, Y, Z], axis=-1)

    else:
        raise ValueError(
            f"Unsupported number of spatial dimensions: {num_spatial_dims}"
        )

    return grid, dx


def _make_grid_from_loader(
    dataloader, min_domain=0.0, max_domain=2 * np.pi
) -> tuple[jnp.ndarray, float]:
    """Create a spatial grid from the first batch of a DataLoader."""
    _, y = next(iter(dataloader))
    data_shape = y.shape
    return _make_grid_from_data_shape(
        data_shape, min_domain=min_domain, max_domain=max_domain
    )


def select_context_points(
    dataloader: DataLoader,
    context_selection: str,
    n_context_points: int = 50,
    n_pca_components: int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: int | None = None,
    time_keep: int | None = None,
    grid_stride: int | tuple[int, ...] | None = None,
) -> tuple[Array, Array, Array | None]:
    """Select context points from a dataset using various strategies.

    Supports single and combined strategies (e.g., 'random+sobol').

    Args:
        dataloader: DataLoader containing the dataset
        context_selection: Strategy name or combination (e.g., 'sobol', 'random+halton')
        n_context_points: Total number of context points to select
        n_pca_components: Number of PCA components for PCA-based strategies
        pca_variance_threshold: Variance threshold for PCA
        seed: Random seed for reproducibility
        time_keep: Number of time steps to keep (if applicable)
        grid_stride: Stride for downsampling the spatial grid

    Returns:
        Tuple of (context_x, context_y, grid)
    """
    if "+" in context_selection:
        strategies = [s.strip() for s in context_selection.split("+")]
        logger.info(
            f"Using combined strategy: {context_selection} "
            f"(splitting {n_context_points} points across {len(strategies)} strategies)"
        )

        points_per_strategy = n_context_points // len(strategies)
        remainder = n_context_points % len(strategies)

        all_context_x = []
        all_context_y = []

        for i, strategy in enumerate(strategies):
            n_points = points_per_strategy + (1 if i < remainder else 0)

            strategy_seed = None if seed is None else seed + i
            cx, cy, _ = select_context_points(
                dataloader=dataloader,
                context_selection=strategy,
                n_context_points=n_points,
                n_pca_components=n_pca_components,
                pca_variance_threshold=pca_variance_threshold,
                seed=strategy_seed,
                time_keep=time_keep,
                grid_stride=None,
            )
            all_context_x.append(cx)
            all_context_y.append(cy)

        context_x = jnp.concatenate(all_context_x, axis=0)
        context_y = jnp.concatenate(all_context_y, axis=0)

        grid, _ = _make_grid_from_loader(dataloader)
        if grid_stride is not None and grid_stride != 1:
            stride = (
                grid_stride
                if isinstance(grid_stride, (tuple, list))
                else (int(grid_stride),)
            )
            if grid.ndim == 1:
                s = max(1, int(stride[0]))
                grid = grid[::s]
                context_x = context_x[:, ::s, ...]
            elif grid.ndim == 3:
                s_x = s_y = max(1, int(stride[0]))
                if len(stride) == 2:
                    s_x, s_y = max(1, int(stride[0])), max(1, int(stride[1]))
                grid = grid[::s_y, ::s_x, :]
                context_x = context_x[:, ::s_y, ::s_x, ...]

        return context_x, context_y, grid

    if context_selection == "random":
        context_x, context_y = _random_context_points(
            dataloader, n_context_points, seed
        )

    elif context_selection == "sobol" or context_selection == "pca_sobol":
        context_x, context_y = _sobol_context_points(
            dataloader, n_context_points, n_pca_components, pca_variance_threshold, seed
        )
    elif context_selection == "halton" or context_selection == "pca_halton":
        context_x, context_y = _halton_context_points(
            dataloader, n_context_points, n_pca_components, pca_variance_threshold, seed
        )
    elif context_selection == "latin_hypercube" or context_selection == "pca_lhs":
        context_x, context_y = _latin_hypercube_context_points(
            dataloader, n_context_points, n_pca_components, pca_variance_threshold, seed
        )
    elif context_selection == "pca":
        context_x, context_y = _sobol_context_points(
            dataloader, n_context_points, n_pca_components, pca_variance_threshold, seed
        )
    else:
        raise ValueError(
            f"Unknown context_selection: {context_selection}. "
            "Choose from 'random', 'sobol', 'halton', 'latin_hypercube', 'pca'"
        )

    if time_keep is not None:
        t_keep = max(1, int(time_keep))
        if context_x.shape[-2] > t_keep:
            context_x = context_x[..., :t_keep, :]

    grid, _ = _make_grid_from_loader(dataloader)

    if grid_stride is not None and grid_stride != 1:
        stride = (
            grid_stride
            if isinstance(grid_stride, (tuple, list))
            else (int(grid_stride),)
        )
        if grid.ndim == 1:
            s = max(1, int(stride[0]))
            grid = grid[::s]
            context_x = context_x[:, ::s, ...]
        elif grid.ndim == 3:
            s_x = s_y = max(1, int(stride[0]))
            if len(stride) >= 2:
                s_x = max(1, int(stride[0]))
                s_y = max(1, int(stride[1]))
            grid = grid[::s_y, ::s_x, :]
            context_x = context_x[:, ::s_x, ::s_y, ...]
        elif grid.ndim == 4:
            s_x = s_y = s_z = max(1, int(stride[0]))
            if len(stride) >= 3:
                s_x = max(1, int(stride[0]))
                s_y = max(1, int(stride[1]))
                s_z = max(1, int(stride[2]))
            grid = grid[::s_x, ::s_y, ::s_z, :]
            context_x = context_x[:, ::s_x, ::s_y, ::s_z, ...]
        else:
            raise ValueError(f"Unsupported grid dimension for striding: {grid.ndim}")

    return context_x, context_y, grid
