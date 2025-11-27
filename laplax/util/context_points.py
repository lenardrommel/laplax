from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from laplax.types import Array, DataLoader, Int


def _load_all_data_from_dataloader(
    dataloader: DataLoader,
) -> tuple[Array, Array]:
    """Load all batches from a DataLoader into JAX arrays.

    Returns:
        Tuple of (all_x, all_y) JAX arrays containing concatenated batches.
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

    Avoids using jax.numpy on Python tuples (which can carry tracers in tests)
    by computing the product with numpy to obtain a plain integer.

    Returns:
        Tuple of (flattened_data, original_shape).
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
    """Standardize features, then run PCA (SVD-backed) and return scores.

    - Centers and scales each original feature to unit variance prior to PCA.
    - Uses sklearn's PCA which centers again internally; pre-scaling is the key.

    Returns:
        Tuple of (transformed_data, fitted_pca_model).
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

    return jax.device_put(transformed), pca


def _pca_transform_input_jax(
    x_data: Array,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[Array, PCA, np.ndarray, np.ndarray]:
    """Standardize input features, then run PCA and return scores.

    Returns:
        Tuple of (transformed_data, fitted_pca_model, mean, std).
    """
    x_np = np.array(x_data)
    feat_mean = x_np.mean(axis=0, keepdims=True)
    feat_std = x_np.std(axis=0, keepdims=True) + 1e-8
    x_np_std = (x_np - feat_mean) / feat_std

    if n_components is None:
        pca = PCA(n_components=variance_threshold, svd_solver="full")
    else:
        pca = PCA(n_components=n_components)

    pca.fit(x_np_std)
    transformed = pca.transform(x_np_std)

    return jax.device_put(transformed), pca, feat_mean, feat_std


def _pca_inverse_transform_jax(
    pca_scores: np.ndarray,
    pca: PCA,
    mean: np.ndarray,
    std: np.ndarray,
) -> Array:
    """Inverse PCA transform: map from PCA space back to original input space.

    Returns:
        Reconstructed data in original input space.
    """
    x_pca_std = pca.inverse_transform(pca_scores)
    x_reconstructed = x_pca_std * std + mean
    return jax.device_put(x_reconstructed)


def _generate_low_discrepancy_sequence(
    n_dims: int,
    n_points: int,
    sequence_type: str = "sobol",
    seed: Int | None = None,
) -> np.ndarray:
    """Generate low-discrepancy quasi-random sequences.

    Returns:
        Array of shape (n_points, n_dims) containing quasi-random points.

    Raises:
        ValueError: If sequence_type is not 'sobol', 'halton', or 'latin_hypercube'.
    """
    if sequence_type.lower() == "sobol":
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        if (n_points & (n_points - 1)) == 0 and n_points > 0:
            points = sampler.random_base2(int(np.log2(n_points)))
        else:
            points = sampler.random(n_points)
    elif sequence_type.lower() == "halton":
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=seed)
        points = sampler.random(n_points)
    elif sequence_type.lower() == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        points = sampler.random(n_points)
    else:
        msg = (
            f"Unknown sequence type: {sequence_type}. "
            "Choose from 'sobol', 'halton', 'latin_hypercube'"
        )
        raise ValueError(msg)

    return points


def _normalize_to_unit_cube(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data to the unit hypercube [0, 1]^d.

    Returns:
        Tuple of (normalized_data, data_min, data_max).
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

    Returns:
        Array of indices of nearest neighbors in data_points.
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(data_points)

    _, indices = nn.kneighbors(query_points)
    return indices.flatten()


def _pca_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    sequence_type: str = "sobol",
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
    *,
    return_pca: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, PCA]:
    """Select context points using PCA-based low-discrepancy sampling.

    Returns:
        Tuple of (context_x, context_y) or (context_x, context_y, pca) if
        return_pca=True.
    """
    all_x, all_y = _load_all_data_from_dataloader(dataloader)
    y_flat, _ = _flatten_spatial_dims(all_y)

    y_pca, pca = _pca_transform_jax(
        y_flat,
        n_components=n_pca_components,
        variance_threshold=pca_variance_threshold,
    )
    y_pca_norm, _, _ = _normalize_to_unit_cube(np.array(y_pca))

    ld_points = _generate_low_discrepancy_sequence(
        n_dims=pca.n_components_,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    if sequence_type.lower() == "sobol":
        centered = 2.0 * (ld_points - 0.5)
        variances = pca.explained_variance_  # shape (n_components,)
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
            indices = np.concatenate([unique_indices, available])
        else:
            rng = np.random.default_rng(seed)
            additional = rng.choice(available, size=remaining, replace=False)
            indices = np.concatenate([unique_indices, additional])
    else:
        indices = unique_indices[:n_context_points]

    context_x = all_x[indices]
    context_y = all_y[indices]

    if return_pca:
        return context_x, context_y, pca
    return context_x, context_y


def _sobol_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Sobol-based PCA context point selection.

    Returns:
        Tuple of (context_x, context_y).
    """
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
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Latin Hypercube-based PCA context point selection.

    Returns:
        Tuple of (context_x, context_y).
    """
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
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Halton-based PCA context point selection.

    Returns:
        Tuple of (context_x, context_y).
    """
    return _pca_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="halton",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _get_input_shape_and_dim(dataloader: DataLoader) -> tuple[tuple, int]:
    """Get input shape and flattened feature dimension from dataloader.

    Returns:
        Tuple of (input_shape, feature_dim).
    """
    x, _ = next(iter(dataloader))
    x_array = jnp.array(x)
    input_shape = x_array.shape[1:]
    feature_dim = int(np.prod(input_shape))
    return input_shape, feature_dim


def _get_input_range(dataloader: DataLoader) -> tuple[Array, Array]:
    """Get min and max values for each input feature dimension.

    Returns:
        Tuple of (min_values, max_values) as 1D arrays.
    """
    all_x, _ = _load_all_data_from_dataloader(dataloader)
    x_flat = all_x.reshape(len(all_x), -1)
    min_vals = jnp.min(x_flat, axis=0)
    max_vals = jnp.max(x_flat, axis=0)
    return min_vals, max_vals


def _generate_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    sequence_type: str = "halton",
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points in input space using low-discrepancy sequences.

    Generates points that look like data but are not from the dataset.

    Returns:
        Tuple of (context_x, context_y) where context_x are synthetic points
        and context_y are dummy values with appropriate shape.
    """
    _, all_y = _load_all_data_from_dataloader(dataloader)
    input_shape, feature_dim = _get_input_shape_and_dim(dataloader)
    min_vals, max_vals = _get_input_range(dataloader)

    ld_points = _generate_low_discrepancy_sequence(
        n_dims=feature_dim,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    ld_points_jax = jnp.array(ld_points)
    context_x_flat = ld_points_jax * (max_vals - min_vals) + min_vals
    context_x = context_x_flat.reshape(n_context_points, *input_shape)

    y_shape = all_y.shape[1:]
    context_y = jnp.zeros((n_context_points, *y_shape), dtype=all_y.dtype)

    return context_x, context_y


def _halton_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using Halton sequence in input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    del n_pca_components, pca_variance_threshold
    return _generate_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="halton",
        seed=seed,
    )


def _latin_hypercube_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using Latin Hypercube in input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    del n_pca_components, pca_variance_threshold
    return _generate_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="latin_hypercube",
        seed=seed,
    )


def _sobol_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using Sobol sequence in input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    del n_pca_components, pca_variance_threshold
    return _generate_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="sobol",
        seed=seed,
    )


def _grid_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate context points on a regular grid in input space.

    Returns:
        Tuple of (context_x, context_y).

    Raises:
        ValueError: If feature dimension is not 1, 2, 3, or 4.
    """
    del n_pca_components, pca_variance_threshold, seed

    _, all_y = _load_all_data_from_dataloader(dataloader)
    input_shape, feature_dim = _get_input_shape_and_dim(dataloader)
    min_vals, max_vals = _get_input_range(dataloader)

    if feature_dim not in {1, 2, 3, 4}:
        msg = (
            f"Grid context selection only works for 1D-4D features, got {feature_dim}D"
        )
        raise ValueError(msg)

    if feature_dim == 1:
        context_x_flat = jnp.linspace(
            min_vals[0], max_vals[0], n_context_points
        ).reshape(-1, 1)
    elif feature_dim == 2:
        n_dim = int(jnp.rint(jnp.sqrt(n_context_points)))
        x1 = jnp.linspace(min_vals[0], max_vals[0], n_dim)
        x2 = jnp.linspace(min_vals[1], max_vals[1], n_dim)
        X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
        context_x_flat = jnp.stack([X1, X2], axis=-1).reshape(-1, 2)
    elif feature_dim == 3:
        n_dim = int(jnp.rint(n_context_points ** (1 / 3)))
        x1 = jnp.linspace(min_vals[0], max_vals[0], n_dim)
        x2 = jnp.linspace(min_vals[1], max_vals[1], n_dim)
        x3 = jnp.linspace(min_vals[2], max_vals[2], n_dim)
        X1, X2, X3 = jnp.meshgrid(x1, x2, x3, indexing="ij")
        context_x_flat = jnp.stack([X1, X2, X3], axis=-1).reshape(-1, 3)
    else:
        n_dim = int(jnp.rint(n_context_points ** (1 / 4)))
        x1 = jnp.linspace(min_vals[0], max_vals[0], n_dim)
        x2 = jnp.linspace(min_vals[1], max_vals[1], n_dim)
        x3 = jnp.linspace(min_vals[2], max_vals[2], n_dim)
        x4 = jnp.linspace(min_vals[3], max_vals[3], n_dim)
        X1, X2, X3, X4 = jnp.meshgrid(x1, x2, x3, x4, indexing="ij")
        context_x_flat = jnp.stack([X1, X2, X3, X4], axis=-1).reshape(-1, 4)

    context_x_flat = context_x_flat[:n_context_points]
    context_x = context_x_flat.reshape(-1, *input_shape)

    y_shape = all_y.shape[1:]
    context_y = jnp.zeros((len(context_x), *y_shape), dtype=all_y.dtype)

    return context_x, context_y


def _generate_pca_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    sequence_type: str = "halton",
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points by sampling in PCA space and mapping back.

    Performs PCA on input data, samples in PCA space using low-discrepancy sequences,
    then maps back to input space using inverse PCA transform. This generates points
    that look like data but are not from the dataset.

    Returns:
        Tuple of (context_x, context_y) where context_x are synthetic points
        and context_y are dummy values with appropriate shape.
    """
    all_x, all_y = _load_all_data_from_dataloader(dataloader)
    input_shape, _ = _get_input_shape_and_dim(dataloader)

    x_flat = all_x.reshape(len(all_x), -1)

    x_pca, pca, mean, std = _pca_transform_input_jax(
        x_flat,
        n_components=n_pca_components,
        variance_threshold=pca_variance_threshold,
    )

    x_pca_np = np.array(x_pca)
    pca_min = x_pca_np.min(axis=0)
    pca_max = x_pca_np.max(axis=0)

    ld_points = _generate_low_discrepancy_sequence(
        n_dims=pca.n_components_,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    ld_points_scaled = ld_points * (pca_max - pca_min) + pca_min

    context_x_flat = _pca_inverse_transform_jax(ld_points_scaled, pca, mean, std)
    context_x = context_x_flat.reshape(n_context_points, *input_shape)

    y_shape = all_y.shape[1:]
    context_y = jnp.zeros((n_context_points, *y_shape), dtype=all_y.dtype)

    return context_x, context_y


def _pca_halton_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using PCA + Halton sequence.

    Samples in PCA space and maps back to input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    return _generate_pca_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="halton",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _pca_latin_hypercube_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using PCA + Latin Hypercube.

    Samples in PCA space and maps back to input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    return _generate_pca_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="latin_hypercube",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _pca_sobol_synthetic_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic context points using PCA + Sobol sequence.

    Samples in PCA space and maps back to input space.

    Returns:
        Tuple of (context_x, context_y).
    """
    return _generate_pca_synthetic_context_points(
        dataloader=dataloader,
        n_context_points=n_context_points,
        sequence_type="sobol",
        n_pca_components=n_pca_components,
        pca_variance_threshold=pca_variance_threshold,
        seed=seed,
    )


def _random_context_points(
    dataloader: DataLoader,
    n_context_points: Int,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
) -> tuple[Array, Array]:
    """Randomly sample context points from the dataset.

    Extra PCA-related arguments are accepted for API compatibility with
    other context selection functions but are ignored.

    Returns:
        Tuple of (context_x, context_y).
    """
    del n_pca_components, pca_variance_threshold

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


ContextSelectionFn = Callable[
    [DataLoader, Int, Int | None, float, Int | None], tuple[Array, Array]
]

CONTEXT_SELECTION_METHODS: dict[str, ContextSelectionFn] = {
    "random": _random_context_points,
    "sobol": _sobol_synthetic_context_points,
    "pca_sobol": _sobol_context_points,
    "halton": _halton_synthetic_context_points,
    "pca_halton": _halton_context_points,
    "latin_hypercube": _latin_hypercube_synthetic_context_points,
    "pca_lhs": _latin_hypercube_context_points,
    "grid": _grid_context_points,
    "pca_synthetic_halton": _pca_halton_synthetic_context_points,
    "pca_synthetic_lhs": _pca_latin_hypercube_synthetic_context_points,
    "pca_synthetic_sobol": _pca_sobol_synthetic_context_points,
    # Alias 'pca' to Sobol-based PCA selection
    "pca": _sobol_context_points,
}


def _make_grid_from_data_shape(
    data_shape, min_domain: float = 0.0, max_domain: float = 2 * np.pi
) -> tuple[jnp.ndarray, float]:
    """Construct a 1D/2D/3D spatial grid from a data shape.

    Returns:
        Tuple of (grid, dx) where grid is the spatial grid and dx is the spacing.

    Raises:
        ValueError: If number of spatial dimensions is not 1, 2, or 3.
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
        msg = f"Unsupported number of spatial dimensions: {num_spatial_dims}"
        raise ValueError(msg)

    return grid, dx


def _make_grid_from_loader(
    dataloader: DataLoader,
    min_domain: float = 0.0,
    max_domain: float = 2 * np.pi,
) -> tuple[jnp.ndarray, float]:
    """Infer grid from a single batch of the dataloader.

    Returns:
        Tuple of (grid, dx) where grid is the spatial grid and dx is the spacing.
    """
    _, y = next(iter(dataloader))
    data_shape = y.shape
    return _make_grid_from_data_shape(
        data_shape, min_domain=min_domain, max_domain=max_domain
    )


def _apply_grid_stride(
    grid: jnp.ndarray,
    context_x: Array,
    grid_stride: Int | Sequence[int] | None,
) -> tuple[jnp.ndarray, Array]:
    """Apply spatial striding to grid and corresponding context inputs.

    Returns:
        Tuple of (strided_grid, strided_context_x).

    Raises:
        ValueError: If grid dimension is not supported for striding.
    """
    if grid_stride is None or grid_stride == 1:
        return grid, context_x

    stride = (
        grid_stride if isinstance(grid_stride, (tuple, list)) else (int(grid_stride),)
    )

    # 1D grid: (S,)
    if grid.ndim == 1:
        s = max(1, int(stride[0]))
        grid = grid[::s]
        # context shape (..., S, T, C) with S at axis=1
        context_x = context_x[:, ::s, ...]
    # 2D grid: (Sy, Sx, 2)
    elif grid.ndim == 3:
        s_x = s_y = max(1, int(stride[0]))
        if len(stride) >= 2:
            s_x = max(1, int(stride[0]))
            s_y = max(1, int(stride[1]))
        # Grid axes are (Sy, Sx, 2) due to indexing="xy"
        grid = grid[::s_y, ::s_x, :]
        # Context axes are (n_ctx, Sx, Sy, T, C)
        context_x = context_x[:, ::s_x, ::s_y, ...]
    # 3D grid: (Sx, Sy, Sz, 3)
    elif grid.ndim == 4:
        s_x = s_y = s_z = max(1, int(stride[0]))
        if len(stride) >= 3:
            s_x = max(1, int(stride[0]))
            s_y = max(1, int(stride[1]))
            s_z = max(1, int(stride[2]))
        grid = grid[::s_x, ::s_y, ::s_z, :]
        context_x = context_x[:, ::s_x, ::s_y, ::s_z, ...]
    else:
        msg = f"Unsupported grid dimension for striding: {grid.ndim}"
        raise ValueError(msg)

    return grid, context_x


def make_grid(
    spatial_dims: int | tuple[int, ...],
    min_domain: float = 0.0,
    max_domain: float = 2 * np.pi,
) -> jnp.ndarray:
    """Create a spatial grid with the specified dimensions.

    Args:
        spatial_dims: Spatial dimensions. Can be:
            - int: Number of points for 1D grid
            - tuple[int, int]: (nx, ny) for 2D grid
            - tuple[int, int, int]: (nx, ny, nz) for 3D grid
        min_domain: Minimum domain value. Defaults to 0.0.
        max_domain: Maximum domain value. Defaults to 2π.

    Returns:
        Grid array with appropriate shape:
            - 1D: (n_points,)
            - 2D: (ny, nx, 2) with coordinates stacked along last axis
            - 3D: (nx, ny, nz, 3) with coordinates stacked along last axis

    Raises:
        ValueError: If spatial_dims is invalid or unsupported.

    Examples:
        >>> grid_1d = make_grid(32)  # 1D grid with 32 points
        >>> grid_2d = make_grid((16, 16))  # 2D grid 16x16
        >>> grid_3d = make_grid((8, 8, 8))  # 3D grid 8x8x8
    """
    if isinstance(spatial_dims, int):
        num_points = spatial_dims
        domain_extent = max_domain - min_domain
        dx = domain_extent / num_points
        grid = jnp.linspace(min_domain, max_domain - dx, num_points)
        return grid

    if isinstance(spatial_dims, tuple):
        num_dims = len(spatial_dims)

        if num_dims == 2:
            num_points_x, num_points_y = spatial_dims
            domain_extent = max_domain - min_domain
            dx = domain_extent / num_points_x
            dy = domain_extent / num_points_y

            x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
            y = jnp.linspace(min_domain, max_domain - dy, num_points_y)

            X, Y = jnp.meshgrid(x, y, indexing="xy")
            grid = jnp.stack([X, Y], axis=-1)
            return grid

        if num_dims == 3:
            num_points_x, num_points_y, num_points_z = spatial_dims
            domain_extent = max_domain - min_domain
            dx = domain_extent / num_points_x
            dy = domain_extent / num_points_y
            dz = domain_extent / num_points_z

            x = jnp.linspace(min_domain, max_domain - dx, num_points_x)
            y = jnp.linspace(min_domain, max_domain - dy, num_points_y)
            z = jnp.linspace(min_domain, max_domain - dz, num_points_z)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
            grid = jnp.stack([X, Y, Z], axis=-1)
            return grid

        msg = (
            f"Unsupported number of spatial dimensions: {num_dims}. "
            "Expected int for 1D, tuple of 2 for 2D, or tuple of 3 for 3D."
        )
        raise ValueError(msg)

    msg = (
        f"Invalid spatial_dims type: {type(spatial_dims)}. "
        "Expected int or tuple of ints."
    )
    raise ValueError(msg)


def select_context_points(
    dataloader: DataLoader,
    context_selection: str,
    n_context_points: Int = 50,
    n_pca_components: Int | None = None,
    pca_variance_threshold: float = 0.95,
    seed: Int | None = None,
    time_keep: Int | None = None,
    grid_stride: Int | None = None,
) -> tuple[Array, Array, Array | None]:
    """Top-level context point selection API.

    Returns:
        Tuple of (context_x, context_y, grid).

    Raises:
        ValueError: If context_selection method is unknown.
    """
    # Handle combined strategies (e.g., "random+sobol", "sobol+latin_hypercube")
    if "+" in context_selection:
        strategies = [s.strip() for s in context_selection.split("+")]

        points_per_strategy = n_context_points // len(strategies)
        remainder = n_context_points % len(strategies)

        all_context_x: list[Array] = []
        all_context_y: list[Array] = []

        for i, strategy in enumerate(strategies):
            n_points = points_per_strategy + (1 if i < remainder else 0)
            strategy_seed = None if seed is None else seed + i
            if strategy not in CONTEXT_SELECTION_METHODS:
                msg = (
                    f"Unknown context_selection: {strategy}. "
                    "Choose from 'random', 'sobol', 'halton', "
                    "'latin_hypercube', 'grid', 'pca', 'pca_sobol', "
                    "'pca_halton', 'pca_lhs', 'pca_synthetic_halton', "
                    "'pca_synthetic_lhs', 'pca_synthetic_sobol'"
                )
                raise ValueError(msg)
            cx, cy = CONTEXT_SELECTION_METHODS[strategy](
                dataloader,
                n_points,
                n_pca_components,
                pca_variance_threshold,
                strategy_seed,
            )
            all_context_x.append(cx)
            all_context_y.append(cy)

        context_x = jnp.concatenate(all_context_x, axis=0)
        context_y = jnp.concatenate(all_context_y, axis=0)

        grid, _ = _make_grid_from_loader(dataloader)
        grid, context_x = _apply_grid_stride(grid, context_x, grid_stride)

        return context_x, context_y, grid

    # Single strategy selection
    if context_selection not in CONTEXT_SELECTION_METHODS:
        msg = (
            f"Unknown context_selection: {context_selection}. "
            "Choose from 'random', 'sobol', 'halton', "
            "'latin_hypercube', 'grid', 'pca', 'pca_sobol', "
            "'pca_halton', 'pca_lhs', 'pca_synthetic_halton', "
            "'pca_synthetic_lhs', 'pca_synthetic_sobol'"
        )
        raise ValueError(msg)

    context_x, context_y = CONTEXT_SELECTION_METHODS[context_selection](
        dataloader,
        n_context_points,
        n_pca_components,
        pca_variance_threshold,
        seed,
    )

    if time_keep is not None:
        t_keep = max(1, int(time_keep))
        if context_x.shape[-2] > t_keep:
            context_x = context_x[..., :t_keep, :]

    grid, _ = _make_grid_from_loader(dataloader)
    grid, context_x = _apply_grid_stride(grid, context_x, grid_stride)

    return context_x, context_y, grid
