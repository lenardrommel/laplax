import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from sklearn.decomposition import PCA

from laplax.api import _is_data_loader, _validate_and_get_transform
from laplax.types import Array, Data, Float, InputArray, Int, Iterable, KeyType


def _load_all_data_from_dataloader(
    dataloader: Iterable,
) -> tuple[InputArray, InputArray]:
    """Load all data from a dataloader into arrays.

    Args:
        dataloader: Iterable yielding batches of data.

    Returns:
        Tuple of (all_inputs, all_targets) concatenated across all batches.
    """
    x_list = []
    y_list = []

    transform = _validate_and_get_transform(next(iter(dataloader)))

    for batch in dataloader:
        data = transform(batch)
        x_list.append(jnp.array(data["input"]))
        y_list.append(jnp.array(data["target"]))

    all_x = jnp.concatenate(x_list, axis=0)
    all_y = jnp.concatenate(y_list, axis=0)

    return all_x, all_y


def _generate_low_discrepancy_sequence(
    n_dims: Int,
    n_points: Int,
    sequence_type: str = "sobol",
    seed: Int | None = None,
) -> Array:
    """Generate a low-discrepancy sequence in [0, 1]^n_dims.

    Args:
        n_dims: Number of dimensions.
        n_points: Number of points to generate.
        sequence_type: Type of sequence ("sobol", "halton", "latin_hypercube").
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_points, n_dims) with values in [0, 1].

    Raises:
        ValueError: If sequence_type is not supported.
    """
    seq_type = sequence_type.lower()

    if seq_type == "sobol":
        m = int(np.ceil(np.log2(max(1, n_points))))
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        points_full = sampler.random_base2(m)
        points = points_full[:n_points]
    elif seq_type == "halton":
        sampler = qmc.Halton(d=n_dims, scramble=True, seed=seed)
        points = sampler.random(n_points)
    elif seq_type == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        points = sampler.random(n_points)
    else:
        msg = (
            f"Unknown sequence type: {sequence_type}. "
            "Choose from 'sobol', 'halton', 'latin_hypercube'"
        )
        raise ValueError(msg)

    return jnp.array(points)


def _sample_uniform_like(
    data: Array,
    n_points: Int,
    key: KeyType,
) -> Array:
    """Sample new points uniformly in the axis-aligned bounding box of data.

    Args:
        data: Array with shape (batch, ...) to compute bounding box from.
        n_points: Number of points to sample.
        key: JAX PRNG key for random sampling.

    Returns:
        Array of shape (n_points, ...) with same non-batch shape as data.
    """
    batch_size = data.shape[0]
    feature_shape = data.shape[1:]

    data_flat = data.reshape(batch_size, -1)
    n_features = data_flat.shape[1]

    feat_min = jnp.min(data_flat, axis=0, keepdims=True)
    feat_max = jnp.max(data_flat, axis=0, keepdims=True)

    samples_flat = jax.random.uniform(
        key=key,
        shape=(n_points, n_features),
        minval=feat_min,
        maxval=feat_max,
    )

    samples = samples_flat.reshape(n_points, *feature_shape)
    return samples


def _fit_pca_on_data(
    data: Array,
    pca_variance_threshold: Float,
) -> tuple[Array, Array, Array, PCA, tuple]:
    """Fit PCA on data and return transformed scores and metadata.

    Args:
        data: Array with shape (batch, ...) to fit PCA on.
        pca_variance_threshold: Variance threshold for PCA (0-1).

    Returns:
        Tuple of (scores, feat_mean, feat_std, pca, original_shape).
    """
    batch_size = data.shape[0]
    original_shape = data.shape[1:]

    data_flat = data.reshape(batch_size, -1)
    data_np = np.array(data_flat)

    feat_mean = data_np.mean(axis=0, keepdims=True) + 1e-8
    feat_std = data_np.std(axis=0, keepdims=True) + 1e-8
    data_std = (data_np - feat_mean) / feat_std

    pca = PCA(n_components=pca_variance_threshold)
    scores = pca.fit_transform(data_std)

    return (
        jnp.array(scores),
        jnp.array(feat_mean),
        jnp.array(feat_std),
        pca,
        original_shape,
    )


def _sample_from_pca(
    scores: Array,
    feat_mean: Array,
    feat_std: Array,
    pca: PCA,
    n_points: Int,
    sequence_type: str,
    seed: Int | None,
    original_shape: tuple,
    jitter_scale: Float = 1e-8,
) -> Array:
    """Sample new points in PCA space using a low-discrepancy sequence.

    Args:
        scores: PCA scores from training data.
        feat_mean: Mean of features used for standardization.
        feat_std: Std of features used for standardization.
        pca: Fitted PCA object.
        n_points: Number of points to sample.
        sequence_type: Type of low-discrepancy sequence.
        seed: Random seed.
        original_shape: Original feature shape (without batch dimension).
        jitter_scale: Scale of jitter noise relative to feature std.

    Returns:
        Array of sampled points with shape (n_points, *original_shape).
    """
    n_dims = pca.n_components_
    ld_unit = _generate_low_discrepancy_sequence(
        n_dims=n_dims,
        n_points=n_points,
        sequence_type=sequence_type,
        seed=seed,
    )

    scores_np = np.array(scores)
    scores_min = scores_np.min(axis=0)
    scores_max = scores_np.max(axis=0)
    sampled_scores = ld_unit * (scores_max - scores_min) + scores_min

    sampled_std = pca.inverse_transform(np.array(sampled_scores))
    feat_mean_np = np.array(feat_mean)
    feat_std_np = np.array(feat_std)
    sampled = sampled_std * feat_std_np + feat_mean_np

    sampled_jax = jax.device_put(sampled)

    if jitter_scale > 0:
        jitter_seed = (seed + 9999) if seed is not None else 42
        jitter_key = jax.random.PRNGKey(jitter_seed)
        feat_std_jax = jnp.array(feat_std)
        jitter = (
            jitter_scale
            * feat_std_jax
            * jax.random.normal(
                jitter_key, shape=sampled_jax.shape, dtype=sampled_jax.dtype
            )
        )
        sampled_jax = sampled_jax + jitter

    n_features = int(np.prod(original_shape))
    sampled_jax = sampled_jax.reshape(n_points, n_features)
    sampled_jax = sampled_jax.reshape(n_points, *original_shape)

    return sampled_jax


def _random_context_points(
    data: Data | Iterable,
    n_context_points: Int,
    key: KeyType,
) -> tuple[InputArray, InputArray]:
    """Generate random context points uniformly in data bounding box.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to generate.
        key: JAX PRNG key.

    Returns:
        Tuple of (context_x, context_y) arrays.
    """
    if _is_data_loader(data):
        all_x, all_y = _load_all_data_from_dataloader(data)
    else:
        transform = _validate_and_get_transform(data)
        batch_data = transform(data)
        all_x = jnp.array(batch_data["input"])
        all_y = jnp.array(batch_data["target"])

    key_x, key_y = jax.random.split(key)
    context_x = _sample_uniform_like(all_x, n_context_points, key_x)
    context_y = _sample_uniform_like(all_y, n_context_points, key_y)

    return context_x, context_y


def _pca_context_points(
    data: Data | Iterable,
    n_context_points: Int,
    sequence_type: str = "sobol",
    pca_variance_threshold: Float = 0.95,
    seed: Int | None = None,
    jitter_scale: Float = 1e-3,
) -> tuple[InputArray, InputArray]:
    """Generate context points using PCA and low-discrepancy sequences.

    Args:
        data: Single batch (dict) or iterable of batches.
        n_context_points: Number of context points to generate.
        sequence_type: Type of low-discrepancy sequence.
        pca_variance_threshold: Variance threshold for PCA.
        seed: Random seed.
        jitter_scale: Scale of jitter noise.

    Returns:
        Tuple of (context_x, context_y) arrays.
    """
    if _is_data_loader(data):
        all_x, all_y = _load_all_data_from_dataloader(data)
    else:
        transform = _validate_and_get_transform(data)
        batch_data = transform(data)
        all_x = jnp.array(batch_data["input"])
        all_y = jnp.array(batch_data["target"])

    x_scores, x_mean, x_std, pca_x, x_shape = _fit_pca_on_data(
        all_x,
        pca_variance_threshold=pca_variance_threshold,
    )

    context_x_flat = _sample_from_pca(
        scores=x_scores,
        feat_mean=x_mean,
        feat_std=x_std,
        pca=pca_x,
        n_points=n_context_points,
        sequence_type=sequence_type,
        seed=seed,
        original_shape=x_shape,
        jitter_scale=jitter_scale,
    )

    context_x = context_x_flat.reshape(n_context_points, *x_shape)

    rng = np.random.default_rng(seed)
    y_seed = rng.integers(0, 2**31)
    key_y = jax.random.PRNGKey(y_seed)
    context_y = _sample_uniform_like(all_y, n_context_points, key_y)

    return context_x, context_y


def select_context_points(
    data: Data | Iterable,
    method: str,
    n_context_points: Int = 50,
    *,
    pca_variance_threshold: Float = 0.95,
    seed: Int | None = None,
    jitter_scale: Float = 1e-3,
) -> tuple[InputArray, InputArray]:
    """Select context points from data using specified method.

    Args:
        data: Single batch (dict) or iterable of batches.
        method: Selection method ("random", "sobol", "halton",
            "latin_hypercube", "pca").
        n_context_points: Number of context points to generate.
        pca_variance_threshold: Variance threshold for PCA-based methods.
        seed: Random seed for reproducibility.
        jitter_scale: Scale of jitter noise for PCA methods.

    Returns:
        Tuple of (context_x, context_y) arrays.

    Raises:
        ValueError: If method is not supported.
    """
    key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)

    method_lower = method.lower()

    if method_lower == "random":
        return _random_context_points(data, n_context_points, key)

    if method_lower in {"sobol", "pca_sobol", "pca"}:
        return _pca_context_points(
            data,
            n_context_points,
            sequence_type="sobol",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    if method_lower in {"halton", "pca_halton"}:
        return _pca_context_points(
            data,
            n_context_points,
            sequence_type="halton",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    if method_lower in {"latin_hypercube", "pca_lhs"}:
        return _pca_context_points(
            data,
            n_context_points,
            sequence_type="latin_hypercube",
            pca_variance_threshold=pca_variance_threshold,
            seed=seed,
            jitter_scale=jitter_scale,
        )

    msg = (
        f"Unknown context selection method: {method}. "
        "Choose from 'random', 'sobol', 'halton', 'latin_hypercube', 'pca'"
    )
    raise ValueError(msg)
