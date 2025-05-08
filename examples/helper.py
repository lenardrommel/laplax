from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

jax.config.update("jax_enable_x64", True)


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return self.X[batch_indices], self.y[batch_indices]


DEFAULT_INTERVALS = [
    (0, 2),
    (4, 5),
    (6, 8),
]


# Function to create the sinusoid dataset
def get_sinusoid_example(
    num_train_data: int = 150,
    num_valid_data: int = 50,
    num_test_data: int = 100,
    sigma_noise: float = 0.3,
    intervals: list[tuple[float, float]] = DEFAULT_INTERVALS,
    rng_key=None,
    dtype=jnp.float32,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Generate a sinusoid dataset.

    Args:
        num_train_data: Number of training data points.
        num_valid_data: Number of validation data points.
        num_test_data: Number of test data points.
        sigma_noise: Standard deviation of the noise.
        intervals: List of tuples containing the intervals for random data generation.
        rng_key: Random number generator key.
        dtype: Data type for the generated arrays.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        X_valid: Validation input data.
        y_valid: Validation target data.
        X_test: Testing input data.
        y_test: Testing target data.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(0)

    # Check if float64 is supported when requested
    if dtype == jnp.float64:
        from jax import config

        if not config.read("jax_enable_x64"):
            import warnings

            warnings.warn(
                "float64 requested but JAX float64 support is not enabled. "
                "Use jax.config.update('jax_enable_x64', True) to enable. "
                "Falling back to float32."
            )
            dtype = jnp.float32

    # Split RNG key for reproducibility
    (
        rng_key,
        rng_x_train,
        rng_x_valid,
        rng_noise_train,
        rng_noise_valid,
        rng_noise_test,
    ) = random.split(rng_key, 6)

    # Convert intervals to array with appropriate dtype
    tuples_as_array = jnp.asarray(intervals, dtype=dtype)

    def f(key):
        key1, key2 = random.split(key, 2)
        interval = random.choice(key1, tuples_as_array, axis=0)
        x = random.uniform(key2, minval=interval[0], maxval=interval[1])
        return jnp.asarray(x, dtype=dtype)

    # Generate random training data
    X_train = jnp.asarray(
        (jax.vmap(f)(random.split(rng_x_train, num_train_data))).reshape(-1, 1),
        dtype=dtype,
    )
    noise = jnp.asarray(
        random.normal(rng_noise_train, X_train.shape) * sigma_noise, dtype=dtype
    )
    y_train = jnp.asarray(jnp.sin(X_train) + noise, dtype=dtype)

    # Generate calibration data
    X_valid = jnp.asarray(
        (jax.vmap(f)(random.split(rng_x_valid, num_valid_data))).reshape(-1, 1),
        dtype=dtype,
    )
    noise = jnp.asarray(
        random.normal(rng_noise_valid, X_valid.shape) * sigma_noise, dtype=dtype
    )
    y_valid = jnp.asarray(jnp.sin(X_valid) + noise, dtype=dtype)

    # Generate testing data
    X_test = jnp.asarray(
        jnp.linspace(0.0, 8.0, num_test_data).reshape(-1, 1), dtype=dtype
    )
    noise = jnp.asarray(
        random.normal(rng_noise_test, X_test.shape) * sigma_noise, dtype=dtype
    )
    y_test = jnp.asarray(jnp.sin(X_test) + noise, dtype=dtype)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
