import jax
from jax import numpy as jnp
from torch.utils.data import DataLoader


def create_sin_data(
    num_train_data=100,
    num_valid_data=100,
    num_test_data=100,
    sigma_noise=0.3,
    intervals=[(0, 2), (4, 5), (6, 12)],
):
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(-2, 2, num_train_data).reshape(-1, 1)
    y = (
        jnp.sin(2 * jnp.pi * x)
        + jax.random.normal(key, (num_train_data, 1)) * sigma_noise
    )
