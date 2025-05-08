import jax
import jax.numpy as jnp
import optax
from flax import nnx
from nosplace.data.sinusoid import get_sinusoid_example, DataLoader
from nosplace.fno._fno import MLP


import jax
import jax.numpy as jnp
import optax
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from plotting import plot_sinusoid_task, plot_gp_prediction
from laplax.curv.cov import Posterior
from laplax.curv.fsp import create_fsp_objective
from matplotlib import pyplot as plt
import pickle
import orbax.checkpoint as ocp

jax.config.update("jax_enable_x64", True)

n_epochs = 1000
key = jax.random.key(0)

# Sample toy data example
num_training_samples = 150
num_calibration_samples = 50
num_test_samples = 150

batch_size = 20
X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
    num_train_data=num_training_samples,
    num_valid_data=num_calibration_samples,
    num_test_data=num_test_samples,
    sigma_noise=0.3,
    intervals=[(0, 8)],
    rng_key=jax.random.key(0),
    dtype=jnp.float64,
)
train_loader = DataLoader(X_train, y_train, batch_size)