import gpjax as gpx
import jax
import jax.numpy as jnp
import matplotlib as mpl
import optax
from helper import get_sinusoid_example
from jax import random
from matplotlib import pyplot as plt
from prior import Prior

key = jax.random.key(0)


num_training_samples = 150
num_calibration_samples = 50
num_test_samples = 150

batch_size = 20
X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
    num_train_data=num_training_samples,
    num_valid_data=num_calibration_samples,
    num_test_data=num_test_samples,
    sigma_noise=0.0,
    intervals=[(0, 2), (4, 5), (6, 8)],
    rng_key=jax.random.key(0),
)
data = {
    "input": X_train,
    "target": y_train,
    "test_input": X_test,
    "test_target": y_test,
}
prior = Prior(kernel=gpx.kernels.Periodic(), key=key, data=data, config={})
prior(X_train, jitter=1e-6)
p = prior._build_standard_prior("pureperiodic")
tuned_prior, _ = prior._tune_hyperparameters(p)
