from laplax.e import get_sinusoid_example
import matplotlib.pyplot as plt

# jax
import jax
from jax import random, tree_util, value_and_grad, numpy as jnp
from jaxtyping import PyTree

# laplacestuff
from functools import partial
from laplax.util.mv import to_dense
from laplax.util.flatten import create_pytree_flattener
from laplax.curv.ggn import create_ggn_mv

# flax for training, optax for optimization
import optax
import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad
from flax import nnx
import math
from jax.scipy.linalg import det

# generate training data
n_data = 150
sigma_noise = 0.3
batch_size = 32
rng_key = random.key(711)

X_train, y_train, train_loader, X_test = get_sinusoid_example(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))

plt.scatter(X_train, jnp.sin(X_train), color='black', label='sin(x)')
plt.scatter(X_train, y_train)


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 50, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(50, 1, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


def create_model(rngs):
    model = MLP(rngs)
    return model


# Mean squared error loss function
def mse_loss(model, X, y):
    pred_y = model(X)
    return jax.numpy.mean((y - pred_y) ** 2)

# Update function using Optax
def update(X, y, model, optimizer):
    loss_fn = lambda model: mse_loss(model, X, y)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

# Training loop
# Assuming `train_loader` is a generator that yields (X, y) batches
def train_model(train_loader, n_epochs, rng_key):
    model = create_model(rng_key)
    optimizer = nnx.Optimizer(model, optax.adam(1e-2))
    for _ in range(n_epochs):
        for X, y in train_loader:
            update(X, y, model, optimizer)
            # You can log the loss here if you want
    return model

n_epochs = 10
rng_key = nnx.Rngs(0)
model = train_model(train_loader, n_epochs, rng_key)

pred = model(X_test)
plt.scatter(X_test, jnp.sin(X_test), color='red')
plt.scatter(X_test, pred)

loss_fn = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2) # noqa: E731

data = {"input": X_train, "target": y_train}

_, params, rest = nnx.split(model, nnx.Param, ...)

def model_fn(input, params):
    nnx.update(model, nnx.GraphState.merge(params, rest))  # Load the model from parameters\n",
    return model(input)

full_fn = lambda p, data: loss_fn(model_fn(data["input"], p), data["target"])


ggn_mv = create_ggn_mv(
    model_fn,
    params,
    data,
    loss_fn="mse",
    num_curv_samples=150,
    num_total_samples=75,
)

from laplax.curv.cov import create_posterior_fn

# Create Posterior
posterior_fn = create_posterior_fn(
    "full",
    mv=ggn_mv,
    layout=params,
)

posterior_state = posterior_fn({"prior_prec": 1.0})

mv = posterior_state.cov_mv(posterior_state.state)

from laplax.util.mv import to_dense
from laplax.util.tree import get_size
from laplax.util.flatten import create_pytree_flattener, wrap_function


flatten, unflatten = create_pytree_flattener(params)
identity = lambda x: x

mv_ = wrap_function(mv, input_fn=identity, output_fn=flatten)
cov = to_dense(mv_, layout=get_size(params))

ggn_det = jnp.linalg.det(cov)
log_p_D_theta_star = jnp.log(full_fn(params, data))

log_marginal_likelihood = log_p_D_theta_star - 0.5 * jnp.log(jax.numpy.abs(((1 / (2 * jnp.pi)) * ggn_det)))
print(log_marginal_likelihood)

# Log-likelihood at θ*
log_p_D_theta_star = jnp.log(full_fn(params, data))
print(log_p_D_theta_star)

# Determinant of Hessian
ggn = to_dense(ggn_mv, params)
flatten_ggn, _ = create_pytree_flattener(ggn)
ggn_dense = flatten_ggn(ggn)

ggn_size = int(jnp.sqrt(ggn_dense.size))

# Reshape the flattened Hessian into a square matrix
ggn_dense = ggn_dense.reshape((ggn_size, ggn_size))
prior = jnp.eye(ggn_size)
ggn = ggn_dense + prior
# invert ggn
ggn_det = jnp.linalg.det(ggn)

print(ggn_det)
log_marginal_likelihood = log_p_D_theta_star - 0.5 * jnp.log(jax.numpy.abs(((1 / (2 * jnp.pi)) * ggn_det)))

print(log_marginal_likelihood)