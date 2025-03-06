from laplax.eval.marginal_likelihood import calculate_marginal_likelihood, calculate_marginal_likelihood_diagonal
from helper import get_sinusoid_example
import matplotlib.pyplot as plt
from laplax.curv.ggn import create_ggn_mv

import jax
import jax.numpy as jnp
import optax
from jax import random
from flax import nnx
from laplax.curv.cov import create_posterior_fn

# generate training data
# generate training data
n_data = 150
sigma_noise = 0.3
batch_size = 32
rng_key = random.key(711)

X_train, y_train, train_loader, X_test = get_sinusoid_example(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))

plt.scatter(X_train, jnp.sin(X_train), color='black', label='sin(x)')
plt.scatter(X_train, y_train)

X_train, y_train, train_loader, X_test = get_sinusoid_example(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))

plt.scatter(X_train, jnp.sin(X_train), color='black', label='sin(x)')
plt.scatter(X_train, y_train)


# Too Simple: Cannot capture sine properly
class MLP_Simple(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 2, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(2, 1, rngs=rngs)
        self.learning_rate = 1e-3
        self.epochs = 1

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# Reasonable Complexity #1: Captures sine wave well
class MLP_Reasonable(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 20, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(20, 1, rngs=rngs)
        self.learning_rate = 1e-3
        self.epochs = 100

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class MLP_Overfit(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 100, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(100, 50, rngs=rngs)
        self.linear3 = nnx.Linear(50, 1, rngs=rngs)
        self.learning_rate = 1e-4
        self.epochs = 1000

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        return x


models = [MLP_Simple, MLP_Reasonable]

for model_class in models:
    def create_model(rngs):
        model = model_class(rngs)
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
    def train_model(train_loader, rng_key):
        model = create_model(rng_key)
        n_epochs = model.epochs
        optimizer = nnx.Optimizer(model, optax.adam(model.learning_rate))
        for _ in range(n_epochs):
            for X, y in train_loader:
                update(X, y, model, optimizer)
                # You can log the loss here if you want
        return model

    rng_key = nnx.Rngs(0)
    model = train_model(train_loader, rng_key)

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


    # Create Posterior
    posterior_fn = create_posterior_fn(
        "full",
        mv=ggn_mv,
        layout=params,
    )

    posterior_state = posterior_fn({"prior_prec": 1.0})

    pred = model(X_test)
    y_test = jnp.sin(X_test)
    test_loss = loss_fn(pred, y_test).item()

    pred_train = model(X_train)
    train_loss = loss_fn(pred_train, y_train).item()

    marginal_likelihood = calculate_marginal_likelihood(posterior_state, params, full_fn, data)

    print("likelihood:", marginal_likelihood)
    print("train loss:", str(train_loss))
    print("test loss:", str(test_loss))
