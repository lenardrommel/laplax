from laplax.eval.marginal_likelihood import calculate_marginal_likelihood, marg_lik_with_hessian, calculate_marginal_likelihood_diagonal
from helper import get_sinusoid_example, get_sinusoid_examples_spaced
import matplotlib.pyplot as plt
from laplax.curv.ggn import create_ggn_mv
from laplax.util.flatten import create_pytree_flattener, wrap_function
import jax
import jax.numpy as jnp
import optax
from jax import random
from flax import nnx
from laplax.curv.cov import create_posterior_fn
import jax
import pickle
import flax.serialization as ser


def create_model(rngs):
    model = model_class(rngs)
    return model

def save_model(name, model):
    # Get model parameters and state
    _, params, rest = nnx.split(model, nnx.Param, ...)

    # Convert parameters and state to bytes for easy saving


    # Save parameters and state to disk with the given name
    with open(f"models/model_params_{name}.pkl", "wb") as f:
        pickle.dump(params,f)

    with open(f"models/model_rest_{name}.pkl", "wb") as f:
        pickle.dump(rest,f)

    print("Model saved successfully!")


# Define a function to load the model
def load_model(name, model_class):
    # Load the saved parameters and state from disk
    with open(f"models/model_params_{name}.pkl", "rb") as f:
        params = pickle.load(f)

    with open(f"models/model_rest_{name}.pkl", "rb") as f:
        rest = pickle.load(f)

    return params, rest



# generate training data
# generate training data
n_data = 150
sigma_noise = 0.2
batch_size = 32
rng_key = random.key(711)
loss_fn = lambda y_pred, y_true: jnp.mean((y_pred - y_true) ** 2)  # noqa: E731

X_train, y_train, train_loader, X_test = get_sinusoid_examples_spaced(n_data, sigma_noise, batch_size, rng_key)
train_loader = list(zip(X_train, y_train, strict=False))

plt.scatter(X_train, jnp.sin(X_train), color='green', label='sin(x)')
plt.scatter(X_train, y_train)


# Too Simple: Cannot capture sine properly
class MLP_Simple(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 2, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(2, 1, rngs=rngs)
        self.learning_rate = 1e-2
        self.epochs = 1

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


# Reasonable Complexity #1: Captures sine wave well
class MLP_Reasonable(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 10, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(10, 1, rngs=rngs)
        self.learning_rate = 1e-2
        self.epochs = 10

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


class MLP_Good_Fit(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 50, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(50, 1, rngs=rngs)
        self.learning_rate = 6e-3
        self.epochs = 150

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

class MLP_Overfit(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(1, 25, rngs=rngs)
        self.tanh = nnx.tanh
        self.linear2 = nnx.Linear(25, 50, rngs=rngs)
        self.linear3 = nnx.Linear(50, 25, rngs=rngs)
        self.linear4 = nnx.Linear(25, 1, rngs=rngs)
        self.learning_rate = 9e-4
        self.epochs = 500

    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        x = self.tanh(x)
        x = self.linear4(x)
        return x



#models = [("simple", MLP_Simple), ("reasonable", MLP_Reasonable), ("good_fit", MLP_Good_Fit), ("overfit"), MLP_Overfit)]
models = [("reasonable", MLP_Reasonable)]
load_saved_model = False

for name, model_class in models:

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
        optimizer = nnx.Optimizer(model, optax.adamw(model.learning_rate))
        for i in range(n_epochs):
            for X, y in train_loader:
                update(X, y, model, optimizer)
                # You can log the loss here if you want
            if i % 10 == 0:
                pred = model(X_test)
                y_test = jnp.sin(X_test)
                test_loss = loss_fn(pred, y_test).item()
                pred_train = model(X_train)
                train_loss = loss_fn(pred_train, y_train).item()
                print("\n")
                print(f"epoch: {str(i)}")
                print("train loss:", str(train_loss))
                print("test loss:", str(test_loss))

        return model

    if load_saved_model:
        params, rest = load_model(name, model_class)
        # Recreate the model by merging parameters and state
        rng = nnx.Rngs(0)
        model = create_model(rng)
        nnx.update(model, nnx.GraphState.merge(params, rest))
    else:
        rng_key = nnx.Rngs(0)
        model = train_model(train_loader, rng_key)
        save_model(name, model)
        _, params, rest = nnx.split(model, nnx.Param, ...)

    data = {"input": X_train, "target": y_train}


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

    posterior_state = posterior_fn({"prior_prec": 1})

    pred = model(X_test)
    y_test = jnp.sin(X_test)
    test_loss = loss_fn(pred, y_test).item()


    pred_train = model(X_train)
    train_loss = loss_fn(pred_train, y_train).item()

    #marginal_likelihood = marg_lik_with_hessian(params, full_fn, data)
    marginal_likelihood = calculate_marginal_likelihood(posterior_state, params, full_fn, data)
    marginal_likelihood_2 = calculate_marginal_likelihood_diagonal(posterior_state, params, full_fn, data)
    marginal_likelihood_3 = marg_lik_with_hessian(params, full_fn, data)
    flatten, _ = create_pytree_flattener(params)
    num_params = len(flatten(params))
    print(f"{name} ({num_params} parameters)")
    print("likelihood:", marginal_likelihood)
    print("train loss:", str(train_loss))
    print("test loss:", str(test_loss))

    plt.scatter(X_train, y_train, color="green")
    plt.scatter(X_test, y_test, color='red')
    pred = pred.primal
    plt.scatter(X_test, pred, color="blue")
    plt.show()
