import jax
import jax.numpy as jnp
import optax
from flax import nnx

import laplax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class ModelWrapper(nnx.Module):
    class ModelWrapper:
        """A wrapper class for a neural network model that integrates parameters and data size.

        Attributes:
            model (Callable): The neural network model to be wrapped.
            param (nnx.Param): A parameter object initialized with the given parameter values.
            N (int): The size of the dataset.

        Methods:
            __call__(x):
                Invokes the wrapped model with the input `x`.

            to_float64():
                Converts the model parameters to float64 precision.

        Args:
            model (Callable): The neural network model to be wrapped.
            param (array-like): The initial parameter values for the model.
            data_size (int): The size of the dataset.
        """

    def __init__(self, model, param, data_size, dtype):
        self.model = model
        self.param = nnx.Param(jnp.asarray(param))
        self.N = data_size
        if dtype is jnp.float64:
            self.model = self.to_float64()

    def __call__(self, x):
        return self.model(x)

    def to_float64(self):  # noqa: F811
        graph_def, params = nnx.split(self.model)
        params = laplax.util.tree.to_dtype(params, jnp.float64)
        return nnx.merge(graph_def, params)


def create_model_():
    config = {
        "in_channels": 1,
        "hidden_channels": 50,
        "out_channels": 1,
        "rngs": nnx.Rngs(jax.random.key(0)),
        "param": jnp.log(0.1),
        "data_size": 150,
        "dtype": jnp.float64,
    }
    in_channels = config.get("in_channels", 1)
    hidden_channels = config.get("hidden_channels", 64)
    out_channels = config.get("out_channels", 1)
    rngs = config.get("rngs", nnx.Rngs(0))

    model = Model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        rngs=rngs,
    )
    return model


def create_model(config=None):
    if config is None:
        config = {
            "in_channels": 1,
            "hidden_channels": 50,
            "out_channels": 1,
            "rngs": nnx.Rngs(jax.random.key(0)),
            "param": jnp.log(0.1),
            "data_size": 150,
            "dtype": jnp.float64,
        }
    in_channels = config.get("in_channels", 1)
    hidden_channels = config.get("hidden_channels", 64)
    out_channels = config.get("out_channels", 1)
    rngs = config.get("rngs", nnx.Rngs(0))
    param = config.get("param", None)
    data_size = config.get("data_size", None)
    dtype = config.get("dtype", jnp.float64)

    model = Model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        rngs=rngs,
    )
    return model
    # graph_def, _ = nnx.split(model)

    # def model_fn(input, params):
    #     return nnx.call((graph_def, params))(input)[0]

    # if param is not None and data_size is not None:
    #     model = ModelWrapper(model, param, data_size, dtype)

    # return model, model_fn, graph_def
