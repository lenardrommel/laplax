import jax
from jax import numpy as jnp

from objective import n_gaussian_log_posterior_objective
from examples.fsp.model import Model


def select_context_points(
    n_context_points,
    context_selection,
    context_points_maxval,
    context_points_minval,
    datapoint_shape,
    key,
    x,
):
    if context_selection == "random":
        context_points = jax.random.uniform(
            key=key,
            shape=(n_context_points,) + datapoint_shape,
            minval=context_points_minval,
            maxval=context_points_maxval,
        )
    return context_points
