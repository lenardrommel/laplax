import jax
import jax.numpy as jnp

import laplax
from laplax.util.mv import to_dense, diagonal
from laplax.util.tree import get_size
from laplax.util.flatten import create_pytree_flattener, wrap_function


def calculate_marginal_likelihood(posterior_state, params, full_fn, data):
    mv = posterior_state.cov_mv(posterior_state.state)
    flatten, unflatten = create_pytree_flattener(params)
    identity = lambda x: x

    mv_ = wrap_function(mv, input_fn=identity, output_fn=flatten)
    cov = to_dense(mv_, layout=get_size(params))

    ggn_det = jnp.linalg.det(cov)
    log_p_D_theta_star = jnp.log(full_fn(params, data))

    log_marginal_likelihood = log_p_D_theta_star - 0.5 * jnp.log(jnp.abs(((1 / (2 * jnp.pi)) * ggn_det)))
    return log_marginal_likelihood


def calculate_marginal_likelihood_diagonal(posterior_state, params, full_fn, data):
    mv = posterior_state.cov_mv(posterior_state.state)
    flatten, unflatten = create_pytree_flattener(params)
    identity = lambda x: x

    mv_ = wrap_function(mv, input_fn=identity, output_fn=flatten)
    diagonal_cov = diagonal(mv_, layout=get_size(params))

    log_ggn_det = jnp.sum(jnp.log(diagonal_cov))

    log_p_D_theta_star = jnp.log(full_fn(params, data))

    log_marginal_likelihood = log_p_D_theta_star - 0.5 * (jnp.log(1 / (2 * jnp.pi)) + log_ggn_det)

    return log_marginal_likelihood