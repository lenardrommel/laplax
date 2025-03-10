import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import laplax
from laplax.util.mv import to_dense, diagonal
from laplax.util.tree import get_size
from laplax.util.flatten import create_pytree_flattener, wrap_function


def calculate_marginal_likelihood(posterior_state, params, full_fn, data):
    mv = posterior_state.cov_mv(posterior_state.state)
    flatten, unflatten = create_pytree_flattener(params)
    identity = lambda x: x

    mv_ = wrap_function(mv, input_fn=identity, output_fn=flatten)
    cov_ = to_dense(mv_, layout=get_size(params))

    ggn_det = jnp.linalg.det(cov_)
    log_det_H = jnp.log(ggn_det)

    regularization_term = 0.5 * log_det_H + 0.5 * jnp.log(2 * jnp.pi) * len(params)


    log_p_D_theta_star = - cov_[0].size * full_fn(params, data)

    log_marginal_likelihood = log_p_D_theta_star - regularization_term

    return log_marginal_likelihood


def calculate_marginal_likelihood_diagonal(posterior_state, params, full_fn, data):
    mv = posterior_state.cov_mv(posterior_state.state)
    flatten, unflatten = create_pytree_flattener(params)
    identity = lambda x: x

    mv_ = wrap_function(mv, input_fn=identity, output_fn=flatten)
    diagonal_cov = diagonal(mv_, layout=get_size(params))

    log_det_H = jnp.sum(jnp.log(diagonal_cov))
    regularization_term = 0.5 * log_det_H + 0.5 * jnp.log(2 * jnp.pi) * len(params)

    log_p_D_theta_star = - diagonal_cov[0].size * full_fn(params, data)

    log_marginal_likelihood = log_p_D_theta_star - regularization_term

    return log_marginal_likelihood


def marg_lik_with_hessian(params, full_fn, data):

    log_p_D_theta_star = - len(data["input"]) * full_fn(params, data)

    H_theta_star = jax.hessian(lambda p: full_fn(p, data))(params)
    H_theta_star = create_pytree_flattener(H_theta_star)[0](H_theta_star)
    size = int(jnp.sqrt(len(H_theta_star)))
    H_theta_star = H_theta_star.reshape(size, size)

    epsilon = 0.1
    H_theta_star += epsilon * jnp.eye(size)
    log_det_H = jnp.log(linalg.det(H_theta_star))


    # Step 4: Calculate the regularization term
    regularization_term = 0.5 * log_det_H + 0.5 * jnp.log(2 * jnp.pi)

    # Step 5: Calculate the final quantity
    return log_p_D_theta_star - regularization_term