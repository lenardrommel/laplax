import jax

import haiku as hk
import jax.experimental
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial
from jax.flatten_util import ravel_pytree
from jax.example_libraries.optimizers import adam

jax.config.update("jax_enable_x64", True)


def sample_laplace_posterior(
    model,
    params,
    dataloader,
    config,
    key,
    mc_samples
):
    """
    Compute covariance under Laplace approximation.

    params:
    - model (Model): neural network.
    - params (jax.tree_util.pytree): parameters of the neural network.
    - dataloader (DataLoader): wrapper for the data.
    - prior (callable): prior distribution.
    - config (dict): configuration dictionary.
    - key (jax.random.PRNGKey): random key.
    - mc_samples (int): number of Monte Carlo samples.

    returns:
    - cov (jnp.array): covariance under Laplace approximation.
    """
    # Get configuration
    lr = config["sampling_laplace"]["inference"]["lr"]
    n_iter = config["sampling_laplace"]["inference"]["n_iter"]
    prior_scale = config["sampling_laplace"]["prior"]["scale_init"]
    n_em_steps = config["sampling_laplace"]["inference"]["n_em_steps"]

    # Keys
    key, key1 = jax.random.split(key)

    # Split parameters 
    dim = hk.data_structures.tree_size(params)
    unravel = ravel_pytree(params)[1]
    print("Number of parameters", dim, flush=True)

    # Parameter samples
    z = jax.random.normal(key1, (mc_samples, dim))
    
    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    opt_state = opt_init(z)
    
    # Optimize
    for i in range(n_em_steps):
        # Collect posterior samples 
        step = 0
        n_samples = len(dataloader.dataset)
        for i in range(n_iter):
            for x, y in dataloader:
                opt_state, loss_value = update(
                    model,
                    opt_state,
                    get_params,
                    opt_update,
                    params,
                    prior_scale, 
                    x,
                    key,
                    n_samples,
                    unravel, 
                    step
                )
                step += 1
            if i % 10 == 0:
                print(f"{i} - Loss: {loss_value.mean()}")
        # Refine prior scale
        z = get_params(opt_state)
        z = jax.vmap(unravel)(z)
        trace = 0.
        for x_b, y_b in dataloader:
            f = lambda p: model.apply_fn(p, key, x_b)
            loss = lambda f: model.loss_fn(f, y_b)
            trace += jax.vmap(
                lambda p: precision_linop(x_b, y_b, x_b, p, f, loss)
            )(z).sum()
        prior_scale = jax.experimental.l2_norm(params) / jnp.sqrt(trace)    

    z = get_params(opt_state)
    z = jax.vmap(unravel)(z)
    
    return z

@partial(jax.jit, static_argnums=(0,))
def precision_linop(
    x_b,
    y_b,
    x, 
    params,
    fun,
    loss
):
    """
    Helper method to compute the GGN-vector product.

    params:
    - x_b (array): batch of inputs.
    - y_b (array): batch of outputs.
    - x (array): input array.

    returns:
    - result (array): output array.
    """
    f = lambda p: fun(p, x_b)
    loss = lambda f: loss(f, y_b)
    f_b, jx = jax.jvp(f, (params,), (x,))
    hjx = jax.jvp(jax.grad(loss), (f_b,), (jx,))[1] # jax.hvp(loss, f_b, jx)
    _, jhjx_fn = jax.vjp(f, params)

    return x @ jhjx_fn(hjx)[0]


@partial(jax.jit, static_argnums=(0,2,3,5,8,9))
def update(
    model,
    opt_state,
    get_params,
    opt_update,
    params,
    prior_scale, 
    x,
    key,
    n_samples,
    unravel, 
    step
):
    """
    Update parameters.

    params:
    """
    # Get parameters
    z = get_params(opt_state)
    
    # Compute gradients
    value, grad = jax.vmap(
        jax.value_and_grad(laplace_sampling_objective), 
        in_axes=(0,None,None,None,None,None,None,None), 
        out_axes=(0,0)
    )(
        z,
        params,
        model,
        prior_scale,
        x,
        key,
        n_samples, 
        unravel
    )
    # Update parameters
    opt_state = jax.vmap(opt_update, in_axes=(None,0,0))(step, grad, opt_state)
    
    return opt_state, value


@partial(jax.jit, static_argnums=(2,3,6,7))
def laplace_sampling_objective(
    z,
    params,
    model,
    prior_scale,
    x,
    key,
    n_samples, 
    unravel
):
    # Get number of outputs
    n_outputs = 1 if model.likelihood == "Gaussian" else model.n_classes

    # Log-likelihood term
    B = -likelihood_hessian(model, params, x, key)
    if model.likelihood == "Gaussian":
        eps = model.ll_scale * jax.random.normal(key, (x.shape[0],))
    elif model.likelihood == "Categorical":
        eps = jnp.concatenate(
            [
                jax.random.multivariate_normal(key, jnp.zeros(n_outputs), B[i]+1e-10*jnp.eye(B[i].shape[0])) 
                for i in range(x.shape[0])
            ]
        )
        B = jsp.linalg.block_diag(*B)
    f = lambda p: model.apply_fn(p, key, x).T.reshape(-1) # (n_batch * n_outputs)
    J_x_z = jax.jvp(f, (params,), (unravel(z),))[1] # (n_outputs * n_batch)

    # Log-prior term
    prior_sample = prior_scale * jax.random.normal(key, (z.shape[0],))
    theta_0 = prior_sample + prior_scale**2 * ravel_pytree(jax.vjp(f, params)[1](B @ eps)[0])[0]

    # Compute sampling loss
    loss = 0.5 * n_samples * J_x_z.T @ B @ J_x_z / x.shape[0]
    loss += 0.5 * jnp.square(z - theta_0).sum() / prior_scale**2

    return loss


@partial(jax.jit, static_argnums=(0,))
def likelihood_hessian(
    model, 
    params,
    x, 
    key
):
    """
    Compute hessian of likelihood with respect to its parameters.

    params:
    - x (jnp.array): input data.
    - ll_scale (float): log-likelihood scale. 

    returns:
    - hessian (jnp.array): hessian of likelihood.
    """
    if model.likelihood == "Gaussian":
        ll_hessian = -1 / model.ll_scale**2 * jnp.eye(x.shape[0])
    elif model.likelihood == "Categorical":
        logits = model.apply_fn(params, key, x) # (n_batch, n_classes)
        probs = jax.nn.softmax(logits, axis=-1)  # (n_batch, n_classes)
        ll_hessian = -jax.vmap(jnp.diag)(probs) + jnp.einsum('bk,bc->bck', probs, probs) # (n_batch, n_classes, n_classes)

    return ll_hessian
