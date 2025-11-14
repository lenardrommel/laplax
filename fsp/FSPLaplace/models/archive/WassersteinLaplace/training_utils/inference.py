import jax
import wandb
import haiku as hk
import jax.numpy as jnp

from functools import partial
from jax.example_libraries.optimizers import adam 

jax.config.update("jax_enable_x64", True)


def wasserstein_laplace_inference(
    model, 
    mean_params, 
    dataloader, 
    prior,
    key, 
    config
):
    """
    Compute the posterior distribution Laplace approximation.
    """
    # Get configuration
    likelihood = "normal" if config["experiment"]["name"] == "regression" else "categorical"
    stochastic_layers = config["FSPLaplace"]["stochastic_layers"]
    ll_scale = config["FSPLaplace"]["likelihood_scale"]
    cov_type = config["FSPLaplace"]["cov_type"]
    feature_dim = config["data"]["feature_dim"]
    n_context_points = config["FSPLaplace"]["training"]["n_context_points"]
    
    # Split parameters 
    stochastic_params, static_params = split_parameters(mean_params, stochastic_layers)
    dim = hk.data_structures.tree_size(stochastic_params)

    # Prior covariance
    key, key1 = jax.random.split(key)
    if cov_type == "full": 
        L = 1e-10*jax.random.normal(key1, shape=(dim, dim))
    elif cov_type == "diag":
        L = 0.1*jax.random.uniform(key1, shape=(dim,))

    # Initialize optimizer 
    opt_init, opt_update, get_params = adam(
            config["FSPLaplace"]["training"]["lr"],
            config["FSPLaplace"]["training"]["b1"],
            config["FSPLaplace"]["training"]["b2"],
            config["FSPLaplace"]["training"]["eps"]
        )
    opt_state = opt_init(L)

    # Compute prior covariance
    for step in range(1000):
        # Split the keys
        key, key1, key2 = jax.random.split(key, 3)
        # Sample from feature space
        x_context = jnp.linspace(-2, 2, n_context_points).reshape(-1, 1)
        # x_context = jax.random.uniform(key1, minval=-2, maxval=2, shape=(n_context_points, feature_dim))
        # GP covariance
        func_prior_cov = prior(x_context)[1]
        # Parameter covariance
        fwd = lambda p: model.apply_fn(join_parameters(p, static_params), key2, x_context)
        if cov_type == "full":
            @jax.jit
            def _w2(L):
                cov = L.T @ L #+ 1e-20 * jnp.eye(dim)
                # Covariance Jacobian product
                unravel = jax.flatten_util.ravel_pytree(stochastic_params)[1]
                pytree_cov = jax.vmap(unravel)(cov)
                cov_J = jax.vmap(jax.jvp, in_axes=(None, None, 0))(fwd, (stochastic_params,), (pytree_cov,))[1] # (p,n,1)
                leaves = jax.tree_util.tree_flatten(cov_J)[0]
                Jt_cov = jnp.concatenate([i.reshape(cov.shape[0], -1) for i in leaves], axis=-1).T # (n,p)
                pytree_Jt_cov = jax.vmap(unravel)(Jt_cov)
                # Jacobian covariance Jacobian product
                weight_prior_cov = jax.vmap(jax.jvp, in_axes=(None, None, 0))(fwd, (stochastic_params,), (pytree_Jt_cov,))[1] # (n,n)
                weight_prior_cov = weight_prior_cov.reshape(x_context.shape[0], x_context.shape[0])
                return wasserstein_2(weight_prior_cov, func_prior_cov)
        elif cov_type == "diag":
            raise NotImplemented("Diagonal covariance not implemented yet.")
        # Update covariance
        w2, w2_grad = jax.jit(jax.value_and_grad(_w2))(L)
        opt_state = opt_update(step, w2_grad, opt_state)
        L = get_params(opt_state)
        print(f"Step {step} - Wasserstein = {w2}", flush=True)
        wandb.log(
            {
                "Train/w2": w2
            }
        )
    prior_cov = L.T @ L
    prior_cov_eigvals = jnp.linalg.eigvalsh(prior_cov)
    print("Prior cov eigenvalues", prior_cov_eigvals, flush=True)
    if prior_cov_eigvals.min() < 0:
        prior_cov += 2*jnp.abs(prior_cov_eigvals.min()) * jnp.eye(dim)
        print("Prior cov eigenvalues after preprocessing", jnp.linalg.eigvalsh(prior_cov), flush=True)
    
    # Contribution of the prior covariance
    precision = jnp.linalg.inv(prior_cov)

    # Contribution of the likelihood to the precision
    for x, y in dataloader:
        # Split the keys
        key, key1, key2 = jax.random.split(key, 3)
        
        # Hessian of the likelihood
        H = -likelihood_hessian(
            x, 
            likelihood, 
            model, 
            mean_params, 
            ll_scale, 
            key1
        )

        if cov_type == "full":
            fwd = lambda p: model.apply_fn(join_parameters(p, static_params), key2, x)
            _vjp = jax.vjp(fwd, stochastic_params)[1]
            # Hessian jacobian product
            H_J = jax.vmap(_vjp)(jnp.expand_dims(H, axis=-1))[0]
            leaves = jax.tree_util.tree_flatten(H_J)[0]
            Jt_H = jnp.concatenate([i.reshape(x.shape[0], -1) for i in leaves], axis=-1).T
            # Jacobian Tr Hessian Jacobian product
            Jt_H_J = jax.vmap(_vjp)(jnp.expand_dims(Jt_H, axis=-1))[0]
            leaves = jax.tree_util.tree_flatten(Jt_H_J)[0]
            Jt_H_J = jnp.concatenate([i.reshape(dim, -1) for i in leaves], axis=-1)
            # Update precision
            precision += Jt_H_J
        elif cov_type == "diag": 
            raise NotImplemented("Diagonal covariance not implemented yet.")

    prec_eigvals = jnp.linalg.eigvalsh(precision)  
    print("Precision eigenvalues", prec_eigvals, flush=True)
    if prec_eigvals.min() < 0:
        precision += 2*jnp.abs(prec_eigvals.min()) * jnp.eye(dim)
        print("Precision eigenvalues after preprocessing", jnp.linalg.eigvalsh(precision), flush=True)
                
    # Compute covariance
    if cov_type == "full":
        cov = jnp.linalg.inv(precision)
    elif cov_type == "diag":
        cov = 1 / precision
    print("Covariance eigenvalues", jnp.linalg.eigvalsh(cov), flush=True)

    return cov


def likelihood_hessian(
    x, 
    likelihood, 
    model, 
    mean_params, 
    ll_scale, 
    key
):
    """
    Compute hessian of likelihood with respect to its parameters.
    """
    if likelihood == "normal":
        hessian = -1 / ll_scale**2 * jnp.eye(x.shape[0])
    elif likelihood == "categorical":
        logits = model.apply_fn(mean_params, key, x) # (batch, n_output)
        probs = jnp.clip(jax.nn.softmax(logits, -1), a_min=1e-7, a_max=1-1e-7)
        outer_prod = jnp.einsum('bj,bk->bjk', probs, probs)
        hessian = jax.vmap(jnp.diag)(probs) - outer_prod
    else:
        raise Exception("Likelihood not recognized.")
    
    return hessian


@jax.jit
def wasserstein_2(
    q_cov, 
    p_cov
):
    """Return wasserstein2_(q || p).

    :param mean_q: mean of Gaussian distribution q.
    :param mean_p: mean of Gaussian distribution p.
    :param cov_q: covariance of Gaussian distribution q, 2-D array.
    :param cov_q: covariance of Gaussian distribution p, 2-D array.
    :return:
        KL divergence.
    """
    w2 = jnp.trace(q_cov) 
    w2 += jnp.trace(p_cov)
    eigs = jnp.array(jax.jit(eigvals, device=jax.devices("gpu")[0])(q_cov @ p_cov))
    w2 -= 2 * jnp.sqrt(eigs).real.sum()

    return w2



def eig(a):
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    eigenvalues_shape = jax.ShapeDtypeStruct(a.shape[:-1], complex)
    eigenvectors_shape = jax.ShapeDtypeStruct(a.shape, complex)
    return jax.experimental.host_callback.call(
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        jax.jit(jnp.linalg.eig, device=jax.devices("cpu")[0]),
        a.astype(complex),
        result_shape=(eigenvalues_shape, eigenvectors_shape),
    )
    

@jax.custom_jvp
def eigvals(a):
    """Wraps jnp.linalg.eig so that it can be jit-ed on a machine with GPUs."""
    eigenvalues_shape = jax.ShapeDtypeStruct(a.shape[:-1], complex)
    return jax.experimental.host_callback.call(
        # We force this computation to be performed on the cpu by jit-ing and
        # explicitly specifying the device.
        jax.jit(jnp.linalg.eigvals, device=jax.devices("cpu")[0]),
        a.astype(complex),
        result_shape=eigenvalues_shape,
    )


@eigvals.defjvp
def eigvals_jvp_rule(primals, tangents):
    a, = primals
    da, = tangents

    w, v = jax.jit(eig, device=jax.devices("gpu")[0])(a)

    dot = partial(jax.lax.dot if a.ndim == 2 else jnp.lax.batch_matmul,
                  precision=jax.lax.Precision.HIGHEST)
    vinv_da_v = dot(jnp.linalg.solve(v, da), v)
    dw = jnp.diagonal(vinv_da_v, axis1=-2, axis2=-1)

    return (w,), (dw,)


def split_parameters(
    mean_params, 
    stochastic_layers
):
    """
    Split the model parameters into two sets: 
    -Parameters on which to perform inference
    -Parameters kept as MAP estimates
    :params mean_params: MAP estimate of the model parameters.
    """

    stochastic_params, static_params = hk.data_structures.partition(
        lambda m, n, p: stochastic_layers[int(m[23:]) if m[23:] else 0], mean_params
    )

    return stochastic_params, static_params
    

def join_parameters(
    stochastic_params, 
    static_params
):
    """
    Join inference and static parameters into full set of parameters.
    :params stochastic_params: parameters on which to perform inference.
    :params static_params: parameters left as MAP estimates.
    """
    return hk.data_structures.merge(stochastic_params, static_params)
