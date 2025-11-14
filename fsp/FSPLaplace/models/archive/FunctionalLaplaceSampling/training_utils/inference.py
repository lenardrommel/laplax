import jax
import optax 

import haiku as hk
import jax.numpy as jnp
import jax.scipy as jsp

from jax.scipy.optimize import minimize
from functools import partial
from jax.example_libraries.optimizers import adam

jax.config.update("jax_enable_x64", True)


def sample_laplace_posterior(
    model,
    params,
    dataloader,
    prior,
    config,
    key,
    mc_samples,
    x_context
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
    lr = config["flaplace_sampling"]["training"]["sampling_lr"]
    max_context_val = config["flaplace_sampling"]["training"]["max_context_val"]
    min_context_val = config["flaplace_sampling"]["training"]["min_context_val"]
    n_context_points = config["flaplace_sampling"]["training"]["n_context_points"]
    context_selection = config["flaplace_sampling"]["training"]["context_selection"]
    nb_iterations_sampling = config["flaplace_sampling"]["training"]["nb_iterations_sampling"]

    # Keys
    key, key1, key2 = jax.random.split(key, 3)

    # Split parameters 
    sto_params, det_params = model.partition_parameters(params)
    dim = hk.data_structures.tree_size(sto_params)
    print("Number of parameters", dim, flush=True)

    # Parameter samples
    sampled_params = jax.random.normal(key1, (mc_samples, dim))
    unravel = jax.flatten_util.ravel_pytree(sto_params)[1]
    sampled_params = jax.vmap(unravel)(sampled_params)

    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    opt_state = opt_init(sampled_params)

    # Sample context points
    x = dataloader.dataset.X[0]
    # x_context = select_context_points(
    #     n_context_points,
    #     context_selection,
    #     max_context_val,
    #     min_context_val,
    #     x.shape,
    #     key2, 
    #     x
    # )
    
    x = dataloader.dataset.X
    keys = jax.random.split(key, mc_samples)
    sample_fn = lambda k: optimize_with_cg(
        model, 
        prior,
        sto_params, 
        det_params, 
        x,
        x_context,
        k
    )
    sampled_params = jax.vmap(sample_fn)(keys)
    sampled_params = jax.vmap(unravel)(sampled_params)


    # # Optimize
    # step = 0
    # n_samples = len(dataloader.dataset)
    # keys = jax.random.split(key, mc_samples)
    # for i in range(nb_iterations_sampling):
    #     for x, y in dataloader:
    #         opt_state, loss_value = update(
    #             model,
    #             opt_state, 
    #             get_params,
    #             opt_update,
    #             sto_params, 
    #             det_params,
    #             prior, 
    #             x,
    #             x_context,
    #             keys,
    #             n_samples,
    #             step
    #         )
    #         step += 1
    #     if i % 100 == 0:
    #         print(f"{i} - Loss: {loss_value.mean()}")

    # # Get parameters
    # sampled_params = get_params(opt_state)

    return sampled_params


def optimize_with_cg(
    model, 
    prior,
    sto_params, 
    det_params, 
    x,
    x_context,
    key
):
    # Split key
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    # Get number of outputs
    n_outputs = 1 if model.likelihood == "Gaussian" else model.n_classes

    # Log-likelihood term
    params = model.merge_parameters(sto_params, det_params)
    B = -likelihood_hessian(model, params, x, key1)
    if model.likelihood == "Gaussian":
        eps = model.ll_scale * jax.random.normal(key2, (x.shape[0], 1)) # (n_batch, n_outputs)
    elif model.likelihood == "Categorical":
        try:
            eps = jax.random.multivariate_normal(key2, mean=jnp.zeros(n_outputs,x.shape[0]), cov=B+1e-10*jnp.eye(B.shape[0]))
        except:
            print("B", jnp.linalg.eigvalsh(B))
    
    Jac_x = jax.jacrev(lambda p: model.apply_fn(model.merge_parameters(p, det_params), key3, x))(sto_params)
    v_Jac_x = jnp.concatenate(
        [l.reshape(x.shape[0],n_outputs,-1) for l in jax.tree_util.tree_leaves(Jac_x)],
        axis=-1
    )

    # Log-prior term
    prior_mean, prior_cov = prior(x_context)
    f_x_ctxt = jnp.stack(
        [
            jax.random.multivariate_normal(
                key4, 
                mean=prior_mean[:,i], 
                cov=prior_cov[:,:,i]
            )
            for i in range(n_outputs)
        ], 
        axis=0
    )
    Jac_x_ctxt = jax.jacrev(lambda p: model.apply_fn(model.merge_parameters(p, det_params), key5, x_context))(sto_params)
    v_Jac_x_ctxt = jnp.concatenate(
        [l.reshape(x_context.shape[0],n_outputs,-1) for l in jax.tree_util.tree_leaves(Jac_x_ctxt)],
        axis=-1
    )
    print("prior_cov", prior_cov.shape)
    print("v_Jac_x_ctxt", v_Jac_x_ctxt.shape)   
    K_inv_jac_x_ctxt = jax.vmap(lambda K, J: jnp.linalg.solve(K, J), in_axes=(2,1))(prior_cov, v_Jac_x_ctxt) # .transpose(1,0,2) 
    print("K_inv_jac_x_ctxt", K_inv_jac_x_ctxt.shape)
    print("f_x_ctxt", f_x_ctxt.shape)
    if model.likelihood == "Gaussian":
        A = 1 / model.ll_scale**2 * jnp.einsum('bcp,bkq->pq', v_Jac_x, v_Jac_x) + jnp.einsum('bcp,bcq->pq', v_Jac_x_ctxt.transpose(1,0,2), K_inv_jac_x_ctxt)
        b = 1 / model.ll_scale**2 * jnp.einsum('bcp,bc->p', v_Jac_x, eps) + jnp.einsum('bcp,bc->p', K_inv_jac_x_ctxt, f_x_ctxt)
    elif model.likelihood == "Categorical":
        A = jnp.einsum('bcp,bck,bkq->pq', v_Jac_x, B, v_Jac_x) + jnp.einsum('bcp,bcq->pq', v_Jac_x, K_inv_jac_x_ctxt)
        b = jnp.einsum('bcp,bc->p', v_Jac_x, eps) + jnp.einsum('bcp,bc->p', K_inv_jac_x_ctxt, f_x_ctxt)
    
    #sampled_params, _ = jax.scipy.sparse.linalg.cg(A, b, x0=sampled_params, maxiter=10000)
    sampled_params= jnp.linalg.solve(A, b)

    return sampled_params



@partial(jax.jit, static_argnums=(0,2,3,6,10))
def update(
    model,
    opt_state, 
    get_params,
    opt_update,
    sto_params, 
    det_params,
    prior, 
    x,
    x_context,
    keys,
    n_samples,
    step
):
    """
    Update parameters.

    params:
    """
    # Get parameters
    sampled_params = get_params(opt_state)
    
    # Compute gradients
    value, grad = jax.vmap(
        jax.value_and_grad(laplace_sampling_objective), 
        in_axes=(0,None,None,None,None,None,None,0,None), 
        out_axes=(0,0)
    )(
        sampled_params,
        sto_params,
        det_params,
        model,
        prior,
        x,
        x_context,
        keys, 
        n_samples
    )
    # Update parameters
    opt_state = jax.vmap(opt_update, in_axes=(None,0,0))(step, grad, opt_state)
    
    return opt_state, value


@partial(jax.jit, static_argnums=(3,4,8))
def laplace_sampling_objective(
    sampled_params,
    sto_params,
    det_params,
    model,
    prior,
    x,
    x_context,
    key, 
    n_samples
):
    unravel = jax.flatten_util.ravel_pytree(sto_params)[1]
    sampled_params = unravel(sampled_params)
    # Split key
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    # Get number of outputs
    n_outputs = 1 if model.likelihood == "Gaussian" else model.n_classes

    # Log-likelihood term
    params = model.merge_parameters(sto_params, det_params)
    B = -likelihood_hessian(model, params, x, key1)
    if model.likelihood == "Gaussian":
        eps = model.ll_scale * jax.random.normal(key2, (x.shape[0],)).reshape(-1) # (n_outputs * n_batch)
    elif model.likelihood == "Categorical":
        try:
            eps = jax.random.multivariate_normal(key2, mean=jnp.zeros(n_outputs*x.shape[0]), cov=B+1e-10*jnp.eye(B.shape[0])).reshape(-1)
        except:
            print("B", jnp.linalg.eigvalsh(B))
    fwd = lambda p: model.apply_fn(model.merge_parameters(p, det_params), key3, x) 
    J_x_z = jax.jvp(fwd, (sto_params,), (sampled_params,))[1].reshape(-1) # (n_outputs * n_batch)

    # Log-prior term
    prior_mean, prior_cov = prior(x_context)
    f_x_ctxt = jnp.concatenate(
        [
            jax.random.multivariate_normal(
                key4, 
                mean=prior_mean[:,i], 
                cov=prior_cov[:,:,i]
            ).reshape(-1)
            for i in range(n_outputs)
        ], 
        axis=0
    )
    K = jsp.linalg.block_diag(*prior_cov.T)
    fwd = lambda p: model.apply_fn(model.merge_parameters(p, det_params), key5, x_context)
    J_x_ctxt_z = jax.jvp(fwd, (sto_params,), (sampled_params,))[1].reshape(-1) #(n_batch, n_outputs)

    # Compute sampling loss
    loss = 0.5 * n_samples * (J_x_z - eps).T @ B @ (J_x_z - eps) / x.shape[0]
    loss += 0.5 * (J_x_ctxt_z - f_x_ctxt).T @ jnp.linalg.solve(K, J_x_ctxt_z - f_x_ctxt)

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


def select_context_points(
	n_context_points,
    context_selection,
	context_points_maxval,
    context_points_minval,
    datapoint_shape,
	key, 
    x
):
    """
    Select context points.

    params:
    - n_context_points (int): number of context points to select.
    - context_selection (str): context selection method.
    - context_points_maxval (float): maximum value of context points.
    - context_points_minval (float): minimum value of context points.
    - x_shape (jnp.array): shape of data.
    - key: random key.

    returns:
    - context points (jnp.array): context points.
    """
    if context_selection == "random":
        context_points = jax.random.uniform(
            key=key,
            shape=(n_context_points,)+datapoint_shape,
            minval=context_points_minval,
            maxval=context_points_maxval,
        )
    elif context_selection == "random_monochrome":
        n, h, w, c = x.shape
        X_reshaped = x.reshape(n, h * w * c)
        random_indices = jax.random.randint(key, shape=(n_context_points, h, w, c), minval=0, maxval=n)
        context_points = X_reshaped[random_indices, jnp.arange(c)].reshape(n_context_points, h, w, c)
    elif context_selection == "grid":
        assert datapoint_shape[-1] in [1,2], "Grid context selection only works for 1D or 2D features."
        if datapoint_shape[-1]  == 1:
            context_points = jnp.linspace(
                context_points_minval, 
                context_points_maxval, 
                n_context_points
            ).reshape(-1, 1)
        elif datapoint_shape[-1]  == 2:
            x1 = jnp.linspace(-1, 1, np.sqrt(n_context_points).astype(int))
            x2 = jnp.linspace(-1, 1, np.sqrt(n_context_points).astype(int))
            x = jnp.meshgrid(x1, x2, indexing='ij')
            context_points = jnp.stack(x, axis=-1).reshape(-1, 2)
    
    return context_points