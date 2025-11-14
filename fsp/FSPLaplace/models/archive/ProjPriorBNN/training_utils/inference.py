import jax
import haiku as hk
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def inference(
    model, 
    mean_params, 
    prior, 
    key, 
    dataloader,
    config
):
    """
    Compute the posterior distribution of the model parameters.
    """
    # Get configuration
    stochastic_layers = config["proj_prior_bnn"]["stochastic_layers"]
    cov_type = config["proj_prior_bnn"]["cov_type"]
    feature_dim = config["data"]["feature_dim"]
    
    # Split parameters 
    stochastic_params, static_params = split_parameters(mean_params, stochastic_layers)
    dim = hk.data_structures.tree_size(stochastic_params)

    # Get data
    X_train, y_train = dataloader.get_data(data_split="train")
    X_train = X_train.reshape(-1, 1, feature_dim)
    
    # Integrate with a Grid of points
    P, A = jnp.zeros((dim, dim)), jnp.zeros((dim, dim))
    c = int(config["proj_prior_bnn"]["mc_samples"]**0.5) 
    for i in range(c):
        print("Sample: ", i, flush=True)
        key, key1 = jax.random.split(key)
        x = -2 + 4 * i * jnp.ones((1,1)) / (c-1)
        fx = lambda p: model.apply_fn(join_parameters(p, static_params), key1, x)
        fx_jac = jax.jacrev(fx)(stochastic_params)
        leaves = jax.tree_util.tree_flatten(fx_jac)[0]
        fx_jac  = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
        # Update P
        P += fx_jac @ fx_jac.T
        # Update A
        for j in range(c):
            # Split the keys
            key, key1 = jax.random.split(key)    
            # Get data
            y =  -2 + 4 * j * jnp.ones((1,1)) / (c-1)
            fy = lambda p: model.apply_fn(join_parameters(p, static_params), key1, y)
            fy_jac = jax.jacrev(fy)(stochastic_params)
            leaves = jax.tree_util.tree_flatten(fy_jac)[0]
            fy_jac = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
            # Compute A 
            A += prior.cross_covariance(x, y) * fx_jac @ fy_jac.T
            ############ Possibly faster implementation #############
            # # Compute A 
            # fx_jvp = lambda v: jax.jvp(fx, (w,), (v,))[1]
            # fx_jtvp = lambda v: jax.linear_transpose(fx_jvp, jnp.ones(dims))(v)[0]
            # fy_vjp = lambda v: jax.vjp(fy, w)[1](v)[0]
            # jx_T = jax.vmap(fx_jtvp)(jnp.eye(n)).T
            # A += jax.vmap(fy_vjp)(jx_T)
            # # Compute P
            # fx_jvp = lambda v: jax.jvp(fx, (w,), (v,))[1]
            # fx_jtvp = lambda v: jax.linear_transpose(fx_jvp, jnp.ones(dims))(v)[0]
            # fx_vjp = lambda v: jax.vjp(fx, w)[1](v)[0]
            # jx_T = jax.vmap(fx_jtvp)(jnp.eye(n)).T
            # P += jax.vmap(fx_vjp)(jx_T)
    # A /= c**2
    # P /= c

    for i, x in enumerate(X_train):
        print("Sample: ", i, flush=True)
        key, key1 = jax.random.split(key)
        fx = lambda p: model.apply_fn(join_parameters(p, static_params), key1, x)
        fx_jac = jax.jacrev(fx)(stochastic_params)
        leaves = jax.tree_util.tree_flatten(fx_jac)[0]
        fx_jac  = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
        # Update P
        P += fx_jac @ fx_jac.T
        # Update A
        for y in X_train:
            # Split the keys
            key, key1 = jax.random.split(key)    
            fy = lambda p: model.apply_fn(join_parameters(p, static_params), key1, y)
            fy_jac = jax.jacrev(fy)(stochastic_params)
            leaves = jax.tree_util.tree_flatten(fy_jac)[0]
            fy_jac = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
            # Compute A 
            A += prior.cross_covariance(x, y) * fx_jac @ fy_jac.T
    A /= c**2 + X_train.shape[0]**2
    P /= c + X_train.shape[0]

    # Integrate with trapezoidal rule on grid of points
    # A, P = [], []
    # c = int(config["proj_prior_bnn"]["mc_samples"]**0.5) 
    # for i in range(c):
    #     print("Sample: ", i, flush=True)
    #     key, key1 = jax.random.split(key)
    #     x = -2 + 4 * i * jnp.ones((1,1)) / (c-1)
    #     fx = lambda p: model.apply_fn(join_parameters(p, static_params), key1, x)
    #     fx_jac = jax.jacrev(fx)(stochastic_params)
    #     leaves = jax.tree_util.tree_flatten(fx_jac)[0]
    #     fx_jac  = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
    #     # Update P
    #     P += [fx_jac @ fx_jac.T] 
    #     # Update A
    #     _A  = []
    #     for j in range(c):
    #         # Split the keys
    #         key, key1 = jax.random.split(key)    
    #         # Get data
    #         y =  -2 + 4 * j * jnp.ones((1,1)) / (c-1)
    #         fy = lambda p: model.apply_fn(join_parameters(p, static_params), key1, y)
    #         fy_jac = jax.jacrev(fy)(stochastic_params)
    #         leaves = jax.tree_util.tree_flatten(fy_jac)[0]
    #         fy_jac = jnp.concatenate([a.reshape(1, -1) for a in leaves], axis=-1).reshape(-1, 1)
    #         # Compute A 
    #         _A += [prior.cross_covariance(x, y) * fx_jac @ fy_jac.T]
    #     _A = jnp.stack(_A, axis=0)
    #     A += [jnp.trapz(_A, dx=1/c, axis=0)]
    # A = jnp.stack(A)
    # A = jnp.trapz(A, dx=1/c, axis=0)
    # P = jnp.stack(P)
    # P = jnp.trapz(P, dx=1/c, axis=0) 
  
    lin_ind_idx = linear_independant_row_idx(P)
    print("Linearly independant rows", lin_ind_idx, flush=True)
    # _P = P[lin_ind_idx,:][:,lin_ind_idx]
    # _A = A[lin_ind_idx,:][:,lin_ind_idx]
    # print("Eigenvalues of _P matrix", jnp.linalg.eigvalsh(_P), flush=True)
    # print("Eigenvalues of _A matrix", jnp.linalg.eigvalsh(_A), flush=True)
    # _P_inv = jnp.linalg.inv(_P)
    # print("Shape _P_inv and _A", _P_inv.shape, _A.shape, flush=True)
    # _cov = _P_inv @ _A @ _P_inv
    # print("Eigenvalues of _cov matrix", jnp.linalg.eigvalsh(_cov), flush=True)

    print("P matrix", P, flush=True)
    P_lam, P_u = jnp.linalg.eigh(P)
    print("Eigenvalues of P matrix", P_lam, flush=True)
    P_lam = P_lam.at[P_lam < 1e-3].set(jnp.inf)
    P_inv = P_u @ jnp.diag(1/P_lam) @ P_u.T
    P += (2*jnp.abs((1/P_lam).min()) + 1e-15)* jnp.eye(dim)
    print("Eigenvalues of P_inv matrix", jnp.linalg.eigvalsh(P_inv), flush=True)

    # print("Eigenvalues of P matrix", jnp.linalg.eigvalsh(P), flush=True)
    # P += 2*jnp.abs(jnp.linalg.eigvalsh(P).min())* jnp.eye(dim)
    # P_inv = jnp.linalg.inv(P)
    # print("Eigenvalues of P_inv matrix", jnp.linalg.eigvalsh(P_inv), flush=True)


    print("Eigenvalues of A matrix", jnp.linalg.eigvalsh(A), flush=True)
    A += 2*jnp.abs(jnp.linalg.eigvalsh(A).min()) * jnp.eye(dim)

    cov = P_inv @ A @ P_inv

    print("Eigenvalues of cov matrix", jnp.linalg.eigvalsh(cov), flush=True)
    cov += 2*jnp.abs(jnp.linalg.eigvalsh(cov).min()) * jnp.eye(dim)
    print("Cov eigenvalues after preprocessing", jnp.linalg.eigvalsh(cov), flush=True)


    ## TMP ##
    x = jnp.arange(-2, 2, 0.1).reshape(-1, 1)
    fx = lambda p: model.apply_fn(join_parameters(p, static_params), key1, x)
    fx_jac = jax.jacrev(fx)(stochastic_params)
    leaves = jax.tree_util.tree_flatten(fx_jac)[0]
    fx_jac  = jnp.concatenate([a.reshape(x.shape[0], -1) for a in leaves], axis=-1).reshape(x.shape[0], -1)
    import matplotlib.pylab as pl
    colors = pl.cm.jet(jnp.linspace(0,1,fx_jac.shape[-1]))
    import matplotlib.pyplot as plt
    for i in range(fx_jac.shape[-1]):
        plt.plot(
            x.reshape(-1), 
            fx_jac[:,i], 
            linewidth=1, 
            color=colors[i]
        )
    plt.xlabel("x")
    plt.ylabel(r"$\phi(x)_i$")
    plt.savefig(
        f"eigenfunctions_{config['proj_prior_bnn']['activation_fn']}.pdf", 
        bbox_inches='tight'
    )
    plt.close()

    return cov


def linear_independant_row_idx(X):
    import jax.scipy as jsp
    R = jsp.linalg.qr(X, mode="r")[0]
    diag_R = jnp.diagonal(R)
    mask = (jnp.abs(diag_R) >= jnp.abs(diag_R).max() * R.shape[0] * jnp.finfo(R.dtype).eps)
    non_zero_idx = jnp.nonzero(mask)[0]

    return non_zero_idx


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
