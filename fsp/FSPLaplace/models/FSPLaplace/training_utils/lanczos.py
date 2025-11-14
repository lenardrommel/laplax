import jax

import jax.numpy as jnp

from haiku.data_structures import merge

def cg(
    A,
    b,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000
):
    """
    Conjugate gradient method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - x0 (array): initial guess.
    - atol (float): absolut tolerance.
    - tol (float): relative tolerance.
    - max_iter (int): maximum number of iterations.
    """
    @jax.jit
    def _step(values):
        x, r, _, ds, i = values
        alpha = jnp.linalg.norm(r)**2
        A_r = A @ r
        # Double gram schmidt
        d = r - ds @ (ds.T @ A_r)  # d = r - ds[:,:i] @ (ds[:,:i].T @ A_r)
        d = d - ds @ (ds.T @ (A @ d))  # d = d - ds[:,:i] @ (ds[:,:i].T @ (A @ d))
        eta = d.T @ A_r
        x = x + alpha * d / eta
        jax.debug.print("eta: {a} - norm = {r}", a=eta, r=jnp.linalg.norm(r))
        ds = ds.at[:,i].set(d / jnp.sqrt(eta))
        r = b - A @ x
        return x, r, eta, ds, i + 1

    def _cond_fun(values):
        _, r, eta, _, i = values
        return (jnp.linalg.norm(r) > tol) & (i < max_iter) & (eta > min_eta)
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    x = jnp.zeros_like(b)
    ds = jnp.zeros((b.shape[0], max_iter))
    tol = tol #max(tol, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta #max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps

    # CG loop
    _, _, _, ds, i = jax.lax.while_loop(_cond_fun, _step, (x, b, jnp.inf, ds, 0))

    return ds[:,:i]


def lanczos_memory_efficient(
    A,
    b,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000
):
    """
    Lanczos method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - tol (float): relative tolerance.
    - min_eta (float): minimum value for the Lanczos vector.
    - max_iter (int): maximum number of iterations.

    returns:
    - ds (array): Lanczos vectors.
    """
    @jax.jit
    def _step(values):
        x, r, _, r_norm_sq, ds, i = values
        alpha = r_norm_sq
        Ar = A @ r
        d = r - ds @ (ds.T @ Ar)
        d = d - ds @ (ds.T @ (A @ d))
        eta = d @ Ar
        x = x + (alpha / eta) * d
        ds = ds.at[:,i].set(d / jnp.sqrt(eta))
        r = b - A @ x
        r_norm_sq = r @ r
        jax.debug.print("eta: {a} - norm = {r}", a=eta, r=jnp.linalg.norm(r))
        return x, r, eta, r_norm_sq, ds, i + 1

    def _cond_fun(values):
        _, _, eta, r_norm_sq, _, i = values
        return (i < max_iter) & (jnp.sqrt(r_norm_sq) > tol) & (eta > min_eta) 
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    x = jnp.zeros_like(b)
    ds = jnp.zeros((b.shape[0], max_iter))
    tol = tol #max(tol, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta #max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf
    r_norm_sq = 1.
    
    # Lanczos iterations
    _, _, _, _, ds, i = jax.lax.while_loop(_cond_fun, _step, (x, b, eta, r_norm_sq, ds, 0))

    return ds[:,:i]


def lanczos_compute_efficient(
    A,
    b,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000,
    overwrite_b=False, 
):
    """
    Conjugate gradient method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - x0 (array): initial guess.
    - atol (float): absolut tolerance.
    - tol (float): relative tolerance.
    - max_iter (int): maximum number of iterations.
    """
    @jax.jit
    def _step(values):
        ds, rs, rs_norm_sq, p, eta, k = values
        # Compute search direction
        true_fn = lambda _p: rs[:, k] + rs_norm_sq[k] / rs_norm_sq[k - 1] * _p
        false_fn = lambda _p: _p
        p = jax.lax.cond(k > 0, true_fn, false_fn, p)

        # Compute modified Lanzcos vector
        w = A @ p
        eta = p @ w
        ds = ds.at[:, k].set(p / jnp.sqrt(eta))

        # Update residual
        mu = rs_norm_sq[k] / eta
        rs_prev_k = rs # rs[:, :k]
        rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)

        # Full reorthogonalization of residual (double Gram-Schmidt)
        rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq))
        rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq))

        rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1].T @ rs[:, k + 1])
        jax.debug.print("eta: {a} - sq_norm = {r}", a=eta, r=rs_norm_sq[k])
        
        return ds, rs, rs_norm_sq, p, eta, k+1

        
    def _cond_fun(values):
        ds, _, rs_norm_sq, _, eta, k = values
        return (rs_norm_sq[k] > sqtol) & (k < max_iter) #& (eta > min_eta)
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter)) # only implemented for order='K'
    rs = jnp.zeros((b.size, max_iter + 1)) #   only implemented for order='K'
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)

    # Initialize loop variables
    sqtol = tol ** 2#max(tol ** 2, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta #max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf # to make sure the first iteration is done
    rs = rs.at[:,0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k = jax.lax.while_loop(_cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0))


    return ds[:,:k]


def cg_precision(
    A,
    b,
    model,
    x_context,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000
):
    """
    Conjugate gradient method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - x0 (array): initial guess.
    - atol (float): absolut tolerance.
    - tol (float): relative tolerance.
    - max_iter (int): maximum number of iterations.
    """
    prior_tr = model.prior.covariance_trace(x_context) # (n_outputs,)
    f = lambda p: model.apply_fn(merge(p, model.other_params), model.state, model.key, x_context, training=False)[0]

    @jax.jit
    def _step(values):
        x, r, _, ds, i, post_tr = values
        alpha = jnp.linalg.norm(r)**2
        A_r = A @ r
        # Double gram schmidt
        d = r - ds @ (ds.T @ A_r)  # d = r - ds[:,:i] @ (ds[:,:i].T @ A_r)
        d = d - ds @ (ds.T @ (A @ d))  # d = d - ds[:,:i] @ (ds[:,:i].T @ (A @ d))
        eta = d.T @ A_r
        x = x + alpha * d / eta
        ds = ds.at[:,i].set(d / jnp.sqrt(eta))
        r = b - A @ x
        # Compute posterior trace
        jvp = jax.jvp(f, (model.mean_params,), (model.unravel_params(ds[:, i]),))[1] # (n_outputs, n_classes)
        post_tr += jnp.sum(jvp**2, axis=0)
        jax.debug.print("eta: {a} - norm = {b} - post_tr: {c} - prior_tr: {d}", a=eta, b=jnp.linalg.norm(r), c=post_tr, d=prior_tr)

        return x, r, eta, ds, i + 1, post_tr

    def _cond_fun(values):
        _, r, eta, _, i, post_tr = values
        return (jnp.linalg.norm(r) > tol) & (i < max_iter) & (eta > min_eta) & (post_tr < prior_tr).all() 
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    x = jnp.zeros_like(b)
    ds = jnp.zeros((b.shape[0], max_iter))
    post_tr = jnp.zeros_like(prior_tr)
    tol = tol # max(tol, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta   #max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps

    # CG loop
    _, _, _, ds, i, post_tr = jax.lax.while_loop(_cond_fun, _step, (x, b, jnp.inf, ds, 0, post_tr))

    return ds[:,:i-1] if not (post_tr < prior_tr).all() else ds[:,:i] 


def lanczos_memory_efficient_precision(
    A,
    b,
    model,
    x_context,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000
):
    """
    Lanczos method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - tol (float): relative tolerance.
    - min_eta (float): minimum value for the Lanczos vector.
    - max_iter (int): maximum number of iterations.

    returns:
    - ds (array): Lanczos vectors.
    """
    prior_tr = model.prior.covariance_trace(x_context) # (n_outputs,)
    f = lambda p: model.apply_fn(merge(p, model.other_params), model.state, model.key, x_context, training=False)[0]

    @jax.jit
    def _step(values):
        x, r, _, r_norm_sq, ds, i, post_tr= values
        alpha = r_norm_sq
        Ar = A @ r
        d = r - ds @ (ds.T @ Ar)
        d = d - ds @ (ds.T @ (A @ d))
        eta = d @ Ar
        x = x + (alpha / eta) * d
        ds = ds.at[:,i].set(d / jnp.sqrt(eta))
        r = b - A @ x
        r_norm_sq = r @ r

        jvp = jax.jvp(f, (model.mean_params,), (model.unravel_params(ds[:, i]),))[1] # (n_outputs, n_classes)
        post_tr += jnp.sum(jvp**2, axis=0)
        jax.debug.print("eta: {a} - sq_norm = {b} - post_tr: {c} - prior_tr: {d}", a=eta, b=r_norm_sq, c=post_tr, d=prior_tr)
        
        return x, r, eta, r_norm_sq, ds, i + 1, post_tr

    def _cond_fun(values):
        _, _, eta, r_norm_sq, _, i, post_tr = values
        return (i < max_iter) & (jnp.sqrt(r_norm_sq) > tol) & (eta > min_eta) & (post_tr < prior_tr).all() 
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    x = jnp.zeros_like(b)
    ds = jnp.zeros((b.shape[0], max_iter))
    post_tr = jnp.zeros_like(prior_tr)
    tol = tol #max(tol, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta # max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf
    r_norm_sq = 1.
    
    # Lanczos iterations
    _, _, _, _, ds, i, post_tr = jax.lax.while_loop(_cond_fun, _step, (x, b, eta, r_norm_sq, ds, 0, post_tr))

    return ds[:,:i-1] if not (post_tr < prior_tr).all() else ds[:,:i] 


def lanczos_compute_efficient_precision(
    A,
    b,
    model,
    x_context,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000,
    overwrite_b=False, 
):
    """
    Conjugate gradient method to solve the linear system Ax = b.

    params:
    - A (callable): linear operator.
    - b (array): right-hand side.
    - x0 (array): initial guess.
    - atol (float): absolut tolerance.
    - tol (float): relative tolerance.
    - max_iter (int): maximum number of iterations.
    """
    prior_tr = model.prior.covariance_trace(x_context) # (n_outputs,)
    f = lambda p: model.apply_fn(merge(p, model.other_params), model.state, model.key, x_context, training=False)[0]

    @jax.jit
    def _step(values):
        ds, rs, rs_norm_sq, p, eta, k, post_tr = values
        # Compute search direction
        true_fn = lambda _p: rs[:, k] + rs_norm_sq[k] / rs_norm_sq[k - 1] * _p
        false_fn = lambda _p: _p
        p = jax.lax.cond(k > 0, true_fn, false_fn, p)

        # Compute modified Lanzcos vector
        w = A @ p
        eta = p @ w
        ds = ds.at[:, k].set(p / jnp.sqrt(eta))

        # Update residual
        mu = rs_norm_sq[k] / eta
        rs_prev_k = rs # rs[:, :k]
        rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)

        # Full reorthogonalization of residual (double Gram-Schmidt)
        rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq))
        rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq))

        rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1].T @ rs[:, k + 1])

        # Update posterior trace
        jvp = jax.jvp(f, (model.mean_params,), (model.unravel_params(ds[:, k]),))[1] # (n_outputs, n_classes)
        post_tr += jnp.sum(jvp**2, axis=0)
        jax.debug.print("eta: {a} - sq_norm = {b} - post_tr: {c} - prior_tr: {d}", a=eta, b=rs_norm_sq[k], c=post_tr, d=prior_tr)
        
        return ds, rs, rs_norm_sq, p, eta, k+1, post_tr

        
    def _cond_fun(values):
        _, _, rs_norm_sq, _, eta, k, post_tr = values
        return (rs_norm_sq[k] > sqtol) & (k < max_iter) & (eta > min_eta) & (post_tr < prior_tr).all() 
    
    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter)) # only implemented for order='K'
    rs = jnp.zeros((b.size, max_iter + 1)) #   only implemented for order='K'
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)
    post_tr = jnp.zeros_like(prior_tr)

    # Initialize loop variables
    sqtol = tol ** 2 #max(tol ** 2, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta # max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf # to make sure the first iteration is done
    rs = rs.at[:,0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k, post_tr = jax.lax.while_loop(_cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0, post_tr))

    
    return ds[:,:k-1] if not (post_tr < prior_tr).all() else ds[:,:k] 