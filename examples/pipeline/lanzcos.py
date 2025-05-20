import jax
import jax.numpy as jnp


def lanczos_compute_efficient(
    A,
    b,
    tol=1e-3,
    min_eta=1e-20,
    max_iter=1000,
    overwrite_b=False,
):
    """Conjugate gradient method to solve the linear system Ax = b.

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
        rs_prev_k = rs  # rs[:, :k]
        rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)

        # Full reorthogonalization of residual (double Gram-Schmidt)
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )
        rs = rs.at[:, k + 1].set(
            rs[:, k + 1] - rs_prev_k @ ((rs_prev_k.T @ rs[:, k + 1]) / rs_norm_sq)
        )

        rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1].T @ rs[:, k + 1])
        jax.debug.print("eta: {a} - sq_norm = {r}", a=eta, r=rs_norm_sq[k])

        return ds, rs, rs_norm_sq, p, eta, k + 1

    def _cond_fun(values):
        ds, _, rs_norm_sq, _, eta, k = values
        return (rs_norm_sq[k] > sqtol) & (k < max_iter)  # & (eta > min_eta)

    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter))  # only implemented for order='K'
    rs = jnp.zeros((b.size, max_iter + 1))  #   only implemented for order='K'
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)

    # Initialize loop variables
    sqtol = (
        tol**2
    )  # max(tol ** 2, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    min_eta = min_eta  # max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf  # to make sure the first iteration is done
    rs = rs.at[:, 0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k = jax.lax.while_loop(
        _cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0)
    )

    return ds[:, :k]
