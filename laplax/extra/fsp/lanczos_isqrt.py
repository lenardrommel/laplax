import jax
import jax.numpy as jnp
from laplax.util import tree


def lanczos_invert_sqrt(
    A,
    b,
    *,
    tol=1e-3,  # 1e-5
    min_eta=1e-20,
    max_iter=500,
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
        w = A @ p  # A(p) or A @ p
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
        ds, rs, rs_norm_sq, p, eta, k = values
        return (rs_norm_sq[k] > tol**2) & (k < max_iter) & (eta > min_eta)

    # Initialization
    b /= jnp.linalg.norm(b, 2)
    ds = jnp.zeros((b.size, max_iter))  # only implemented for order='K'
    rs = jnp.zeros((b.size, max_iter + 1))  #   only implemented for order='K'
    rs_norm_sq = jnp.ones_like(rs, shape=max_iter + 1)

    # Initialize loop variables
    # tol = jnp.finfo(A.dtype).eps
    sqtol = (
        tol**2
    )  # max(tol ** 2, jnp.finfo(A.dtype).eps) if tol is not None else jnp.finfo(A.dtype).eps
    # min_eta = min_eta  # max(min_eta, jnp.finfo(A.dtype).eps) if min_eta is not None else jnp.finfo(A.dtype).eps
    eta = jnp.inf  # to make sure the first iteration is done
    rs = rs.at[:, 0].set(b)
    p = b if overwrite_b else b.copy()

    # Lanczos iterations
    ds, _, _, _, _, k = jax.lax.while_loop(
        _cond_fun, _step, (ds, rs, rs_norm_sq, p, eta, 0)
    )

    return ds[:, :k]


def lanczos_jacobian_initialization(
    model_fn,
    params,
    data,
    *,
    lanczos_initialization_batch_size: int = 20,
):
    """Initialize Lanczos vector using model Jacobian.

    This is the simple initialization for standard FSP on regression tasks.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    data : Data
        Input data for initialization
    lanczos_initialization_batch_size : int
        Batch size for initialization (not currently used)

    Returns
    -------
    jnp.ndarray
        Normalized initial vector
    """
    # Define model Jacobian vector product
    initial_vec = jax.jvp(
        lambda w: model_fn(data, params=w),
        (params,),
        (tree.ones_like(params),),
    )[1]
    initial_vec = initial_vec / jnp.linalg.norm(initial_vec, 2)

    return initial_vec.squeeze(-1)


def lanczos_hosvd_initialization(
    model_fn,
    params,
    data,
    *,
    num_chunks: int = 1,
):
    """Initialize Lanczos using Higher-Order SVD (HOSVD).

    For operator learning with spatial structure, this computes initial
    vectors by performing SVD along each mode of the tensor output.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Model parameters
    data : Data
        Input data with shape (B, S1, S2, ..., C)
        where B is batch/function dimension, S1, S2, ... are spatial dimensions,
        and C is channel dimension
    num_chunks : int
        Number of chunks for memory efficiency

    Returns
    -------
    tuple
        (initial_vectors_function, initial_vectors_spatial)
        Initial vectors for function space and spatial modes
    """
    assert (
        data.ndim >= 4
    ), f"Input must have shape (B, S1, S2, ..., C), but got {data.shape}"

    ones_pytree = tree.ones_like(params)

    model_jvp = jax.vmap(
        lambda x: jax.jvp(
            lambda w: model_fn(x, w),
            (params,),
            (ones_pytree,),
        )[1],
        in_axes=0,
        out_axes=0,
    )

    # Compute JVP in chunks for memory efficiency
    b = jnp.concatenate(
        [model_jvp(data_batch) for data_batch in jnp.split(data, num_chunks, axis=0)],
        axis=0,
    )

    # Extract spatial dimensions (exclude batch and possibly channel dimensions)
    # Assuming shape is (B, S1, S2, ..., [C]) where C is channels (last dim)
    spatial_dims = tuple(s for s in data.shape[1:-1] if s > 1)
    n_function = data.shape[0]

    # Reshape to (n_function, *spatial_dims)
    # If output has channels, take the first one or squeeze
    if b.ndim > len(spatial_dims) + 1:
        b = b[..., 0]  # Take first channel if multiple
    b = b.reshape((n_function,) + spatial_dims)

    initial_vectors = []

    # HOSVD: compute SVD along each mode
    for mode in range(len(b.shape)):
        # Unfold along mode
        n_mode = b.shape[mode]
        b_unfolded = jnp.moveaxis(b, mode, 0).reshape(n_mode, -1)

        # SVD
        u, s, v = jnp.linalg.svd(b_unfolded, full_matrices=False)

        # Take dominant singular vector and normalize
        vec = u[:, 0] / jnp.linalg.norm(u[:, 0])
        initial_vectors.append(vec)

    # Split into function and spatial
    initial_vectors_function = [initial_vectors[0]]
    initial_vectors_spatial = initial_vectors[1:]

    return initial_vectors_function, initial_vectors_spatial


# def test_lanczos_compute_efficient():
#     # Create a simple positive definite matrix
#     n = 100
#     np.random.seed(42)
#     A_np = np.random.randn(n, n)
#     A_np = A_np @ A_np.T + n * np.eye(n)  # Make it positive definite
#     A_jax = jnp.array(A_np)

#     # Define the linear operator
#     def linear_op(x):
#         return A_jax @ x

#     # Create a random vector b
#     b = jnp.ones(n) / np.sqrt(n)  # Normalized vector

#     # Run the Lanczos algorithm
#     ds = lanczos_compute_efficient(linear_op, b, tol=1e-8, max_iter=100)

#     print(f"Lanczos vectors shape: {ds.shape}")

#     # Verify orthogonality of Lanczos vectors
#     D = ds.T @ ds
#     print("Orthogonality check (should be close to identity):")
#     print(np.round(D, 5))

#     # Verify the tridiagonal matrix T = D.T @ A @ D
#     T = ds.T @ (A_jax @ ds)
#     print("Tridiagonal matrix T:")
#     print(np.round(T, 5))

#     # Optional: Compare with a direct eigendecomposition
#     eigvals_lanczos = np.linalg.eigvalsh(T)
#     eigvals_direct = np.linalg.eigvalsh(A_np)[: ds.shape[1]]
#     print("Eigenvalues from L anczos:")
#     print(np.sort(eigvals_lanczos))
#     print("Smallest eigenvalues from direct computation:")
#     print(np.sort(eigvals_direct)[: ds.shape[1]])


# if __name__ == "__main__":
#     test_lanczos_compute_efficient()
