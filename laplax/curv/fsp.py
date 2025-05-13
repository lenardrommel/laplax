import jax
import jax.flatten_util
import jax.numpy as jnp
from flax import nnx

import laplax
from laplax.curv.lanczos_isqrt import lanczos_isqrt
from laplax.enums import LossFn
from laplax.types import Callable, Data, Float, ModelFn, Params, PredArray
from laplax.util import tree


# --------------------------------------------------------------------------------------
# FSP training utilities
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# FSP training objective
# --------------------------------------------------------------------------------------


def create_loss_mse(model_fn: ModelFn):
    """Create the MSE loss function for FSP training."""

    def loss_mse(data: Data, params: Params) -> Float:
        pred = model_fn(data["inputs"], params)
        return jnp.mean(jax.numpy.square(pred - data["targets"]))

    return loss_mse


def create_loss_nll(
    model_fn: ModelFn,
    prior_arguments: dict = None,
    num_training_samples: int = 150,
):
    r"""Create the NLL loss function for FSP training.
    $$
    - \log p(f(X) + \epsilon = y) = \frac{1}{2}(y - \mu_{\theta}(X))^{T} (\Sigma_{\theta}(X, X) + \sigma^2I)^\dagger (y - \mu_{\theta}(X)) - \frac{1}{2} \log |(\Sigma_{\theta}(X, X) + \sigma^2I)| + \frac{d}{2} \log(2\pi)
    $$.
    """  # noqa: D205

    def loss_nll(data: Data, params: Params) -> Float:
        scale_param = params["param"]
        model_params = params["model"]
        noise_scale = scale_param.value
        preds = jax.vmap(model_fn, in_axes=(0, None))(data["inputs"], model_params)
        sq_diff = jnp.square(preds - data["targets"])
        log_term = 0.5 * jnp.log(2 * jnp.pi * noise_scale**2)
        precision_term = 0.5 * sq_diff / (noise_scale**2)
        nll = log_term + precision_term

        N = num_training_samples
        return jnp.mean(nll) * N

    return loss_nll


def create_loss_reg(
    model_fn: ModelFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
):
    def loss_reg(context_points, params: Params) -> Float:
        r"""FSP regularization loss.

        $$1/2 (f(c^{i}) - m)^{T} K^{-1}(c^{i}, c^{i}) (f(c^{i}) - m)$$
        """
        params = params["model"]
        f_c = jax.vmap(model_fn, in_axes=(0, None))(context_points, params)
        K_c_c = prior_cov_kernel(context_points, context_points)
        left = jax.numpy.linalg.solve(K_c_c, f_c)
        return 0.5 * jax.numpy.einsum("ij,ij->", f_c, left)

    return loss_reg


def create_fsp_objective(
    model_fn: ModelFn,
    loss_fn: LossFn,
    prior_mean: PredArray,
    prior_cov_kernel: Callable,
    num_training_samples: int = 150,
    batch_size: int = 20,
):
    """Create FSP objective using the wrapped model with learnable scale."""
    # Create loss functions
    loss_nll = create_loss_nll(model_fn)
    loss_reg = create_loss_reg(model_fn, prior_mean, prior_cov_kernel)

    # Create objective
    def fsp_objective(data: Data, context_points: PredArray, params: Params) -> Float:
        return loss_nll(data, params) + loss_reg(context_points, params)

    return fsp_objective


# --------------------------------------------------------------------------------------
# FSP Laplace approximation
# --------------------------------------------------------------------------------------


def lanczos_jacobian_initialization(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    has_batch_dim: bool = True,
    *,
    lanczos_initialization_batch_size: int = 32,
):
    # Define model Jacobian vector product
    if has_batch_dim:
        model_jvp = jax.vmap(
            lambda x: jax.jvp(
                lambda w: model_fn(x, params=w),
                (params,),
                (tree.ones_like(params),),
            )[1],
            in_axes=0,
            out_axes=0,
        )

        initial_vec = jax.lax.map(
            model_jvp,
            data["inputs"],
            batch_size=lanczos_initialization_batch_size,
        ).reshape(-1)
    else:
        initial_vec = jax.jvp(
            lambda w: model_fn(data["inputs"], params=w),
            (params,),
            (tree.ones_like(params),),
        )[1]

    # Normalize
    initial_vec = initial_vec / jnp.linalg.norm(initial_vec, 2)

    return initial_vec.squeeze(-1)


def compute_matrix_jacobian_product(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    matrix: PredArray,
    has_batch_dim: bool = True,
):
    """Compute the matrix Jacobian product."""
    if has_batch_dim:
        flatten, unflatten = laplax.util.create_partial_pytree_flattener(params)

        raise NotImplementedError(
            "Batch dimension not implemented for matrix Jacobian product."
        )

    M_tree = jax.vmap(
        jax.vjp(
            lambda p: jnp.reshape(
                model_fn(data["inputs"], params=p), (matrix.shape[0],)
            ),
            params,
        )[1],
        in_axes=1,  # 0
        out_axes=1,  # 0
    )(matrix)

    M, unravel_fn = jax.flatten_util.ravel_pytree(
        M_tree,
    )
    return M, unravel_fn


def compute_curvature_fn(
    model_fn: ModelFn,
    params: Params,
    data: Data,  # maybe change to test_data
    ggn: PredArray,
    prior_var: PredArray,
    u: PredArray,
) -> PredArray:
    _eigvals, _eigvecs = jnp.linalg.eigh(ggn)
    eps = jnp.finfo(ggn.dtype).eps
    tol = eps * (_eigvals.max() ** 0.5) * _eigvals.shape[0]
    eigvals = _eigvals[_eigvals > tol]
    eigvecs = _eigvecs[:, _eigvals > tol]
    eigvals = jnp.flip(eigvals, axis=0)
    eigvecs = jnp.flip(eigvecs, axis=1)

    def create_scan_fn(
        unflatten_fn, _u, eigvecs, eigvals, params, model_fn, data, prior_var
    ):
        def scan_fn(carry, i):
            post_var, valid_indices = carry

            new_cov = _u @ (eigvecs[:, i] * (1 / eigvals[i] ** 0.5))
            lr_fac_i = unflatten_fn(new_cov)  # Use captured unflatten

            all_inputs = jnp.array(
                data["test_inputs"]
            )  # maybe change to data["test_inputs"] used to be data["inputs"]

            def compute_jvp_squared(x):
                jvp_result = jax.jvp(lambda p: model_fn(x, p), (params,), (lr_fac_i,))[
                    1
                ]
                return jnp.square(jvp_result)

            jvp_squared_results = jax.vmap(compute_jvp_squared)(all_inputs)
            jvp_squared_concat = jnp.reshape(jvp_squared_results, post_var.shape)

            new_post_var = post_var + jvp_squared_concat
            is_valid = jnp.all(new_post_var < prior_var)

            new_valid_indices = jax.lax.cond(
                is_valid,
                lambda _: valid_indices.at[i].set(True),
                lambda _: valid_indices,
                None,
            )

            return (new_post_var, new_valid_indices), (new_cov, is_valid)

        return scan_fn

    _, unflatten = laplax.util.flatten.create_pytree_flattener(params)

    scan_fn = create_scan_fn(
        unflatten, u, eigvecs, eigvals, params, model_fn, data, prior_var
    )

    init_post_var = jnp.zeros((prior_var.shape[0],))
    init_valid_indices = jnp.zeros(eigvals.shape[0], dtype=jnp.bool_)

    (final_post_var, final_valid_indices), (covs, validity) = jax.lax.scan(
        scan_fn, (init_post_var, init_valid_indices), jnp.arange(eigvals.shape[0])
    )
    cumulative_validity = jnp.cumprod(validity)

    n_valid = jnp.sum(cumulative_validity)

    n_valid_int = jnp.minimum(
        jnp.array(eigvals.shape[0], dtype=jnp.int32),
        jnp.array(n_valid, dtype=jnp.int32),
    )
    valid_covs = jax.lax.dynamic_slice(covs, (0, 0), (n_valid_int, covs.shape[1]))
    return jnp.transpose(valid_covs)


def fsp_laplace(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    prior_mean: PredArray,
    prior_cov_kernel: Callable[[PredArray, PredArray], Float],
    context_points: PredArray,
    **kwargs,
):
    """FSP Laplace approximation."""
    # Initial vector

    # Define cov operator
    op = prior_cov_kernel(context_points, context_points)
    op = op if isinstance(op, Callable) else lambda x: op @ x

    # Compute low rank terms
    # USE INVERSE OF COV MATRIX
    initial_vec = lanczos_jacobian_initialization(
        model_fn,
        params,
        data,
        **kwargs,
    )
    L = lanczos_isqrt(op, initial_vec, **kwargs)

    # Compute
    model_vjp = jax.vmap(
        lambda x, v: jax.vjp(lambda w: model_fn(x, w), params)[1](v),
        in_axes=(0, 0),
        out_axes=0,
    )(context_points, L)
    S = S
    # Posterior state
    posterior_state: PosteriorState = {"scale_sqrt": S}

    posterior = Posterior(
        state=posterior_state,
        cov_mv=lambda state: lambda x: state["scale_sqrt"] @ state["scale_sqrt"].T @ x,
        scale_mv=lambda state: lambda x: state["scale_sqrt"] @ x,
    )

    def posterior_fn(*args, **kwargs):
        del args, kwargs
        return posterior

    return posterior_fn


# lambda prior_args: posteiror_fn(prior_args)
