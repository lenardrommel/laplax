import jax
from jax import numpy as jnp
import laplax
from laplax.extra.fsp.objective import select_context_points

from laplax.types import ModelFn, Params, Data, PredArray


def compute_curvature_fn(
    model_fn: ModelFn,
    params: Params,
    data: Data,
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
    x_context = select_context_points(
        int(data["input"].shape[0]),
        "grid",
        data["input"].max(axis=0),
        data["input"].min(axis=0),
        data["input"].shape,
        key=jax.random.key(0),
    )

    def create_scan_fn(
        unflatten_fn, _u, eigvecs, eigvals, params, model_fn, data, prior_var
    ):
        def scan_fn(carry, i):
            post_var, valid_indices = carry

            new_cov = _u @ (eigvecs[:, i] * (1 / eigvals[i] ** 0.5))
            lr_fac_i = unflatten_fn(new_cov)  # Use captured unflatten

            all_inputs = jnp.array(x_context)

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
        jnp.array(eigvals.shape[0], dtype=jnp.int64),
        jnp.array(n_valid, dtype=jnp.int64),
    )
    valid_covs = jax.lax.dynamic_slice(covs, (0, 0), (n_valid_int, covs.shape[1]))
    return jnp.transpose(valid_covs)
