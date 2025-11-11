"""Utilities for operator learning with FSP.

This module provides utilities for function-space posterior inference
on operator learning tasks with spatial/temporal structure.
"""

import functools
import operator
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from laplax.types import InputArray, ModelFn, Params, PredArray


@partial(jax.jit, static_argnames=("model_fn",))
def _model_vjp(
    model_fn: ModelFn, params: Params, xs: InputArray, vs: PredArray
) -> jax.Array:
    """Compute multiple vector-Jacobian products of the model at the mode.

    ```
    res[b] == vjp(f(xs[b], :), vs[b])
    ```

    Parameters
    ----------
    xs : jnp.array with shape `(B,) + input_shape`
        Primals
    vs : jnp.array with shape `(B,) + output_shape`
        Tangents

    Returns
    -------
    jnp.array with shape `(B,) + param_shape`
        Batch of vector-Jacobian products.
    """
    return jax.vmap(
        lambda x, v: jax.vjp(lambda w: model_fn(x, w), params)[1](v),
        in_axes=(0, 0),
        out_axes=0,
    )(xs, vs)


@partial(jax.jit, static_argnames=("model_fn",))
def compute_M_batch(
    model_fn: ModelFn, params: Params, xs: InputArray, L: PredArray
):
    """Compute batch matrix-Jacobian product M = J^T L.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Parameters
    xs : InputArray
        Input data with shape (batch, ...)
    L : PredArray
        Low-rank matrix with shape (batch, ..., rank)

    Returns
    -------
    PyTree
        Matrix-Jacobian product with rank dimension
    """
    return jax.vmap(
        lambda vs: jax.tree.map(
            lambda param: jnp.sum(param, axis=0),
            _model_vjp(model_fn, params, xs, vs),
        ),
        in_axes=-1,
        out_axes=-1,
    )(L)


@partial(jax.jit, static_argnums=(0,), static_argnames=("num_chunks",))
def hosvd_lanczos_init(
    model_fn: ModelFn, params: Params, xs: InputArray, num_chunks: int = 1
):
    """Initialize Lanczos using Higher-Order SVD (HOSVD).

    For operator learning with spatial structure, this computes initial
    vectors by performing SVD along each mode of the tensor output.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Parameters
    xs : InputArray
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
        xs.ndim >= 4
    ), f"Input must have shape (B, S1, S2, ..., C), but got {xs.shape}"

    ones_pytree = jax.tree.map(lambda x: jnp.ones_like(x), params)

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
        [model_jvp(xs_batch) for xs_batch in jnp.split(xs, num_chunks, axis=0)],
        axis=0,
    )

    # Extract spatial dimensions (exclude batch and channel dimensions)
    spatial_dims = tuple(s for s in xs.shape[1:-1] if s > 1)
    n_function = xs.shape[0]

    # Reshape to (n_function, *spatial_dims)
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


def compute_M_batch_chunked(
    model_fn: ModelFn,
    params: Params,
    x_chunks: Sequence[InputArray],
    k_chunks: Sequence[PredArray],
) -> Params:
    """Compute M_batch in chunks and accumulate results.

    Parameters
    ----------
    model_fn : ModelFn
        Model function
    params : Params
        Parameters
    x_chunks : Sequence[InputArray]
        Sequence of input chunks
    k_chunks : Sequence[PredArray]
        Sequence of kernel chunks

    Returns
    -------
    Params
        Accumulated matrix-Jacobian product
    """
    return functools.reduce(
        lambda params1, params2: jax.tree.map(operator.add, params1, params2),
        (
            compute_M_batch(model_fn, params, x_c, k_c)
            for x_c, k_c in zip(x_chunks, k_chunks)
        ),
    )


def reshape_for_spatial_structure(
    data: PredArray, n_function: int, output_shape: tuple
) -> PredArray:
    """Reshape data to separate function and spatial dimensions.

    Parameters
    ----------
    data : PredArray
        Data with shape (n_function * spatial_prod, rank)
    n_function : int
        Number of functions in batch
    output_shape : tuple
        Target output shape (n_function, *spatial_dims, rank)

    Returns
    -------
    PredArray
        Reshaped data
    """
    return data.reshape(n_function, *output_shape[1:], -1)
