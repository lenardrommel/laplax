# /laplax/util/interop.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import linear_operator as lo
import numpy as np
import torch
from linox import Matrix
from probnum.linops import LambdaLinearOperator

if TYPE_CHECKING:
    from collections.abc import Callable

Array = jax.Array


@dataclass(frozen=True)
class JaxMVP:
    """Tiny JAX-native operator wrapper around a matvec."""

    mv: Callable[[Array], Array]
    shape: tuple[int, int]
    dtype: jnp.dtype

    def matvec(self, x: Array) -> Array:
        return self.mv(x)

    def matmat(self, X: Array) -> Array:
        # X: (n, k) -> (m, k)
        return jax.vmap(self.mv, in_axes=1, out_axes=1)(X)

    @property
    def T(self) -> JaxMVP:
        m, n = self.shape
        # Need a primal with correct shape
        x0 = jnp.zeros((n,), dtype=self.dtype)
        transpose_fun = jax.linear_transpose(self.mv, x0)

        def mv_T(y: Array) -> Array:
            # returns a tuple of cotangents for primals
            (xbar,) = transpose_fun(y)
            return xbar

        return JaxMVP(mv=mv_T, shape=(n, m), dtype=self.dtype)

    def todense(self) -> Array:
        _m, n = self.shape
        I = jnp.eye(n, dtype=self.dtype)  # (n, n)
        return self.matmat(I)  # (m, n)


def as_probnum(op: JaxMVP):
    import inspect

    m, n = op.shape
    dtype = np.dtype(op.dtype)

    def matmul(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            y = op.matvec(jnp.asarray(X))
            return np.asarray(jax.device_get(y))
        if X.ndim == 2:
            Y = op.matmat(jnp.asarray(X))  # (m, k)
            return np.asarray(jax.device_get(Y))
        msg = f"matmul expects 1D or 2D, got {X.ndim}D"
        raise ValueError(msg)

    def todense() -> np.ndarray:
        return np.asarray(jax.device_get(op.todense()))

    # Only pass supported kwargs (ProbNum versions differ)
    sig = inspect.signature(LambdaLinearOperator)
    kwargs = {"shape": (m, n), "matmul": matmul}
    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    if "todense" in sig.parameters:
        kwargs["todense"] = todense

    return LambdaLinearOperator(**kwargs)


def as_linox_dense(op: JaxMVP):
    return Matrix(op.todense())


def as_linear_operator_dense(op: JaxMVP):
    A = np.asarray(jax.device_get(op.todense()))
    A_t = torch.tensor(A)
    return lo.to_linear_operator(A_t)


def _as_2d(x, colvec: bool) -> tuple[jax.Array, bool]:
    """Return (X2d, was_vector). If colvec=True, treat vector as (n,1), else (1,n).

    Raises:
        ValueError: If x is not 1D or 2D.
    """
    x = jnp.asarray(x)
    if x.ndim == 1:
        return (x[:, None] if colvec else x[None, :]), True
    if x.ndim == 2:
        return x, False
    msg = f"Expected 1D/2D, got {x.ndim}D"
    raise ValueError(msg)


def _to_numpy(x: Any) -> np.ndarray:
    # JAX -> numpy
    if isinstance(x, jax.Array):
        return np.asarray(jax.device_get(x))
    # torch -> numpy (optional)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_jax(x: Any) -> jax.Array:
    if isinstance(x, jax.Array):
        return x
    # torch -> numpy -> jax (optional)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)


def _as_2d_col(x: jax.Array) -> tuple[jax.Array, bool]:
    x = jnp.asarray(x)
    if x.ndim == 1:
        return x[:, None], True  # (n,1)
    if x.ndim == 2:
        return x, False  # (n,k)
    raise ValueError(f"Expected 1D/2D, got {x.ndim}D")


@dataclass(frozen=True)
class AnyLinOp:
    shape: tuple[int, int]
    matmul_fn: Callable[[Any], Any]
    solve_fn: Callable[[Any], Any] | None = None
    host_fallback: bool = True

    def matmul(self, x: jax.Array) -> jax.Array:
        X, was_vec = _as_2d_col(x)
        try:
            Y = self.matmul_fn(X)  # try JAX first
        except TypeError:
            if not self.host_fallback:
                raise
            Y = self.matmul_fn(_to_numpy(X))  # fallback via numpy
        Y = _to_jax(Y)
        return Y[:, 0] if was_vec else Y

    def solve(self, b: jax.Array) -> jax.Array:
        if self.solve_fn is None:
            raise NotImplementedError("solve not available for this operator.")
        B, was_vec = _as_2d_col(b)
        try:
            X = self.solve_fn(B)  # try JAX first
        except TypeError:
            if not self.host_fallback:
                raise
            X = self.solve_fn(_to_numpy(B))  # fallback via numpy
        X = _to_jax(X)
        return X[:, 0] if was_vec else X

    def __matmul__(self, x: jax.Array) -> jax.Array:
        return self.matmul(x)


def wrap_linear_operator(
    op: Any,
    *,
    shape: tuple[int, int] | None = None,
    host_fallback: bool = True,
) -> AnyLinOp:
    shp = shape or getattr(op, "shape", None)
    if shp is None:
        raise TypeError("Need shape=(m,n) or op.shape")
    m, n = shp

    # Choose a right-matmul implementation. Support common conventions.
    if hasattr(op, "matmul") and callable(op.matmul):
        matmul_fn = lambda X: op.matmul(X)
    elif hasattr(op, "_matmul") and callable(op._matmul):
        # probnum-style internal API
        matmul_fn = lambda X: op._matmul(X)
    elif hasattr(op, "__matmul__"):
        matmul_fn = lambda X: op @ X
    elif callable(op):
        # interpret callable as matvec; vectorize over columns
        matmul_fn = lambda X: jax.vmap(op, in_axes=1, out_axes=1)(X)
    else:
        raise TypeError("Operator must provide matmul/_matmul/@ or be callable.")

    solve_fn = None
    if hasattr(op, "solve") and callable(op.solve):
        solve_fn = lambda B: op.solve(B)
    elif hasattr(op, "_solve") and callable(op._solve):
        solve_fn = lambda B: op._solve(B)

    return AnyLinOp(
        shape=(m, n),
        matmul_fn=matmul_fn,
        solve_fn=solve_fn,
        host_fallback=host_fallback,
    )
