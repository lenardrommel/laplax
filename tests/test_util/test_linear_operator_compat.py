# /tests/test_util/test_linear_operator_compat.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.interop import (
    JaxMVP,
    as_linear_operator_dense,
    as_linox_dense,
    as_probnum,
    wrap_linear_operator,
)


def _make_op(n=8, seed=0):
    key = jax.random.PRNGKey(seed)
    A = jax.random.normal(key, (n, n), dtype=jnp.float32)

    def mv(x):
        return A @ x

    op = JaxMVP(mv=mv, shape=A.shape, dtype=A.dtype)
    return A, op


def test_jaxmvp_basic_matvec_and_transpose():
    A, op = _make_op()
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (A.shape[1],), dtype=A.dtype)
    y = jax.random.normal(key, (A.shape[0],), dtype=A.dtype)

    assert jnp.allclose(op.matvec(x), A @ x)
    assert jnp.allclose(op.T.matvec(y), A.T @ y)


def test_jaxmvp_todense_roundtrip():
    A, op = _make_op()
    assert jnp.allclose(op.todense(), A)


def test_linox_compat_dense():
    pytest.importorskip("linox")
    A, op = _make_op()

    lop = as_linox_dense(op)
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (A.shape[1],), dtype=A.dtype)

    # linox supports applying operators via @ (and often callable usage) per README.
    assert jnp.allclose(lop @ x, A @ x)


def test_probnum_compat():
    pytest.importorskip("probnum")
    A, op = _make_op()

    pop = as_probnum(op)
    x = np.random.default_rng(0).normal(size=(A.shape[1],)).astype(np.float32)

    y = pop @ x
    y_ref = np.asarray(jax.device_get(A @ jnp.asarray(x)))
    assert np.allclose(y, y_ref)


def test_linear_operator_compat_dense():
    pytest.importorskip("torch")
    pytest.importorskip("linear_operator")
    A, op = _make_op()

    lop = as_linear_operator_dense(op)
    A_dense = lop.to_dense().detach().cpu().numpy()

    assert np.allclose(A_dense, np.asarray(jax.device_get(A)))


def test_wrap_linear_operator_user_defined():
    key = jax.random.PRNGKey(0)
    n = 6
    A = jax.random.normal(key, (n, n))

    class MyOp:
        shape = (n, n)

        def matmul(self, X):  # X: (n,k)
            return A @ X

        def solve(self, B):  # optional
            return jnp.linalg.solve(A, B)

    from probnum.linops import Matrix

    op = Matrix(A)

    lop = wrap_linear_operator(op)
    x = jax.random.normal(jax.random.PRNGKey(1), (n,))
    assert jnp.allclose(lop @ x, A @ x)

    b = jax.random.normal(jax.random.PRNGKey(2), (n,))
    assert jnp.allclose(lop.solve(b), jnp.linalg.solve(A, b))
