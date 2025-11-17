import jax
import jax.numpy as jnp
import numpy as np
import pytest

from laplax.util.flatten import create_pytree_flattener, wrap_function
from laplax.util.mv import diagonal, kronecker, kronecker_product_factors, to_dense
from laplax.util.tree import get_size


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_dense(n):
    key = jax.random.PRNGKey(123)
    A = jax.random.normal(key, (n, n))

    # Calling diagonal on a dense matrix should return jnp.diag(A)
    diag_computed = diagonal(A)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("n", [2, 5])
def test_diagonal_and_to_dense_flat_mvp(n):
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # 1) Compare diagonal(mv, layout=n) to jnp.diag(A)
    diag_computed = diagonal(mv, layout=n)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=n) to A
    dense_computed = to_dense(mv, layout=n)
    np.testing.assert_allclose(dense_computed, A, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("n", [2, 5, 20, 100])
def test_diagonal_low_rank(n):
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Rank-1: V is n x 1, VV^T is rank-1 positive semidefinite
    u = jax.random.normal(key1, (n,))
    rank1_factor = u.reshape(-1, 1)  # n x 1 matrix
    rank1_matrix = rank1_factor @ rank1_factor.T  # uu^T, not uv^T

    key3 = jax.random.split(key2)[0]
    V_rank2 = jax.random.normal(key3, (n, 2))
    rank2_matrix = V_rank2 @ V_rank2.T

    key4 = jax.random.split(key3)[0]
    V_rank5 = jax.random.normal(key4, (n, 5))
    rank5_matrix = V_rank5 @ V_rank5.T

    def compare_diagonals(full_matrix, factor_matrix):
        assert jnp.allclose(
            jnp.diag(full_matrix), diagonal(factor_matrix, layout=n, low_rank=True)
        ), "Diagonal computation mismatch for low-rank matrix"

    compare_diagonals(rank1_matrix, rank1_factor)
    compare_diagonals(rank2_matrix, V_rank2)
    compare_diagonals(rank5_matrix, V_rank5)


@pytest.mark.parametrize(("n1", "n2"), [(2, 3), (3, 4)])
def test_diagonal_and_to_dense_pytree_mvp(n1, n2):
    key = jax.random.PRNGKey(999)
    layout = {
        "x": jnp.zeros(n1),
        "y": jnp.zeros(n2),
    }
    example_flat, tree_def = jax.tree.flatten(layout)
    sizes = [leaf.size for leaf in example_flat]
    total_dim = sum(sizes)

    # Create a random (total_dim x total_dim) matrix A
    A = jax.random.normal(key, (total_dim, total_dim))

    def mv(pytree_vec):
        leaves, _ = jax.tree.flatten(pytree_vec)
        x_flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
        y_flat = A @ x_flat

        # Split back into the shapes
        split_indices = np.cumsum(sizes[:-1])
        y_splits = jnp.split(y_flat, split_indices)

        # Reshape each split to match original leaf shape
        y_leaves = []
        for s, leaf_shape in zip(
            y_splits, [leaf.shape for leaf in example_flat], strict=True
        ):
            y_leaves.append(s.reshape(leaf_shape))

        # Unflatten back
        return jax.tree.unflatten(tree_def, y_leaves)

    # 1) Compare diagonal(mv, layout=layout) to jnp.diag(A)
    diag_computed = diagonal(mv, layout)
    diag_expected = jnp.diag(A)
    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-7, rtol=1e-7)

    # 2) Compare to_dense(mv, layout=layout) to A
    dense_computed = to_dense(mv, layout)

    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)

    mv_wrapped = wrap_function(mv, input_fn=unflatten, output_fn=flatten)
    dense_computed = to_dense(mv_wrapped, get_size(layout))
    flatten, unflatten = create_pytree_flattener(layout)
    np.testing.assert_allclose(flatten(dense_computed).reshape(*A.shape), A)


def test_diagonal_kronecker_structure():
    """Test diagonal computation with Kronecker product structure.

    Tests the diagonal of a matrix with Kronecker product structure:
    A ⊗ B where A is m×m and B is n×n.
    """
    m, n = 3, 4
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, (m, m))
    B = jax.random.normal(key2, (n, n))

    # Kronecker product: A ⊗ B
    K = jnp.kron(A, B)

    def kron_mv(x):
        """Matrix-vector product using Kronecker structure."""
        return K @ x

    size = m * n
    diag_computed = diagonal(kron_mv, layout=size)
    diag_expected = jnp.diag(K)

    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-6, rtol=1e-6)


def test_to_dense_kronecker_structure():
    """Test to_dense with Kronecker product structure."""
    m, n = 2, 3
    key = jax.random.PRNGKey(123)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, (m, m))
    B = jax.random.normal(key2, (n, n))

    K = jnp.kron(A, B)

    def kron_mv(x):
        return K @ x

    size = m * n
    dense_computed = to_dense(kron_mv, layout=size)

    np.testing.assert_allclose(dense_computed, K, atol=1e-6, rtol=1e-6)


def test_diagonal_separable_kronecker():
    """Test diagonal with separable Kronecker structure.

    For separable structure, diag(A ⊗ B) can be computed more efficiently.
    """
    m, n = 4, 5
    key = jax.random.PRNGKey(999)
    key1, key2 = jax.random.split(key)

    # Use symmetric matrices for simplicity
    A_half = jax.random.normal(key1, (m, m))
    B_half = jax.random.normal(key2, (n, n))
    A = A_half @ A_half.T
    B = B_half @ B_half.T

    K = jnp.kron(A, B)

    # Expected diagonal using Kronecker property:
    # diag(A ⊗ B) = vec(diag(A) ⊗ diag(B)) = diag(A) ⊗ diag(B)
    diag_A = jnp.diag(A)
    diag_B = jnp.diag(B)
    expected_diag = jnp.kron(diag_A, diag_B)

    # Compare with actual diagonal
    actual_diag = jnp.diag(K)
    np.testing.assert_allclose(actual_diag, expected_diag, atol=1e-6, rtol=1e-6)

    # Now test our diagonal function
    def kron_mv(x):
        return K @ x

    computed_diag = diagonal(kron_mv, layout=m * n)
    np.testing.assert_allclose(computed_diag, expected_diag, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("m", "n"), [(2, 2), (3, 4), (5, 3)])
def test_diagonal_kronecker_various_sizes(m, n):
    """Test diagonal computation with various Kronecker product sizes."""
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, (m, m))
    B = jax.random.normal(key2, (n, n))

    K = jnp.kron(A, B)

    def kron_mv(x):
        return K @ x

    size = m * n
    diag_computed = diagonal(kron_mv, layout=size)
    diag_expected = jnp.diag(K)

    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-6, rtol=1e-6)


def test_low_rank_kronecker_structure():
    """Test low-rank diagonal computation with Kronecker-like structure."""
    m, n = 3, 4
    rank = 2

    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Create low-rank factors
    U = jax.random.normal(key1, (m * n, rank))
    # V V^T representation
    V = U  # Use same for simplicity

    # Full low-rank matrix
    low_rank_matrix = V @ V.T

    # Test diagonal
    diag_computed = diagonal(V, layout=m * n, low_rank=True)
    diag_expected = jnp.diag(low_rank_matrix)

    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-6, rtol=1e-6)


def test_mv_operations_composition():
    """Test that mv operations can be composed."""
    n = 5
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    A = jax.random.normal(key1, (n, n))
    B = jax.random.normal(key2, (n, n))

    def mv_A(x):
        return A @ x

    def mv_B(x):
        return B @ x

    def mv_composed(x):
        """Compute (A + B) @ x."""
        return mv_A(x) + mv_B(x)

    # Test diagonal
    diag_computed = diagonal(mv_composed, layout=n)
    diag_expected = jnp.diag(A + B)

    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-6, rtol=1e-6)

    # Test to_dense
    dense_computed = to_dense(mv_composed, layout=n)
    dense_expected = A + B

    np.testing.assert_allclose(dense_computed, dense_expected, atol=1e-6, rtol=1e-6)


def test_diagonal_batch_size_parameter():
    """Test that diagonal_batch_size parameter works correctly."""
    n = 10
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # Test with different batch sizes
    diag_full = diagonal(mv, layout=n)
    diag_batched = diagonal(mv, layout=n, diagonal_batch_size=2)

    np.testing.assert_allclose(diag_full, diag_batched, atol=1e-7, rtol=1e-7)


def test_kronecker_product_factors_two_factors():
    """Test kronecker_product_factors with two factors (reduces to basic kronecker)."""
    key = jax.random.PRNGKey(42)
    k1, k2, kv = jax.random.split(key, 3)

    n1, n2 = 3, 4
    A = jax.random.normal(k1, (n1, n1))
    B = jax.random.normal(k2, (n2, n2))
    v = jax.random.normal(kv, (n1 * n2,))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    # Using kronecker_product_factors
    mv_kron = kronecker_product_factors([mv_a, mv_b], [n1, n2])

    result = mv_kron(v)
    expected = jnp.kron(A, B) @ v

    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_kronecker_product_factors_three_factors():
    """Test kronecker_product_factors with three factors."""
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 4)

    n1, n2, n3 = 2, 3, 2
    A = jax.random.normal(keys[0], (n1, n1))
    B = jax.random.normal(keys[1], (n2, n2))
    C = jax.random.normal(keys[2], (n3, n3))
    v = jax.random.normal(keys[3], (n1 * n2 * n3,))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    def mv_c(x):
        return C @ x

    # Using kronecker_product_factors
    mv_kron = kronecker_product_factors([mv_a, mv_b, mv_c], [n1, n2, n3])

    result = mv_kron(v)

    # Compute reference: (A ⊗ B ⊗ C) @ v
    AB = jnp.kron(A, B)
    ABC = jnp.kron(AB, C)
    expected = ABC @ v

    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_kronecker_product_factors_single_factor():
    """Test kronecker_product_factors with a single factor (identity operation)."""
    key = jax.random.PRNGKey(99)
    k1, kv = jax.random.split(key)

    n = 5
    A = jax.random.normal(k1, (n, n))
    v = jax.random.normal(kv, (n,))

    def mv_a(x):
        return A @ x

    # Single factor should just return the original mv
    mv_kron = kronecker_product_factors([mv_a], [n])

    result = mv_kron(v)
    expected = A @ v

    np.testing.assert_allclose(result, expected, atol=1e-7, rtol=1e-7)


def test_kronecker_product_factors_four_factors():
    """Test kronecker_product_factors with four factors."""
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 5)

    dims = [2, 2, 2, 2]
    matrices = [jax.random.normal(keys[i], (d, d)) for i, d in enumerate(dims)]
    total_dim = int(jnp.prod(jnp.array(dims)))
    v = jax.random.normal(keys[4], (total_dim,))

    mvs = [lambda x, M=M: M @ x for M in matrices]

    # Using kronecker_product_factors
    mv_kron = kronecker_product_factors(mvs, dims)

    result = mv_kron(v)

    # Compute reference by sequential kronecker products
    K = matrices[0]
    for M in matrices[1:]:
        K = jnp.kron(K, M)
    expected = K @ v

    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_kronecker_product_factors_various_sizes():
    """Test kronecker_product_factors with varying factor sizes."""
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 4)

    dims = [2, 5, 3]
    matrices = [jax.random.normal(keys[i], (d, d)) for i, d in enumerate(dims)]
    total_dim = int(jnp.prod(jnp.array(dims)))
    v = jax.random.normal(keys[3], (total_dim,))

    mvs = [lambda x, M=M: M @ x for M in matrices]

    mv_kron = kronecker_product_factors(mvs, dims)

    result = mv_kron(v)

    # Compute reference
    K = matrices[0]
    for M in matrices[1:]:
        K = jnp.kron(K, M)
    expected = K @ v

    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_kronecker_separable_diagonal():
    """Test diagonal computation with separable Kronecker structure.

    For Kronecker products, diag(A ⊗ B) = diag(A) ⊗ diag(B).
    """
    key = jax.random.PRNGKey(999)
    k1, k2 = jax.random.split(key)

    m, n = 4, 5
    A = jax.random.normal(k1, (m, m))
    B = jax.random.normal(k2, (n, n))

    K = jnp.kron(A, B)

    # Expected diagonal using Kronecker property
    diag_A = jnp.diag(A)
    diag_B = jnp.diag(B)
    expected_diag = jnp.kron(diag_A, diag_B)

    # Compare with actual diagonal
    actual_diag = jnp.diag(K)
    np.testing.assert_allclose(actual_diag, expected_diag, atol=1e-6, rtol=1e-6)

    # Test our diagonal function
    def kron_mv(x):
        return K @ x

    computed_diag = diagonal(kron_mv, layout=m * n)
    np.testing.assert_allclose(computed_diag, expected_diag, atol=1e-6, rtol=1e-6)


def test_kronecker_product_factors_to_dense():
    """Test that kronecker_product_factors produces correct dense matrix."""
    key = jax.random.PRNGKey(2024)
    keys = jax.random.split(key, 3)

    dims = [3, 4]
    A = jax.random.normal(keys[0], (dims[0], dims[0]))
    B = jax.random.normal(keys[1], (dims[1], dims[1]))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    mv_kron = kronecker_product_factors([mv_a, mv_b], dims)

    # Convert to dense
    total_dim = int(jnp.prod(jnp.array(dims)))
    dense_result = to_dense(mv_kron, layout=total_dim)

    # Expected
    expected = jnp.kron(A, B)

    np.testing.assert_allclose(dense_result, expected, atol=1e-6, rtol=1e-6)


def test_kronecker_composition():
    """Test that kronecker operations can be composed."""
    key = jax.random.PRNGKey(111)
    keys = jax.random.split(key, 5)

    n = 3
    m = 2

    A = jax.random.normal(keys[0], (n, n))
    B = jax.random.normal(keys[1], (m, m))
    C = jax.random.normal(keys[2], (n, n))
    D = jax.random.normal(keys[3], (m, m))

    v = jax.random.normal(keys[4], (n * m,))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    def mv_c(x):
        return C @ x

    def mv_d(x):
        return D @ x

    # Create two Kronecker products and add them
    mv_kron1 = kronecker_product_factors([mv_a, mv_b], [n, m])
    mv_kron2 = kronecker_product_factors([mv_c, mv_d], [n, m])

    def mv_sum(x):
        return mv_kron1(x) + mv_kron2(x)

    result = mv_sum(v)

    # Expected
    K1 = jnp.kron(A, B)
    K2 = jnp.kron(C, D)
    expected = (K1 + K2) @ v

    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_diagonal_batch_size_parameter():
    """Test that diagonal_batch_size parameter works correctly."""
    n = 10
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # Test with different batch sizes
    diag_full = diagonal(mv, layout=n)
    diag_batched = diagonal(mv, layout=n, diagonal_batch_size=2)

    np.testing.assert_allclose(diag_full, diag_batched, atol=1e-7, rtol=1e-7)


def test_to_dense_batch_size_parameter():
    """Test that to_dense_batch_size parameter works correctly."""
    n = 8
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n, n))

    def mv(x):
        return A @ x

    # Test with different batch sizes
    dense_full = to_dense(mv, layout=n)
    dense_batched = to_dense(mv, layout=n, to_dense_batch_size=3)

    np.testing.assert_allclose(dense_full, dense_batched, atol=1e-7, rtol=1e-7)


def test_diagonal_with_complex_pytree():
    """Test diagonal with more complex PyTree structure."""
    key = jax.random.PRNGKey(123)

    # More complex layout with nested structure
    layout = {
        "layer1": {"w": jnp.zeros(3), "b": jnp.zeros(2)},
        "layer2": jnp.zeros(4),
    }

    total_size = 3 + 2 + 4  # 9

    # Create random matrix
    A = jax.random.normal(key, (total_size, total_size))

    def mv(pytree_vec):
        # Flatten pytree
        leaves, tree_def = jax.tree.flatten(pytree_vec)
        x_flat = jnp.concatenate([leaf.ravel() for leaf in leaves])

        # Apply matrix
        y_flat = A @ x_flat

        # Reconstruct pytree
        sizes = [leaf.size for leaf in leaves]
        split_indices = np.cumsum(sizes[:-1])
        y_splits = jnp.split(y_flat, split_indices)

        y_leaves = []
        for s, leaf in zip(y_splits, leaves, strict=True):
            y_leaves.append(s.reshape(leaf.shape))

        return jax.tree.unflatten(tree_def, y_leaves)

    diag_computed = diagonal(mv, layout)
    diag_expected = jnp.diag(A)

    np.testing.assert_allclose(diag_computed, diag_expected, atol=1e-6, rtol=1e-6)


def test_kronecker_numerical_stability():
    """Test numerical stability of Kronecker products with varying scales."""
    key = jax.random.PRNGKey(2025)
    keys = jax.random.split(key, 3)

    n1, n2 = 3, 4

    # Create matrices with different scales
    A = jax.random.normal(keys[0], (n1, n1)) * 0.1
    B = jax.random.normal(keys[1], (n2, n2)) * 10.0
    v = jax.random.normal(keys[2], (n1 * n2,))

    def mv_a(x):
        return A @ x

    def mv_b(x):
        return B @ x

    mv_kron = kronecker_product_factors([mv_a, mv_b], [n1, n2])

    result = mv_kron(v)

    # Check no NaN or Inf
    assert jnp.all(jnp.isfinite(result))

    # Verify against reference
    expected = jnp.kron(A, B) @ v
    np.testing.assert_allclose(result, expected, atol=1e-5, rtol=1e-5)
