"""Basic functionality tests for FSP modules (can run without pytest)."""

import jax
import jax.numpy as jnp
from flax import nnx

from laplax.extra.fsp import (
    lanczos_hosvd_initialization,
    lanczos_jacobian_initialization,
    select_context_points,
)
from laplax.extra.fsp.operator import (
    compute_M_batch,
    hosvd_lanczos_init,
)

jax.config.update("jax_enable_x64", True)


class SimpleModel(nnx.Module):
    """Simple MLP for testing."""

    def __init__(self, in_features, hidden_features, out_features, rngs):
        self.linear1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.linear2(h)


def test_imports():
    """Test that all imports work."""
    print("✓ All imports successful")


def test_lanczos_initialization():
    """Test standard Lanczos initialization."""
    print("\n" + "=" * 60)
    print("Testing Lanczos Jacobian Initialization")
    print("=" * 60)

    # Create model
    model = SimpleModel(
        in_features=2, hidden_features=10, out_features=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Create data
    key = jax.random.PRNGKey(0)
    x_context = jax.random.normal(key, (20, 2))

    # Initialize
    v = lanczos_jacobian_initialization(model_fn, params, x_context)

    # Verify
    print(f"  Initial vector shape: {v.shape}")
    print(f"  Vector norm: {jnp.linalg.norm(v):.6f} (should be ~1.0)")
    print(f"  All finite: {jnp.isfinite(v).all()}")

    assert v.ndim == 1
    assert jnp.isfinite(v).all()
    assert jnp.abs(jnp.linalg.norm(v) - 1.0) < 1e-5

    print("✓ Lanczos initialization test passed")


def test_hosvd_initialization():
    """Test HOSVD initialization for operator learning."""
    print("\n" + "=" * 60)
    print("Testing HOSVD Initialization")
    print("=" * 60)

    # Create model
    model = SimpleModel(
        in_features=8, hidden_features=10, out_features=8, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Create 4D data (B, S1, S2, C)
    key = jax.random.PRNGKey(42)
    x_context = jax.random.normal(key, (5, 8, 1, 1))

    # Initialize
    vectors_function, vectors_spatial = lanczos_hosvd_initialization(
        model_fn, params, x_context, num_chunks=1
    )

    # Verify
    print(f"  Function vectors: {len(vectors_function)}")
    print(f"  Spatial vectors: {len(vectors_spatial)}")

    for i, vec in enumerate(vectors_function):
        norm = jnp.linalg.norm(vec)
        print(f"  Function vector {i} - shape: {vec.shape}, norm: {norm:.6f}")
        assert jnp.abs(norm - 1.0) < 1e-5
        assert jnp.isfinite(vec).all()

    for i, vec in enumerate(vectors_spatial):
        norm = jnp.linalg.norm(vec)
        print(f"  Spatial vector {i} - shape: {vec.shape}, norm: {norm:.6f}")
        assert jnp.abs(norm - 1.0) < 1e-5
        assert jnp.isfinite(vec).all()

    print("✓ HOSVD initialization test passed")


def test_context_selection():
    """Test context point selection strategies."""
    print("\n" + "=" * 60)
    print("Testing Context Point Selection")
    print("=" * 60)

    key = jax.random.PRNGKey(123)
    n_context = 15

    for method in ["random", "grid", "sobol"]:
        print(f"\n  Method: {method}")

        context_points = select_context_points(
            n_context_points=n_context,
            context_selection=method,
            context_points_minval=[-1.0, -1.0],
            context_points_maxval=[1.0, 1.0],
            datapoint_shape=(2,),
            key=key,
        )

        print(f"    Shape: {context_points.shape}")
        print(f"    Min: {context_points.min():.3f}, Max: {context_points.max():.3f}")
        print(f"    All finite: {jnp.isfinite(context_points).all()}")

        assert context_points.shape[1] == 2
        assert jnp.isfinite(context_points).all()
        assert jnp.all(context_points >= -1.0)
        assert jnp.all(context_points <= 1.0)

    print("\n✓ Context point selection tests passed")


def test_M_batch_computation():
    """Test batch matrix-Jacobian product."""
    print("\n" + "=" * 60)
    print("Testing M Batch Computation")
    print("=" * 60)

    # Create model
    model = SimpleModel(
        in_features=2, hidden_features=8, out_features=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Create data
    key = jax.random.PRNGKey(456)
    n_context = 10
    rank = 3
    x_context = jax.random.normal(key, (n_context, 2))
    L = jax.random.normal(key, (n_context, 1, rank))

    # Compute M
    M = compute_M_batch(model_fn, params, x_context, L)

    # Verify
    first_leaf = jax.tree_util.tree_leaves(M)[0]
    print(f"  First leaf shape: {first_leaf.shape}")
    print(
        f"  All finite: {jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(), M))}"
    )

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(), M)
    )
    assert first_leaf.shape[-1] == rank

    print("✓ M batch computation test passed")
