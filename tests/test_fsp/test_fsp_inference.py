"""Tests for FSP inference on regression and operator learning tasks."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from laplax.extra.fsp import (
    fsp_inference,
    fsp_operator_inference,
    lanczos_hosvd_initialization,
    lanczos_jacobian_initialization,
    select_context_points,
)
from laplax.extra.fsp.operator import (
    compute_M_batch,
    hosvd_lanczos_init,
)

jax.config.update("jax_enable_x64", True)


class SimpleRegression(nnx.Module):
    """Simple MLP for regression tasks."""

    def __init__(self, in_features, hidden_features, out_features, rngs):
        self.linear1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.linear2(h)


class SimpleOperator(nnx.Module):
    """Simple CNN for operator learning tasks."""

    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.conv1 = nnx.Conv(in_channels, hidden_channels, kernel_size=(3,), rngs=rngs)
        self.conv2 = nnx.Conv(hidden_channels, out_channels, kernel_size=(3,), rngs=rngs)

    def __call__(self, x):
        # x shape: (spatial, channels)
        x = x[jnp.newaxis, ...]  # Add batch dim
        h = nnx.relu(self.conv1(x))
        y = self.conv2(h)
        return y[0]  # Remove batch dim


def rbf_kernel(x1, x2=None, lengthscale=1.0):
    """RBF kernel for GP prior."""
    if x2 is None:
        x2 = x1
    sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * sq_dist / lengthscale**2)


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    key = jax.random.PRNGKey(0)
    n_train = 100
    n_test = 50

    # Generate data
    x_train = jax.random.uniform(key, (n_train, 1), minval=-2.0, maxval=2.0)
    key, subkey = jax.random.split(key)
    y_train = jnp.sin(2 * jnp.pi * x_train) + 0.1 * jax.random.normal(subkey, (n_train, 1))

    x_test = jax.random.uniform(key, (n_test, 1), minval=-2.0, maxval=2.0)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "key": key,
    }


@pytest.fixture
def operator_data():
    """Create synthetic operator learning data."""
    key = jax.random.PRNGKey(42)
    n_functions = 10
    spatial_size = 32
    channels = 1

    # Generate spatial data
    x = jax.random.normal(key, (n_functions, spatial_size, channels))
    y = jax.random.normal(key, (n_functions, spatial_size, channels))

    class SimpleDataLoader:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __iter__(self):
            yield self.x, self.y

    return {
        "x": x,
        "y": y,
        "loader": SimpleDataLoader(x, y),
        "key": key,
    }


def test_lanczos_jacobian_initialization(regression_data):
    """Test standard Lanczos initialization."""
    # Create model
    model = SimpleRegression(
        in_features=1, hidden_features=10, out_features=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Initialize
    x_context = regression_data["x_train"][:20]
    v = lanczos_jacobian_initialization(model_fn, params, x_context)

    # Check properties
    assert v.ndim == 1
    assert v.shape[0] > 0
    assert jnp.isfinite(v).all()
    assert jnp.abs(jnp.linalg.norm(v) - 1.0) < 1e-6  # Should be normalized


def test_hosvd_initialization(operator_data):
    """Test HOSVD-based Lanczos initialization for operator learning."""
    # Create model
    model = SimpleOperator(
        in_channels=1, hidden_channels=4, out_channels=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Initialize (need at least 4D input)
    x_context = operator_data["x"]
    # Reshape to (B, S, 1, C) for proper operator structure
    x_context_4d = x_context[:, :, jnp.newaxis, :]  # (10, 32, 1, 1)

    vectors_function, vectors_spatial = lanczos_hosvd_initialization(
        model_fn, params, x_context_4d, num_chunks=2
    )

    # Check properties
    assert len(vectors_function) == 1  # One function space vector
    assert len(vectors_spatial) >= 1  # At least one spatial vector

    for vec in vectors_function:
        assert jnp.isfinite(vec).all()
        assert jnp.abs(jnp.linalg.norm(vec) - 1.0) < 1e-6

    for vec in vectors_spatial:
        assert jnp.isfinite(vec).all()
        assert jnp.abs(jnp.linalg.norm(vec) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "context_selection",
    ["random", "grid", "sobol"],
)
def test_context_point_selection(context_selection, regression_data):
    """Test different context point selection strategies."""
    n_context = 20
    key = regression_data["key"]

    context_points = select_context_points(
        n_context_points=n_context,
        context_selection=context_selection,
        context_points_minval=[-2.0],
        context_points_maxval=[2.0],
        datapoint_shape=(1,),
        key=key,
    )

    # Check properties
    assert context_points.shape[0] == n_context or context_points.shape[0] >= n_context
    assert context_points.shape[1] == 1
    assert jnp.isfinite(context_points).all()
    assert jnp.all(context_points >= -2.0)
    assert jnp.all(context_points <= 2.0)


def test_fsp_inference_regression(regression_data):
    """Test FSP inference on regression task."""
    # Create and train model
    model = SimpleRegression(
        in_features=1, hidden_features=20, out_features=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x).squeeze(-1)

    # Define prior kernel
    def prior_kernel(x1, x2=None):
        return rbf_kernel(x1, x2, lengthscale=0.5)

    # Run FSP inference
    data_dict = {
        "input": regression_data["x_train"],
        "target": regression_data["y_train"].squeeze(-1),
    }

    posterior = fsp_inference(
        model_fn=model_fn,
        params=params,
        data=data_dict,
        prior_cov_kernel=prior_kernel,
        context_selection="grid",
        n_context_points=30,
        key=jax.random.PRNGKey(42),
        truncate_to_prior_var=True,
        context_points_minval=[-2.0],
        context_points_maxval=[2.0],
        datapoint_shape=(1,),
    )

    # Check posterior properties
    assert posterior.rank > 0
    assert posterior.state["scale_sqrt"].shape[1] == posterior.rank
    assert jnp.isfinite(posterior.state["scale_sqrt"]).all()

    # Test that we can apply the posterior
    flatten_fn, unflatten_fn = jax.flatten_util.ravel_pytree(params)
    params_flat = flatten_fn(params)
    test_vec = jax.random.normal(jax.random.PRNGKey(123), (posterior.rank,))

    # Apply scale_mv
    scale_result = posterior.scale_mv(posterior.state)(test_vec)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(), scale_result)
    )

    # Check rank is reasonable
    assert posterior.rank < params_flat.shape[0]
    assert posterior.rank > 1


def test_compute_M_batch(regression_data):
    """Test batch matrix-Jacobian product computation."""
    # Create model
    model = SimpleRegression(
        in_features=1, hidden_features=10, out_features=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Create random L matrix
    n_context = 20
    rank = 5
    x_context = regression_data["x_train"][:n_context]
    L = jax.random.normal(jax.random.PRNGKey(123), (n_context, 1, rank))

    # Compute M
    M = compute_M_batch(model_fn, params, x_context, L)

    # Check properties
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(), M)
    )

    # Check that M has rank dimension
    first_leaf = jax.tree_util.tree_leaves(M)[0]
    assert first_leaf.shape[-1] == rank


def test_hosvd_lanczos_init_function(operator_data):
    """Test the hosvd_lanczos_init utility function."""
    # Create model
    model = SimpleOperator(
        in_channels=1, hidden_channels=4, out_channels=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Prepare data with proper shape
    x_data = operator_data["x"]
    # Need shape (B, S1, S2, ..., C) with at least 4D
    x_4d = x_data[:, :, jnp.newaxis, :]  # (10, 32, 1, 1)

    # Test HOSVD initialization
    vectors_func, vectors_spatial = hosvd_lanczos_init(
        model_fn, params, x_4d, num_chunks=2
    )

    # Verify structure
    assert len(vectors_func) == 1
    assert len(vectors_spatial) >= 1

    # Verify normalization
    for v in vectors_func + vectors_spatial:
        assert jnp.abs(jnp.linalg.norm(v) - 1.0) < 1e-6


@pytest.mark.skip(reason="Operator learning FSP requires more complex setup")
def test_fsp_operator_inference(operator_data):
    """Test FSP inference on operator learning task.

    Note: This test is currently skipped because it requires proper
    kernel construction for Kronecker products, which is non-trivial.
    """
    # Create model
    model = SimpleOperator(
        in_channels=1, hidden_channels=4, out_channels=1, rngs=nnx.Rngs(0)
    )
    graph_def, params = nnx.split(model)

    def model_fn(x, params):
        return nnx.merge(graph_def, params)(x)

    # Define dummy kernels
    def spatial_kernel(grid1, grid2=None):
        if grid2 is None:
            grid2 = grid1
        return jnp.eye(grid1.shape[0])

    def function_kernel(x1, x2=None):
        if x2 is None:
            x2 = x1
        return jnp.eye(x1.shape[0])

    # Run FSP operator inference
    posterior = fsp_operator_inference(
        model_fn=model_fn,
        params=params,
        data_loader=operator_data["loader"],
        spatial_kernels=[spatial_kernel],
        function_kernels=[function_kernel],
        context_selection="dataloader",
        n_context_points=5,
        grid_stride=2,
        n_chunks=2,
        key=jax.random.PRNGKey(42),
        truncate_to_prior_var=False,
    )

    # Check posterior properties
    assert posterior.rank > 0
    assert jnp.isfinite(posterior.state["scale_sqrt"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
