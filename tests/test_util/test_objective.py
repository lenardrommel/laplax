import jax
import pytest
from flax import linen as nn
from jax import numpy as jnp

from laplax.enums import CovarianceStructure
from laplax.types import Callable, Float, ModelFn, Params, PredArray
from laplax.util.objective import (
    create_fsp_objective,
    create_loss_nll,
    create_loss_reg,
)


def create_kernel_fn(case: str) -> Callable[[PredArray, PredArray], jnp.ndarray]:
    """Create a kernel function for testing.

    Args:
        case: Either "inner_product" for simple inner product kernel or
            "gpjax" for GPJAX RBF kernel.

    Returns:
        Kernel function that takes two arrays and returns a covariance matrix.

    Raises:
        ValueError: If case is not "inner_product" or "gpjax".
    """
    if case == "inner_product":

        def kernel_fn(x1: PredArray, x2: PredArray | None = None) -> jnp.ndarray:
            if x2 is None:
                x2 = x1
            return jnp.dot(x1, x2.T)

        return kernel_fn

    else:
        raise ValueError(f"Invalid case: {case}")


def create_model_fn() -> tuple[ModelFn, Params]:
    """Create a simple linear model function.

    Returns:
        Tuple of (model_fn, params) where model_fn takes (x, params) -> y.
    """
    key = jax.random.PRNGKey(42)

    class SimpleModel(nn.Module):
        """Simple linear model for testing."""

        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)

    model = SimpleModel()
    dummy_input = jnp.ones((1, 1))
    variables = model.init(key, dummy_input)
    params = variables["params"]

    def model_fn(x: PredArray, p: Params) -> PredArray:
        return model.apply({"params": p}, x)

    return model_fn, params


@pytest.mark.parametrize("kernel_case", ["inner_product"])
def test_create_fsp_objective_creation(kernel_case: str):
    """Test that FSP objective is created successfully."""
    model_fn, _params = create_model_fn()
    kernel_fn = create_kernel_fn(kernel_case)

    dataset_size = 50
    n_context = 20

    # Prior mean should match number of context points
    prior_mean = jnp.zeros(n_context)

    fsp_objective = create_fsp_objective(model_fn, dataset_size, prior_mean, kernel_fn)

    assert fsp_objective is not None, "FSP objective should not be None"
    assert callable(fsp_objective), "FSP objective should be callable"


@pytest.mark.parametrize("kernel_case", ["inner_product"])
def test_create_fsp_objective_output_dimensions(kernel_case: str):
    """Test that FSP objective output has correct dimensions."""
    model_fn, params = create_model_fn()
    kernel_fn = create_kernel_fn(kernel_case)

    dataset_size = 50
    n_context = 20
    input_dim = 1

    # Prior mean should match number of context points
    prior_mean = jnp.zeros(n_context)

    fsp_objective = create_fsp_objective(
        model_fn,
        dataset_size,
        prior_mean,
        kernel_fn,
        structure=CovarianceStructure.NONE,
    )

    # Create test data
    key = jax.random.PRNGKey(123)
    data = {
        "input": jax.random.normal(key, (dataset_size, input_dim)),
        "target": jax.random.normal(key, (dataset_size,)),
    }
    context_points = {
        "context": jax.random.normal(key, (n_context, input_dim)),
        # "grid": jax.random.normal(key, (n_context, input_dim)),
    }

    # Evaluate objective (with default scale)
    loss = fsp_objective(data, context_points, params, scale=1.0)

    # Check output properties
    assert loss is not None, "Loss should not be None"
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert isinstance(loss, jnp.ndarray), "Loss should be a JAX array"
    assert jnp.isfinite(loss), "Loss should be finite"


@pytest.mark.parametrize("kernel_case", ["inner_product"])
def test_create_fsp_objective_with_scale(kernel_case: str):
    """Test FSP objective with scale parameter."""
    model_fn, params = create_model_fn()
    kernel_fn = create_kernel_fn(kernel_case)

    dataset_size = 30
    n_context = 15
    input_dim = 1
    scale = 0.5

    prior_mean = jnp.zeros(n_context)

    fsp_objective = create_fsp_objective(model_fn, dataset_size, prior_mean, kernel_fn)

    key = jax.random.PRNGKey(456)
    data = {
        "input": jax.random.normal(key, (dataset_size, input_dim)),
        "target": jax.random.normal(key, (dataset_size,)),
    }
    context_points = {
        "context": jax.random.normal(key, (n_context, input_dim)),
        "grid": jax.random.normal(key, (n_context, input_dim)),
    }

    loss = fsp_objective(data, context_points, params, scale=scale)

    assert loss.shape == (), "Loss should be scalar"
    assert jnp.isfinite(loss), "Loss should be finite"


def test_create_loss_nll():
    """Test NLL loss creation and evaluation."""
    model_fn, params = create_model_fn()

    dataset_size = 40
    loss_nll = create_loss_nll(model_fn, dataset_size)

    key = jax.random.PRNGKey(789)
    data = {
        "input": jax.random.normal(key, (dataset_size, 1)),
        "target": jax.random.normal(key, (dataset_size,)),
    }

    loss = loss_nll(data, params, scale=1.0)

    assert loss.shape == (), "NLL loss should be scalar"
    assert jnp.isfinite(loss), "NLL loss should be finite"
    assert loss > 0, "NLL loss should be positive"


@pytest.mark.parametrize("kernel_case", ["inner_product"])
def test_create_loss_reg(kernel_case: str):
    """Test regularization loss creation and evaluation."""
    model_fn, params = create_model_fn()
    kernel_fn = create_kernel_fn(kernel_case)

    n_context = 15
    input_dim = 1
    prior_mean = jnp.zeros(n_context)

    loss_reg = create_loss_reg(model_fn, prior_mean, kernel_fn)

    key = jax.random.PRNGKey(321)
    context_points = {
        "context": jax.random.normal(key, (n_context, input_dim)),
        "grid": jax.random.normal(key, (n_context, input_dim)),
    }

    loss = loss_reg(context_points, params)

    assert loss.shape == (), "Regularization loss should be scalar"
    assert jnp.isfinite(loss), "Regularization loss should be finite"
    assert loss >= 0, "Regularization loss should be non-negative"
