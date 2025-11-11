"""Tests for FSP calibration and metrics."""

import jax
import jax.numpy as jnp
import pytest

from laplax.curv.utils import LowRankTerms
from laplax.extra.fsp import (
    PriorHyperparameters,
    RBFKernel,
    SimpleGPPrior,
    calibrate_gp_prior,
    compute_fsp_metrics,
    create_kernel_from_config,
    log_determinant,
    mahalanobis_distance,
    negative_log_predictive_density,
)

jax.config.update("jax_enable_x64", True)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_regression_data():
    """Create simple regression dataset."""
    key = jax.random.PRNGKey(42)
    n = 50

    # Generate data
    x = jax.random.uniform(key, (n, 1), minval=-3.0, maxval=3.0)
    key, subkey = jax.random.split(key)

    # True function: sin(x)
    y_true = jnp.sin(x.ravel())
    y = y_true + 0.1 * jax.random.normal(subkey, (n,))

    return {"x": x, "y": y, "y_true": y_true, "key": key}


@pytest.fixture
def low_rank_posterior():
    """Create synthetic low-rank posterior for testing."""
    n = 100
    rank = 10

    key = jax.random.PRNGKey(123)

    # Random low-rank factorization
    U = jax.random.normal(key, (n, rank))
    U, _ = jnp.linalg.qr(U)  # Orthonormalize

    key, subkey = jax.random.split(key)
    S = jax.random.uniform(subkey, (rank,), minval=0.1, maxval=2.0)

    return LowRankTerms(U=U, S=S, scalar=0.0)


# =============================================================================
# CALIBRATION TESTS
# =============================================================================


def test_prior_hyperparameters():
    """Test hyperparameter container."""
    hparams = PriorHyperparameters(lengthscale=-2.0, variance=0.5, noise_variance=-1.0)

    # Test to_dict
    d = hparams.to_dict()
    assert "lengthscale" in d
    assert "variance" in d
    assert "noise_variance" in d

    # Test from_dict
    hparams2 = PriorHyperparameters.from_dict(d)
    assert hparams2.lengthscale == hparams.lengthscale
    assert hparams2.variance == hparams.variance

    # Test transform
    transformed = hparams.transform()
    assert jnp.allclose(transformed["lengthscale"], jnp.exp(-2.0))
    assert jnp.allclose(transformed["variance"], jnp.exp(0.5))


def test_simple_gp_prior(simple_regression_data):
    """Test SimpleGPPrior class."""
    x = simple_regression_data["x"]
    y = simple_regression_data["y"]

    # Create GP prior
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)
    prior = SimpleGPPrior(kernel, noise_variance=0.01)

    # Test log likelihood
    nll = prior.log_likelihood(x, y)
    assert jnp.isfinite(nll)
    assert nll > 0  # Should be positive (negative log likelihood)

    # Test prediction
    x_test = jnp.linspace(-3, 3, 20).reshape(-1, 1)
    pred_mean, pred_var = prior.predict(x, y, x_test)

    assert pred_mean.shape == (20,)
    assert pred_var.shape == (20,)
    assert jnp.isfinite(pred_mean).all()
    assert jnp.isfinite(pred_var).all()
    assert jnp.all(pred_var > 0)  # Variance should be positive


def test_calibrate_gp_prior(simple_regression_data):
    """Test GP prior calibration."""
    x = simple_regression_data["x"]
    y = simple_regression_data["y"]

    # Create kernel
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)

    # Calibrate (use fewer iterations for speed)
    hparams, prior = calibrate_gp_prior(
        kernel, x, y, max_iter=10, tol=1e-2  # Reduced for test speed
    )

    # Check that hyperparameters are reasonable
    assert jnp.isfinite(hparams.lengthscale)
    assert jnp.isfinite(hparams.variance)
    assert jnp.isfinite(hparams.noise_variance)

    # Check that kernel was updated
    assert jnp.isfinite(kernel.lengthscale)
    assert kernel.lengthscale > 0

    # Test prediction with calibrated prior
    x_test = jnp.linspace(-3, 3, 20).reshape(-1, 1)
    pred_mean, pred_var = prior.predict(x, y, x_test)

    assert jnp.isfinite(pred_mean).all()
    assert jnp.isfinite(pred_var).all()


def test_create_kernel_from_config():
    """Test kernel creation from config."""
    # Test RBF
    config = {"type": "rbf", "lengthscale": 1.5, "variance": 2.0}
    kernel = create_kernel_from_config(config)
    assert isinstance(kernel, RBFKernel)
    assert kernel.lengthscale == 1.5
    assert kernel.variance == 2.0

    # Test composite
    composite_config = {
        "type": "additive",
        "components": [
            {"type": "rbf", "lengthscale": 1.0},
            {"type": "matern52", "lengthscale": 0.5},
        ],
    }
    kernel = create_kernel_from_config(composite_config)
    assert hasattr(kernel, "kernels")  # Should be composite


# =============================================================================
# METRICS TESTS
# =============================================================================


def test_mahalanobis_distance(low_rank_posterior):
    """Test Mahalanobis distance computation."""
    n = 100
    key = jax.random.PRNGKey(456)

    # Create synthetic predictions and targets
    pred_mean = jax.random.normal(key, (n,))
    key, subkey = jax.random.split(key)
    target = jax.random.normal(subkey, (n,))

    noise_variance = 0.1

    # Compute distance
    dist = mahalanobis_distance(pred_mean, target, low_rank_posterior, noise_variance)

    # Check properties
    assert jnp.isfinite(dist)
    assert dist >= 0  # Distance should be non-negative

    # Distance to self should be zero
    dist_self = mahalanobis_distance(
        pred_mean, pred_mean, low_rank_posterior, noise_variance
    )
    assert jnp.allclose(dist_self, 0.0, atol=1e-6)


def test_log_determinant(low_rank_posterior):
    """Test log determinant computation."""
    noise_variance = 0.1

    log_det = log_determinant(low_rank_posterior, noise_variance)

    # Check properties
    assert jnp.isfinite(log_det)

    # Log det should increase with noise
    log_det_high_noise = log_determinant(low_rank_posterior, 1.0)
    assert log_det_high_noise > log_det


def test_negative_log_predictive_density(low_rank_posterior):
    """Test NLPD computation."""
    n = 100
    key = jax.random.PRNGKey(789)

    pred_mean = jax.random.normal(key, (n,))
    key, subkey = jax.random.split(key)
    target = jax.random.normal(subkey, (n,))

    noise_variance = 0.1

    nlpd = negative_log_predictive_density(
        pred_mean, target, low_rank_posterior, noise_variance
    )

    # Check properties
    assert jnp.isfinite(nlpd)
    assert nlpd > 0  # Should be positive

    # NLPD for perfect predictions should be smaller
    nlpd_perfect = negative_log_predictive_density(
        pred_mean, pred_mean, low_rank_posterior, noise_variance
    )
    assert nlpd_perfect < nlpd


def test_compute_fsp_metrics(low_rank_posterior):
    """Test comprehensive metrics computation."""
    n = 100
    key = jax.random.PRNGKey(321)

    pred_mean = jax.random.normal(key, (n,))
    key, subkey = jax.random.split(key)
    target = pred_mean + 0.1 * jax.random.normal(subkey, (n,))  # Small noise

    noise_variance = 0.1
    pred_var = jnp.ones(n) * 0.5

    metrics = compute_fsp_metrics(
        pred_mean, target, low_rank_posterior, noise_variance, pred_var=pred_var
    )

    # Check all metrics are present
    assert "rmse" in metrics
    assert "mahalanobis" in metrics
    assert "log_det" in metrics
    assert "nlpd" in metrics
    assert "marginal_nlpd" in metrics

    # Check all are finite
    for key, value in metrics.items():
        assert jnp.isfinite(value), f"Metric {key} is not finite"

    # RMSE should be small (targets are close to predictions)
    assert metrics["rmse"] < 0.2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_end_to_end_workflow(simple_regression_data):
    """Test complete calibration + prediction + metrics workflow."""
    x_train = simple_regression_data["x"]
    y_train = simple_regression_data["y"]
    key = simple_regression_data["key"]

    # 1. Create and calibrate kernel
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)
    hparams, prior = calibrate_gp_prior(
        kernel, x_train, y_train, max_iter=10, tol=1e-2
    )

    # 2. Make predictions
    x_test = jax.random.uniform(key, (30, 1), minval=-3.0, maxval=3.0)
    y_test = jnp.sin(x_test.ravel())

    pred_mean, pred_var = prior.predict(x_train, y_train, x_test)

    # 3. Create low-rank posterior (simplified)
    rank = 10
    U = jax.random.normal(jax.random.PRNGKey(999), (30, rank))
    U, _ = jnp.linalg.qr(U)
    S = jnp.ones(rank) * 0.5
    low_rank = LowRankTerms(U=U, S=S, scalar=0.0)

    # 4. Compute metrics
    noise_var = jnp.exp(hparams.noise_variance)
    metrics = compute_fsp_metrics(
        pred_mean, y_test, low_rank, noise_var, pred_var=pred_var
    )

    # 5. Verify metrics are reasonable
    assert metrics["rmse"] < 2.0  # Should be reasonably accurate
    assert jnp.isfinite(metrics["nlpd"])
    assert metrics["mahalanobis"] >= 0


if __name__ == "__main__":
    # Run tests without pytest
    print("=" * 70)
    print("FSP CALIBRATION AND METRICS TESTS")
    print("=" * 70)

    # Create fixtures
    key = jax.random.PRNGKey(42)
    n = 50
    x = jax.random.uniform(key, (n, 1), minval=-3.0, maxval=3.0)
    y = jnp.sin(x.ravel()) + 0.1 * jax.random.normal(jax.random.PRNGKey(43), (n,))
    simple_data = {"x": x, "y": y, "y_true": jnp.sin(x.ravel()), "key": key}

    # Low-rank posterior
    U = jax.random.normal(jax.random.PRNGKey(123), (100, 10))
    U, _ = jnp.linalg.qr(U)
    S = jax.random.uniform(jax.random.PRNGKey(124), (10,), minval=0.1, maxval=2.0)
    low_rank = LowRankTerms(U=U, S=S, scalar=0.0)

    print("\nTest 1: Prior Hyperparameters...")
    test_prior_hyperparameters()
    print("✓ Passed")

    print("\nTest 2: Simple GP Prior...")
    test_simple_gp_prior(simple_data)
    print("✓ Passed")

    print("\nTest 3: Kernel from Config...")
    test_create_kernel_from_config()
    print("✓ Passed")

    print("\nTest 4: Mahalanobis Distance...")
    test_mahalanobis_distance(low_rank)
    print("✓ Passed")

    print("\nTest 5: Log Determinant...")
    test_log_determinant(low_rank)
    print("✓ Passed")

    print("\nTest 6: NLPD...")
    test_negative_log_predictive_density(low_rank)
    print("✓ Passed")

    print("\nTest 7: Compute FSP Metrics...")
    test_compute_fsp_metrics(low_rank)
    print("✓ Passed")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70 + "\n")
