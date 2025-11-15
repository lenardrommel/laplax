"""FSP Laplace on Truncated Sine Regression.

This example demonstrates FSP (Function-Space Prior) Laplace approximation
for regression on a truncated sine function, showing how FSP captures
uncertainty in extrapolation regions.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from laplax.curv import KernelStructure, create_fsp_posterior

# Note: jax_enable_x64 disabled to avoid dtype mismatches
# jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Data Generation
# ==============================================================================


def generate_truncated_sine(key, n_samples=100, feature_dim=1, noise_std=0.1):
    """Generate truncated sine data.

    Generates data from a sine function on [-1, -0.5] U [0.5, 1],
    creating a gap in the middle for extrapolation testing.

    Args:
        key: JAX random key
        n_samples: Number of samples
        feature_dim: Dimensionality of input features
        noise_std: Standard deviation of observation noise

    Returns:
        Tuple of (X, y) arrays
    """
    key1, key2, key3 = jax.random.split(key, num=3)

    # Features: split between two regions
    X1 = jax.random.uniform(
        key1, minval=-1.0, maxval=-0.5, shape=(n_samples // 2, feature_dim)
    )
    X2 = jax.random.uniform(
        key2, minval=0.5, maxval=1.0, shape=(n_samples // 2, feature_dim)
    )
    X = jnp.concatenate([X1, X2], axis=0)

    # Targets: sine function + noise
    eps = noise_std * jax.random.normal(key3, shape=(n_samples,))
    y = jnp.sin(2 * jnp.pi * X.mean(axis=-1)) + eps

    return X, y.reshape(-1, 1)


# ==============================================================================
# Model Definition (Flax NNX)
# ==============================================================================


class MLP(nnx.Module):
    """Simple MLP for regression using Flax NNX."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        """Initialize MLP.

        Args:
            hidden_dims: List of hidden layer dimensions
            rngs: Random number generator
        """
        in_dim = 1  # 1D input

        # Build layers - use attributes to avoid tuple storage issue
        for i, hidden_dim in enumerate(hidden_dims):
            setattr(self, f"hidden_{i}", nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim

        # Output layer (single value for regression)
        self.output_layer = nnx.Linear(in_dim, 1, rngs=rngs)
        self.n_hidden = len(hidden_dims)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through MLP.

        Args:
            x: Input array of shape (1,) or (batch, 1)

        Returns:
            Prediction output
        """
        for i in range(self.n_hidden):
            x = jnp.tanh(getattr(self, f"hidden_{i}")(x))
        x = self.output_layer(x)
        return x.squeeze()


def split_model(model: MLP):
    """Split Flax NNX model into function and parameters."""
    graphdef, params = nnx.split(model, nnx.Param)

    def model_fn(x, params):
        """Model function that takes input and parameters."""
        model_copy = nnx.merge(graphdef, params)
        return model_copy(x)

    return model_fn, params


# ==============================================================================
# Training
# ==============================================================================


def mse_loss(model, x_batch, y_batch):
    """Mean squared error loss."""
    preds = jax.vmap(model)(x_batch)
    return jnp.mean((preds - y_batch.squeeze()) ** 2)


def train_mlp(model, x_train, y_train, num_epochs=1000, learning_rate=0.01):
    """Train MLP with Adam optimizer."""
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Detect Flax version to use correct optimizer API
    import inspect
    update_sig = inspect.signature(optimizer.update)
    # Flax < 0.11.0: update(grads), Flax >= 0.11.0: update(model, grads)
    needs_model_arg = len(update_sig.parameters) > 1

    if needs_model_arg:
        @nnx.jit
        def train_step(model, optimizer, x_batch, y_batch):
            """Single training step."""
            loss, grads = nnx.value_and_grad(mse_loss)(model, x_batch, y_batch)
            optimizer.update(model, grads)
            return loss
    else:
        @nnx.jit
        def train_step(model, optimizer, x_batch, y_batch):
            """Single training step."""
            loss, grads = nnx.value_and_grad(mse_loss)(model, x_batch, y_batch)
            optimizer.update(grads)
            return loss

    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, x_train, y_train)
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model


# ==============================================================================
# FSP Posterior with RBF Kernel
# ==============================================================================


def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    """RBF kernel function."""
    sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-sq_dist / (2 * lengthscale**2))


def create_kernel_matrix(x_context, lengthscale=1.0, variance=1.0):
    """Create RBF kernel matrix."""
    K = rbf_kernel(x_context, x_context, lengthscale, variance)
    # Add jitter for numerical stability
    K = K + 1e-6 * jnp.eye(K.shape[0])
    return K


# ==============================================================================
# Main Example
# ==============================================================================


def main():
    """Run FSP Laplace on truncated sine regression."""
    print("=" * 70)
    print("FSP Laplace: Truncated Sine Regression")
    print("=" * 70)

    # Generate training data
    print("\n1. Generating truncated sine data...")
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    X_train, y_train = generate_truncated_sine(
        data_key, n_samples=100, feature_dim=1, noise_std=0.1
    )
    print(f"   Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Train model
    print("\n2. Training MLP...")
    key, model_key = jax.random.split(key)
    model = MLP(hidden_dims=[50, 50], rngs=nnx.Rngs(int(model_key[0])))
    model = train_mlp(model, X_train, y_train, num_epochs=1000, learning_rate=0.01)

    # Evaluate training loss
    train_loss = mse_loss(model, X_train, y_train)
    print(f"   Final training loss: {train_loss:.4f}")

    # Create FSP posterior
    print("\n3. Computing FSP posterior...")

    # Use all training data as context
    x_context = X_train

    # Kernel hyperparameters
    lengthscale = 0.3
    variance = 1.0

    def kernel_fn(v):
        """Kernel matrix-vector product."""
        K = create_kernel_matrix(x_context, lengthscale=lengthscale, variance=variance)
        return K @ v

    # Compute prior variance
    prior_cov = create_kernel_matrix(
        x_context, lengthscale=lengthscale, variance=variance
    )
    prior_variance = jnp.diag(prior_cov)

    print(
        f"   Prior variance range: [{prior_variance.min():.4f}, {prior_variance.max():.4f}]"
    )

    # Split model for laplax
    model_fn, trained_params = split_model(model)

    # Create FSP posterior
    posterior = create_fsp_posterior(
        model_fn=model_fn,
        params=trained_params,
        x_context=x_context,
        kernel_structure=KernelStructure.NONE,
        kernel=kernel_fn,
        prior_variance=prior_variance,
        n_chunks=2,
        max_iter=50,
    )

    print(f"   FSP posterior rank: {posterior.rank}")

    # Make predictions with uncertainty
    print("\n4. Making predictions with uncertainty...")

    # Create test grid including extrapolation region
    x_test = jnp.linspace(-1.5, 1.5, 300).reshape(-1, 1)

    # Mean predictions
    mean_preds = jax.vmap(model)(x_test)

    # Sample from posterior
    print("   Sampling from FSP posterior...")
    key, sample_key = jax.random.split(key)
    n_samples = 50
    samples = []

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)

        sample_preds = jax.vmap(lambda x: model_fn(x, sample_params))(x_test)
        samples.append(sample_preds)

    samples = jnp.stack(samples)
    pred_mean = jnp.mean(samples, axis=0)
    pred_std = jnp.std(samples, axis=0)

    print(f"   Prediction std range: [{pred_std.min():.3f}, {pred_std.max():.3f}]")

    # Visualize results
    print("\n5. Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True function for reference
    x_true = jnp.linspace(-1.5, 1.5, 1000)
    y_true = jnp.sin(2 * jnp.pi * x_true)

    # Plot 1: Predictions with uncertainty
    ax = axes[0]
    ax.plot(x_true, y_true, "k--", alpha=0.3, label="True function", linewidth=2)
    ax.scatter(X_train, y_train, c="blue", alpha=0.6, s=30, label="Training data")
    ax.plot(x_test, pred_mean, "r-", label="Posterior mean", linewidth=2)
    ax.fill_between(
        x_test.squeeze(),
        pred_mean - 2 * pred_std,
        pred_mean + 2 * pred_std,
        alpha=0.3,
        color="red",
        label="±2σ (95% CI)",
    )

    # Highlight extrapolation regions
    ax.axvspan(-1.5, -1.0, alpha=0.1, color="gray", label="Extrapolation")
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="gray")
    ax.axvspan(1.0, 1.5, alpha=0.1, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"FSP Laplace Predictions (rank={posterior.rank})", fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Uncertainty quantification
    ax = axes[1]
    ax.plot(x_test, pred_std, "b-", linewidth=2, label="Predictive std")
    ax.scatter(X_train, jnp.zeros_like(y_train), c="blue", alpha=0.3, s=20)

    # Highlight extrapolation regions
    ax.axvspan(-1.5, -1.0, alpha=0.1, color="gray", label="Extrapolation")
    ax.axvspan(-0.5, 0.5, alpha=0.1, color="gray")
    ax.axvspan(1.0, 1.5, alpha=0.1, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("Predictive standard deviation")
    ax.set_title("Uncertainty Quantification", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fsp_sin_regression.png", dpi=150, bbox_inches="tight")
    print("   Saved visualization to 'fsp_sin_regression.png'")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"FSP Posterior:")
    print(f"  - Rank: {posterior.rank}")
    print(f"  - Mean prediction std: {pred_std.mean():.4f}")
    print(f"  - Max prediction std: {pred_std.max():.4f}")
    print(f"  - Training loss: {train_loss:.4f}")
    print("\nNote: Uncertainty increases in extrapolation regions ([-1.5,-1] ∪ [-0.5,0.5] ∪ [1,1.5])")
    print("=" * 70)


if __name__ == "__main__":
    main()
