"""Two Moons Classification with FSP Laplace Approximation.

This example demonstrates how to use FSP (Function-Space Prior) Laplace
approximation for binary classification on the two moons dataset.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from laplax.curv import KernelStructure, create_fsp_posterior

jax.config.update("jax_enable_x64", True)


# ==============================================================================
# Model Definition
# ==============================================================================


def init_mlp_params(layer_sizes, key):
    """Initialize MLP parameters."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        w = jax.random.normal(subkey, (layer_sizes[i], layer_sizes[i + 1])) * 0.1
        b = jnp.zeros(layer_sizes[i + 1])
        params.append({"w": w, "b": b})
    return params


def mlp_forward(x, params):
    """Forward pass through MLP."""
    for i, layer in enumerate(params[:-1]):
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    # Output layer (no activation)
    x = x @ params[-1]["w"] + params[-1]["b"]
    return x.squeeze()


# ==============================================================================
# Training
# ==============================================================================


def binary_cross_entropy_loss(params, x_batch, y_batch):
    """Binary cross-entropy loss."""
    logits = jax.vmap(lambda x: mlp_forward(x, params))(x_batch)
    # BCE with logits (negative log likelihood to minimize)
    return -jnp.mean(jax.nn.log_sigmoid(logits) * y_batch +
                     jax.nn.log_sigmoid(-logits) * (1 - y_batch))


def train_mlp(params, x_train, y_train, num_epochs=100, learning_rate=0.01):
    """Train MLP with gradient descent."""

    @jax.jit
    def update(params, x_batch, y_batch):
        loss, grads = jax.value_and_grad(binary_cross_entropy_loss)(
            params, x_batch, y_batch
        )
        # Gradient descent update
        params = jax.tree.map(
            lambda p, g: p - learning_rate * g, params, grads
        )
        return params, loss

    for epoch in range(num_epochs):
        params, loss = update(params, x_train, y_train)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return params


# ==============================================================================
# FSP Posterior
# ==============================================================================


def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    """RBF kernel function."""
    sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-sq_dist / (2 * lengthscale**2))


def create_simple_kernel(x_context, lengthscale=1.0, variance=1.0):
    """Create a simple RBF kernel matrix."""
    K = rbf_kernel(x_context, x_context, lengthscale, variance)
    # Add jitter for numerical stability
    K = K + 1e-6 * jnp.eye(K.shape[0])
    return K


# ==============================================================================
# Main Example
# ==============================================================================


def main():
    """Run two moons classification with FSP Laplace."""
    print("=" * 70)
    print("Two Moons Classification with FSP Laplace")
    print("=" * 70)

    # Generate two moons dataset
    print("\n1. Generating two moons dataset...")
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # Convert to JAX arrays
    X_jax = jnp.array(X)
    y_jax = jnp.array(y)

    # Initialize and train model
    print("\n2. Training MLP...")
    key = jax.random.PRNGKey(0)
    layer_sizes = [2, 32, 32, 1]
    params = init_mlp_params(layer_sizes, key)

    trained_params = train_mlp(params, X_jax, y_jax, num_epochs=100, learning_rate=0.1)

    # Evaluate training accuracy
    logits = jax.vmap(lambda x: mlp_forward(x, trained_params))(X_jax)
    predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    train_acc = jnp.mean(predictions == y_jax)
    print(f"   Training accuracy: {train_acc:.2%}")

    # Select context points - for classification, just randomly sample from training data
    print("\n3. Selecting context points...")
    n_context = 50
    np.random.seed(42)
    context_indices = np.random.choice(len(X), size=n_context, replace=False)
    x_context = X_jax[context_indices]
    print(f"   Selected {x_context.shape[0]} context points")

    # Create FSP posterior
    print("\n4. Computing FSP posterior...")

    # Create a simple kernel function (unstructured)
    def kernel_fn(v):
        """Kernel matrix-vector product."""
        K = create_simple_kernel(x_context, lengthscale=1.0, variance=1.0)
        return K @ v

    # Compute prior variance
    prior_cov = create_simple_kernel(x_context, lengthscale=1.0, variance=1.0)
    prior_variance = jnp.diag(prior_cov)

    # Create FSP posterior with unstructured kernel
    posterior = create_fsp_posterior(
        model_fn=mlp_forward,
        params=trained_params,
        x_context=x_context,
        kernel_structure=KernelStructure.NONE,
        kernel=kernel_fn,
        prior_variance=prior_variance,
        n_chunks=2,
        max_iter=50,
    )

    print(f"   Posterior rank: {posterior.rank}")

    # Make predictions with uncertainty
    print("\n5. Making predictions with uncertainty...")

    # Create a grid for visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    # Mean predictions
    mean_logits = jax.vmap(lambda x: mlp_forward(x, trained_params))(grid_jax)
    mean_probs = jax.nn.sigmoid(mean_logits)

    # Sample from posterior to get uncertainty estimates
    key = jax.random.PRNGKey(42)
    n_samples = 20
    posterior_samples = []

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        # Sample from posterior
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)

        # Predict with sampled params
        sample_logits = jax.vmap(lambda x: mlp_forward(x, sample_params))(grid_jax)
        posterior_samples.append(jax.nn.sigmoid(sample_logits))

    posterior_samples = jnp.stack(posterior_samples)
    predictive_std = jnp.std(posterior_samples, axis=0)

    print(f"   Mean probability range: [{mean_probs.min():.3f}, {mean_probs.max():.3f}]")
    print(f"   Std range: [{predictive_std.min():.3f}, {predictive_std.max():.3f}]")

    # Visualize results
    print("\n6. Creating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Decision boundary
    ax = axes[0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape),
        levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30)
    ax.scatter(x_context[:, 0], x_context[:, 1], c="green", marker="x", s=100,
               label="Context points")
    ax.set_title("Mean Predictions")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    plt.colorbar(contour, ax=ax)

    # Plot 2: Predictive uncertainty
    ax = axes[1]
    contour = ax.contourf(
        xx, yy, predictive_std.reshape(xx.shape),
        levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30)
    ax.set_title("Predictive Uncertainty (Std)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    # Plot 3: Confidence (1 - uncertainty)
    ax = axes[2]
    confidence = 1 - 2 * predictive_std  # Higher std = lower confidence
    contour = ax.contourf(
        xx, yy, confidence.reshape(xx.shape),
        levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30)
    ax.set_title("Model Confidence")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    plt.tight_layout()
    plt.savefig("two_moons_fsp_laplace.png", dpi=150, bbox_inches="tight")
    print("   Saved visualization to 'two_moons_fsp_laplace.png'")

    print("\n" + "=" * 70)
    print("Done! FSP Laplace successfully applied to two moons classification.")
    print("=" * 70)


if __name__ == "__main__":
    main()
