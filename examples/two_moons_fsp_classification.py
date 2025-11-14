"""Two Moons Classification: FSP Laplace vs Standard Laplace.

This example demonstrates the difference between FSP (Function-Space Prior) Laplace
and standard parameter-space Laplace approximation for binary classification on
the two moons dataset.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from laplax.api import GGN
from laplax.curv import (
    KernelStructure,
    create_fsp_posterior,
    estimate_curvature,
    set_posterior_fn,
)
from laplax.enums import CurvApprox, LossFn
from laplax.util.flatten import create_pytree_flattener

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
    """Run two moons classification comparing FSP Laplace vs Standard Laplace."""
    print("=" * 70)
    print("Two Moons Classification: FSP Laplace vs Standard Laplace")
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
    # Use smaller prior variance to make it more informative
    kernel_lengthscale = 0.5
    kernel_variance = 0.01  # Reduced from 0.1 to make prior tighter

    def kernel_fn(v):
        """Kernel matrix-vector product."""
        K = create_simple_kernel(x_context, lengthscale=kernel_lengthscale, variance=kernel_variance)
        return K @ v

    # Compute prior variance
    prior_cov = create_simple_kernel(x_context, lengthscale=kernel_lengthscale, variance=kernel_variance)
    prior_variance = jnp.diag(prior_cov)

    print(f"   Prior variance range: [{prior_variance.min():.4f}, {prior_variance.max():.4f}]")

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

    print(f"   FSP posterior rank: {posterior.rank}")

    # Compute standard Laplace posterior for comparison
    print("\n5. Computing standard Laplace posterior...")

    # Create GGN matrix-vector product for standard Laplace
    flatten_fn, unflatten_fn = create_pytree_flattener(trained_params)

    def model_fn_flat(input, params):
        """Model function with flattened parameters (GGN expects keyword args)."""
        params_pytree = unflatten_fn(params)
        # For binary classification, return 2 logits (class 0, class 1)
        logit = mlp_forward(input, params_pytree)
        # Convert single logit to 2-class logits: [logit_class0, logit_class1]
        return jnp.stack([jnp.zeros_like(logit), logit], axis=-1)

    # Create GGN on full dataset - needs to be dict with 'input' and 'target' keys
    # Convert targets to integers for cross-entropy
    data = {"input": X_jax, "target": y_jax.astype(jnp.int32)}
    ggn_mv = GGN(
        model_fn_flat,
        flatten_fn(trained_params),
        data,
        loss_fn=LossFn.CROSS_ENTROPY,
        vmap_over_data=True,
    )

    # Estimate curvature with low-rank approximation
    max_rank = 50
    curv_estimate = estimate_curvature(
        curv_type=CurvApprox.LANCZOS,
        mv=ggn_mv,
        layout=flatten_fn(trained_params).shape[0],
        rank=max_rank,
        key=jax.random.key(42),
        has_batch=True,
    )

    # Set posterior function (this creates a factory for posterior objects)
    posterior_fn = set_posterior_fn(
        curv_type=CurvApprox.LANCZOS,
        curv_estimate=curv_estimate,
        layout=flatten_fn(trained_params).shape[0],
    )

    # Create the actual posterior with prior precision
    prior_args = {"prior_prec": 1.0}
    standard_posterior = posterior_fn(prior_args)

    print(f"   Standard Laplace rank: {curv_estimate.U.shape[1]}")

    # Make predictions with uncertainty
    print("\n6. Making predictions with uncertainty...")

    # Create a grid for visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    # Mean predictions (same for both methods)
    mean_logits = jax.vmap(lambda x: mlp_forward(x, trained_params))(grid_jax)
    mean_probs = jax.nn.sigmoid(mean_logits)

    # Sample from FSP posterior
    print("   Sampling from FSP posterior...")
    key = jax.random.PRNGKey(42)
    n_samples = 20
    fsp_samples = []

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        # Sample from FSP posterior
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)

        # Predict with sampled params
        sample_logits = jax.vmap(lambda x: mlp_forward(x, sample_params))(grid_jax)
        fsp_samples.append(jax.nn.sigmoid(sample_logits))

    fsp_samples = jnp.stack(fsp_samples)
    fsp_std = jnp.std(fsp_samples, axis=0)

    # Sample from standard Laplace posterior
    print("   Sampling from standard Laplace posterior...")
    key = jax.random.PRNGKey(42)
    standard_samples = []

    # For low-rank posterior, need to sample in full parameter space
    param_size = flatten_fn(trained_params).shape[0]

    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        # Sample from standard Laplace posterior - need full parameter space vector
        z = jax.random.normal(subkey, (param_size,))
        delta_params_flat = standard_posterior.scale_mv(standard_posterior.state)(z)
        delta_params = unflatten_fn(delta_params_flat)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)

        # Predict with sampled params
        sample_logits = jax.vmap(lambda x: mlp_forward(x, sample_params))(grid_jax)
        standard_samples.append(jax.nn.sigmoid(sample_logits))

    standard_samples = jnp.stack(standard_samples)
    standard_std = jnp.std(standard_samples, axis=0)

    print(f"   FSP uncertainty - Std range: [{fsp_std.min():.3f}, {fsp_std.max():.3f}]")
    print(f"   Standard uncertainty - Std range: [{standard_std.min():.3f}, {standard_std.max():.3f}]")

    # Visualize results
    print("\n7. Creating comparison visualizations...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: FSP Laplace
    # Plot 1: Mean predictions
    ax = axes[0, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape),
        levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.scatter(x_context[:, 0], x_context[:, 1], c="green", marker="x", s=100,
               label="Context points", linewidths=2)
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    plt.colorbar(contour, ax=ax)

    # Plot 2: FSP uncertainty
    ax = axes[0, 1]
    contour = ax.contourf(
        xx, yy, fsp_std.reshape(xx.shape),
        levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(f"FSP Laplace Uncertainty (rank={posterior.rank})", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    # Plot 3: FSP confidence
    ax = axes[0, 2]
    fsp_confidence = 1 - 2 * fsp_std
    contour = ax.contourf(
        xx, yy, fsp_confidence.reshape(xx.shape),
        levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("FSP Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    # Row 2: Standard Laplace
    # Plot 4: Placeholder (using mean predictions again)
    ax = axes[1, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape),
        levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax)

    # Plot 5: Standard Laplace uncertainty
    ax = axes[1, 1]
    contour = ax.contourf(
        xx, yy, standard_std.reshape(xx.shape),
        levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(f"Standard Laplace Uncertainty (rank={curv_estimate.U.shape[1]})", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    # Plot 6: Standard Laplace confidence
    ax = axes[1, 2]
    standard_confidence = 1 - 2 * standard_std
    contour = ax.contourf(
        xx, yy, standard_confidence.reshape(xx.shape),
        levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Standard Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    plt.tight_layout()
    plt.savefig("two_moons_comparison.png", dpi=150, bbox_inches="tight")
    print("   Saved comparison to 'two_moons_comparison.png'")

    # Compute and display quantitative comparison
    print("\n" + "=" * 70)
    print("Comparison Summary:")
    print("=" * 70)
    print(f"FSP Laplace:")
    print(f"  - Rank: {posterior.rank}")
    print(f"  - Mean uncertainty: {fsp_std.mean():.4f}")
    print(f"  - Max uncertainty: {fsp_std.max():.4f}")
    print(f"\nStandard Laplace:")
    print(f"  - Rank: {curv_estimate.U.shape[1]}")
    print(f"  - Mean uncertainty: {standard_std.mean():.4f}")
    print(f"  - Max uncertainty: {standard_std.max():.4f}")
    print(f"\nUncertainty difference (FSP - Standard):")
    print(f"  - Mean: {(fsp_std - standard_std).mean():.4f}")
    print(f"  - Max absolute: {jnp.abs(fsp_std - standard_std).max():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
