"""FSP vs Standard Laplace on Two Moons (Binary Classification).

This script mirrors the Jupyter notebook `fsp_two_moons_classification.ipynb` but as
an executable Python script. It trains a small MLP on the noiseless two-moons
dataset, constructs an FSP posterior with an RBF kernel on context points, and
compares predictive uncertainty against a standard Laplace (low‑rank/Lanczos) posterior.

It saves a visualization to `fsp_two_moons_classification.png`.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from sklearn.datasets import make_moons

from laplax.api import GGN
from laplax.curv import KernelStructure, create_fsp_posterior, estimate_curvature, set_posterior_fn
from laplax.enums import CurvApprox, LossFn
from laplax.util.flatten import create_pytree_flattener


# ==============================================================================
# Model (Flax NNX)
# ==============================================================================


class MLP(nnx.Module):
    """Simple MLP for binary classification."""

    def __init__(self, hidden_dims: list[int], *, rngs: nnx.Rngs):
        in_dim = 2  # Two moons features
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        layers.append(nnx.Linear(in_dim, 1, rngs=rngs))

        self.hidden_layers = tuple(layers[:-1])
        self.output_layer = layers[-1]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.hidden_layers:
            x = jnp.tanh(layer(x))
        x = self.output_layer(x)
        return x.squeeze()


def split_model(model: MLP):
    """Split Flax NNX model into function and parameters."""
    graphdef, params = nnx.split(model, nnx.Param)

    def model_fn(x, params):
        model_copy = nnx.merge(graphdef, params)
        return model_copy(x)

    return model_fn, params


# ==============================================================================
# Training
# ==============================================================================


def binary_cross_entropy_loss(model, x_batch, y_batch):
    logits = jax.vmap(model)(x_batch)
    return -jnp.mean(
        jax.nn.log_sigmoid(logits) * y_batch
        + jax.nn.log_sigmoid(-logits) * (1 - y_batch)
    )


def train_mlp(model, x_train, y_train, num_epochs=100, learning_rate=0.1):
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        loss, grads = nnx.value_and_grad(binary_cross_entropy_loss)(
            model, x_batch, y_batch
        )
        optimizer.update(model, grads)
        return loss

    for epoch in range(num_epochs):
        loss = train_step(model, optimizer, x_train, y_train)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model


# ==============================================================================
# Kernels (RBF) for FSP
# ==============================================================================


def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    sq_dist = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    return variance * jnp.exp(-sq_dist / (2 * lengthscale**2))


def create_kernel_matrix(x_context, lengthscale=1.0, variance=1.0):
    K = rbf_kernel(x_context, x_context, lengthscale, variance)
    return K + 1e-6 * jnp.eye(K.shape[0])


# ==============================================================================
# Main
# ==============================================================================


def main():
    print("=" * 70)
    print("FSP vs Standard Laplace: Two Moons Classification (noise=0.0)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------
    print("1) Generating two moons dataset (noise=0.0)...")
    X, y = make_moons(n_samples=300, noise=0.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_jax = jnp.array(X)
    y_jax = jnp.array(y)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")

    # ------------------------------------------------------------------
    # 2) Train model
    # ------------------------------------------------------------------
    print("\n2) Training MLP...")
    model = MLP(hidden_dims=[32, 32], rngs=nnx.Rngs(0))
    model = train_mlp(model, X_jax, y_jax, num_epochs=100, learning_rate=0.1)

    logits = jax.vmap(model)(X_jax)
    predictions = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    train_acc = jnp.mean(predictions == y_jax)
    print(f"   Training accuracy: {float(train_acc):.2%}")

    # ------------------------------------------------------------------
    # 3) FSP posterior
    # ------------------------------------------------------------------
    print("\n3) Building FSP posterior with RBF kernel on context points...")

    # Select context points
    n_context = 50
    rng = np.random.default_rng(42)
    context_indices = rng.choice(len(X), size=n_context, replace=False)
    x_context = X_jax[context_indices]
    print(f"   Context points: {x_context.shape[0]}")

    # Kernel hyperparameters
    lengthscale = 0.5
    variance = 0.01

    def kernel_fn(v):
        K = create_kernel_matrix(x_context, lengthscale=lengthscale, variance=variance)
        return K @ v

    prior_cov = create_kernel_matrix(x_context, lengthscale=lengthscale, variance=variance)
    prior_variance = jnp.diag(prior_cov)
    print(
        f"   Prior variance range: [{float(prior_variance.min()):.4f}, {float(prior_variance.max()):.4f}]"
    )

    # Split for Laplax
    model_fn, trained_params = split_model(model)

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

    # ------------------------------------------------------------------
    # 4) Standard Laplace (Lanczos low-rank curvature)
    # ------------------------------------------------------------------
    print("\n4) Estimating standard Laplace curvature (Lanczos)...")
    flatten_fn, unflatten_fn = create_pytree_flattener(trained_params)

    def model_fn_flat(inp, flat_params):
        params_pytree = unflatten_fn(flat_params)
        logit = model_fn(inp, params_pytree)
        return jnp.stack([jnp.zeros_like(logit), logit], axis=-1)

    data = {"input": X_jax, "target": y_jax.astype(jnp.int32)}
    ggn_mv = GGN(
        model_fn_flat,
        flatten_fn(trained_params),
        data,
        loss_fn=LossFn.CROSS_ENTROPY,
        vmap_over_data=True,
    )

    max_rank = 50
    curv_estimate = estimate_curvature(
        curv_type=CurvApprox.LANCZOS,
        mv=ggn_mv,
        layout=flatten_fn(trained_params).shape[0],
        rank=max_rank,
        key=jax.random.key(42),
        has_batch=True,
    )

    posterior_fn = set_posterior_fn(
        curv_type=CurvApprox.LANCZOS,
        curv_estimate=curv_estimate,
        layout=flatten_fn(trained_params).shape[0],
    )
    standard_posterior = posterior_fn({"prior_prec": 1.0})
    print(f"   Standard Laplace rank: {curv_estimate.U.shape[1]}")

    # ------------------------------------------------------------------
    # 5) Predictions & Sampling
    # ------------------------------------------------------------------
    print("\n5) Creating prediction grid and sampling posteriors...")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    # Mean prediction
    mean_logits = jax.vmap(model)(grid_jax)
    mean_probs = jax.nn.sigmoid(mean_logits)

    # FSP sampling
    print("   Sampling from FSP posterior...")
    key = jax.random.PRNGKey(42)
    n_samples = 20
    fsp_samples = []
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (posterior.rank,))
        delta_params = posterior.scale_mv(posterior.state)(z)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)
        sample_logits = jax.vmap(lambda x: model_fn(x, sample_params))(grid_jax)
        fsp_samples.append(jax.nn.sigmoid(sample_logits))
    fsp_samples = jnp.stack(fsp_samples)
    fsp_std = jnp.std(fsp_samples, axis=0)

    # Standard Laplace sampling
    print("   Sampling from Standard Laplace posterior...")
    key = jax.random.PRNGKey(42)
    standard_samples = []
    param_size = int(flatten_fn(trained_params).shape[0])
    for _ in range(n_samples):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (param_size,))
        delta_params_flat = standard_posterior.scale_mv(standard_posterior.state)(z)
        delta_params = unflatten_fn(delta_params_flat)
        sample_params = jax.tree.map(lambda p, dp: p + dp, trained_params, delta_params)
        sample_logits = jax.vmap(lambda x: model_fn(x, sample_params))(grid_jax)
        standard_samples.append(jax.nn.sigmoid(sample_logits))
    standard_samples = jnp.stack(standard_samples)
    standard_std = jnp.std(standard_samples, axis=0)

    print(
        f"   FSP std range: [{float(fsp_std.min()):.3f}, {float(fsp_std.max()):.3f}]"
    )
    print(
        "   Standard std range: "
        f"[{float(standard_std.min()):.3f}, {float(standard_std.max()):.3f}]"
    )

    # ------------------------------------------------------------------
    # 6) Visualization
    # ------------------------------------------------------------------
    print("\n6) Saving visualizations to 'fsp_two_moons_classification.png'...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Mean predictions (shared)
    ax = axes[0, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.scatter(
        x_context[:, 0],
        x_context[:, 1],
        c="green",
        marker="x",
        s=100,
        label="Context points",
        linewidths=2,
    )
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    plt.colorbar(contour, ax=ax)

    # Row 1: FSP uncertainty and confidence
    ax = axes[0, 1]
    contour = ax.contourf(
        xx, yy, fsp_std.reshape(xx.shape), levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(
        f"FSP Laplace Uncertainty (rank={posterior.rank})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    ax = axes[0, 2]
    fsp_confidence = 1 - 2 * fsp_std
    contour = ax.contourf(
        xx, yy, fsp_confidence.reshape(xx.shape), levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("FSP Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    # Row 2: Standard Laplace uncertainty and confidence
    ax = axes[1, 0]
    contour = ax.contourf(
        xx, yy, mean_probs.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Mean Predictions (Both Methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax)

    ax = axes[1, 1]
    contour = ax.contourf(
        xx, yy, standard_std.reshape(xx.shape), levels=20, cmap="viridis", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title(
        f"Standard Laplace Uncertainty (rank={curv_estimate.U.shape[1]})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Std")

    ax = axes[1, 2]
    standard_confidence = 1 - 2 * standard_std
    contour = ax.contourf(
        xx, yy, standard_confidence.reshape(xx.shape), levels=20, cmap="plasma", alpha=0.8
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=30, alpha=0.6)
    ax.set_title("Standard Laplace Confidence", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(contour, ax=ax, label="Confidence")

    plt.tight_layout()
    plt.savefig("fsp_two_moons_classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   Saved 'fsp_two_moons_classification.png'.")

    # ------------------------------------------------------------------
    # 7) Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("FSP Laplace:")
    print(f"  - Rank: {posterior.rank}")
    print(f"  - Mean uncertainty: {float(fsp_std.mean()):.4f}")
    print(f"  - Max uncertainty: {float(fsp_std.max()):.4f}")
    print("\nStandard Laplace:")
    print(f"  - Rank: {curv_estimate.U.shape[1]}")
    print(f"  - Mean uncertainty: {float(standard_std.mean()):.4f}")
    print(f"  - Max uncertainty: {float(standard_std.max()):.4f}")
    print("\nUncertainty difference (FSP - Standard):")
    diff = fsp_std - standard_std
    print(f"  - Mean: {float(diff.mean()):.4f}")
    print(f"  - Max absolute: {float(jnp.abs(diff).max()):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

