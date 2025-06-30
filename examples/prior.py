import jax
from jax import numpy as jnp
import math
from jax import Array
from matplotlib import pyplot as plt
from typing import Callable, Dict, Tuple
from jax import jit, grad, vmap
from jax.scipy.linalg import cholesky, solve_triangular
import optax


def periodic_kernel(x, y, lengthscale, period, variance=1.0):
    arg = jnp.pi * jnp.abs(x - y) / period
    sin2 = jnp.sin(arg) ** 2
    return variance * jnp.exp(jnp.sum(sin2, axis=-1) / (-2 * lengthscale**2))


def matern12_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    return variance * jnp.exp(-r / lengthscale)


def matern52_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    sqrt5 = jnp.sqrt(5.0)
    sr = sqrt5 * r / lengthscale
    return variance * (1.0 + sr + sr**2 / 3.0) * jnp.exp(-sr)


def rbf_kernel(x, y, lengthscale, variance=1.0):
    r = jnp.linalg.norm(x - y, axis=-1)
    return variance * jnp.exp(-(r**2) / (2 * lengthscale**2))


def composite_kernel(x, y, params):
    X = x[:, None, :]
    Y = y[None, :, :]
    k_per = periodic_kernel(
        X, Y, params["per_ls"], params["per_p"], params.get("per_var", 1.0)
    )
    k_matern52 = matern52_kernel(X, Y, params.get("matern52_ls", 1.0), 1.0)
    k_matern12 = matern12_kernel(
        X, Y, params.get("matern12_ls", 1.0), params.get("matern12_var", 0.0)
    )
    return k_per * k_matern52 + k_matern12


def gram(x: jnp.ndarray, params: dict, kernel_fn, jitter: float = 1e-6) -> jnp.ndarray:
    K = kernel_fn(x, x, params)
    return K + jitter * jnp.eye(x.shape[0])


# === HYPERPARAMETER OPTIMIZATION ===


def log_marginal_likelihood(
    params: Dict,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
) -> float:
    """
    Compute log marginal likelihood: log p(y | X, θ)
    = -0.5 * y^T K^{-1} y - 0.5 * log|K| - 0.5 * n * log(2π)
    """
    n = X.shape[0]

    # Build covariance matrix
    K = kernel_fn(X, X, params)
    K_noise = K + params["noise_var"] * jnp.eye(n)

    try:
        # Cholesky decomposition for numerical stability
        L = cholesky(K_noise, lower=True)

        # Solve L α = y  =>  α = L^{-1} y
        alpha = solve_triangular(L, y, lower=True)

        # Log determinant: log|K| = 2 * sum(log(diag(L)))
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # Data fit term: y^T K^{-1} y = α^T α
        data_fit = jnp.dot(alpha, alpha)

        # Log marginal likelihood
        log_ml = -0.5 * data_fit - 0.5 * log_det - 0.5 * n * jnp.log(2 * jnp.pi)

        return log_ml

    except Exception:
        # Return large negative value if Cholesky fails
        return -1e10


def negative_log_marginal_likelihood(
    params: Dict,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
) -> float:
    """Negative log marginal likelihood for minimization."""
    return -log_marginal_likelihood(params, X, y, kernel_fn)


# Parameter transformation functions (for constrained optimization)
def transform_params(raw_params: Dict) -> Dict:
    """Transform unconstrained parameters to constrained parameter space."""
    return {
        "per_ls": jnp.exp(raw_params["log_per_ls"]),  # lengthscale > 0
        "per_p": jnp.exp(raw_params["log_per_p"]),  # period > 0
        "per_var": jnp.exp(raw_params["log_per_var"]),  # variance > 0
        "matern52_ls": jnp.exp(raw_params["log_matern52_ls"]),  # lengthscale > 0
        "matern12_ls": jnp.exp(raw_params["log_matern12_ls"]),  # lengthscale > 0
        "matern12_var": jnp.exp(raw_params["log_matern12_var"]),  # variance > 0
        "noise_var": jnp.exp(raw_params["log_noise_var"]),  # noise variance > 0
    }


def inverse_transform_params(transformed_params: Dict) -> Dict:
    """Inverse transform from constrained to unconstrained parameter space."""
    return {
        "log_per_ls": jnp.log(transformed_params["per_ls"]),
        "log_per_p": jnp.log(transformed_params["per_p"]),
        "log_per_var": jnp.log(transformed_params["per_var"]),
        "log_matern52_ls": jnp.log(transformed_params["matern52_ls"]),
        "log_matern12_ls": jnp.log(transformed_params["matern12_ls"]),
        "log_matern12_var": jnp.log(transformed_params["matern12_var"]),
        "log_noise_var": jnp.log(transformed_params["noise_var"]),
    }


def objective(
    raw_params: Dict,
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
) -> float:
    """Objective function with parameter transformation."""
    params = transform_params(raw_params)
    return negative_log_marginal_likelihood(params, X, y, kernel_fn)


# Optimization setup
def optimize_hyperparameters(
    X: jnp.ndarray,
    y: jnp.ndarray,
    kernel_fn: Callable,
    initial_params: Dict,
    learning_rate: float = 0.01,
    num_steps: int = 1000,
) -> Tuple[Dict, list]:
    """
    Optimize kernel hyperparameters using gradient descent.

    Args:
        X: Input locations [N, D]
        y: Target values [N]
        kernel_fn: Kernel function
        initial_params: Initial hyperparameters (in log space)
        learning_rate: Learning rate for optimization
        num_steps: Number of optimization steps
        noise_var: Observation noise variance

    Returns:
        Optimized parameters and loss history
    """

    # Set up optimizer
    optimizer = optax.adam(learning_rate)

    # Compile the gradient function

    grad_fn = grad(objective, argnums=0)

    # Initialize
    params = inverse_transform_params(initial_params)
    opt_state = optimizer.init(params)
    loss_history = []

    print("Starting optimization...")

    for step in range(num_steps):
        # Compute loss and gradients
        loss = objective(params, X, y, kernel_fn)
        grads = grad_fn(params, X, y, kernel_fn)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        loss_history.append(float(loss))

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
            # Print current hyperparameters
            current_params = transform_params(params)
            print(
                f"  per_ls: {current_params['per_ls']:.4f}, per_p: {current_params['per_p']:.4f}, "
                f"per_var: {current_params['per_var']:.4f}, noise_var: {current_params['noise_var']:.4f}"
            )
            # print(f"  rbf_ls: {current_params['rbf_ls']:.4f}")

    return transform_params(params), loss_history


# Example usage and testing
def create_test_data():
    """Create synthetic test data with periodic + smooth trend."""
    key = jax.random.PRNGKey(0)
    X = jnp.linspace(-2, 2, 150).reshape(-1, 1)

    # True function: periodic + smooth trend + noise
    true_f = jnp.sin(2 * jnp.pi * X.squeeze())
    noise = 0.1 * jax.random.normal(key, (len(X),))
    y = true_f + noise

    return X, y, true_f


def gp_predict(X_train, y_train, X_test, params, kernel_fn):
    """
    Compute GP predictions with proper error handling and debugging.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    print(f"GP Predict - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Params: {params}")

    # Build covariance matrices
    K_train = kernel_fn(X_train, X_train, params)
    print(
        f"K_train shape: {K_train.shape}, range: [{K_train.min():.4f}, {K_train.max():.4f}]"
    )

    # Add noise to diagonal
    K_train_noise = K_train + params.get("noise_var", 1e-2) * jnp.eye(n_train)

    # Cross-covariance: note the order matters for shapes
    K_cross = kernel_fn(X_train, X_test, params)  # [n_train, n_test]
    print(f"K_cross shape: {K_cross.shape}")

    # Test covariance (for uncertainty)
    K_test = kernel_fn(X_test, X_test, params)

    try:
        # Solve using Cholesky for numerical stability
        L = jnp.linalg.cholesky(K_train_noise)

        # Solve L @ alpha = y for alpha
        alpha = jnp.linalg.solve(L, y_train)

        # Solve L.T @ beta = alpha for beta, so beta = K^{-1} y
        beta = jnp.linalg.solve(L.T, alpha)

        # Mean prediction: f* = K*T @ K^{-1} @ y = K*T @ beta
        f_mean = K_cross.T @ beta  # [n_test]

        # For variance: solve L @ v = K*T for v
        v = jnp.linalg.solve(L, K_cross)  # [n_train, n_test]

        # Predictive variance: K** - K*T @ K^{-1} @ K*
        f_var = jnp.diag(K_test) - jnp.sum(v**2, axis=0)  # [n_test]
        f_var = jnp.maximum(f_var, 1e-12)  # Ensure positive
        f_std = jnp.sqrt(f_var)

        print(f"Prediction range: [{f_mean.min():.4f}, {f_mean.max():.4f}]")
        print(f"Std range: [{f_std.min():.4f}, {f_std.max():.4f}]")

        return f_mean, f_std

    except Exception as e:
        print(f"GP prediction failed: {e}")
        # Fallback to simple prediction
        f_mean = jnp.zeros(n_test)
        f_std = jnp.ones(n_test)
        return f_mean, f_std


def plot_optimization_results(X, y, true_f, best_params, all_histories):
    """Plot the optimization results with better GP prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Data and true function
    axes[0, 0].scatter(X.squeeze(), y, alpha=0.7, label="Noisy observations", s=30)
    axes[0, 0].plot(X.squeeze(), true_f, "r-", label="True function", linewidth=2)
    axes[0, 0].set_title("Training Data")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss curves
    for i, history in enumerate(all_histories):
        axes[0, 1].plot(history, alpha=0.8, label=f"Restart {i + 1}", linewidth=2)
    axes[0, 1].set_title("Optimization Loss Curves")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Negative Log Marginal Likelihood")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: GP prediction with optimized hyperparameters
    X_test = jnp.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)

    print("\n=== GP Prediction Debug ===")
    f_mean, f_std = gp_predict(X, y, X_test, best_params, composite_kernel)

    axes[1, 0].fill_between(
        X_test.squeeze(),
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        alpha=0.2,
        color="blue",
        label="95% CI",
    )
    axes[1, 0].plot(X_test.squeeze(), f_mean, "b-", label="GP Mean", linewidth=2)
    axes[1, 0].scatter(
        X.squeeze(),
        y,
        alpha=0.8,
        color="red",
        label="Training data",
        s=40,
        edgecolors="darkred",
    )

    # Also plot true function on test points for comparison
    X_test_true = X_test.squeeze()
    true_f_test = (
        jnp.sin(X_test_true) + 0.1 * X_test_true + 0.3 * jnp.sin(0.5 * X_test_true)
    )
    axes[1, 0].plot(
        X_test.squeeze(),
        true_f_test,
        "g--",
        label="True function",
        linewidth=2,
        alpha=0.7,
    )

    axes[1, 0].set_title("GP Prediction with Optimized Hyperparameters")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Hyperparameter values
    param_names = list(best_params.keys())
    param_values = [best_params[name] for name in param_names]

    bars = axes[1, 1].bar(range(len(param_names)), param_values, alpha=0.7)
    axes[1, 1].set_xticks(range(len(param_names)))
    axes[1, 1].set_xticklabels(param_names, rotation=45, ha="right")
    axes[1, 1].set_title("Optimized Hyperparameters")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    # Print final hyperparameters
    print("\nOptimized Hyperparameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")

    # Print some diagnostics
    print(f"\nDiagnostics:")
    print(f"  Training data range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Prediction range: [{f_mean.min():.3f}, {f_mean.max():.3f}]")
    print(f"  Average uncertainty: {f_std.mean():.3f}")

    # Compute and print final log marginal likelihood
    final_lml = log_marginal_likelihood(best_params, X, y, composite_kernel, 1e-3)
    print(f"  Final log marginal likelihood: {final_lml:.3f}")


# Additional debugging functions
def debug_kernel_behavior(X, params, kernel_fn):
    """Debug kernel behavior with current parameters."""
    print("\n=== Kernel Debug ===")
    print(f"Input shape: {X.shape}")

    # Test kernel evaluation
    K = kernel_fn(X, X, params)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel matrix range: [{K.min():.6f}, {K.max():.6f}]")
    print(f"Kernel matrix diagonal: {jnp.diag(K)[:5]}...")
    print(f"Kernel matrix condition number: {jnp.linalg.cond(K):.2e}")

    # Check if kernel is positive definite
    try:
        jnp.linalg.cholesky(K + 1e-6 * jnp.eye(len(K)))
        print("Kernel is positive definite ✓")
    except:
        print("Kernel is NOT positive definite ✗")

    return K


# Main execution
if __name__ == "__main__":
    # Create test data
    X, y, true_f = create_test_data()

    print("\n=== Testing with original data ===")

    # Test with reasonable initial parameters before optimization
    reasonable_params = {
        "per_ls": 1.0,
        "per_p": 1.0,  # Should match the main period in the data
        "per_var": 1.0,
        "matern52_ls": 4.0,
        "matern12_ls": 0.25,
        "matern12_var": 1e-3,
        "noise_var": 1e-2,  # Small noise variance
    }

    print("Testing kernel with reasonable parameters:")
    debug_kernel_behavior(X, reasonable_params, composite_kernel)

    key = jax.random.key(0)
    X_train1 = jnp.linspace(-1, -0.5, 75).reshape(-1, 1)
    X_train2 = jnp.linspace(0.5, 1, 75).reshape(-1, 1)
    X_train = jnp.concatenate([X_train1, X_train2], axis=0)
    y_train = jnp.reshape(
        jnp.sin(X_train * 2 * jnp.pi) + jax.random.normal(key, (150, 1)) * 0.1, (-1, 1)
    )[:, 0]

    # Test prediction before optimization
    X_test_debug = jnp.linspace(-3, 3, 200).reshape(-1, 1)
    f_mean_test, f_std_test = gp_predict(
        X_train, y_train, X_test_debug, reasonable_params, composite_kernel
    )

    plt.figure(figsize=(10, 6))
    plt.fill_between(
        X_test_debug.squeeze(),
        f_mean_test - 2 * f_std_test,
        f_mean_test + 2 * f_std_test,
        alpha=0.3,
    )
    plt.plot(X_test_debug.squeeze(), f_mean_test, "b-", label="GP mean (before opt)")
    plt.scatter(X_train.squeeze(), y_train, color="red", label="Training data")
    plt.plot(X.squeeze(), true_f, "g-", label="True function")
    plt.legend()
    plt.title("GP with Reasonable Initial Parameters")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Now run optimization
    print("\n=== Running Optimization ===")
    best_params, losses = optimize_hyperparameters(
        X_train,
        y_train,
        composite_kernel,
        initial_params=reasonable_params,
        learning_rate=0.01,
        num_steps=500,  # Reduced for faster debugging
    )

    print(f"\nBest final loss: {losses[-1]:.4f}")

    # Debug optimized parameters
    debug_kernel_behavior(X, best_params, composite_kernel)

    # Plot results
    plot_optimization_results(X_train, y_train, true_f, best_params, [losses])
