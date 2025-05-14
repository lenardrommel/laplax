import datetime
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from loguru import logger
from orbax import checkpoint as ocp
from tueplots import bundles, fonts

from laplax import laplace
from laplax.types import Callable, Float, PriorArguments

# ------------------------------------------------------------------------------
# TASK Setup
# ------------------------------------------------------------------------------


DEFAULT_INTERVALS = [(0, 2), (4, 5), (6, 8)]


def build_sinusoid_data(
    num_train=150,
    num_valid=50,
    num_test=150,
    sigma_noise=0.3,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    rng_seed=0,
):
    key = jax.random.key(rng_seed)
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_sinusoid_example(
        num_train_data=num_train,
        num_valid_data=num_valid,
        num_test_data=num_test,
        sigma_noise=sigma_noise,
        intervals=intervals,
        rng_key=key,
    )

    train_loader = DataLoader(X_train, y_train, batch_size)
    valid_loader = DataLoader(X_valid, y_valid, batch_size)
    test_loader = DataLoader(X_test, y_test, batch_size)

    return train_loader, valid_loader, test_loader


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.linear2(h)


@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)  # Call methods directly
        return jnp.sum((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


def train_map_model(
    model,
    train_loader,
    n_epochs,
    *,
    lr=1e-3,
    verbose=True,
):
    optimizer = nnx.Optimizer(model, optax.adamw(lr))
    loss = 0.0
    for epoch in range(n_epochs):
        for xb, yb in train_loader:
            loss = train_step(model, optimizer, xb, yb)
        if verbose and epoch % (n_epochs // 10) == 0:
            logger.info(f"Epoch {epoch}/{n_epochs}, loss={loss:.4f}")
    if verbose:
        logger.info(f"Final training loss: {loss:.4f}")
    return model


# ------------------------------------------------------------------------------
# Plotting and Logging Functions
# ------------------------------------------------------------------------------

def plot_regression_with_uncertainty_icml(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    X_pred=None,
    y_pred=None,
    y_std=None,
    title=None,
):
    """Plot regression data with optional prediction and uncertainty.

    Args:
        X_train: Training input data (shape: n_samples, 1)
        y_train: Training target data (shape: n_samples, 1)
        X_test: Test input data (shape: n_samples, 1), optional
        y_test: Test target data (shape: n_samples, 1), optional
        X_pred: Prediction input data (shape: n_samples, 1), if None uses X_test
        y_pred: Prediction mean (shape: n_samples, 1), optional
        y_std: Prediction standard deviation (shape: n_samples, 1), optional
        title: Plot title, optional
    """
    # Apply ICML formatting
    with plt.rc_context({
        **bundles.icml2022(column="half", nrows=1, ncols=1),
        **fonts.icml2022_tex(),
    }):
        fig, ax = plt.subplots()

        # Convert to numpy arrays if they are JAX arrays
        if hasattr(X_train, "device_buffer"):
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            if X_test is not None:
                X_test = np.array(X_test)
                y_test = np.array(y_test)
            if X_pred is not None and y_pred is not None:
                X_pred = np.array(X_pred)
                y_pred = np.array(y_pred)
                if y_std is not None:
                    y_std = np.array(y_std)

        # Plot training data
        ax.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data", s=10)

        # Plot test data if provided
        if X_test is not None and y_test is not None:
            ax.scatter(X_test, y_test, color="green", alpha=0.6, label="Test data", s=10)

        # Plot prediction with uncertainty if provided
        if y_pred is not None:
            # If X_pred is not provided but X_test is, use X_test for predictions
            X_plot = X_pred if X_pred is not None else X_test

            # Only proceed if we have points to plot predictions for
            if X_plot is not None:
                # Sort X for proper line plotting
                sort_idx = np.argsort(X_plot.flatten())
                X_plot_sorted = X_plot[sort_idx]
                y_pred_sorted = y_pred[sort_idx]

                ax.plot(X_plot_sorted, y_pred_sorted, color="red", label="Prediction", linewidth=1.5)

                # Plot uncertainty if provided
                if y_std is not None:
                    y_std_sorted = y_std[sort_idx]
                    ax.fill_between(
                        X_plot_sorted.flatten(),
                        (y_pred_sorted - 2 * y_std_sorted).flatten(),
                        (y_pred_sorted + 2 * y_std_sorted).flatten(),
                        color="red",
                        alpha=0.2,
                        label="95% confidence interval",
                    )

        # Plot true function
        x_true = np.linspace(0, 8, 1000).reshape(-1, 1)
        y_true = np.sin(x_true)
        ax.plot(x_true, y_true, color="black", linestyle="--", label="True function", linewidth=1.5)

        # Add labels and title
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Regression with Uncertainty")

        ax.legend(frameon=True, fancybox=False, edgecolor="black")
        ax.grid(True, alpha=0.3, linestyle="--")

    return fig


def save_results_to_csv(
    results,
    experiment_name,
    output_dir="results",
    *,
    log_args=None,
):
    """Save experiment results to a CSV file by appending to a single file."""
    # Create output directory if it doesn't exist
    if log_args is None:
        log_args = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract evaluation metrics
    eval_results = results.get("evaluation", {})

    # Create a dictionary with evaluation results and add metadata
    log_args = {} if log_args is None else log_args
    data = {
        **eval_results,
        **log_args,
        "experiment_name": experiment_name,
        "timestamp": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add NLL if available
    if "nll" in results:
        data["nll"] = results["nll"]

    # Create DataFrame from the data
    df = pd.DataFrame([data])

    # Define the CSV file path (single file for all experiments)
    csv_path = output_path / "regression_experiments.csv"

    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()

    # Append to CSV (or create if it doesn't exist)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    logger.info(f"Results appended to {csv_path}")

    return csv_path


def generate_experiment_name(
    curv_type,
    hidden_channels,
    num_train,
    *,
    calibration=False,
    calibration_objective=None,
):
    """Generate a descriptive name for the experiment."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    name_parts = [
        f"reg_{curv_type}",
        f"h{hidden_channels}",
        f"n{num_train}"
    ]

    if calibration:
        name_parts.append("calibrated")
        if calibration_objective:
            name_parts.append(str(calibration_objective).split('.')[-1].lower())

    return f"{timestamp}_{'_'.join(name_parts)}"


# ------------------------------------------------------------------------------
# LAPLACE support functions
# ------------------------------------------------------------------------------


def optimize_prior_prec_gradient(
    objective: Callable[[PriorArguments], float],
    *,
    init_prior_prec: Float | None = None,
    init_sigma_noise: Float | None = None,
    num_epochs: int = 20,
    learning_rate: float = 1e-2,
    optimizer_fn: Callable = None,
    **kwargs,
) -> Float:
    """Optimize prior precision using gradient descent.

    Args:
        objective: A callable objective function that takes `PriorArguments` as input
            and returns a float result.
        init_prior_prec: Initial prior precision value (default: None)
        init_sigma_noise: Initial noise standard deviation value (default: None)
        num_epochs: Number of optimization epochs (default: 20)
        learning_rate: Learning rate for the optimizer (default: 1e-3)
        optimizer_fn: Function to create the optimizer (default: optax.adam)
        **kwargs: Additional arguments

    Returns:
        The optimized prior precision value.
    """
    del kwargs

    # Default optimizer if none provided
    if optimizer_fn is None:
        def optimizer_fn(lr):
            return optax.adam(lr)

    # Initialize in log space for optimization over positive values
    prior_args = {}
    if init_prior_prec is None and init_sigma_noise is None:
        msg = "init_prior_prec and init_sigma_noise cannot both be None"
        raise ValueError(msg)
    if init_prior_prec is not None:
        prior_args["prior_prec"] = jnp.log(init_prior_prec)
    if init_sigma_noise is not None:
        prior_args["sigma_noise"] = jnp.log(init_sigma_noise)

    # Create optimizer
    optimizer = optimizer_fn(learning_rate)
    opt_state = optimizer.init(prior_args)

    # Optimization loop
    for epoch in range(num_epochs):
        # Compute value and gradient
        val, grads = jax.value_and_grad(
            lambda p: objective(jax.tree.map(jnp.exp, p))
        )(prior_args)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        prior_args = optax.apply_updates(prior_args, updates)

        logger.info(f"Epoch {epoch}: objective = {val:.6f}, "
                   f"log_prior_prec = {prior_args['prior_prec']:.6f}")

    # Transform back from log space
    final_prior_prec = jnp.exp(prior_args["prior_prec"]) \
        if "prior_prec" in prior_args else None
    final_sigma_noise = jnp.exp(prior_args["sigma_noise"]) \
        if "sigma_noise" in prior_args else None
    if final_prior_prec is not None:
        logger.info(f"Final prior precision: {final_prior_prec:.6f}")
    if final_sigma_noise is not None:
        logger.info(f"Final sigma noise: {final_sigma_noise:.6f}")

    return jax.tree.map(jnp.exp, prior_args)


# ------------------------------------------------------------------------------
# Checkpointing Functions
# ------------------------------------------------------------------------------


def save_model_checkpoint(
    model,
    checkpoint_path: str | Path = "./tests/test-checkpoints",
):
    """Save model checkpoint using Orbax."""
    ckpt_dir = Path(checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Split model into graph and params for checkpointing
    _, state = nnx.split(model)

    # Save the checkpoint
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir.resolve(), state, force=True)
    checkpointer.wait_until_finished()
    logger.info(f"Model checkpoint saved to {ckpt_dir}")
    return ckpt_dir


def load_model_checkpoint(
    model_class,
    model_kwargs,
    checkpoint_path,
):
    """Load model checkpoint using Orbax."""
    model = model_class(**model_kwargs, rngs=nnx.Rngs(0))
    graph_def, abstract_state = nnx.split(model)

    # Restore the checkpoint
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        abstract_state,
    )

    # Merge into model
    model = nnx.merge(graph_def, state_restored)

    logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    return model, graph_def, state_restored


# ------------------------------------------------------------------------------
# Training experiment
# ------------------------------------------------------------------------------

def train_sinusoid_model(
    *,
    # Data settings
    num_train=150,
    num_valid=50,
    num_test=150,
    sigma_noise=0.3,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    rng_seed=0,
    data_seed=42,
    # Model settings
    hidden_channels=64,
    # Training settings
    n_epochs=1000,
    lr=1e-3,
    # Output settings
    checkpoint_dir="./checkpoints/",
):
    """Train a MAP model on sinusoid data and save the checkpoint."""
    # Generate experiment name for the checkpoint
    experiment_name = f"map_model_h{hidden_channels}_n{num_train}_seed{rng_seed}"

    logger.info(f"Starting model training: {experiment_name}")

    # Prepare data
    train_loader, valid_loader, test_loader = build_sinusoid_data(
        num_train, num_valid, num_test,
        sigma_noise, intervals, batch_size, data_seed
    )

    # Initialize MAP model
    rngs = nnx.Rngs(rng_seed)
    model = Model(
        in_channels=1,
        hidden_channels=hidden_channels,
        out_channels=1,
        rngs=rngs,
    )

    # Train the model
    model = train_map_model(model, train_loader, n_epochs=n_epochs, lr=lr)

    # Save checkpoint
    checkpoint_path = save_model_checkpoint(
        model,
        Path(checkpoint_dir) / experiment_name,
    )

    # Return data loaders and checkpoint path for potential immediate use
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "checkpoint_path": checkpoint_path,
        "experiment_name": experiment_name,
    }


# ------------------------------------------------------------------------------
# LAPLACE experiment
# ------------------------------------------------------------------------------


def run_laplace_approximation(
    *,
    # Checkpoint information
    checkpoint_path,
    # Data loaders (optional, will be loaded if not provided)
    train_loader=None,
    valid_loader=None,
    test_loader=None,
    # Data settings (used only if loaders not provided)
    num_train=150,
    num_valid=50,
    num_test=150,
    sigma_noise=0.3,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    rng_seed=0,
    # Laplace settings
    laplace_kwargs=None,
    # Calibration settings (None to skip)
    calibration_kwargs=None,
    # Evaluation settings
    eval_metrics="regression",
    # Output settings
    output_dir="results",
    save_plots=True,
    save_logs=True,
    data_seed=42,
):
    """Run Laplace approximation on a pre-trained model."""
    # Load the model checkpoint
    _, graph_def, params = load_model_checkpoint(
        Model,
        model_kwargs={
            "in_channels": 1,
            "hidden_channels": 64,
            "out_channels": 1,
        },
        checkpoint_path=checkpoint_path,
    )

    # Create model function for Laplace
    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    # Load data if not provided
    if train_loader is None or valid_loader is None or test_loader is None:
        train_loader, valid_loader, test_loader = build_sinusoid_data(
            num_train, num_valid, num_test,
            sigma_noise, intervals, batch_size, data_seed
        )

    # Extract model information from checkpoint path
    checkpoint_parts = Path(checkpoint_path).name.split('_')
    hidden_channels = int(next(p[1:] for p in checkpoint_parts if p.startswith('h')))
    num_train_actual = int(next(p[1:] for p in checkpoint_parts if p.startswith('n')))

    # Generate experiment name
    has_calibration = calibration_kwargs is not None
    calibration_obj = None
    if has_calibration and "calibration_objective" in calibration_kwargs:
        calibration_obj = calibration_kwargs["calibration_objective"]
    curv_type = laplace_kwargs.get("curv_type", "diagonal") \
        if laplace_kwargs is not None else "diagonal"
    laplace_kwargs = {} if laplace_kwargs is None else laplace_kwargs
    experiment_name = generate_experiment_name(
        curv_type,
        hidden_channels,
        num_train_actual,
        calibration=has_calibration,
        calibration_objective=calibration_obj,
    )

    logger.info(f"Starting Laplace approximation: {experiment_name}")

    # Set default laplace kwargs
    default_lap = {
        "loss_fn": "mse",
        "curv_type": laplace_kwargs.get("curv_type", "diagonal"),
        "num_curv_samples": laplace_kwargs.get("num_curv_samples", 150),
        "num_total_samples": laplace_kwargs.get("num_total_samples", 150),
        "key": jax.random.key(rng_seed + 12),
    }
    lap_kwargs = default_lap if laplace_kwargs is None \
        else {**default_lap, **laplace_kwargs}
    logger.debug("Laplace arguments: {}", lap_kwargs)

    # Run Laplace approximation
    posterior_fn, curv_est = laplace.laplace(
        model_fn=model_fn,
        params=params,
        data=train_loader,
        **lap_kwargs,
    )

    results = {
        "curv_estimate": curv_est,
        "curv_type": lap_kwargs["curv_type"],
    }

    # Calibration (optional)
    valid_batch = {"input": valid_loader.X, "target": valid_loader.y}
    if calibration_kwargs is not None:
        prior_args, _ = laplace.calibration(
            posterior_fn=posterior_fn,
            model_fn=model_fn,
            params=params,
            data=valid_batch,
            curv_estimate=curv_est,
            **calibration_kwargs,
        )
        results["prior_arguments"] = prior_args
        # Store pushforward type if available
        if "pushforward_type" in calibration_kwargs:
            results["pushforward_type"] = calibration_kwargs["pushforward_type"]
    else:
        logger.info("No calibration performed. Using default prior precision.")
        prior_args = {"prior_prec": 1.0}

    # Evaluation on test set
    eval_input = {"input": test_loader.X, "target": test_loader.y}
    eval_res, prob_predictive = laplace.evaluation(
        posterior_fn=posterior_fn,
        model_fn=model_fn,
        params=params,
        arguments=prior_args,
        data=eval_input,
        metrics=eval_metrics,
    )
    results["evaluation"] = eval_res

    # Extract NLL if available
    nll = None
    if isinstance(eval_res, dict):
        nll = eval_res.get("nll", eval_res.get("neg_log_likelihood", None))
    results["nll"] = nll

    # Log results
    logger.info(f"Evaluation results: {eval_res}")
    if nll is not None:
        logger.info(f"NLL: {nll:.4f}")

    # Save results to CSV if requested
    if save_logs:
        save_results_to_csv(results, experiment_name, output_dir, log_args={
            "curv_type": lap_kwargs["curv_type"],
            "hidden_channels": hidden_channels,
            "num_train": num_train_actual,
            "num_valid": num_valid,
            "num_test": num_test,
            "sigma_noise": sigma_noise,
            "calibration_objective": calibration_obj,
            "pushforward_type": calibration_kwargs.get("pushforward_type", None),
            "calibration_method": calibration_kwargs.get("calibration_method", None),
        })

    # Generate and save plots if requested
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate predictions with uncertainty
        X_pred = jnp.linspace(0, 8, 200).reshape(200, 1)
        pred = jax.vmap(prob_predictive)(X_pred)

        # Plot predictions with uncertainty
        fig = plot_regression_with_uncertainty_icml(
            X_train=train_loader.X, y_train=train_loader.y,
            X_pred=X_pred,
            y_pred=pred["pred_mean"][:, 0],
            y_std=pred["pred_std"][:, 0],
            title=f"Laplace Approximation ({curv_type})"
        )
        fig.savefig(output_path / f"{experiment_name}_prediction.png")
        plt.close(fig)

        logger.info(f"Plots saved to {output_dir}")

    return results, experiment_name


# Keep the original function for backward compatibility
def run_sinusoid_experiment(
    *,
    # Data settings
    num_train=150,
    num_valid=50,
    num_test=150,
    sigma_noise=0.3,
    intervals=DEFAULT_INTERVALS,
    batch_size=20,
    rng_seed=0,
    # Model settings
    hidden_channels=64,
    # Training settings
    n_epochs=1000,
    lr=1e-3,
    # Laplace settings
    laplace_kwargs=None,
    # Calibration settings (None to skip)
    calibration_kwargs=None,
    # Evaluation settings
    eval_metrics="regression",
    # Output settings
    output_dir="results",
    save_plots=True,
    save_logs=True,
    checkpoint_dir="checkpoints",
):
    """Run the full experiment (training + Laplace) in one go."""
    # Train the model and get checkpoint
    train_result = train_sinusoid_model(
        num_train=num_train,
        num_valid=num_valid,
        num_test=num_test,
        sigma_noise=sigma_noise,
        intervals=intervals,
        batch_size=batch_size,
        rng_seed=rng_seed,
        hidden_channels=hidden_channels,
        n_epochs=n_epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
    )

    # Run Laplace approximation
    return run_laplace_approximation(
        checkpoint_path=train_result["checkpoint_path"],
        train_loader=train_result["train_loader"],
        valid_loader=train_result["valid_loader"],
        test_loader=train_result["test_loader"],
        laplace_kwargs=laplace_kwargs,
        calibration_kwargs=calibration_kwargs,
        eval_metrics=eval_metrics,
        output_dir=output_dir,
        save_plots=save_plots,
        save_logs=save_logs,
    )


if __name__ == "__main__":
    # Register calibration methods
    laplace.register_calibration_method(
        "gradient_descent",
        optimize_prior_prec_gradient,
    )

    # Example usage - split approach
    # 1. Train the model once
    train_result = train_sinusoid_model(
        n_epochs=10,
        hidden_channels=64,
        checkpoint_dir="checkpoints",
    )

    # 2. Run multiple Laplace approximations with different settings
    curv_types = ["diagonal", "full"]
    clbr_methods = ["gradient_descent", "grid_search"]
    clbr_objs = [
        laplace.CalibrationObjective.NLL,
        laplace.CalibrationObjective.MARGINAL_LOG_LIKELIHOOD,
    ]

    for curv_type, clbr_method, clbr_obj in product(curv_types, clbr_methods, clbr_objs):
        logger.info(f"Running Laplace with curvature type: {curv_type}")
        res, exp_name = run_laplace_approximation(
            checkpoint_path=train_result["checkpoint_path"],
            train_loader=train_result["train_loader"],
            valid_loader=train_result["valid_loader"],
            test_loader=train_result["test_loader"],
            laplace_kwargs={"curv_type": curv_type, "num_total_samples": 150},
            calibration_kwargs={
                "loss_fn": "mse",
                "curv_type": curv_type,
                "predictive_type": "none",
                "pushforward_type": "linear",
                "pushforward_fns": [],
                "calibration_objective": clbr_obj,
                "calibration_method": clbr_method,
                "init_prior_prec": 1.0,
                "init_sigma_noise": 1.0,
                "learning_rate": 1e-4
            },
            eval_metrics="regression",
            output_dir="results",
            save_plots=True,
            save_logs=True,
        )
        msg = f"Experiment {exp_name} completed with NLL: {res['nll']:.4f}"
        logger.info(msg)

        with open("results.txt", "a") as f:
            f.write(msg + "\n")