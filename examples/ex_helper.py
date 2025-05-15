import datetime
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax import nnx
from loguru import logger
from orbax import checkpoint as ocp

from laplax.types import Callable, Float, PriorArguments

# ---------------------------------------------------------------------
# Checkpoint Helper
# ---------------------------------------------------------------------

def save_with_pickle(obj, path):
    """Save object to pickle file."""
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_with_pickle(path):
    """Load object from pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


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


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class CSVLogger:
    def __init__(
        self, output_dir="results", file_name="regression_experiments.csv", *, force: bool=True
    ):
        """A CSV logger for experiments.

        Args:
            output_dir (str or Path): Directory in which to store the CSV.
            file_name (str): Name of the CSV file.
            force (bool): If True, delete any existing CSV at initialization. 
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / file_name

        if force and self.csv_path.exists():
            self.csv_path.unlink()
            logger.info(f"Existing log {self.csv_path} removed (force=True)")

        # Track whether to write headers on next write
        self._write_header = not self.csv_path.exists()

    def log(self, results: dict, experiment_name: str, *, log_args: dict = None) -> Path:
        """Append a single experiment's results to the CSV.

        Args:
            results (dict): A dict that may contain an "evaluation" sub-dict and optional "nll" field.
            experiment_name (str): A unique name or identifier for this run.
            log_args (dict, optional): Any additional metadata to record.

        Returns:
            Path: The path to the CSV file.
        """
        log_args = {} if log_args is None else dict(log_args)

        # Build the row data
        row = {
            **results,
            **log_args,
            "experiment_name": experiment_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }

        df = pd.DataFrame([row])
        df.to_csv(
            self.csv_path,
            mode="a",
            header=self._write_header,
            index=False,
        )

        if self._write_header:
            self._write_header = False

        logger.info(f"Logged experiment '{experiment_name}' to {self.csv_path}")
        return self.csv_path

    def log_samples(self, results: dict, experiment_name: str):
        """Storing samples."""
        sample_path = self.output_dir / ("samples_" + experiment_name)
        save_with_pickle(results, path=sample_path)


def generate_experiment_name(**kwargs):
    """Generate a descriptive name for the experiment."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    name_parts = [f"{k}={v}" for k, v in kwargs.items()]
    return f"{timestamp}_{'_'.join(name_parts)}"

# ---------------------------------------------------------------------
# DataLoader Helper
# ---------------------------------------------------------------------


class LimitedLoader:
    """DataLoader wrapper that limits the number of batches."""

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        batch_iter = iter(self.loader)
        for _ in range(self.max_batches):
            yield next(batch_iter)

    def __len__(self):
        return self.max_batches


# ---------------------------------------------------------------------
# Last-layer-only helper
# ---------------------------------------------------------------------


def split_model(model, *, last_layer_only=False):
    """Split model into graph and params."""
    if last_layer_only:
        graph_def, relevant_params, remaining_params = nnx.split(
            model, lambda n, _: "final_layer" in n, ...
        )

        def model_fn(input, params):
            return nnx.call((graph_def, params, remaining_params))(input)[0]

        return model_fn, relevant_params

    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    return model_fn, params


# ---------------------------------------------------------------------
# Train Helper
# ---------------------------------------------------------------------


def train_map_model(
    model,
    train_loader,
    n_epochs,
    *,
    lr=1e-3,
    verbose=True,
    log_every_n_epochs=10,
    loss_type="mse",
    test_loader=None,
):
    optimizer = nnx.Optimizer(model, optax.adamw(lr))
    loss = 0.0

    if loss_type == "mse":

        def loss_fn(y_pred, y):
            return jnp.sum((y_pred - y) ** 2)

    elif loss_type == "cross_entropy":

        def loss_fn(y_pred, y):
            return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()

    else:
        msg = f"Unknown loss type: {loss_type}"
        raise ValueError(msg)

    @nnx.jit
    def train_step(model, optimizer, x, y):
        def forward(model):
            y_pred = nnx.vmap(model)(x)
            return loss_fn(y_pred, y)

        loss, grads = nnx.value_and_grad(forward)(model)
        optimizer.update(grads)  # Inplace updates

        return loss

    @nnx.jit
    def eval_step(model, x, y):
        y_pred = nnx.vmap(model)(x)
        pred_class = jnp.argmax(y_pred, axis=1)
        return jnp.mean(pred_class == y)

    for epoch in range(1, n_epochs + 1):
        for xb, yb in train_loader:
            loss = train_step(model, optimizer, xb, yb)
        if verbose and epoch % log_every_n_epochs == 0:
            logger.info(f"Epoch {epoch}/{n_epochs}, loss={loss:.4f}")
    if verbose:
        logger.info(f"Final training loss: {loss:.4f}")

    if loss_type == "cross_entropy" and test_loader is not None:
        total_acc = 0.0
        n_batches = 0
        for xb, yb in test_loader:
            acc = eval_step(model, xb, yb)
            total_acc += acc
            n_batches += 1
        avg_acc = total_acc / n_batches
        logger.info(f"Test accuracy: {avg_acc:.4f}")

    return model


def optimize_prior_prec_gradient(
    objective: Callable[[PriorArguments], float],
    init_prior_prec: Float | None = None,
    init_sigma_noise: Float | None = None,
    *,
    num_epochs: int = 20,
    learning_rate: float = 1e-2,
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

    # Validate inputs
    if init_prior_prec is None and init_sigma_noise is None:
        msg = "Provide at least one of init_prior_prec or init_sigma_noise."
        raise ValueError(msg)

    # Initialize log-parameters
    params = {}
    if init_prior_prec is not None:
        params["prior_prec"] = jnp.array(jnp.log(init_prior_prec))
    if init_sigma_noise is not None:
        params["sigma_noise"] = jnp.array(jnp.log(init_sigma_noise))

    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Single optimization step
    @jax.jit
    def step(params, opt_state):
        # Compute loss and gradients w.r.t. log-params
        loss, grads = jax.value_and_grad(
            lambda p: objective(jax.tree.map(jnp.exp, p))
            )(params)

        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    # Optimization loop
    for epoch in range(1, num_epochs + 1):
        params, opt_state, loss = step(params, opt_state)
        logger.debug(
            f"Epoch {epoch:02d}: loss={loss:.6f}, "
        )

    # Convert back from log-domain
    params = jax.tree.map(jnp.exp, params)
      
    return params
