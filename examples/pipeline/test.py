import os
import json
import time
import pickle
from copy import copy
from pathlib import Path

import jax
from jax import numpy as jnp
from flax import nnx
import pandas as pd
import optax
from orbax import checkpoint as ocp
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from nosplace.fno._data import get_ood_data_loader
from nosplace.fno._fno import FNO, FNO_Wrapper
from nosplace.fsp.calibrate import load_hparams, GPrior


class Trainer:
    def __init__(
        self,
        model_hparams: dict[str, any],
        optimizer_hparams: dict[str, any],
        seed: int = 42,
        debug: bool = False,
        save_dir: str = "model/small_model",
        model_id: int = 1,
        logger_type="csv",
    ):
        self.config = {
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "seed": seed,
        }

        self.rng = nnx.Rngs(seed)
        self.debug = debug
        self.model_name = f"model_{model_id}"
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

        self.save_dir = Path(save_dir) / f"s={seed}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = (self.save_dir / "checkpoint").resolve()


class PDETrainer:
    def __init__(
        self,
        prior,
        model_hparams: dict[str, any],
        optimizer_hparams: dict[str, any],
        seed: int = 42,
        debug: bool = True,
        plot_only: bool = False,
        save_dir: str = "model/small_model",
        model_id: int = 1,
        logger_type="csv",
    ):
        self.config = {
            "model_hparams": model_hparams,
            "optimizer_hparams": optimizer_hparams,
            "seed": seed,
        }

        self.warmup_epochs = 5

        self.rng = nnx.Rngs(seed)
        self.prior = prior

        # boolean flags
        self.debug = debug
        self.plot_only = plot_only

        self.model_name = f"model_{model_id}"
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.model = FNO_Wrapper(
            FNO(**model_hparams, rngs=self.rng),
            prior=prior,
            param=jnp.array(-1.4),
            dataset_size=1000,
        )

        self.save_dir = Path(os.path.join("model", "small_model", f"s={seed}"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = (self.save_dir / "checkpoint").resolve()

        # Call the initialization methods
        self.init_checkpointer()
        self._init_logger(logger_type)
        self.metrics_config = self._default_metrics_config()
        self.loss_config = self._default_loss_config()
        required_metrics = self._get_required_metrics()
        self.init_metrics(
            self._filter_metrics_config(self.metrics_config, required_metrics)
        )

        self.print_params()

        if self.plot_only:
            self.model = self.load_model_from_checkpoint(self.checkpoint_path)

    def _init_logger(self, logger_type):
        """Initialize the PyTorch Lightning logger."""
        self.log_dir = self.save_dir / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)

        if logger_type.lower() == "csv":
            self.logger = CSVLogger(
                save_dir=str(self.log_dir),
                name=self.model_name,
            )
        elif logger_type.lower() == "wandb":
            self.logger = WandbLogger(
                save_dir=str(self.log_dir),
                name=self.model_name,
                project="pde_training",
            )
        else:
            raise ValueError(f"Unsupported logger type: {logger_type}")

    def _default_metrics_config(self):
        return {
            "loss": {"type": "Average", "name": "loss", "fun": None},
            "nll": {"type": "Average", "name": "nll", "fun": None},
            "reg": {"type": "Average", "name": "reg", "fun": None},
            # "mse": {"type": "Average", "name": "mse", "fun": None},
            "param": {
                "type": "Average",
                "name": "obs_noise_std",
                "fun": jax.nn.softplus,
            },
        }

    def _default_loss_config(self):
        """Default configuration for loss functions."""
        return {
            "mse": {
                "enabled": True,
                "weight": 1.0,
                "function": self.model.mse,
                "requires_metrics": ["mse"],
            },
            "nll": {
                "enabled": True,
                "weight": 1.0,
                "function": self.model.nll,
                "requires_metrics": ["nll"],
            },
            "reg": {
                "enabled": True,
                "weight": 1.0,
                "function": self.model.reg,
                "requires_metrics": ["reg"],
            },
        }

    def build_loss_function(self, loss_config, is_warmup=False):
        """
        Builds a combined loss function based on configuration.

        Args:
            loss_config: Optional custom loss configuration to override default.

        Returns:
            A function that computes the combined loss and returns individual components.
        """
        loss_config = loss_config or self._default_loss_config()

        enabled_losses = {k: v for k, v in loss_config.items() if v["enabled"]}

        def combined_loss(model, batch, metrics=None):
            total_loss = 0.0
            individual_losses = {}

            for name, config in enabled_losses.items():
                loss_fn = getattr(model, f"{name}")
                base_weight = config.get("weight", 1.0)

                weight = base_weight
                if is_warmup:
                    if name == "reg" and metrics is not None:
                        weight = base_weight * jnp.min(
                            jnp.array([
                                1.0,
                                metrics.reg.count.value / self.warmup_epochs,
                            ])
                        )

                loss_value = loss_fn(batch)
                weighted_loss = weight * loss_value

                total_loss += weighted_loss

                individual_losses[name] = loss_value

            return total_loss, individual_losses

        return combined_loss

    def _get_required_metrics(self):
        """Get set of required metrics based on enabled loss functions."""
        required = set()

        for name, config in self.loss_config.items():
            if config["enabled"]:
                required.update(config.get("requires_metrics", []))

        # Always include param metric
        required.add("param")

        return required

    def _filter_metrics_config(self, metrics_config, required_metrics):
        """Filter metrics configuration to only include required metrics."""
        return {k: v for k, v in metrics_config.items() if k in required_metrics}

    def update_metrics_from_losses(self, metrics, model, loss_dict):
        """
        Updates metrics based on loss dictionary.

        Args:
            metrics: Metrics object to update
            model: Model with parameters
            loss_dict: Dictionary of individual losses
        """
        # Create update dictionary
        update_dict = {}

        # Add all losses from the loss dictionary
        update_dict.update(loss_dict)

        # Calculate total loss if not already in dict
        if "loss" not in update_dict and loss_dict:
            total_loss = sum(loss_dict.values())
            update_dict["loss"] = total_loss

        # Filter to only include metrics we're tracking
        update_dict = {k: v for k, v in update_dict.items() if k in self.metrics_keys}

        update_dict["obs_noise_std"] = model.param.value
        metrics.update(**update_dict)

    def init_metrics(self, metrics_config):
        """
        Initializes metrics dynamically based on the provided configuration.

        Args:
            metrics_config: Dictionary mapping metric keys to their configuration
        """
        metrics_dict = {}
        self.metric_transforms = {}

        for key, config in metrics_config.items():
            metric_type = config.get("type", "Average")
            metric_name = config.get("name", key)
            transform_fun = config.get("fun", None)

            self.metric_transforms[key] = transform_fun

            if metric_type == "Average":
                metrics_dict[key] = nnx.metrics.Average(metric_name)
            elif metric_type == "Sum":
                metrics_dict[key] = nnx.metrics.Sum(metric_name)
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

        self.metrics = nnx.MultiMetric(**metrics_dict)
        self.metrics_keys = set(metrics_dict.keys())

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        """This method initializes the optimizer Adam."""
        hparams = copy(self.config["optimizer_hparams"])

        lr = hparams.pop("lr", 1e-4)
        grad_clip = hparams.pop("gradient_clip", 1.0)
        optimizer_name = hparams.pop("optimizer", "adam")

        # Create learning rate schedule
        total_steps = int(num_epochs * num_steps_per_epoch)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=int(total_steps * 0.1),  # 10% warmup
            decay_steps=total_steps,
            end_value=lr * 0.01,  # End at 1% of peak
        )

        if optimizer_name.lower() == "adam":
            opt = optax.adam(lr_schedule)
        elif optimizer_name.lower() == "sgd":
            opt = optax.sgd(lr_schedule, momentum=hparams.get("momentum", 0.9))
        elif optimizer_name.lower() == "adamw":
            opt = optax.adamw(
                lr_schedule, weight_decay=hparams.get("weight_decay", 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), opt)

        self.optimizer = nnx.Optimizer(self.model, optimizer)

    def _log_metrics(self, train_metrics, eval_metrics, epoch):
        """Log metrics using the Lightning logger with support for dynamic metrics and transformations."""

        metrics = {"epoch": epoch}

        for k, v in train_metrics.items():
            if k == "train/loss":
                metrics["train_loss"] = v
            else:
                metrics[k] = v

        for k, v in eval_metrics.items():
            if k in self.metric_transforms and self.metric_transforms[k] is not None:
                v = self.metric_transforms[k](v)

            if k == "loss":
                metrics["eval_loss"] = v
            else:
                metrics[f"eval_{k}"] = v

        self.logger.log_metrics(metrics, step=epoch)

    def init_checkpointer(self):
        self.checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "metadata")
        )

    def create_functions(self):
        """Create training and evaluation functions dynamically based on loss config."""

        # Build the combined loss function
        loss_fn = self.build_loss_function(self.loss_config)

        def train_step(model, optimizer, batch, metrics):
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (total_loss, individual_losses), grads = grad_fn(model, batch)
            optimizer.update(grads)
            self.update_metrics_from_losses(metrics, model, individual_losses)

        def eval_step(model, batch, metrics):
            total_loss, individual_losses = loss_fn(model, batch)
            self.update_metrics_from_losses(metrics, model, individual_losses)

        if not self.debug:
            train_step = nnx.jit(train_step)
            eval_step = nnx.jit(eval_step)

        return train_step, eval_step

    def create_warmup_functions(self):
        def nll_loss(model, batch):
            return model.compute_loss(batch)

        def mse_loss(model, batch):
            inputs, targets, _ = batch
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets.reshape(*preds.shape)) ** 2)

        def reg_loss(model, batch):
            inputs, targets, grid = batch
            preds = jax.vmap(model)(inputs)  # batch, nx, 1, 1
            if preds.shape[0] != 32:
                return 0.0
            return self.prior.rkhs_loss(
                preds[:, ::10, :, :], inputs[:, ::10, :, :], grid[::10]
            )

        def combined_loss(model, batch, alpha):
            loss_nll = nll_loss(model, batch)
            loss_reg = reg_loss(model, batch)

            return loss_nll + alpha * loss_reg, (loss_nll, alpha * loss_reg)

        def train_step_warmup(model, optimizer, batch, metrics):
            grad_fn = nnx.value_and_grad(combined_loss, has_aux=True)
            alpha = jnp.min(jnp.array([1.0, metrics.reg.count.value / 5]))
            (total_loss, (loss_nll, loss_reg)), grads = grad_fn(model, batch, alpha)
            optimizer.update(grads)
            metrics.update(
                loss=total_loss,
                nll=loss_nll,
                reg=loss_reg,
                obs_noise_std=model.param.value,
            )

        def eval_step_warmup(model, batch, metrics):
            alpha = jnp.min(jnp.array([1.0, metrics.reg.count.value / 5]))
            total_loss, (loss_nll, loss_reg) = combined_loss(model, batch, alpha)
            metrics.update(
                loss=total_loss,
                nll=loss_nll,
                reg=loss_reg,
                obs_noise_std=model.param.value,
            )

        if not self.debug:
            train_step_warmup = nnx.jit(train_step_warmup)
            eval_step_warmup = nnx.jit(eval_step_warmup)

        return train_step_warmup, eval_step_warmup

    def create_functions_partitioned(self):
        def nll_loss(model, batch):
            return model.compute_loss(batch)
            """inputs, targets, _ = batch
            preds = jax.vmap(model)(inputs)

            # Get the standard deviation parameter (after softplus activation)
            sigma = jnp.log(1 + jnp.exp(model.param))

            # Calculate the difference between targets and predictions
            diff = targets - preds

            # Calculate the negative log likelihood using the analytical formula
            # log(P(x|μ,σ)) = -0.5*log(2π) - log(σ) - 0.5*(x-μ)²/σ²
            log_likelihood = (
                -0.5 * jnp.log(2 * jnp.pi)  # Constant term
                - jnp.log(sigma)  # Log of standard deviation
                - 0.5 * (diff**2) / (sigma**2)  # Squared error term
            )"""

            # Mean over all data points and scale by dataset size
            # return -self._dataset_size * jnp.mean(log_likelihood)
            """return (
                -self._dataset_size
                * jsp.stats.norm.logpdf(
                    targets,
                    loc=preds,
                    scale=jax.nn.softplus(model.param),
                ).mean()
            )"""

        def mse_loss(model, batch):
            inputs, targets, _ = batch
            preds = jax.vmap(model)(inputs)
            return jnp.mean((preds - targets.reshape(*preds.shape)) ** 2)

        def reg_loss(model, batch):
            inputs, targets, grid = batch
            preds = jax.vmap(model)(inputs)  # batch, nx, 1, 1
            if preds.shape[0] != 32:
                return 0.0

            # rkhs = self.prior.rkhs_loss(preds, inputs, grid)

            # def rkhs_loss(f_hat, prior, params, x, y, grid):
            #     kernel_op = prior(params, x, y, grid)  # Sigma(C, C)
            #     f_hat = f_hat.reshape(-1)  # f_hat is a vector
            #     _right = lo.lsolve(kernel_op, f_hat)  # Sigma(C, C)^{-1} f_hat
            #     return (
            #         0.5 * f_hat @ _right.T
            #     )  # Sigma(C, C)^{-1} f_hat @ f_hat^T Sigma(C, C)^{-1}

            return self.prior.rkhs_loss(
                preds[:, ::10, :, :], inputs[:, ::10, :, :], grid[::10]
            )

        def combined_loss(model, batch):
            loss_nll = nll_loss(model, batch)
            loss_reg = reg_loss(model, batch)

            return loss_nll + loss_reg, (loss_nll, loss_reg)

        def train_step(model, optimizer, batch, metrics):
            grad_fn = nnx.value_and_grad(combined_loss, has_aux=True)
            (total_loss, (loss_nll, loss_reg)), grads = grad_fn(model, batch)
            optimizer.update(grads)
            metrics.update(
                loss=total_loss,
                nll=loss_nll,
                reg=loss_reg,
                obs_noise_std=model.param.value,
            )

        def eval_step(model, batch, metrics):
            total_loss, (loss_nll, loss_reg) = combined_loss(model, batch)
            metrics.update(
                loss=total_loss,
                nll=loss_nll,
                reg=loss_reg,
                obs_noise_std=model.param.value,
            )

        if not self.debug:
            train_step = nnx.jit(train_step)
            eval_step = nnx.jit(eval_step)

        return train_step, eval_step

    def save_metrics(self, filename: str, metrics: dict[str, any]):
        """
        Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        os.makedirs(os.path.join(self.log_dir, "metrics/"), exist_ok=True)
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def train_epoch(self, epoch, train_loader):
        """
        Trains a model for one epoch.
        Args:
            epoch: The current epoch number.
            train_loader: Data loader of the training set.

        Returns:
            A dictionary of the average training metrics over all batches for logging.
        """

        start_time = time.time()
        self.metrics.reset()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            self.train_step(self.model, self.optimizer, batch, self.metrics)

        metrics = {"train/" + k: v.item() for k, v in self.metrics.compute().items()}
        metrics["epoch_time"] = time.time() - start_time

        return metrics

    def warmup_epoch(self, warmup_epoch, train_loader):
        start_time = time.time()
        self.metrics.reset()
        for batch in tqdm(
            train_loader, desc=f"Epoch {warmup_epoch} Training", leave=False
        ):
            self.train_step_warmup(self.model, self.optimizer, batch, self.metrics)

    def eval_model(self, data_loader):
        self.metrics.reset()

        num_elements = 0

        for batch in data_loader:
            self.eval_step(self.model, batch, self.metrics)

            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            num_elements += batch_size
        metrics = self.metrics.compute()
        metrics = {k: v.item() for k, v in metrics.items()}

        return metrics

    def warmup_eval(self, data_loader):
        self.metrics.reset()

        num_elements = 0

        for batch in data_loader:
            self.eval_step_warmup(self.model, batch, self.metrics)

            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            num_elements += batch_size
        metrics = self.metrics.compute()
        metrics = {k: v.item() for k, v in metrics.items()}

        return metrics

    def train_model(self, train_loader, test_loader, num_epochs, num_train_steps):
        self._dataset_size = len(train_loader.dataset)
        self.num_train_steps = num_train_steps
        self.init_optimizer(num_epochs, num_train_steps)

        # self.train_step, self.eval_step = self.create_warmup_functions()

        # for i in range(self.warmup_epochs):
        #     train_metrics = self.train_epoch(i, train_loader)
        #     eval_metrics = self.eval_model(test_loader)
        #     self._log_metrics(train_metrics, eval_metrics, i)
        #     self.save_metrics(f"epoch_{i}", train_metrics)
        #     print(
        #         f"Warm up iteration {i}: "
        #         f"total_loss={eval_metrics['loss']:.6f}, "
        #         f"nll={eval_metrics['nll']:.6f}, "
        #         f"reg={eval_metrics['reg']:.6f}"
        #     )

        self.train_step, self.eval_step = self.create_functions()

        for epoch in range(self.warmup_epochs, num_epochs + self.warmup_epochs + 1):
            train_metrics = self.train_epoch(epoch, train_loader)
            eval_metrics = self.eval_model(test_loader)
            self._log_metrics(train_metrics, eval_metrics, epoch)
            self.save_metrics(f"epoch_{epoch}", train_metrics)
            print(
                f"Epoch {epoch}: "
                f"total_loss={eval_metrics['loss']:.6f}, "
                f"nll={eval_metrics['nll']:.6f}, "
                f"reg={eval_metrics['reg']:.6f}"
            )

        self.save_model()
        self.logger.finalize("success")

    def print_params(self):
        """
        Print the number of parameters in the model.
        """
        _, state = nnx.split(self.model)
        print(f"Parameter count: {sum(p.size for p in jax.tree.leaves(state))}")

    def _plot_results(self, test_loader):
        """
        Plot the training results using metrics from the logger.
        """
        if self.plot_only:
            print("Plotting results...")
        # For CSV logger, load the metrics from the CSV file
        if isinstance(self.logger, CSVLogger):
            metrics_path = Path(self.logger.log_dir) / "metrics.csv"

            if metrics_path.exists():
                df = pd.read_csv(metrics_path)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

                ax1.plot(df["epoch"], df["train_loss"], label="Training Loss")
                ax1.plot(df["epoch"], df["eval_loss"], label="Evaluation Loss")
                ax1.plot(df["epoch"], df["reg"], label="Regularization Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_title("Total Loss")
                ax1.legend(loc="upper left")
                ax1.legend()
                ax1.grid(True)

                ax1_r = ax1.twinx()
                ax1_r.plot(
                    df["epoch"],
                    df["obs_noise_std"],
                    label="Observation Noise STD",
                    color="tab:green",
                    linestyle="--",
                )
                ax1_r.set_ylabel("Noise STD")
                ax1_r.legend(loc="lower right")

                # ax2.plot(df["epoch"], df["train_loss"], label="Training Loss")

                u0, uT, grid = next(iter(test_loader))
                idx = jax.random.randint(jax.random.key(1), (1,), 0, u0.shape[0])[0]
                ax3.plot(grid, uT[idx].reshape(-1), label="Target")
                ax3.plot(grid, self.model(u0[idx]).reshape(-1), label="Prediction")
                ax3.set_xlabel("x")
                ax3.set_ylabel("u(x)")
                ax3.set_title("Prediction vs Target")
                ax3.legend()

                plt.tight_layout()
                plt.show()

                figures_dir = self.save_dir / "figures"
                figures_dir.mkdir(exist_ok=True, parents=True)
                fig.savefig(figures_dir / "training_results.png", dpi=300)

                plt.show()

        elif isinstance(self.logger, WandbLogger):
            u0, uT, grid = next(iter(test_loader))
            idx = jax.random.randint(jax.random.key(1), (1,), 0, u0.shape[0])[0]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grid, uT[idx].reshape(-1), label="Target")
            ax.plot(grid, self.model(u0[idx]).reshape(-1), label="Prediction")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x)")
            ax.set_title("Prediction vs Target")
            ax.legend()

            plt.tight_layout()

            figures_dir = self.save_dir / "figures"
            figures_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(figures_dir / "prediction_target.png", dpi=300)

            plt.show()

            self.logger.experiment.log({"prediction_plot": fig})

    def save_model(self, step: int = 0, checkpoint_path=""):
        """
        Saves current training state at certain training iteration.

        Args:
            step: Index of the step to save the model at, e.g. epoch.
            checkpoint_path: Optional custom path for checkpoint. If None, uses default path.
        """
        save_path = checkpoint_path or self.checkpoint_path
        self.checkpointer.save(
            save_path.resolve(),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(nnx.state(self.model)),
                metadata=ocp.args.JsonSave(self.config),
            ),
            force=True,
        )

        if checkpoint_path:
            model_card_path = Path(checkpoint_path).parent / "params"
            model_card_path.mkdir(parents=True, exist_ok=True)
            with open(model_card_path / "model_card.pkl", "wb") as f:
                pickle.dump(
                    {
                        "model_hparams": self.model_hparams,
                        "optimizer_hparams": self.optimizer_hparams,
                        "seed": self.seed,
                        "step": step,
                    },
                    f,
                )

    def load_checkpoint(self):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        restored = self.checkpointer.restore(self.checkpoint_path)
        return restored

    def load_model_from_checkpoint(self, checkpoint_path) -> any:
        print(f"Loading model from checkpoint: {self.checkpoint_path}")
        graph_def, state = nnx.split(self.model)

        checkpointer = ocp.StandardCheckpointer()
        restored_state = checkpointer.restore(
            (checkpoint_path / "state").resolve(), target=state
        )
        print("Model loaded successfully")
        return nnx.merge(graph_def, restored_state)


def run_training():
    """This function runs the training loop."""

    train_loader = get_ood_data_loader(folder_path="nosplace/data/big_wave_data.npz")
    test_loader = get_ood_data_loader(
        folder_path="nosplace/data/big_wave_data_test.npz"
    )

    num_train_steps = len(train_loader)

    model_hparams = {
        "modes": 8,
        "width": 8,
        "num_layers": 2,
        "dims": 1,
        "in_channels": 1,
        "out_channels": 1,
    }

    optimizer_hparams = {
        "lr": 1e-3,
        "optimizer": "adam",
    }

    prior_hparams = {
        "spatial_kernel": {"lengthscale": -3.178490994605825},
        "function_kernel": {"bias": -8.0},
        "output_scale": {"lengthscale": -5.480685896558868},
        "sigma": {"lengthscale": -1.0085474424827798},
    }

    hps = load_hparams("trained_osp_hparams.json")
    prior = GPrior(hps)

    trainer = PDETrainer(
        prior=prior, model_hparams=model_hparams, optimizer_hparams=optimizer_hparams
    )

    trainer.train_model(
        train_loader, test_loader, num_epochs=20, num_train_steps=num_train_steps
    )

    checkpoint_path = Path("model/small_model/s=42/checkpoint")
    trainer.load_model_from_checkpoint(checkpoint_path)
    trainer._plot_results(test_loader)


if __name__ == "__main__":
    run_training()


def create_functions(self):
    def mse_loss(model, batch):
        inputs, targets = batch
        preds = jax.vmap(model)(inputs)
        return jnp.mean((preds - targets.reshape(*preds.shape)) ** 2)

    def train_step(model, optimizer, batch, metrics):
        # Compute loss and gradients
        grad_fn = nnx.value_and_grad(mse_loss)
        loss, grads = grad_fn(model, batch)
        optimizer.update(grads)
        metrics.update(loss=loss)

    def eval_step(model, batch, metrics):
        loss = mse_loss(model, batch)
        metrics.update(loss=loss)

    return train_step, eval_step
