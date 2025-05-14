import datetime
from itertools import product
from os import readlink
from pathlib import Path

import jax
import jax.numpy as jnp
from laplax.eval import pushforward
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from flax import nnx
from helper import DataLoader, get_sinusoid_example
from loguru import logger
from orbax import checkpoint as ocp
from tueplots import bundles, fonts

from laplax.laplace import (
    laplace, 
    CalibrationObjective,
    PushforwardType, 
    PredictiveType,
    calibration,
    evaluation,
    lin_setup,
    lin_pred_mean,
    lin_pred_std,
    DEFAULT_REGRESSION_METRICS,
    register_calibration_method
)
from laplax.eval.pushforward import lin_samples
from laplax.enums import CurvApprox, LossFn
from laplax.types import Callable, Float, PriorArguments


from ex_helper import ( # isort: skip
    generate_experiment_name,
    load_model_checkpoint, 
    save_model_checkpoint, 
    train_map_model, 
    split_model,
    CSVLogger,
    optimize_prior_prec_gradient
)

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
        self.final_layer = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        return self.final_layer(h)


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
    model_seed=0,
    data_seed=42,
    # Model settings
    hidden_channels=64,
    # Training settings
    n_epochs=1000,
    lr=1e-3,
    # Output settings
    ckpt_dir="./checkpoints/",
):
    """Train a MAP model on sinusoid data and save the checkpoint."""
    # Set model
    model = Model(
        in_channels=1,
        hidden_channels=hidden_channels,
        out_channels=1,
        rngs=nnx.Rngs(model_seed)
    )

    # Set data
    train_loader, valid_loader, test_loader = build_sinusoid_data(
        num_train, num_valid, num_test,
        sigma_noise, intervals, batch_size, data_seed
    )
        
    # Generate experiment name for the checkpoint
    model_name = f"regression_model_h{hidden_channels}_n{num_train}_seed{model_seed}"
    logger.info(f"Starting model training: {model_name}")

    # Train the model
    model = train_map_model(model, train_loader, n_epochs=n_epochs, lr=lr)

    # Save checkpoint
    checkpoint_path = save_model_checkpoint(
        model, Path(ckpt_dir) / model_name,
    )

    # Return data loaders and checkpoint path for potential immediate use
    return checkpoint_path


#------------------------------------------------------------------------------
# LAPLACE experiment
# ------------------------------------------------------------------------------


def evaluate_regression_example(
    ckpt_dir: str = "./checkpoints/",
    model_name: str = "regression_model_h64_n150_seed0",
    *,
    # Laplace settings
    laplace_kwargs,
    # Pushforward
    pushforward_kwargs,
    # Calibration
    clbr_kwargs=None,
    # Evaluation
    eval_kwargs=None,
    # Output settings
    output_dir="results",
    save_logs=True,
    save_samples=True,
    csv_logger: CSVLogger | None = None
):
    """Evaluate regression example."""
    # Load map model
    ckpt_path = Path(ckpt_dir) / model_name
    model, _, _ = load_model_checkpoint(
        Model,
        model_kwargs={
            "in_channels": 1, 
            "hidden_channels": 64, 
            "out_channels": 1
        },
        checkpoint_path=ckpt_path    
    )

    # Load data
    train_loader, valid_loader, test_loader = build_sinusoid_data()

    # Start evaluation
    results = {}
    csv_logger = CSVLogger(force=True) if csv_logger is None else csv_logger
    
    # Extract parameters
    last_layer_only=laplace_kwargs.get('last_layer_only', False)
    curv_type = laplace_kwargs.get('curv_type')
    low_rank_rank = laplace_kwargs.get('low_rank_rank', 100)
    low_rank_seed = laplace_kwargs.get('low_rank_seed', 1950)
    sample_seed = pushforward_kwargs.get("sample_seed", 21904)
    pushforward_type = pushforward_kwargs.get('pushforward_type', PushforwardType.LINEAR)
    clbr_obj = clbr_kwargs.get("calibration_objective")
    clbr_mthd = clbr_kwargs.get("calibration_method")
    
    experiment_name = generate_experiment_name(
        ct=curv_type,
        ll=last_layer_only,
        co=clbr_obj,
        cm=clbr_mthd,
        pt=pushforward_type
    )
    model_fn, params = split_model(
        model, 
        last_layer_only=last_layer_only
    )

    # Approximate curvature
    posterior_fn, curv_est = laplace(
        model_fn=model_fn,
        params=params,
        data=train_loader, 
        loss_fn=LossFn.MSE,
        curv_type=curv_type,
        rank=low_rank_rank,
        key=jax.random.key(low_rank_seed),
        has_batch=True
    )

    # Calibration
    prior_args = {"prior_args": 1.0}
    if clbr_kwargs is not None:
        prior_args, _ = calibration(
            posterior_fn=posterior_fn,
            model_fn=model_fn,
            params=params,
            data={"input": valid_loader.X, "target": valid_loader.y},
            curv_estimate=curv_est,
            curv_type=curv_type,
            loss_fn=LossFn.MSE,
            # Pushforward
            predictive_type=PredictiveType.NONE,
            pushforward_type=pushforward_type,
        )
        
    # Evaluation
    results, _ = evaluation(
        posterior_fn=posterior_fn,
        model_fn=model_fn,
        params=params,
        arguments=prior_args,
        data={"input": test_loader.X, "target": test_loader.y},
        metrics=DEFAULT_REGRESSION_METRICS,
        predictive_type=PredictiveType.NONE,
        pushforward_type=pushforward_type,
        pushforward_fns=[
            lin_setup,
            lin_pred_mean,
            lin_pred_std,
            lin_samples,
        ] if pushforward_type is PushforwardType.LINEAR else None,
        sample_key=jax.random.key(sample_seed),
        num_samples=20,
    )

    logger.info(f"Eval: {results}")
    csv_logger.log(
        results, 
        experiment_name=experiment_name, 
        log_args={
            "curv_type": curv_type,
            "last_layer_only": last_layer_only,
            "pushforward_type": pushforward_type,
            "calibration_objective": clbr_obj,
            "calibration_method": clbr_mthd,  
        }
    
    )    
    csv_logger.log_samples(
        results,
        experiment_name=experiment_name
    )


if __name__ == "__main__":
    # Register calibration methods
    register_calibration_method(
        "gradient_descent",
        optimize_prior_prec_gradient,
    )

    # Train model
    train_result = train_sinusoid_model(n_epochs=1000)

    # Start evaluation
    csv_logger = CSVLogger(force=True)

    curv_types = [CurvApprox.DIAGONAL, CurvApprox.FULL, CurvApprox.LANCZOS]
    no_last_layer = [False]
    clbr_methods = ["gradient_descent", "grid_search"]
    clbr_objs = [
        CalibrationObjective.NLL,
        CalibrationObjective.MARGINAL_LOG_LIKELIHOOD,
    ]
    push_methods = [PushforwardType.LINEAR, PushforwardType.NONLINEAR]

    settings = list(
        product(curv_types, clbr_methods, clbr_objs, no_last_layer, push_methods)
    ) + list(
        product(["full"], clbr_methods, clbr_objs, [True], push_methods)
    )

    for curv_type, clbr_method, clbr_obj, last_layer_only, push_mthd in settings[:]:
        logger.info(f"Running Laplace with curvature type: {curv_type}")

        evaluate_regression_example(
            laplace_kwargs={
                "curv_type": curv_type,
                "last_layer_only": last_layer_only,
            },
            pushforward_kwargs={
                "pushforward_type": push_mthd,
            },
            clbr_kwargs={
                "": None,
                "calibration_objective": clbr_obj,
                "calibration_method": clbr_method,
            },
            csv_logger=csv_logger
        )                
                        
