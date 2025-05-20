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
