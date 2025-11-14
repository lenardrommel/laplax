import jax 
import time
import wandb

import numpy as np
import jax.scipy as jsp

from pathlib import Path

from models.model import Model
from data_utils.dataset import Dataset
from data_utils.utils import read_image_data
from data_utils.dataloader import DataLoader


def predictive_prior_check(
    config 
):
    """
    Run toy regression.

    params:
    - config (dict): configuration dictionary.
    """
    # Define random key 
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, num=3)

    # Load configuaration
    mc_samples = 100
    dataset_name = config["data"]["name"]
    batch_size = config["data"]["batch_size"]
    model_name = config["model"]["name"].lower()
    
    # Update config
    config[model_name]["likelihood"]["n_classes"] = 10
    config[model_name]["neural_net"]["validation_freq"] = 10
    config[model_name]["likelihood"]["model"] = "Categorical"

    # Initialize wandb
    init_wandb(config)

    # Load data
    print("Loading data...", flush=True)
    X_train, _, y_train, _ = read_image_data(dataset_name)

    dataset = Dataset(X_train, y_train)
    dataloader = DataLoader(
        key1, 
        dataset, 
        batch_size, 
        shuffle=True, 
        replacement=False
    )

    if model_name in ["fsplaplace", "flaplace_sampling"]:
        # Set model configuration   
        x_min = float(dataset.X.min())
        x_max = float(dataset.X.max())
        config[model_name]["training"]["min_context_val"] = x_min - 0.5 * (x_max - x_min)
        config[model_name]["training"]["max_context_val"] = x_max + 0.5 * (x_max - x_min)

    # Load model
    print("Loading model...", flush=True)
    model = Model(key2, config)

    # Fit model 
    print("Fitting the model...", flush=True)
    model.fit_to_prior(X_train[0].shape) 

    # Predictive prior check
    print("Predictive prior check...", flush=True)
    predictive_entropy = []
    mean_softmax = []
    for i, (x, y) in enumerate(dataloader):
        if i % 50 == 0:
            print(f"Batch {i} / {len(dataloader)}", flush=True)
        key, key1 = jax.random.split(key3)
        probs = model.predict_y(x, key1, mc_samples).mean(0) # (batch_size, n_classes)
        entropy = jsp.special.entr(probs).sum(-1)
        predictive_entropy += entropy.tolist()
        mean_softmax += probs.tolist()

    print(f"Predictive entropy: {np.mean(predictive_entropy)} +/- {np.std(predictive_entropy)}" , flush=True)
        
    # Log predictive entropy
    Path(f"predictive_prior_check").mkdir(parents=True, exist_ok=True)
    np.savez(
        f'predictive_prior_check/{config["model"]["name"]}_{config["data"]["name"]}_predictive_prior.npz',
        predictive_entropy=predictive_entropy, 
        mean_softmax=mean_softmax
    )

    # Close wandb
    wandb.finish()


def init_wandb(
    config
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            wandb.init(
                project="flaplace",
                config=config, 
                name=f'{config["model"]["name"]}_{config["data"]["name"]}', 
                group=f'predictive_prior_check', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)