import jax 
import time
import wandb

import numpy as np

from sklearn.model_selection import train_test_split 

from models.model import Model
from data_utils.dataset import Dataset
from data_utils.utils import read_toy_data
from data_utils.dataloader import DataLoader


TOY_DATASETS = [
    "truncated_sine",
    "GP_RBF",
    "GP_Polynomial",
    "GP_Matern12",
    "GP_Matern32",
    "GP_Matern52",
    "two_moons"
]

def toy_data_experiments(
    config 
):
    """
    Run toy regression.

    params:
    - config (dict): configuration dictionary.
    """
    # Define random key 
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, num=4)

    # Load configuaration
    dataset_name = config["data"]["name"]
    assert dataset_name in TOY_DATASETS, f"Dataset {dataset_name} not found."

    # Initialize wandb
    init_wandb(config)
    
    # Load data
    print("Loading data...", flush=True)
    X, y = read_toy_data(config)
    print(X.min(), X.max(), y.min(), y.max(), X.shape, flush=True)

    # Split data
    dataset_size = X.shape[0] // 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dataset_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=dataset_size, random_state=42
    )
    
    # Build datasets
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    # Build dataloaders
    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(
        key1, 
        train_dataset, 
        batch_size, 
        shuffle=True, 
        replacement=False
    ) 
    val_loader = DataLoader(
        key2, 
        val_dataset, 
        batch_size, 
        shuffle=True, 
        replacement=False
    )
    test_loader = DataLoader(
        key3, 
        test_dataset, 
        batch_size, 
        shuffle=True, 
        replacement=False
    ) 

    # Load model
    print("Loading model...", flush=True)
    model = Model(key4, config)

    # Fit model 
    print("Fitting the model...", flush=True)
    start_time = time.time()
    model.fit(train_loader, val_loader, train_loader)
    print(f"Training time: {time.time() - start_time}", flush=True)

    # Evaluate 
    print("Evaluating the model...", flush=True)
    model.evaluate(test_loader)

    # Plot 
    print("Plot functions...", flush=True)
    model.plot(train_loader)

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
                group=f'toy_data', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)