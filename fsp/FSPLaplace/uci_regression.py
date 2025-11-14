import gc
import jax 
import time
import wandb 

import numpy as np 

from datetime import datetime
from sklearn.model_selection import KFold, train_test_split


from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.dataloader import DataLoader
from data_utils.utils import read_uci_data, standardize_data


DATASET_CONFIG = {
    "boston": {"feature_dim": 13, "patience": 1000},
    "concrete": {"feature_dim": 8, "patience": 1000},  
    "energy": {"feature_dim": 8, "patience": 1000}, 
    "kin8nm": {"feature_dim": 8, "patience": 1000}, 
    "naval": {"feature_dim": 16, "patience": 500}, 
    "power": {"feature_dim": 4, "patience": 1000}, 
    "protein": {"feature_dim": 9, "patience": 1000}, 
    "wine": {"feature_dim": 11, "patience": 1000}, 
    "yacht": {"feature_dim": 6, "patience": 1000}, 
    "wave": {"feature_dim": 48, "patience": 100},
    "denmark": {"feature_dim": 2, "patience": 100},
}
def uci_regression(
    config
):
    """
    """
    # Define random key 
    key = jax.random.PRNGKey(0)

    # Load configuaration
    dataset_name = config["data"]["name"]
    k_folds = config["data"]["k_folds"]
    batch_size = config["data"]["batch_size"]
    model_name = config["model"]["name"].lower()

    # Get current time
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."
            
    # Update config
    if model_name != "gp":
        config[model_name]["neural_net"]["validation_freq"] = 100
    config[model_name]["likelihood"]["model"] = "Gaussian"
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    
    # Load UCI data
    X, y = read_uci_data(dataset_name)

    # K-fold cross-validation
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    test_losses, val_losses = [], []
    for fold, (train_idx, test_idx) in enumerate(splits.split(X)):
        # Split keys
        key, key1, key2, key3, key4 = jax.random.split(key, num=5)
        
        print(f"Fold: {fold} / {k_folds}", flush=True)

        # Initialize wandb
        init_wandb(config, date)

        # Get train and validation indices
        print("Loading data...", flush=True)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42, shuffle=True)

        # Pre-process data
        X_train, X_val, X_test = standardize_data(X[train_idx,:], X[val_idx,:], X[test_idx,:])
        y_train, y_val, y_test = standardize_data(y[train_idx,:], y[val_idx,:], y[test_idx,:])

        # Build datasets
        train_dataset = Dataset(X_train, y_train)
        val_dataset = Dataset(X_val, y_val)
        test_dataset = Dataset(X_test, y_test)
    
        # Build dataloaders
        train_loader = DataLoader(key1, train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(key2, val_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(key3, test_dataset, batch_size, shuffle=True)

        # Load model
        print("Loading model...", flush=True)
        model = Model(key4, config)

        # Fit model 
        print("Fitting the model...", flush=True)
        val_loss = model.fit(train_loader, val_loader, train_loader)
        
        # Evaluate 
        print("Evaluating the model...", flush=True)
        test_loss = model.evaluate(test_loader)

        # Log losses 
        test_losses.append(test_loss)
        val_losses.append(val_loss)

        # Close wandb
        wandb.finish()
        
        del model, train_loader, val_loader, test_loader
        gc.collect()

    # Print results
    for k in val_losses[0].keys():
        # Get losses
        val_loss = [val_losses[i][k] for i in range(k_folds)]
        # Print mean and std-div of results
        print(f"{k} - val: {np.mean(val_loss)} +/- {np.std(val_loss)}", flush=True)
    
    # Print results
    for k in test_losses[0].keys():
        # Get losses
        test_loss = [test_losses[i][k] for i in range(k_folds)]
        # Print mean and std-div of results
        print(f"{k} - test: {np.mean(test_loss)} +/- {np.std(test_loss)}", flush=True)          


def init_wandb(
    config, 
    date
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            name = f'uci_{config["model"]["name"]}_{config["data"]["name"]}'
            wandb.init(
                project="flaplace",
                config=config, 
                name=name, 
                group=f'uci_{date}',
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)