import ot
import jax 
import time
import yaml
import wandb

import numpy as np

from datetime import datetime
from sklearn.model_selection import KFold, train_test_split

from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.dataloader import DataLoader
from data_utils.utils import read_uci_data, standardize_data


DATASET_CONFIG = {
    "boston": {"feature_dim": 13, "patience": 2000},
    "concrete": {"feature_dim": 8, "patience": 2000},  
    "energy": {"feature_dim": 8, "patience": 2000}, 
    "wine": {"feature_dim": 11, "patience": 2000}, 
    "yacht": {"feature_dim": 6, "patience": 2000}
}


def posterior_comparaison_all(
    config 
):
    """
    Run all OOD detection experiments.

    params:
    - config (dict): configuration dictionary.
    """
    # Load configuaration
    with open("opt_config.yml", "r") as f:
        opt_config = yaml.safe_load(f)

    for dataset_name in DATASET_CONFIG.keys():

        for model in [""]: # "GFSVI", "GP_sparse", "FVI"
            # Run experiment
            config["data"]["name"] = dataset_name
            config["model"]["name"] = model
            posterior_comparaison(config, opt_config)



def posterior_comparaison(
    config, 
    opt_config,
):
    """
    params:
    - config (dict): configuration dictionary.
    """
    # Define random key 
    key = jax.random.PRNGKey(0)

    # Get current time
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load configuaration
    dataset_name = config["data"]["name"]
    model_name = config["model"]["name"]
    k_folds = config["data"]["k_folds"]
    batch_size = config["data"]["batch_size"]

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."
    print(f"Dataset: {dataset_name} - Model: {model_name}", flush=True)
    
    # Update config
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    if model_name == "GP_sparse":
        config["gp"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    else:
        config[model_name.lower()]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    
    # Load UCI data
    X, y = read_uci_data(dataset_name)

    # K-fold cross-validation
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    val_losses, w2_means = [], []
    for fold, (train_idx, test_idx) in enumerate(splits.split(X)):

        # Split keys
        key, key1, key2, key3, key4, key5 = jax.random.split(key, num=6)
        
        print(f"Fold: {fold} / {k_folds}", flush=True)

        # Initialize wandb
        init_wandb(config, date, fold)
        
        # Get train and validation indices
        print("Loading data...", flush=True)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

        # Pre-process data
        X_train, X_val, X_test = standardize_data(X[train_idx,:], X[val_idx,:], X[test_idx,:])
        y_train, y_val, y_test = standardize_data(y[train_idx,:], y[val_idx,:], y[test_idx,:])

        # Build datasets
        train_dataset = Dataset(X_train, y_train)
        val_dataset = Dataset(X_val, y_val)
        test_dataset = Dataset(X_test, y_test)
    
        # Build dataloaders
        train_loader = DataLoader(key1, train_dataset, batch_size, shuffle=True, replacement=False)
        val_loader = DataLoader(key2, val_dataset, batch_size, shuffle=True, replacement=False)
        test_loader = DataLoader(key3, test_dataset, batch_size, shuffle=True, replacement=False)

        # Fit model 
        print(f"Fitting {model_name} model...", flush=True)
        config["model"]["name"] = model_name
        config = update_config(config, opt_config, model_name, dataset_name)
        model = Model(key4, config)
        val_loss = model.fit(train_loader, val_loader)
        
        # Fit GP
        print("Fitting GP...", flush=True)
        config["model"]["name"] = "GP"
        config = update_config(config, opt_config, "GP", dataset_name)
        gp_model = Model(key5, config)
        gp_model.fit(train_loader, val_loader, train_loader)

        # Posterior comparaison
        print("Posterior comparaison...", flush=True)
        w2_mean = run_posterior_comparaison(model, gp_model, test_loader, config)
        print(f"W2 model and gp posterior mean: {w2_mean}", flush=True)

        # Log losses 
        val_losses.append(val_loss)
        w2_means.append(w2_mean)

        # Close wandb
        wandb.finish()

    # Print results
    for k in val_losses[0].keys():
        # Get losses
        val_loss = [val_losses[i][k] for i in range(k_folds)]
        # Print mean and std-div of results
        print(f"{k} - val: {np.mean(val_loss)} +/- {np.std(val_loss)}", flush=True)

    print(f"W2 means accuracy: {np.mean(w2_means)} +/- {np.std(w2_means)}", flush=True)


def run_posterior_comparaison(
    model, 
    gp_model,
    test_dataloader, 
    config
):
    """
    Run OOD detection.

    params:
    - model (Model): model.
    - test_dataloader (DataLoader): test dataloader.
    - config (dict): configuration dictionary.

    returns:
    - mean_acc (float): mean accuracy.
    """
    key = jax.random.PRNGKey(0)

    mc_samples = 1000
    X = test_dataloader.dataset.X.reshape(-1, 1, config["data"]["feature_dim"])
    w2 = 0
    for x in X:
        key, key1, key2 = jax.random.split(key, num=3)
        gp_preds = gp_model.predict_f(x, key1, mc_samples).reshape(-1)
        model_preds = model.predict_f(x, key2, mc_samples).reshape(-1)
        loss = ot.emd2_1d(np.array(gp_preds), np.array(model_preds), metric='sqeuclidean')
        w2 += loss
    w2 /= X.shape[0]      

    return w2


def update_config(
    config, 
    opt_config, 
    model_name,
    dataset_name
):
    """
    All models have the same Kernel as the GP model used as reference.
    """
    # Update config
    config["data"]["name"] = dataset_name
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    
    if model_name == "GFSVI":
        config["model"]["name"] = "GFSVI"
        config["gfsvi"]["training"]["lr"] = 0.0005 #0.001 # opt_config[dataset_name]["gfsvi"]["lr"]
        config["gfsvi"]["prior"]["kernel"] = "RBF" #opt_config[dataset_name]["gp"]["kernel"]
        config["gfsvi"]["neural_net"]["activation_fn"] = "tanh" # opt_config[dataset_name]["gfsvi"]["activation_fn"]
        config["gfsvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    elif model_name == "GP":
        config["model"]["name"] = "GP"
        config["gp"]["posterior"]["type"] = "GP"
        config["gp"]["training"]["nb_epochs"] = opt_config[dataset_name]["gp"]["nb_epochs"]
        config["gp"]["training"]["lr"] = opt_config[dataset_name]["gp"]["lr"]
        config["gp"]["prior"]["kernel"] = "RBF" # opt_config[dataset_name]["gp"]["kernel"]
        config["gp"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    elif model_name == "GP_sparse":
        config["model"]["name"] = "GP"
        config["gp"]["posterior"]["type"] = "SVGP"
        config["gp"]["training"]["nb_epochs"] = opt_config[dataset_name]["gp_sparse"]["nb_epochs"]
        config["gp"]["training"]["lr"] = opt_config[dataset_name]["gp_sparse"]["lr"]
        config["gp"]["prior"]["kernel"] = "RBF" # opt_config[dataset_name]["gp"]["kernel"]
        config["gp"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["gp"]["posterior"]["n_inducing_pts"] = 100
    elif model_name == "FVI":
        config["model"]["name"] = "FVI"
        config["fvi"]["training"]["lr"] = 0.0005  #0.001 #opt_config[dataset_name]["fvi"]["lr"]
        config["fvi"]["prior"]["kernel"] = "RBF" # opt_config[dataset_name]["gp"]["kernel"]
        config["fvi"]["neural_net"]["activation_fn"] = "tanh" # opt_config[dataset_name]["fvi"]["activation_fn"]
        config["fvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    else:
        raise Exception("Unknown model")
    
    return config

    
def init_wandb(
    config, 
    date, 
    fold
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
                name=f'{config["model"]["name"]}_{config["data"]["name"]}_fold_{fold}', 
                group=f'gp_comparison_{date}', 
                mode="disabled"
            )
            wandb_init = True
        except:
            time.sleep(10)