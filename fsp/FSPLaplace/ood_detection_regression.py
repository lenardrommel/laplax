import gc
import jax 
import yaml
import time
import wandb

import numpy as np
import jax.numpy as jnp

from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.dataloader import DataLoader
from data_utils.utils import read_uci_data, standardize_data


DATASET_CONFIG = {
    "boston": {"feature_dim": 13, "patience": 2000},
    "yacht": {"feature_dim": 6, "patience": 1000}, 
    "energy": {"feature_dim": 8, "patience": 1000}, 
    "concrete": {"feature_dim": 8, "patience": 1000},  
    "wine": {"feature_dim": 11, "patience": 1000}, 
    "kin8nm": {"feature_dim": 8, "patience": 1000}, 
    "power": {"feature_dim": 4, "patience": 1000}, 
    "protein": {"feature_dim": 9, "patience": 1000}, 
    "naval": {"feature_dim": 16, "patience": 500}, 
    "wave": {"feature_dim": 48, "patience": 200},
    "denmark": {"feature_dim": 2, "patience": 100},
}


def ood_detection_regression_all(
    config 
):
    """
    Run all OOD detection experiments.

    params:
    - config (dict): configuration dictionary.
    """
    # Load configuaration
    # with open("opt_config.yml", "r") as f:
    #     opt_config = yaml.safe_load(f)

    for dataset_name in DATASET_CONFIG.keys():

        for model in ["FSPLaplace"]: #

            print(f"Dataset: {dataset_name} - Model: {model}", flush=True)

            # if opt_config[dataset_name][model.lower()] is None:
            #     continue

            # Update config
            config["data"]["name"] = dataset_name
            # config = update_config(config, opt_config, model, dataset_name)
            
            # Run experiment
            ood_detection_regression(config)



def ood_detection_regression(
    config 
):
    """
    Follow setup from Uncertainty in gradient boosting via ensembles,
    Malinin et al. (2020)

    params:
    - config (dict): configuration dictionary.
    """
    # Define random key 
    key = jax.random.PRNGKey(0)

    # Get current time
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load configuaration
    dataset_name = config["data"]["name"]
    model_name = config["model"]["name"].lower()
    k_folds = config["data"]["k_folds"]
    batch_size = config["data"]["batch_size"]

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."

    # Update config
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    
    # Load UCI data
    X, y = read_uci_data(dataset_name)

    # K-fold cross-validation
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    test_losses, val_losses, ood_scores, ood_auroc_scores = [], [], [], []
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

        # Load model
        print("Loading model...", flush=True)
        model = Model(key4, config)

        # Fit model 
        print("Fitting the model...", flush=True)
        val_loss = model.fit(train_loader, val_loader, train_loader)
        
        # Evaluate 
        print("Evaluating the model...", flush=True)
        test_loss = model.evaluate(test_loader)

        # OOD detection
        print("OOD detection...", flush=True)
        ood_score, ood_auroc_score = run_ood_detection_regression(model, test_loader, key5, config)

        # Log losses 
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        ood_scores.append(ood_score)
        ood_auroc_scores.append(ood_auroc_score)

        # Close wandb
        wandb.finish()

        # Garbage collection
        del model, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
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

    print(f"OOD detection accuracy: {np.mean(ood_scores)} +/- {np.std(ood_scores)}", flush=True)
    print(f"OOD detection ood_auroc_score: {np.mean(ood_auroc_scores)} +/- {np.std(ood_auroc_scores)}", flush=True)


def run_ood_detection_regression(
    model, 
    test_dataloader, 
    key, 
    config
):
    """
    Run OOD detection.

    params:
    - model (Model): model.
    - test_dataloader (DataLoader): test dataloader.
    - key (jax.random.PRNGKey): random key.
    - config (dict): configuration dictionary.

    returns:
    - mean_acc (float): mean accuracy.
    """
    # For MFVI use 100 MC samples to estimate predictive variance
    mc_samples = 100

    # Load configuration
    batch_size = 100 # config["data"]["batch_size"]

    # Load OOD data of same size as test dataset
    key, key1 = jax.random.split(key)
    X_ood, y_ood = read_uci_data("song")   
    n_samples = len(test_dataloader.dataset)
    feature_dim = test_dataloader.dataset.X.shape[-1]
    X_ood = X_ood[:n_samples,:feature_dim]
    y_ood = y_ood[:n_samples,:]
    # Standardize OOD data
    X_ood = StandardScaler().fit_transform(X_ood) 
    X_ood = X_ood[~np.isnan(X_ood).any(axis=1)] # drop rows with NaNs
    ood_dataset = Dataset(X_ood, y_ood)
    ood_dataloader = DataLoader(key1, ood_dataset, batch_size, shuffle=True, replacement=False)

    # Change batch size to avoid OOD
    ood_dataloader.batch_size = 100
    test_dataloader.batch_size = 100

    # Build OOD detection dataset
    X, y = [], []
    
    # Compute predictive variance for each in-distribution sample
    for x, _ in test_dataloader:
        key, key1 = jax.random.split(key)
        f_var = model.f_distribution_mean_var(x, key1, mc_samples)[1] # (batch_size, 1)
        X += f_var.tolist()
        y += [0] * x.shape[0]   # label is 0 for in-distribution

    # Compute predictive variance for each out-of-distribution sample
    for x, _ in ood_dataloader:
        key, key1 = jax.random.split(key)
        f_var = model.f_distribution_mean_var(x, key1, mc_samples)[1] 
        X += f_var.tolist()
        y += [1] * x.shape[0]   # label is 1 for out-of-distribution

    # Format data
    X, y = jnp.array(X), jnp.array(y)
    X, y = X.reshape(-1, 1), y.reshape(-1)

    # Shuffle data
    idx = jax.random.permutation(key, X.shape[0])
    X, y = X[idx], y[idx]

    # OOD detection
    clf = DecisionTreeClassifier(
        criterion="log_loss", 
        max_leaf_nodes=2, 
        max_depth=1,
        random_state=0
    )
    clf = clf.fit(X, y)

    # Evaluate
    y_hat = clf.predict(X)
    y_proba = clf.predict_proba(X)
    mean_acc = jnp.mean(y_hat == y)
    auroc_score = roc_auc_score(y, y_proba[:,1])

    print(f"OOD detection accuracy: {mean_acc}", flush=True)
    print(f"OOD detection auroc_score: {auroc_score}", flush=True)
    print(f"Predictive variance classification threshold: {clf.tree_.threshold[0]}", flush=True)

    return mean_acc, auroc_score


def update_config(
    config, 
    opt_config, 
    model_name,
    dataset_name
):
    """
    """
    # Update config
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]

    if model_name == "GFSVI":
        config["model"]["name"] = "GFSVI"
        config["gfsvi"]["training"]["lr"] = opt_config[dataset_name]["gfsvi"]["lr"]
        config["gfsvi"]["prior"]["kernel"] = opt_config[dataset_name]["gfsvi"]["kernel"]
        config["gfsvi"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["gfsvi"]["activation_fn"]
        config["gfsvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["gfsvi"]["neural_net"]["validation_freq"] = 100
        config["gfsvi"]["likelihood"]["model"] = "Gaussian"
        if dataset_name == "wave" or dataset_name == "denmark":
            config["gfsvi"]["prior"]["nb_epochs"] = 500
    elif model_name == "GFSVI_layerwise":
        config["model"]["name"] = "GFSVI_layerwise"
        config["gfsvi_layerwise"]["training"]["lr"] = opt_config[dataset_name]["gfsvi"]["lr"]
        config["gfsvi_layerwise"]["prior"]["kernel"] = opt_config[dataset_name]["gfsvi"]["kernel"]
        config["gfsvi_layerwise"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["gfsvi"]["activation_fn"]
        config["gfsvi_layerwise"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["gfsvi_layerwise"]["neural_net"]["validation_freq"] = 100
        config["gfsvi_layerwise"]["likelihood"]["model"] = "Gaussian"
        if dataset_name == "wave" or dataset_name == "denmark":
            config["gfsvi_layerwise"]["prior"]["nb_epochs"] = 500
    elif model_name == "GP":
        config["model"]["name"] = "GP"
        config["gp"]["posterior"]["type"] = "GP"
        config["gp"]["training"]["nb_epochs"] = opt_config[dataset_name]["gp"]["nb_epochs"]
        config["gp"]["training"]["lr"] = opt_config[dataset_name]["gp"]["lr"]
        config["gp"]["prior"]["kernel"] = opt_config[dataset_name]["gp"]["kernel"]
        config["gp"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    elif model_name == "GP_sparse":
        config["model"]["name"] = "GP"
        config["gp"]["posterior"]["type"] = "SVGP"
        config["gp"]["training"]["nb_epochs"] = opt_config[dataset_name]["gp_sparse"]["nb_epochs"]
        config["gp"]["training"]["lr"] = opt_config[dataset_name]["gp_sparse"]["lr"]
        config["gp"]["prior"]["kernel"] = opt_config[dataset_name]["gp_sparse"]["kernel"]
        config["gp"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        if dataset_name == "wave":
            config["gp"]["n_inducing_pts"] = 80
            config["gp"]["training"]["nb_epochs"] = 20000
        if dataset_name == "denmark":
            config["gp"]["n_inducing_pts"] = 100
            config["gp"]["training"]["nb_epochs"] = 20000
    elif model_name == "Laplace_diag":
        config["model"]["name"] = "Laplace"
        config["laplace"]["neural_net"]["cov_type"] = "diag"
        config["laplace"]["training"]["lr"] = opt_config[dataset_name]["laplace_diag"]["lr"]
        config["laplace"]["prior"]["scale"] =  opt_config[dataset_name]["laplace_diag"]["prior_scale"]
        config["laplace"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["laplace_diag"]["activation_fn"]
        config["laplace"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["laplace"]["neural_net"]["validation_freq"] = 100
        config["laplace"]["likelihood"]["model"] = "Gaussian"
    elif model_name == "MFVI":
        config["model"]["name"] = "MFVI"
        config["mfvi"]["training"]["lr"] = opt_config[dataset_name]["mfvi"]["lr"]
        config["mfvi"]["prior"]["scale"] =  opt_config[dataset_name]["mfvi"]["prior_scale"]
        config["mfvi"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["mfvi"]["activation_fn"]
        config["mfvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["mfvi"]["neural_net"]["validation_freq"] = 100
        config["mfvi"]["likelihood"]["model"] = "Gaussian"
    elif model_name == "TFSVI":
        config["model"]["name"] = "TFSVI"
        config["tfsvi"]["training"]["lr"] = opt_config[dataset_name]["tfsvi"]["lr"]
        config["tfsvi"]["prior"]["scale"] =  opt_config[dataset_name]["tfsvi"]["prior_scale"]
        config["tfsvi"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["tfsvi"]["activation_fn"]
        config["tfsvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["tfsvi"]["neural_net"]["validation_freq"] = 100
        config["tfsvi"]["likelihood"]["model"] = "Gaussian"
    elif model_name == "FVI":
        config["model"]["name"] = "FVI"
        config["fvi"]["training"]["lr"] = opt_config[dataset_name]["fvi"]["lr"]
        config["fvi"]["prior"]["kernel"] = opt_config[dataset_name]["fvi"]["kernel"]
        config["fvi"]["neural_net"]["activation_fn"] = opt_config[dataset_name]["fvi"]["activation_fn"]
        config["fvi"]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
        config["fvi"]["neural_net"]["validation_freq"] = 100
        config["fvi"]["likelihood"]["model"] = "Gaussian"
        if dataset_name == "wave" or dataset_name == "denmark":
            config["fvi"]["prior"]["nb_epochs"] = 1000
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
                group=f'ood-detection-regression_{date}', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)