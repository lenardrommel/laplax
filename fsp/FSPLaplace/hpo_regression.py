import gc
import sys
import jax 
import time
import wandb 
import traceback

import numpy as np 

from sklearn.model_selection import KFold, train_test_split


from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.dataloader import DataLoader
from data_utils.utils import read_uci_data, standardize_data

#################### TMP ####################
from ood_detection_regression import run_ood_detection_regression
###########################################

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

METRIC_MAP = {
    "FSPLaplace": "expected_ll",
    "Laplace": "expected_ll",
    "SNGP": "expected_ll",
    "GP": "expected_ll",
    "GWI": "expected_ll",
    "SamplingLaplace": "expected_ll", 
    "FVI": "expected_ll"
}
    
SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize', 
        'name': 'mean_val_metric'
    },
    'parameters': {}
}    

FLAPLACE_PARAMS = {
    'kernel': {
        "distribution": "categorical", 
        "values": ["RBF", "Matern12", "Matern32", "Matern52", "RationalQuadratic"],
    },
    'lr': {
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-3
    },
    'activation_fn': {
        "distribution": "categorical", 
        "values": ["relu", "tanh", "lrelu"],
    },
    "cov_context_selection": {
        "distribution": "categorical",
        "values": ["latin_hypercube", "train_val_latin", "halton", "train_val_halton", "val_latin", "val_halton"] # "sobol", 
    },
    'n_context_points_training': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 250
    },
} 


FVI_PARAMS = {
    'kernel': {
        "distribution": "categorical", 
        "values": ["RBF", "Matern12", "Matern32", "Matern52", "RationalQuadratic"],
    },
    'lr': {
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-3
    },
    'activation_fn': {
        "distribution": "categorical", 
        "values": ["relu", "tanh", "lrelu"],
    },
    'n_context_points_training': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 250
    },
} 

LAPLACE_PARAMS = {
    'lr': {
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-3
    },
    'activation_fn': {
        "distribution": "categorical", 
        "values": ["relu", "tanh", "lrelu"],
    }
}

GP_PARAMS = {
    'kernel': {
        "distribution": "categorical", 
        "values": ["RBF", "Matern12", "Matern32", "Matern52", "RationalQuadratic"],
    },
    'lr': {
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-1
    }
} 

GWI_PARAMS = {
    'kernel': {
        "distribution": "categorical", 
        "values": ["RBF", "Matern12", "Matern32", "Matern52", "RationalQuadratic"],
    },
    'lr': {
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-3
    },
    'activation_fn': {
        "distribution": "categorical", 
        "values": ["relu", "tanh", "lrelu"],
    }
} 

def hpo_regression(
    config
):
    """
    """
    # Get configuaration
    model_name = config["model"]["name"].lower()
    dataset_name = config["data"]["name"]
    sweep_id = config["experiment"]["sweep_id"]

    # Update config
    if model_name != "gp":
        config[model_name]["neural_net"]["validation_freq"] = 100
    config[model_name]["likelihood"]["model"] = "Gaussian"
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
   
    # Update sweep config
    if model_name == "fsplaplace":
        SWEEP_CONFIG["parameters"] = FLAPLACE_PARAMS
    elif model_name == "laplace":
        SWEEP_CONFIG["parameters"] = LAPLACE_PARAMS
    elif model_name == "gp":
        SWEEP_CONFIG["parameters"] = GP_PARAMS
    elif model_name == "gwi":
        SWEEP_CONFIG["parameters"] = GWI_PARAMS
    elif model_name == "fvi":
        SWEEP_CONFIG["parameters"] = FVI_PARAMS
    else:
        raise Exception("Unknown model")
    
    SWEEP_CONFIG['name'] = f"{dataset_name}_{model_name}"

    if model_name == "laplace":
        SWEEP_CONFIG['name'] += config["laplace"]["inference"]["cov_type"]
    elif model_name == "gp":
        SWEEP_CONFIG['name'] += config["gp"]["posterior"]["type"]

    # Initialize sweep by passing in config
    wandb_init = False
    while not wandb_init:
        try:
            if sweep_id == "":
                sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="flaplace")
            # Start sweep job
            wandb.agent(sweep_id, function=lambda: k_folds(config), count=30, project="flaplace")
            wandb_init = True
        except Exception as e:
            print(e, flush=True)
            time.sleep(30)

    
def k_folds(
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

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."

    # Initialize wandb
    init_wandb(config)

    # Update config
    config = update_config(config, wandb.config)
    
    # Load UCI data
    X, y = read_uci_data(dataset_name)

    # K-fold cross-validation
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    ################## TMP ##################
    ood_scores, ood_auroc_scores = [], []
    ###################################
    test_losses, val_losses = [], []
    for fold, (train_idx, test_idx) in enumerate(splits.split(X)):
        # Split keys
        key, key1, key2, key3, key4 = jax.random.split(key, num=5)
        
        print(f"Fold: {fold} / {k_folds}", flush=True)

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
        train_loader = DataLoader(key1, train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(key2, val_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(key3, test_dataset, batch_size, shuffle=True)

        # Load model
        print("Loading model...", flush=True)
        model = Model(key4, config)

        try:
            # Fit model 
            print("Fitting the model...", flush=True)
            val_loss = model.fit(train_loader, val_loader, train_loader)
            
            # Evaluate 
            print("Evaluating the model...", flush=True)
            test_loss = model.evaluate(test_loader)

            #########################    TMP #############################
            print("OOD detection...", flush=True)
            key, key1 = jax.random.split(key)
            ood_score, ood_auroc_score = run_ood_detection_regression(model, test_loader, key1, config)
            ood_scores.append(ood_score)
            ood_auroc_scores.append(ood_auroc_score)
            ###############################################################)

            # Log losses 
            test_losses.append(test_loss)
            val_losses.append(val_loss)
        except Exception:
            print(traceback.print_exc(), file=sys.stderr)

        del model
        gc.collect()
        jax.clear_caches()

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

    ###########################   TMP #############################
    print(f"OOD detection accuracy: {np.mean(ood_scores)} +/- {np.std(ood_scores)}", flush=True)
    print(f"OOD detection ood_auroc_scores: {np.mean(ood_auroc_scores)} +/- {np.std(ood_auroc_scores)}", flush=True)
    wandb.log({"mean_ood_score": np.mean(ood_scores)})
    wandb.log({"mean_ood_auroc_score": np.mean(ood_auroc_scores)})
    ###############################################################
    
    metric_name = METRIC_MAP[config["model"]["name"]]
    mean_val_metric = np.mean([val_losses[i][metric_name] for i in range(k_folds)])
    wandb.log({"mean_val_metric": mean_val_metric})
    print("Mean validation metric: ", mean_val_metric, flush=True)
    


def update_config(
    config, 
    wandb_config
):
    """
    """
    # Get configuaration
    model_name = config["model"]["name"].lower()
    dataset_name = config["data"]["name"]

    # Update config
    config[model_name]["training"]["lr"] = wandb_config.lr
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    
    if model_name == "fsplaplace":
        print(f"FSPLaplace - kernel:{wandb_config.kernel} - activation:{wandb_config.activation_fn} - context:{wandb_config.cov_context_selection}", flush=True)
        config["fsplaplace"]["prior"]["kernel"] = wandb_config.kernel
        config["fsplaplace"]["neural_net"]["activation_fn"] = wandb_config.activation_fn
        config["fsplaplace"]["inference"]["cov_context_selection"] = wandb_config.cov_context_selection
        config["fsplaplace"]["training"]["n_context_points"] = wandb_config.n_context_points_training
    elif model_name == "fvi":
        print(f"FVI - kernel:{wandb_config.kernel} - activation:{wandb_config.activation_fn}", flush=True)
        config["fvi"]["prior"]["kernel"] = wandb_config.kernel
        config["fvi"]["neural_net"]["activation_fn"] = wandb_config.activation_fn
        config["fvi"]["training"]["n_context_points"] = wandb_config.n_context_points_training
    elif model_name == "gwi":
        print(f"GWI - kernel:{wandb_config.kernel} - activation:{wandb_config.activation_fn}", flush=True)
        config["gwi"]["prior"]["kernel"] = wandb_config.kernel
        config["gwi"]["neural_net"]["activation_fn"] = wandb_config.activation_fn
    elif model_name == "gp":
        print(f"GP - kernel:{wandb_config.kernel}", flush=True)
        config["gp"]["prior"]["kernel"] = wandb_config.kernel
        if dataset_name == "wave" or dataset_name == "denmark":
            config["gp"]["posterior"]["n_inducing_pts"] = 100
            config["gp"]["training"]["nb_epochs"] = 20000
    elif model_name == "laplace":
        print(f"Laplace - activation:{wandb_config.activation_fn}", flush=True)
        config[model_name]["neural_net"]["activation_fn"] = wandb_config.activation_fn
        if dataset_name in ["naval", "wave", "denmark"]:
            config["laplace"]["training"]["patience"] = 300
    else:
        raise Exception("Unknown model")
    
    return config


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
                config=config, 
                group=f'hpo_{config["data"]["name"]}_{config["model"]["name"]}', 
                project="flaplace"
            )
            wandb_init = True

        except wandb.errors.UsageError:
            time.sleep(5)
        except wandb.errors.CommError:
            time.sleep(5)
        except wandb.errors.WaitTimeoutError:
            time.sleep(5)            
        except Exception as e:
            print(e, flush=True)
            time.sleep(30)