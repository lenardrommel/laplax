import jax 
import time
import wandb 

import numpy as np 

from datetime import datetime
from sklearn.model_selection import train_test_split


from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.utils import read_image_data, standardize_image_data
from data_utils.dataloader import DataLoader


DATASET_CONFIG = {
    "mnist": {"feature_dim": 784, "patience": 100},
    "fashion_mnist": {"feature_dim": 784, "patience": 100},  
    "cifar10": {"feature_dim": 1024, "patience": 100}, 
    "svhn": {"feature_dim": 1024, "patience": 100}
}

def image_classification(
    config
):
    """
    """
    # Load configuaration
    dataset_name = config["data"]["name"]
    cv_iterations = config["data"]["k_folds"]
    batch_size = config["data"]["batch_size"]

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."

    # Update config
    model_name = config["model"]["name"].lower()
    config[model_name]["likelihood"]["n_classes"] = 10
    config[model_name]["neural_net"]["validation_freq"] = 10
    config[model_name]["likelihood"]["model"] = "Categorical"
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    
    # Initialize wandb
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    init_wandb(config, date)

    # Load image data
    X_train, X_test, y_train, y_test = read_image_data(dataset_name)

    test_losses, val_losses = [], []
    for iter in range(cv_iterations):
        # Define random key 
        key = jax.random.PRNGKey(iter)
        key, key1, key2, key3, key4 = jax.random.split(key, num=5)
        
        print(f"Iter: {iter} / {cv_iterations}", flush=True)

        # Get train and validation indices
        print("Loading data...", flush=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, 
            y_train, 
            test_size=0.1, 
            random_state=42
        )

        # Standardize data
        X_train, X_val, X_test = standardize_image_data(X_train, X_val, X_test)
        
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

    # Print results
    for k in val_losses[0].keys():
        # Get losses
        val_loss = [val_losses[i][k] for i in range(cv_iterations)]
        # Print mean and std-div of results
        print(f"{k} - val: {np.mean(val_loss)} +/- {np.std(val_loss)}", flush=True)
    
    # Print results
    for k in test_losses[0].keys():
        # Get losses
        test_loss = [test_losses[i][k] for i in range(cv_iterations)]
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
            name = f'image_{config["model"]["name"]}_{config["data"]["name"]}'
            wandb.init(
                project="flaplace",
                config=config, 
                name=name, 
                group=f'image_{date}', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)