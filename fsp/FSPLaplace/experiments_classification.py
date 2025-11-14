import gc 
import jax 
import yaml
import time
import wandb
import pickle    

import numpy as np
import scipy as sp
import jax.numpy as jnp
import jax.scipy as jsp
import scipy.ndimage as nd

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.model import Model
from data_utils.dataset import Dataset 
from data_utils.utils import read_image_data, standardize_image_data
from data_utils.dataloader import DataLoader


DATASET_CONFIG = {
    "mnist": {"feature_dim": 784, "patience": 20},
    "fashion_mnist": {"feature_dim": 784, "patience": 5},  # 20
    "cifar10": {"feature_dim": 1024, "patience": 20}, 
}


def experiments_classification(
    config 
):
    """
    params:
    - config (dict): configuration dictionary.
    """
    # Load configuaration
    cv_iterations = config["data"]["k_folds"]
    dataset_name = config["data"]["name"].lower()
    model_name = config["model"]["name"].lower()
    batch_size = config["data"]["batch_size"]

    assert dataset_name in DATASET_CONFIG.keys(), f"Dataset {dataset_name} not found."
    print(f"Dataset: {dataset_name} - Model: {model_name}", flush=True)

    # Update config
    if model_name != "gp":
        config[model_name]["neural_net"]["validation_freq"] = 1
        if dataset_name in ["mnist", "fashion_mnist"]:
            config[model_name]["neural_net"]["type"] = "CNN1"
        elif dataset_name in ["svhn", "cifar10"]:
            config[model_name]["neural_net"]["type"] = "CNN2"
    else:
        config["gp"]["training"]["validation_freq"] = 1
    config[model_name]["likelihood"]["model"] = "Categorical"
    config[model_name]["likelihood"]["n_classes"] = 10
    config["data"]["feature_dim"] = DATASET_CONFIG[dataset_name]["feature_dim"]
    config[model_name]["training"]["patience"] = DATASET_CONFIG[dataset_name]["patience"]

    # Load image data
    X_train, X_test, y_train, y_test = read_image_data(dataset_name)

    test_losses, val_losses, ood_scores, ood_auroc_scores = [], [], [], []
    for it in range(cv_iterations):
        # Define random key 
        key = jax.random.PRNGKey(it)
        key, key1, key2, key3, key4, key5, key6 = jax.random.split(key, num=7)
        
        print(f"Iter: {it+1} / {cv_iterations}", flush=True)

        # Initialize wandb
        init_wandb(config, it)
        
        # Get train and validation indices
        print("Loading data...", flush=True)
        _X_train, _X_val, _y_train, _y_val = train_test_split(
            X_train, 
            y_train, 
            test_size=0.1, 
            random_state=42
        )
        _X_test, _y_test = jnp.copy(X_test), jnp.copy(y_test)

        # Standardize data
        _X_train, _X_val, _X_test = standardize_image_data(_X_train, _X_val, _X_test)

        # Build datasets
        train_dataset = Dataset(_X_train, _y_train)
        val_dataset = Dataset(_X_val, _y_val)
        test_dataset = Dataset(_X_test, _y_test)
    
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

        # Garbage collection
        del train_dataset, val_dataset, train_loader, val_loader, _X_train, _X_val, _y_train, _y_val
        gc.collect()

        # OOD detection
        print("OOD detection...", flush=True)
        ood_score, ood_auroc_score = run_ood_detection_classification(model, test_loader, key5, config, it)

        # Rotated MNIST
        if dataset_name in ["mnist", "fashion_mnist"]:
            print(f"Rotated {dataset_name}...", flush=True)
            rotating_image_classification(model, test_loader, key6, config, it)

        # Log losses 
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        ood_scores.append(ood_score)
        ood_auroc_scores.append(ood_auroc_score)

        # Close wandb
        wandb.finish()

        # Garbage collection
        del model, test_dataset, test_loader, _X_test, _y_test
        gc.collect()
        jax.clear_caches()

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

    print(f"OOD detection accuracy: {np.mean(ood_scores)} +/- {np.std(ood_scores)}", flush=True)
    print(f"OOD detection auroc_score: {np.mean(ood_auroc_scores)} +/- {np.std(ood_auroc_scores)}", flush=True)



def run_ood_detection_classification(
    model, 
    test_dataloader, 
    key, 
    config, 
    fold 
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
    mc_samples = 10

    # Load configuration
    dataset_name = config["data"]["name"].lower()
    batch_size = config["data"]["batch_size"]
    
    # Load OOD data of same size as test dataset
    key, key1 = jax.random.split(key)
    if dataset_name == "mnist":
        X_ood, _, y_ood, _ = read_image_data("fashion_mnist")   
    elif dataset_name == "fashion_mnist":
        X_ood, _, y_ood, _ = read_image_data("mnist")
    elif dataset_name == "cifar10":
        test_svhn = sp.io.loadmat("../Data/svhn/test_32x32.mat")
        X_ood = np.transpose(test_svhn['X'], [3, 0, 1, 2])
        y_ood = test_svhn['y'].reshape(-1, 1)
        y_ood[np.where(y_ood==10)] = 0
        X_ood = jnp.float64(X_ood) / 255
    else:
        raise Exception("Unknown dataset")
    n_samples = len(test_dataloader.dataset)
    X_ood = X_ood[:n_samples,:,:,:]
    y_ood = y_ood[:n_samples,:]

    gc.collect()

    # Standardize data
    mean = np.mean(X_ood, axis=(0,), keepdims=True)
    std = np.std(X_ood, axis=(0,), keepdims=True) + 1e-10
    X_ood = (X_ood - mean) / std
    
    # Build dataloader
    most_uncertain_finetuned = Dataset(X_ood, y_ood)
    ood_dataloader = DataLoader(key1, most_uncertain_finetuned, batch_size, shuffle=True, replacement=False)

    # Build OOD detection dataset
    X, y = [], []

    # Compute predictive variance for each out-of-distribution sample
    for x, _ in ood_dataloader:
        key, key1 = jax.random.split(key)
        probs = model.predict_y(x, key1, mc_samples).mean(0) # (batch_size, n_classes)
        entropy = entropy_categorical(probs).sum(-1)#jsp.special.entr(probs).sum(-1)
        X += entropy.tolist()
        y += [1] * x.shape[0]   # label is 1 for out-of-distribution

    del X_ood, y_ood, ood_dataloader, most_uncertain_finetuned
    
    # Compute predictive variance for each in-distribution sample
    for x, _ in test_dataloader:
        key, key1 = jax.random.split(key)
        probs = model.predict_y(x, key1, mc_samples).mean(0) # (batch_size, n_classes)
        entropy = entropy_categorical(probs).sum(-1) # jsp.special.entr(probs).sum(-1) # (batch_size,)
        X += entropy.tolist()
        y += [0] * x.shape[0]   # label is 0 for in-distribution

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

    # Save data
    Path(f"ood_dataset").mkdir(parents=True, exist_ok=True)
    save_path = f"ood_dataset/{dataset_name}_{config['model']['name']}_{fold}.npz"
    np.savez(save_path, X=X, y=y)

    # Garbage collection
    del X, y, clf
    gc.collect()

    return mean_acc, auroc_score


def entropy_categorical(
    probs
):
    """
    Compute entropy of a categorical distribution.
    """
    x_logx_fn = lambda x: x * jnp.where(x > 0, jnp.log(x), 0)
    entropy_fn = lambda x: jnp.where(x < 0, -jnp.inf, -x_logx_fn(x))
    return entropy_fn(probs)


def rotating_image_classification(
    model, 
    test_dataloader, 
    key, 
    config, 
    fold
):
    """
    https://github.com/dougbrion/pytorch-classification-uncertainty
    From paper http://arxiv.org/abs/1806.01768
    """
    mc_samples = 10
    max_rotation = 180 
    degree_incr = int(max_rotation/20)+1
    dataset_name = config["data"]["name"].lower()

    results = {}
    for degree in np.linspace(0, max_rotation, degree_incr):
        accuracy, entropy, expected_ll = 0., 0., 0.
        for x, y in test_dataloader: 
            key, key1 = jax.random.split(key)
            # Rotate image
            x = nd.rotate(x, degree, axes=(1,2), reshape=False)
            # Predict
            f = model.predict_f(x, key1, mc_samples) # (mc_samples, batch_size, n_classes)
            probs = jax.nn.softmax(f, axis=-1)
            # Compute expected log-likelihood
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=probs.shape[-1])
            expected_ll += jnp.mean(
                jnp.sum(
                    one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_samples, n_batch, n_classes)
                    axis=-1
                ), # (n_samples, n_batch)
                axis=0
            ).sum()
            # Compute entropy
            entropy += entropy_categorical(probs.mean(0)).sum() # jsp.special.entr(probs.mean(0)).sum()
            # Compute accuracy
            accuracy += jnp.sum(jnp.argmax(probs.mean(0), axis=-1) == y.reshape(-1))
        accuracy /= len(test_dataloader.dataset)
        entropy /= len(test_dataloader.dataset)
        expected_ll /= len(test_dataloader.dataset)
        print(f"Degree: {degree} - Accuracy: {accuracy} - Entropy: {entropy} - Expected LL: {expected_ll}", flush=True)
        results[degree] = {"entropy":entropy, "accuracy":accuracy, "expected_ll":expected_ll}

    Path(f"rotated_{dataset_name}_final").mkdir(parents=True, exist_ok=True)
    save_path = f"rotated_{dataset_name}/{config['model']['name']}_{fold}.pkl"
    with open(save_path,'wb') as file:
        pickle.dump(results, file)

    gc.collect()


def init_wandb(
    config, 
    it
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
                name=f'{config["model"]["name"]}_{config["data"]["name"]}_it_{it}', 
                group=f'ood_{config["model"]["name"]}_{config["data"]["name"]}',
            )
            wandb_init = True
        except Exception as e:
            print(e, flush=True)
            time.sleep(30)