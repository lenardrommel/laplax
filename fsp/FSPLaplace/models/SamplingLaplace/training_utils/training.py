import os
import jax 
import wandb
import pickle

import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp

from pathlib import Path
from functools import partial
from jax.example_libraries.optimizers import adam 

from models.SamplingLaplace.training_utils.objective import (
    n_gaussian_log_posterior_objective, 
    n_categorical_log_posterior_objective
)

def fit_model(
    model,
    train_dataloader,
    val_dataloader
):
    """
    Fit the model.

    params:
    - model (LaplaceBNN): neural network.
    - train_dataloader (DataLoader): wrapper for the training data. 
    - val_dataloader (DataLoader): wrapper for the validation data.

    returns: 
    - params (jax.tree_util.pytree): updated neural network parameters.
    - prior_scale (jax.numpy.array): prior scale.
    - acc (float): validation accuracy.
    """
    assert train_dataloader.replacement == val_dataloader.replacement == False, "Data should be sampled without replacement"    

    # Read configuration
    key = model.key
    config = model.config
    dataset_name = config["data"]["name"]
    
    # Log-posterior training configuration
    lr = config["sampling_laplace"]["training"]["lr"]
    prior_scale = config["sampling_laplace"]["prior"]["scale_init"]
    nb_epochs = config["sampling_laplace"]["training"]["nb_epochs"]
    save_weights = config["sampling_laplace"]["training"]["save_weights"]
    early_stopping_patience = config["sampling_laplace"]["training"]["patience"]
    validation_freq = config["sampling_laplace"]["training"]["validation_freq"]

    # Path for checkpoint
    if save_weights:
        counter = 1
        dir_path = f"checkpoints/{dataset_name}"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
        while os.path.exists(checkpoint_path):
            checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
            counter += 1
        print(f"Checkpoint path for weights: {checkpoint_path}", flush=True)

    # Initialize log-posterior optimizer
    lp_opt_init, lp_opt_update, lp_get_params = adam(step_size=lr)
    if model.likelihood == "Gaussian":
        update_fn = update_gaussian_nn
        lp_params_init = (model.mean_params, jnp.log(jnp.exp(model.ll_scale)-1))
    elif model.likelihood == "Categorical":
        update_fn = update_categorical_nn
        lp_params_init = model.mean_params
    lp_opt_state = lp_opt_init(lp_params_init)

    # Number of training and validation samples
    n_train_samples = len(train_dataloader.dataset)
    n_val_samples = len(val_dataloader.dataset)

    # Early stopping initialization
    optimal_lp, no_improve_count_lp, optimal_state_lp = -jnp.inf, 0, lp_opt_state
    
    # Training loop 
    lp_step = model.training_steps
    for epoch in range(nb_epochs):
        log_likelihood, log_posterior = 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1, key2 = jax.random.split(key, 3)
            # Update the model
            lp_opt_state, loss_info = update_fn(
                model,
                lp_opt_state,
                lp_get_params,
                lp_opt_update,
                prior_scale, 
                n_train_samples,
                x,
                y, 
                key2,
                lp_step
            )
            log_likelihood += loss_info["log_likelihood"]
            log_posterior += loss_info["log_posterior"]
            lp_step += 1
        # Log training loss
        wandb.log({"Train/log_likelihood": log_likelihood, "Train/log_posterior": log_posterior})

        # Evaluation
        if epoch % validation_freq == 0 or epoch == nb_epochs-1:
            log_likelihood, log_posterior, val_acc = 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1 = jax.random.split(key)
                # Get prior scale
                # Prediction
                if model.likelihood == "Gaussian":
                    params, ll_rho = lp_get_params(lp_opt_state)
                    val_loss, val_info = n_gaussian_log_posterior_objective(
                        params,
                        ll_rho,
                        model,
                        x,
                        y,
                        key1,
                        prior_scale, 
                        n_val_samples
                    )
                    log_likelihood += val_info["log_likelihood"]
                    log_posterior += val_info["log_posterior"]
                elif model.likelihood == "Categorical":
                    params = lp_get_params(lp_opt_state)
                    val_loss, val_info = n_categorical_log_posterior_objective(
                        params,
                        model,
                        x,
                        y,
                        key1,
                        prior_scale, 
                        n_val_samples
                    )
                    probs = jax.nn.softmax(model.apply_fn(params, key2, x))
                    val_acc += jnp.sum(jnp.argmax(probs, axis=-1) == y.reshape(-1))
                    log_likelihood += val_info["log_likelihood"]
                    log_posterior += val_info["log_posterior"]
            # Log validation loss 
            log_dict = {"Val/log_posterior": log_posterior, "Val/log_likelihood": log_likelihood}
            log_str = f"Epoch {epoch} - val log_posterior: {log_posterior} - val log_likelihood {log_likelihood}"
            if model.likelihood == "Categorical":
                val_acc /= n_val_samples
                log_dict["Val/acc"] = val_acc
                log_str += f" - acc: {val_acc}"
            print(log_str, flush=True)
            wandb.log(log_dict)
            
            # Early stopping
            if log_posterior > optimal_lp:
                optimal_lp = log_posterior
                optimal_state_lp = lp_opt_state
                no_improve_count_lp = 0
                # Save weights
                if save_weights:
                    with open(checkpoint_path, "wb") as file:
                        pickle.dump(
                            {"lp_opt_state": lp_opt_state, "lp_step": lp_step}, file
                        )
            else:
                no_improve_count_lp += validation_freq
                if no_improve_count_lp >= early_stopping_patience:
                    lp_opt_state = optimal_state_lp
                    print("Early stopping.", flush=True)
                    break

            # Update prior params with marginal likelihood

    # Update model
    model.training_steps = lp_step
    if model.likelihood == "Gaussian":
        params, ll_rho = lp_get_params(lp_opt_state)
        ll_scale = jax.nn.softplus(ll_rho)
    elif model.likelihood == "Categorical":
        params = lp_get_params(lp_opt_state)
        ll_scale = 1.

    return params, ll_scale


@partial(jax.jit, static_argnums=(0,2,3,4,5))
def update_gaussian_nn(
    model,
    lp_opt_state,
    lp_get_params,
    lp_opt_update,
    prior_scale,
    n_samples,
    x,
    y, 
    key,
    step
):
    """
    Gradient update step.

    params:
    - model (Model): NN model.
    - lp_opt_state (jax.tree_util.pytree): optimizer state.
    - lp_get_params (jax.tree_util.pytree): function to get parameters.
    - lp_opt_update (callable): optimizer update function.
    - prior_scale (jax.numpy.array): prior scale.
    - n_samples (int): number of training samples.
    - x (jax.numpy.ndarray): a batch of input images.
    - y (jax.numpy.ndarray): a batch of labels.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - lp_opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    params, ll_rho = lp_get_params(lp_opt_state)

    # Compute gradients
    grads, other_info = jax.grad(n_gaussian_log_posterior_objective, argnums=(0,1),has_aux=True)(
        params,
        ll_rho,
        model,
        x,
        y,
        key,
        prior_scale, 
        n_samples
    )

    # Update parameters
    lp_opt_state = lp_opt_update(step, grads, lp_opt_state)

    return lp_opt_state, other_info


@partial(jax.jit, static_argnums=(0,2,3,4,5))
def update_categorical_nn(
    model,
    lp_opt_state,
    lp_get_params,
    lp_opt_update,
    prior_scale,
    n_samples,
    x, 
    y, 
    key,  
    step
):
    """
    Gradient update step on Categorical model params.

    params:
    - model (Model): BNN model.
    - lp_opt_state (jax.tree_util.pytree): optimizer state.
    - lp_get_params (jax.tree_util.pytree): function to get parameters.
    - lp_opt_update (callable): optimizer update_gaussian_nn function.
    - prior_scale (jax.numpy.array): prior scale.
    - n_samples (int): total number of training samples
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - lp_opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    params = lp_get_params(lp_opt_state)

    # Compute gradients
    grads, other_info = jax.grad(n_categorical_log_posterior_objective, has_aux=True)(
        params,
        model,
        x,
        y,
        key,
        prior_scale, 
        n_samples
    )
    
    # Update parameters
    lp_opt_state = lp_opt_update(step, grads, lp_opt_state)

    return lp_opt_state, other_info


def evaluate_model(
    key, 
    model, 
    dataloader, 
    mc_samples=10
):
    """
    Evaluate the model.

    params:
    - key (jax.random.PRNGKey): JAX random seed.
    - model (Model): BNN model.
    - test_dataloader (DataLoader): data loader.

    returns:
    - test_loss (dict): test loss.
    """
    assert dataloader.replacement == False, "Data should be sampled without replacement"
    
    if model.likelihood == "Gaussian":
        # Get likelihood scale
        ll_scale = model.ll_scale
        # Evaluate
        expected_ll, mse = 0., 0.
        for x, y in dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)
            # prediction
            f = model.predict_f(x, key1, mc_samples) # (n_samples, n_batch, n_classes)
            expected_ll += jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=ll_scale
            ).mean(0).sum()
            mse += jnp.sum((f.mean(0).reshape(-1) - y.reshape(-1))**2)
        mse /= len(dataloader.dataset)
        expected_ll /= len(dataloader.dataset)
        # Log
        out = {"expected_ll": expected_ll, "mse": mse}
        wandb.log({"Test/expected_ll": expected_ll, "Test/mse": mse})
        print(f"Expected log-likelihood: {expected_ll} - MSE: {mse}", flush=True)
    elif model.likelihood == "Categorical":
        # Load test data    
        y_one_hot_list, probs_list = [], []
        expected_ll, acc = 0., 0.
        for x, y in dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)
            # Prediction
            f = model.predict_f(x, key1, mc_samples) # (n_samples, n_batch, n_classes)
            probs = jax.nn.softmax(f, axis=-1)
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=probs.shape[-1])
            expected_ll += jnp.mean(
                jnp.sum(
                    one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_samples, n_batch, n_classes)
                    axis=-1
                ), # (n_samples, n_batch)
                axis=0
            ).sum()
            acc += jnp.sum(jnp.argmax(probs.mean(0), axis=-1) == y.reshape(-1))
            y_one_hot_list += [one_hot_y]
            probs_list += [probs.mean(0)]
        expected_ll /= len(dataloader.dataset)
        acc /= len(dataloader.dataset)
        # Calibration metrics
        one_hot_y = np.concatenate(y_one_hot_list, axis=0)
        probs = np.concatenate(probs_list, axis=0)
        ece, mce = calibration_metrics(one_hot_y, probs)
        # from models.Laplace.training_utils.calibration_metrics import compute_all_metrics
        # print(compute_all_metrics(y[:,0], probs))
        # Log
        out = {"expected_ll": expected_ll, "acc": acc, "ece": ece, "mce": mce}
        wandb.log({"Test/expected_ll": expected_ll, "Test/acc": acc, "Test/ece": ece, "Test/mce": mce})
        print(f"Expected log-likelihood: {expected_ll} - Accuracy: {acc} - ECE: {ece} - MCE: {mce}", flush=True)

    return out


def calibration_metrics(
    y, 
    p_mean, 
    num_bins=10
):
    """
    Compute calibration metrics.
    References:
    https://arxiv.org/abs/1706.04599
    https://arxiv.org/abs/1807.00263
    
    params:
    - y (jnp.array): one-hot encoding of the true classes, size (?, num_classes).
    - p_mean (jnp.array): numpy array, size (?, num_classes).
        containing the mean output predicted probabilities.
    - num_bins (jnp.array): number of bins.
    Returns:
    - ece (float): Expected Calibration Error.
    - mce (float): Maximum Calibration Error.
    """
    # Compute for every test sample x, the predicted class.
    class_pred = np.argmax(p_mean, axis=1)
    # and the confidence (probability) associated with it.
    conf = np.max(p_mean, axis=1)
    # Convert y from one-hot encoding to the number of the class
    y = np.argmax(y, axis=1)
    # Storage
    acc_tab = np.zeros(num_bins)  # empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # predicted confidence
    nb_items_bin = np.zeros(num_bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
    for i in np.arange(num_bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))

    return ece, mce