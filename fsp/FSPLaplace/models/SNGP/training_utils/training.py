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

from models.SNGP.training_utils.objective import (
    n_gaussian_log_posterior_objective, 
    n_categorical_log_posterior_objective
)

def fit_model(
    key, 
    params, 
    nn_state,
    model, 
    config, 
    train_dataloader, 
    val_dataloader
):
    """
    Fit the model.

    params:
    - key (jax.random.PRNGKey): random key.
    - params (jax.tree_util.pytree): mean parameters of the BNN.
    - nn_state (jax.tree_util.pytree): state of the neual network.
    - model (Model): BNN.
    - config (dict): configuration dictionary.
    - train_dataloader (DataLoader): train data loader.
    - val_dataloader (DataLoader): val data loader.
    - prior (Prior): prior.

    returns: 
    - params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - loss (dict): validation losses.
    """
    assert train_dataloader.replacement == val_dataloader.replacement == False, "Data should be sampled without replacement"

    # Read configuration
    n_outputs = model.architecture[-1]
    dataset_name = config["data"]["name"]
    lr = config["sngp"]["training"]["lr"]
    n_rff = config["sngp"]["inference"]["n_rff"]
    ll_scale = config["sngp"]["likelihood"]["scale"]
    likelihood = config["sngp"]["likelihood"]["model"]
    nb_epochs = config["sngp"]["training"]["nb_epochs"]
    rff_scale = config["sngp"]["inference"]["rff_scale"]
    save_weights = config["sngp"]["training"]["save_weights"]
    early_stopping_patience = config["sngp"]["training"]["patience"]
    validation_freq = config["sngp"]["neural_net"]["validation_freq"]

    # Path for checkpoint
    if save_weights:
        dir_path = f"checkpoints/{dataset_name}"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        counter = 1
        checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
        while os.path.exists(checkpoint_path):
            checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
            counter += 1

    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    if likelihood == "Gaussian":
        params_init = (params, jnp.log(jnp.exp(ll_scale)-1))
        update_fn = update_gaussian_nn
    elif likelihood == "Categorical":
        params_init = params
        update_fn = update_categorical_nn
    opt_state = opt_init(params_init)

    # Number of training and validation samples
    n_train_samples = len(train_dataloader.dataset)
    n_val_samples = len(val_dataloader.dataset)

    # Early stopping initialization
    optimal_lpost, no_improve_count, optimal_state = -jnp.inf, 0, None

    # Training loop 
    print("Previous training steps: ", model.training_steps, flush=True)
    step = model.training_steps
    for epoch in range(nb_epochs):
        train_ll, train_lpost = 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)

            # Update the model
            opt_state, loss_info = update_fn(
                model,
                nn_state,
                opt_state,
                get_params,
                opt_update,
                rff_scale,
                n_train_samples,
                x, 
                y, 
                key,
                step
            )
            nn_state = loss_info["state"]
            train_ll += loss_info["log_likelihood"]
            train_lpost += loss_info["log_posterior"]
            step += 1
        
        # Log training loss
        wandb.log(
            {
                "Train/log_likelihood": train_ll,
                "Train/log_posterior": train_lpost
            }
        )

        # Evaluation
        if epoch % validation_freq == 0:
            val_ll, val_lpost, val_acc = 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1 = jax.random.split(key)

                # Prediction
                if likelihood == "Gaussian":
                    params, ll_rho = get_params(opt_state)
                    val_loss, val_info = n_gaussian_log_posterior_objective(
                        params,
                        ll_rho,
                        model,
                        nn_state,
                        x,
                        y,
                        key,
                        rff_scale,
                        n_val_samples, 
                        training=False
                    )
                elif likelihood == "Categorical":
                    params = get_params(opt_state)
                    val_loss, val_info = n_categorical_log_posterior_objective(
                        params,
                        model,
                        nn_state,
                        x,
                        y,
                        key,
                        rff_scale,
                        n_val_samples,
                        training=False
                    )
                    probs = jax.nn.softmax(model.apply_fn(params, nn_state, key1, x, training=False)[0][0])
                    val_acc += jnp.sum(jnp.argmax(probs, axis=-1) == y.reshape(-1))
                val_acc /= n_val_samples
                val_ll += val_info["log_likelihood"]
                val_lpost += val_info["log_posterior"]
            
            # Log validation loss 
            log_dict = {
                "Val/log_posterior": val_lpost,
                "Val/log_likelihood": val_ll,
            }
            log_str = f"Epoch {epoch} - val log_posterior: {val_lpost} - val log_likelihood {val_ll}"
            if likelihood == "Categorical":
                log_dict["Val/accuracy"] = val_acc
                log_str += f" - val accuracy {val_acc}"
            wandb.log(log_dict)
            print(log_str, flush=True)

            # Early stopping
            if val_lpost > optimal_lpost:
                optimal_lpost = val_lpost
                optimal_state = opt_state, nn_state
                no_improve_count = 0
                # Save weights
                if save_weights:
                    with open(checkpoint_path, "wb") as file:
                        pickle.dump(
                            {
                                "opt_state": opt_state, 
                                "step": step
                            }, 
                            file
                        )
            else:
                no_improve_count += validation_freq
                if no_improve_count >= early_stopping_patience:
                    opt_state, nn_state = optimal_state
                    print("Early stopping.", flush=True)
                    break
    
    # Update model
    model.training_steps = step
    if likelihood == "Gaussian":
        params, ll_rho = get_params(opt_state)
        model.ll_scale = jax.nn.softplus(ll_rho)
        print("Likelihood scale:", model.ll_scale, flush=True)
    elif likelihood == "Categorical":
        params = get_params(opt_state)

    # Compute precision matrix
    precision = jnp.ones((n_rff, n_rff, n_outputs)) / rff_scale**2
    for x, y in train_dataloader:
        key, key1 = jax.random.split(key)
        f_mean, phi = model.apply_fn(params, nn_state, key1, x, training=False)[0]  # (n_batch, n_classes), (n_batch, features_out)
        if likelihood == "Gaussian":
            precision += (phi.T @ phi / ll_scale**2).reshape(precision.shape)
        elif likelihood == "Categorical":
            p = jax.nn.softmax(f_mean, axis=-1) # (n_batch, n_classes)
            precision += jax.vmap(lambda _p: phi.T @ jnp.diag(_p * (1 - _p)) @ phi, in_axes=-1, out_axes=-1)(p)

    # Compute covariance matrix
    cov = jax.vmap(lambda _p: jnp.linalg.inv(_p + 1e-10*jnp.eye(_p.shape[0])), in_axes=-1, out_axes=-1)(precision)
    print("eigvals cov", jnp.linalg.eigvalsh(cov[:,:,0]), flush=True)
    
    return params, nn_state, cov 


@partial(jax.jit, static_argnums=(0,3,4,5,6))
def update_gaussian_nn(
    model,
    nn_state,
    opt_state,
    get_params,
    opt_update,
    rff_scale,
    n_samples,
    x, 
    y, 
    key,
    step
):
    """
    Gradient update step on Gaussian model params.
    """
    # Get parameters
    params, ll_rho = get_params(opt_state)

    # Compute gradients
    grads, other_info = jax.grad(n_gaussian_log_posterior_objective, argnums=(0,1), has_aux=True)(
        params,
        ll_rho,
        model,
        nn_state,
        x,
        y,
        key,
        rff_scale,
        n_samples, 
        training=True
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


@partial(jax.jit, static_argnums=(0,3,4,5,6))
def update_categorical_nn(
    model,
    nn_state,
    opt_state,
    get_params,
    opt_update,
    rff_scale,
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
    - opt_state (jax.tree_util.pytree): optimizer state.
    - get_params (jax.tree_util.pytree): function to get parameters.
    - opt_update (callable): optimizer update_gaussian_nn function.
    - rff_scale (float): scale of the prior.
    - n_samples (int): total number of training samples
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - params (jax.tree_util.pytree): updated parameters.
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    params = get_params(opt_state)

    # Update
    grads, other_info = jax.grad(n_categorical_log_posterior_objective, has_aux=True)(
        params,
        model,
        nn_state,
        x,
        y,
        key,
        rff_scale,
        n_samples,
        training=True
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


def evaluate_model(
    key, 
    model, 
    dataloader, 
    mc_samples=100
):
    """
    Evaluate the model on the test set.

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
            mean, diag_cov = model.f_distribution_mean_var(x, key1, mc_samples)
            expected_ll += jsp.stats.norm.logpdf(
                y, 
                loc=mean, 
                scale=ll_scale
            ).sum() 
            expected_ll -= 0.5 * diag_cov.sum() / ll_scale**2
            mse += jnp.sum((mean.reshape(-1) - y.reshape(-1))**2)
        mse /= len(dataloader.dataset)
        expected_ll /= len(dataloader.dataset)
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