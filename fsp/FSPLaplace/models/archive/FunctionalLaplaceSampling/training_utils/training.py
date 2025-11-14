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

from data_utils.dataset import Dataset 
from data_utils.utils import read_image_data
from data_utils.dataloader import DataLoader

from models.FSPLaplace.training_utils.objective import (
    n_gaussian_log_posterior_objective, 
    n_categorical_log_posterior_objective
)

def fit_model(
    key, 
    mean_params, 
    model, 
    config, 
    train_dataloader, 
    val_dataloader,
    prior
):
    """
    Fit the model.

    params:
    - key (jax.random.PRNGKey): random key.
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - model (Model): BNN.
    - config (dict): configuration dictionary.
    - train_dataloader (DataLoader): train data loader.
    - val_dataloader (DataLoader): val data loader.
    - prior (Prior): prior.

    returns: 
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - loss (dict): validation losses.
    """
    assert train_dataloader.replacement == val_dataloader.replacement == False, "Data should be sampled without replacement"

    # Read configuration
    dataset_name = config["data"]["name"]
    lr = config["flaplace_sampling"]["training"]["lr"]
    ll_scale = config["flaplace_sampling"]["likelihood"]["scale"]
    likelihood = config["flaplace_sampling"]["likelihood"]["model"]
    nb_epochs = config["flaplace_sampling"]["training"]["nb_epochs"]
    save_weights = config["flaplace_sampling"]["training"]["save_weights"]
    early_stopping_patience = config["flaplace_sampling"]["training"]["patience"]
    validation_freq = config["flaplace_sampling"]["neural_net"]["validation_freq"]
    n_context_points = config["flaplace_sampling"]["training"]["n_context_points"]
    context_selection = config["flaplace_sampling"]["training"]["context_selection"]
    context_points_minval = config["flaplace_sampling"]["training"]["min_context_val"]
    context_points_maxval = config["flaplace_sampling"]["training"]["max_context_val"]

    # Path for checkpoint
    if save_weights:
        dir_path = f"checkpoints/{dataset_name}"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        counter = 1
        checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
        while os.path.exists(checkpoint_path):
            checkpoint_path = dir_path + f"/laplace_{counter}.pkl"
            counter += 1

    # Initalize context point distribution
    if context_selection in ["kmnist", "cifar100"]:
        # Read data 
        X_train, X_test, y_train, y_test = read_image_data(context_selection)
        X_context = jnp.concatenate([X_train, X_test], axis=0)
        y_context = jnp.concatenate([y_train, y_test], axis=0)
        # Pre-process data
        mean = jnp.mean(X_context, axis=(0,), keepdims=True)
        std = jnp.std(X_context, axis=(0,), keepdims=True) + 1e-10
        X_context = (X_context - mean) / std    
        # Create iterator
        context_dataset = Dataset(X_context, y_context)
        context_loader = DataLoader(key, context_dataset, n_context_points, shuffle=True, replacement=False)
        context_point_iter = iter(context_loader)

    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    if likelihood == "Gaussian":
        params_init = (mean_params, jnp.log(jnp.exp(ll_scale)-1))
    elif likelihood == "Categorical":
        params_init = mean_params
    opt_state = opt_init(params_init)

    # Number of training and validation samples
    n_train_samples = len(train_dataloader.dataset)
    n_val_samples = len(val_dataloader.dataset)

    # Early stopping initialization
    optimal_lpost, no_improve_count, optimal_state = jnp.NINF, 0, None

    # Training loop 
    print("Previous training steps: ", model.training_steps, flush=True)
    step = model.training_steps
    for epoch in range(nb_epochs):
        train_ll, train_sq_rkhs, train_lpost = 0., 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)

            # Get context points
            if context_selection in ["kmnist", "cifar100"]:
                try:
                    x_context, _ = next(context_loader)
                except StopIteration:
                    context_point_iter = iter(context_loader)
                    x_context, _ = next(context_point_iter)
            else:
                x_context = select_context_points(
                    n_context_points,
                    context_selection,
                    context_points_maxval,
                    context_points_minval,
                    x.shape[1:],
                    key1, 
                    x
                )

            # Update the model
            if likelihood == "Gaussian":
                opt_state, loss_info = update_gaussian_nn(
                    model,
                    opt_state,
                    get_params,
                    opt_update,
                    prior,
                    n_train_samples,
                    x, 
                    y, 
                    x_context,
                    key,
                    step
                )
            elif likelihood == "Categorical":
                opt_state, loss_info = update_categorical_nn(
                    model,
                    opt_state,
                    get_params,
                    opt_update,
                    prior,
                    n_train_samples,
                    x, 
                    y, 
                    x_context,
                    key,
                    step
                )
            train_ll += loss_info["log_likelihood"]
            train_lpost += loss_info["log_posterior"]
            train_sq_rkhs += loss_info["sq_rkhs_norm"]
            step += 1
        
        # Log training loss
        wandb.log(
            {
                "Train/log_likelihood": train_ll,
                "Train/log_posterior": train_lpost,
                "Train/sq_rkhs_norm": train_sq_rkhs,
            }
        )

        # Evaluation
        if epoch % 100 == 0:
            val_ll, val_sq_rkhs, val_lpost = 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1 = jax.random.split(key)

                # Get context points
                if context_selection in ["kmnist", "cifar100"]:
                    try:
                        x_context, _ = next(context_loader)
                    except StopIteration:
                        context_point_iter = iter(context_loader)
                        x_context, _ = next(context_point_iter)
                else:
                    x_context = select_context_points(
                        n_context_points,
                        context_selection,
                        context_points_maxval,
                        context_points_minval,
                        x.shape[1:],
                        key1, 
                        x
                    )

                # Prediction
                if likelihood == "Gaussian":
                    mean_params, ll_rho = get_params(opt_state)
                    val_loss, val_info = n_gaussian_log_posterior_objective(
                        mean_params,
                        ll_rho,
                        model,
                        x,
                        y,
                        x_context,
                        key,
                        prior,
                        n_val_samples
                    )
                elif likelihood == "Categorical":
                    mean_params = get_params(opt_state)
                    val_loss, val_info = n_categorical_log_posterior_objective(
                        mean_params,
                        model,
                        x,
                        y,
                        x_context,
                        key,
                        prior,
                        n_val_samples
                    )
                val_ll += val_info["log_likelihood"]
                val_lpost += val_info["log_posterior"]
                val_sq_rkhs += val_info["sq_rkhs_norm"]
            
            # Log validation loss 
            wandb.log(
                {
                    "Val/log_posterior": val_lpost,
                    "Val/log_likelihood": val_ll,
                    "Val/sq_rkhs_norm": val_sq_rkhs,
                }
            )
            print(f"Epoch {epoch} - val log_posterior: {val_lpost} - val log_likelihood {val_ll}, val sq_rkhs_norm {val_sq_rkhs}", flush=True)

            # Early stopping
            if val_lpost > optimal_lpost:
                optimal_lpost = val_lpost
                optimal_state = opt_state
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
                    opt_state = optimal_state
                    print("Early stopping.", flush=True)
                    break
    
    # Update model
    model.training_steps = step
    if likelihood == "Gaussian":
        mean_params, ll_rho = get_params(opt_state)
        model.ll_scale = jax.nn.softplus(ll_rho)
        print("Likelihood scale:", model.ll_scale, flush=True)
    elif likelihood == "Categorical":
        mean_params = get_params(opt_state)

    return mean_params


@partial(jax.jit, static_argnums=(0,2,3,4,5))
def update_gaussian_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    prior,
    n_samples,
    x, 
    y, 
    x_context,
    key,
    step
):
    """
    Gradient update step on Gaussian model params.

    params:
    - model (Model): BNN model.
    - opt_state (jax.tree_util.pytree): optimizer state.
    - get_params (jax.tree_util.pytree): function to get parameters.
    - opt_update (callable): optimizer update_gaussian_nn function.
    - prior (Prior): prior.
    - n_samples (int): total number of training samples
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - x_context (jnp.array): a batch of context points to estimate the 
        RKHS norm.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    mean_params, ll_rho = get_params(opt_state)

    # Compute gradients
    grads, other_info = jax.grad(n_gaussian_log_posterior_objective, argnums=(0,1), has_aux=True)(
        mean_params,
        ll_rho,
        model,
        x,
        y,
        x_context,
        key,
        prior,
        n_samples
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


@partial(jax.jit, static_argnums=(0,2,3,4,5))
def update_categorical_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    prior,
    n_samples,
    x, 
    y, 
    x_context,
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
    - prior_scale (float): scale of the prior.
    - n_samples (int): total number of training samples
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - mean_params (jax.tree_util.pytree): updated parameters.
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    mean_params = get_params(opt_state)

    # Update
    grads, other_info = jax.grad(n_categorical_log_posterior_objective, has_aux=True)(
        mean_params,
        model,
        x,
        y,
        x_context,
        key,
        prior,
        n_samples
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


@partial(jax.jit, static_argnums=(0,1,2,3,4))
def select_context_points(
	n_context_points,
    context_selection,
	context_points_maxval,
    context_points_minval,
    datapoint_shape,
	key, 
    x
):
    """
    Select context points.

    params:
    - n_context_points (int): number of context points to select.
    - context_selection (str): context selection method.
    - context_points_maxval (float): maximum value of context points.
    - context_points_minval (float): minimum value of context points.
    - x_shape (jnp.array): shape of data.
    - key: random key.

    returns:
    - context points (jnp.array): context points.
    """
    if context_selection == "random":
        context_points = jax.random.uniform(
            key=key,
            shape=(n_context_points,)+datapoint_shape,
            minval=context_points_minval,
            maxval=context_points_maxval,
        )
    elif context_selection == "random_monochrome":
        n, h, w, c = x.shape
        X_reshaped = x.reshape(n, h * w * c)
        random_indices = jax.random.randint(key, shape=(n_context_points, h, w, c), minval=0, maxval=n)
        context_points = X_reshaped[random_indices, jnp.arange(c)].reshape(n_context_points, h, w, c)
    elif context_selection == "grid":
        assert datapoint_shape[-1] in [1,2], "Grid context selection only works for 1D or 2D features."
        if datapoint_shape[-1]  == 1:
            context_points = jnp.linspace(
                context_points_minval, 
                context_points_maxval, 
                n_context_points
            ).reshape(-1, 1)
        elif datapoint_shape[-1]  == 2:
            x1 = jnp.linspace(-1, 1, np.sqrt(n_context_points).astype(int))
            x2 = jnp.linspace(-1, 1, np.sqrt(n_context_points).astype(int))
            x = jnp.meshgrid(x1, x2, indexing='ij')
            context_points = jnp.stack(x, axis=-1).reshape(-1, 2)
    
    return context_points


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
            f = model.predict_f(x, key1, mc_samples) # (n_samples, n_batch, 1)
            expected_ll += jnp.mean(
                jsp.stats.norm.logpdf(y, loc=f, scale=ll_scale),
                axis=0
            ).sum()
            mse += jnp.sum((f.mean(0).reshape(-1) - y.reshape(-1))**2)
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