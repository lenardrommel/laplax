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
    n_gaussian_log_posterior_objective
)

def fit_model(
    key, 
    mean_params, 
    nn_state,
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
    - nn_state (jax.tree_util.pytree): state of the neual network.
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
    lr = config["fsplaplace"]["training"]["lr"]
    ll_scale = config["fsplaplace"]["likelihood"]["scale"]
    likelihood = config["fsplaplace"]["likelihood"]["model"]
    nb_epochs = config["fsplaplace"]["training"]["nb_epochs"]
    save_weights = config["fsplaplace"]["training"]["save_weights"]
    early_stopping_patience = config["fsplaplace"]["training"]["patience"]
    validation_freq = config["fsplaplace"]["neural_net"]["validation_freq"]
    n_context_points = config["fsplaplace"]["training"]["n_context_points"]
    context_selection = config["fsplaplace"]["training"]["context_selection"]
    context_points_minval = config["fsplaplace"]["inference"]["min_context_val"]
    context_points_maxval = config["fsplaplace"]["inference"]["max_context_val"]

    # Path for checkpoint
    if save_weights:
        dataset_name = config["data"]["name"]
        dir_path = f"checkpoints/{dataset_name}"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        counter = 1
        checkpoint_path = dir_path + f"/flaplace_{counter}.pkl"
        while os.path.exists(checkpoint_path):
            checkpoint_path = dir_path + f"/flaplace_{counter}.pkl"
            counter += 1

    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    if likelihood == "Gaussian":
        params_init = (mean_params, jnp.log(jnp.exp(ll_scale)-1))
        update_fn = update_gaussian_nn
    elif likelihood == "Categorical":
        params_init = mean_params
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
        train_ll, train_sq_rkhs, train_lpost = 0., 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)
            # Get context points
            # print(context_points_maxval, context_points_minval)
            
            x_context = select_context_points(
                n_context_points,
                context_selection,
                context_points_maxval,
                context_points_minval,
                x.shape[1:],
                key1, 
                x, 
            )

            # Update the model
            opt_state, loss_info = update_fn(
                model,
                nn_state,
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
            nn_state = loss_info["state"]
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
        if epoch % validation_freq == 0:
            val_ll, val_sq_rkhs, val_lpost, val_acc = 0., 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1 = jax.random.split(key)

                # Get context points
                
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
                    ll_rho = jnp.clip(ll_rho, a_min=jnp.log(jnp.exp(1e-2)-1))
                    val_loss, val_info = n_gaussian_log_posterior_objective(
                        mean_params,
                        ll_rho,
                        model,
                        nn_state,
                        x,
                        y,
                        x_context,
                        key,
                        prior,
                        n_val_samples, 
                        training=False
                    )
                val_ll += val_info["log_likelihood"]
                val_lpost += val_info["log_posterior"]
                val_sq_rkhs += val_info["sq_rkhs_norm"]
            val_acc /= n_val_samples
            
            # Log validation loss 
            log_dict = {
                "Val/log_posterior": val_lpost,
                "Val/log_likelihood": val_ll,
                "Val/sq_rkhs_norm": val_sq_rkhs,
            }
            log_str = f"Epoch {epoch} - val log_posterior: {val_lpost} - val log_likelihood {val_ll}, val sq_rkhs_norm {val_sq_rkhs}"
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
        mean_params, ll_rho = get_params(opt_state)
        model.ll_scale = jax.nn.softplus(ll_rho) #jnp.clip(jax.nn.softplus(ll_rho), a_min=1e-2)
        print("Likelihood scale:", model.ll_scale, flush=True)
    elif likelihood == "Categorical":
        mean_params = get_params(opt_state)

    return mean_params, nn_state


@partial(jax.jit, static_argnums=(0,3,4,5,6))
def update_gaussian_nn(
    model,
    nn_state,
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
    #ll_rho = jnp.clip(ll_rho, a_min=jnp.log(jnp.exp(1e-2)-1))
    
    
    # Compute gradients
    grads, other_info = jax.grad(n_gaussian_log_posterior_objective, argnums=(0,1), has_aux=True)(
        mean_params,
        ll_rho,
        model,
        nn_state,
        x,
        y,
        x_context,
        key,
        prior,
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
    pass


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
    elif context_selection == "ocean_current_modeling":
        x1 = jnp.linspace(context_points_minval[0], context_points_maxval[0], 10) # 34
        x2 = jnp.linspace(context_points_minval[1], context_points_maxval[1], 8) # 15
        x3 = jnp.array([0., 1.])
        context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, indexing='ij'), axis=-1).reshape(-1, 3)
    elif context_selection == "random_monochrome":
        n, h, w, c = x.shape
        X_reshaped = x.reshape(n, h * w * c)
        random_indices = jax.random.randint(key, shape=(n_context_points, h, w, c), minval=0, maxval=n)
        context_points = X_reshaped[random_indices, jnp.arange(c)].reshape(n_context_points, h, w, c)
    elif context_selection == "grid":
        assert datapoint_shape[-1] in [1,2,3,4], "Grid context selection only works for 1D or 2D features."
        feature_dim = datapoint_shape[-1]
        if feature_dim == 1:
            context_points = jnp.linspace(
                context_points_minval[0], 
                context_points_maxval[0],
                n_context_points
            ).reshape(-1, 1)
        elif feature_dim == 2:
            n_dim = jnp.rint(n_context_points**0.5).astype(int)
            x1 = jnp.linspace(context_points_minval[0], context_points_maxval[0], n_dim)
            x2 = jnp.linspace(context_points_minval[1], context_points_maxval[1], n_dim)
            x = jnp.meshgrid(x1, x2, indexing='ij')
            context_points = jnp.stack(x, axis=-1).reshape(-1, 2)
        elif feature_dim == 3:
            n_dim = jnp.ceil(n_context_points**(1/3)).astype(int)
            x1 = jnp.linspace(context_points_minval[0], context_points_maxval[0], n_dim)
            x2 = jnp.linspace(context_points_minval[1], context_points_maxval[1], n_dim)
            x3 = jnp.linspace(context_points_minval[2], context_points_maxval[2], n_dim)
            context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, indexing='ij'), axis=-1).reshape(-1, 3)
        elif feature_dim== 4:
            n_dim = jnp.ceil(n_context_points**(1/4)).astype(int).item()
            x1 = jnp.linspace(context_points_minval[0], context_points_maxval[0], n_dim)
            x2 = jnp.linspace(context_points_minval[1], context_points_maxval[1], n_dim)
            x3 = jnp.linspace(context_points_minval[2], context_points_maxval[2], n_dim)
            x4 = jnp.linspace(context_points_minval[3], context_points_maxval[3], n_dim)
            context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, x4, indexing='ij'), axis=-1).reshape(-1, 4)

    return context_points


def evaluate_model(
    key, 
    model, 
    dataloader, 
    mc_samples=10
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
            acc += jnp.sum(jnp.argmax(probs.mean(0), axis=-1).reshape(-1) == y.reshape(-1))
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