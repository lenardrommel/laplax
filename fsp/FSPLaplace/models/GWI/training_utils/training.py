import os 
import jax 
import wandb
import pickle

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

from pathlib import Path
from functools import partial
from jax.example_libraries.optimizers import adam 



from models.GWI.training_utils.objective import (
    n_gelbo_gaussian_objective, 
    n_gelbo_categorical_objective
)


def fit_model(
    key, 
    mean_params, 
    L_params,
    model, 
    config, 
    train_dataloader, 
    val_dataloader,
    inducing_points,
    prior
):
    """
    Fit the model.

    params:
    - key (jax.random.PRNGKey): random key.
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - L_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - model (Model): BNN.
    - config (dict): configuration dictionary.
    - train_dataloader (DataLoader): train data loader.
    - val_dataloader (DataLoader): val data loader.
    - prior (Prior): prior.

    returns: 
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - L_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - loss (dict): validation losses.
    """
    assert train_dataloader.replacement == val_dataloader.replacement == False, "Data should be sampled without replacement"

    # Read configuration
    dataset_name = config["data"]["name"]
    lr = config["gwi"]["training"]["lr"]
    nb_epochs = config["gwi"]["training"]["nb_epochs"]
    likelihood = config["gwi"]["likelihood"]["model"]
    mc_samples = config["gwi"]["training"]["mc_samples"]
    validation_freq = config["gwi"]["neural_net"]["validation_freq"]
    early_stopping_patience = config["gwi"]["training"]["patience"]
    n_xs = config["gwi"]["inference"]["n_xs"]

    # Path for checkpoint
    dir_path = f"checkpoints/{dataset_name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    counter = 1
    checkpoint_path = dir_path + f"/gwi_{counter}.pkl"
    while os.path.exists(checkpoint_path):
        checkpoint_path = dir_path + f"/gwi_{counter}.pkl"
        counter += 1

    # Initialize parameter optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    if likelihood == "Gaussian":
        ll_rho = jnp.log(jnp.exp(model.ll_scale)-1)
        opt_state = opt_init((mean_params, L_params, ll_rho))
        update_fn = update_gaussian_nn
    elif likelihood == "Categorical":
        opt_state = opt_init((mean_params, L_params))
        update_fn = update_categorical_nn

    # Number of training and valiation samples
    n_train_samples = len(train_dataloader.dataset)
    n_val_samples = len(val_dataloader.dataset)
    n_xs = min(n_train_samples, n_xs)

    # Early stopping initialization
    optimal_gelbo, no_improve_count, optimal_state = jnp.NINF, 0, None

    # Training loop 
    step = model.training_steps
    print("Previous training steps: ", step, flush=True)
    for epoch in range(nb_epochs):
        # Train losses 
        train_expected_ll, train_gelbo, train_w2 = 0., 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1, key2 = jax.random.split(key, num=3)

            idxs = jax.random.choice(key1, n_train_samples, shape=(n_xs,), replace=False)
            x_s = train_dataloader.dataset.X[idxs,...]

            # Update the model
            opt_state, loss_info = update_fn(
                model,
                opt_state,
                get_params,
                opt_update,
                mc_samples,
                prior,
                n_train_samples,
                x,
                y,
                x_s,
                inducing_points,
                key,  
                step
            )
            train_expected_ll += loss_info["expected_ll"]
            train_gelbo += loss_info["gelbo"]
            train_w2 += loss_info["w2"]
            step += 1
        
        # Log training loss
        wandb.log(
            {
                "Train/gelbo": train_gelbo,
                "Train/expected_ll": train_expected_ll, 
                "Train/w2": train_w2, 
            }
        )

        # Evaluation
        if epoch % validation_freq == 0 or epoch == nb_epochs-1:
            val_expected_ll, val_gelbo, val_w2, val_acc = 0., 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1, key2 = jax.random.split(key, num=3)
                idxs = jax.random.choice(key1, n_train_samples, shape=(n_xs,), replace=False)
                x_s = train_dataloader.dataset.X[idxs,...]

                # Prediction
                if likelihood == "Gaussian":
                    mean_params, L_params, ll_rho = get_params(opt_state)
                    val_loss, val_info = n_gelbo_gaussian_objective(
                        mean_params,
                        L_params,
                        ll_rho,
                        model,
                        prior,
                        x,
                        y,
                        x_s,
                        inducing_points,
                        key1,
                        n_val_samples
                    )
                elif likelihood == "Categorical":
                    mean_params, L_params = get_params(opt_state)
                    val_loss, val_info = n_gelbo_categorical_objective(
                        mean_params,
                        L_params,
                        model,
                        prior,
                        x,
                        y,
                        x_s,
                        inducing_points,
                        key1,
                        n_val_samples, 
                        mc_samples
                    )
                    probs = model.predict_y(mean_params, L_params, prior, inducing_points, x, key2, mc_samples)
                    val_acc += jnp.sum(jnp.argmax(probs.mean(0), axis=-1) == y.reshape(-1))
                val_expected_ll += val_info["expected_ll"]
                val_gelbo += val_info["gelbo"]
                val_w2 += val_info["w2"]
            val_acc /= n_val_samples

            # Log validation loss 
            val_log = {"Val/gelbo": val_gelbo, "Val/expected_ll": val_expected_ll, "Val/w2": val_w2}
            val_str = f"Epoch {epoch} - gelbo: {val_gelbo} - expected log-likelihood {val_expected_ll} - w2 {val_w2}"
            if likelihood == "Categorical":
                val_log["Val/acc"] = val_acc
                val_str += f" - acc: {val_acc}"
            wandb.log(val_log)
            print(val_str, flush=True)

            # Early stopping
            if val_gelbo > optimal_gelbo:
                optimal_gelbo = val_gelbo
                optimal_state = opt_state
                no_improve_count = 0
                # Save weights
                with open(checkpoint_path, "wb") as file:
                    pickle.dump(
                        {
                            "opt_state": opt_state, 
                            "prior_params": prior.params, 
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
    
    # Update model parameters
    model.training_steps = step
    if likelihood == "Gaussian":
        mean_params, L_params, ll_rho = get_params(opt_state)
        model.ll_scale = jax.nn.softplus(ll_rho)
        print("Likelihood scale:", model.ll_scale, flush=True)
    elif likelihood == "Categorical":
        mean_params, L_params = get_params(opt_state)

    # Evaluate model
    key, key1 = jax.random.split(key)
    val_loss = evaluate_model(
        key, 
        mean_params, 
        L_params,
        model, 
        val_dataloader, 
        prior, 
        inducing_points,
        mc_samples=100
    )
    
    return mean_params, L_params, val_loss


@partial(jax.jit, static_argnums=(0,2,3,4,5,6))
def update_gaussian_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    mc_samples,
    prior,
    n_samples,
    x,
    y,
    x_s,
    inducing_points,
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
    - kl_gamma (float): KL divergence regularizer.    
    - n_samples (int): total number of training samples
    - n_context_points (int): number of context points.
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - x_context (jnp.array): a batch of context points to 
        evaluate the regularized KL-divergence term in the ELBO objective.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    mean_params, L_params, ll_rho = get_params(opt_state)
    grads, other_info = jax.grad(n_gelbo_gaussian_objective, argnums=(0,1,2), has_aux=True)(
        mean_params,
        L_params,
        ll_rho,
        model,
        prior,
        x,
        y,
        x_s,
        inducing_points,
        key,
        n_samples
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


@partial(jax.jit, static_argnums=(0,2,3,4,5,6))
def update_categorical_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    mc_samples,
    prior,
    n_samples,
    x,
    y,
    x_s,
    inducing_points,
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
    - mc_samples (int): number of Monte Carlo samples.
    - prior (Prior): prior.
    - kl_gamma (float): KL divergence regularizer.    
    - n_samples (int): total number of training samples.
    - n_context_points (int): number of context points.
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - x_context (jnp.array): a batch of context points to 
        evaluate the regularized KL-divergence term in the ELBO objective.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.
    
    returns:
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    mean_params, L_params = get_params(opt_state)
    grads, other_info = jax.grad(n_gelbo_categorical_objective, argnums=(0,1), has_aux=True)(
        mean_params,
        L_params,
        model,
        prior,
        x,
        y,
        x_s,
        inducing_points,
        key,
        n_samples, 
        mc_samples
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


def evaluate_model(
    key, 
    mean_params, 
    L_params,
    model, 
    dataloader, 
    prior, 
    inducing_points,
    mc_samples=100
):
    """
    Evaluate the model.

    params:
    - key (jax.random.PRNGKey): JAX random seed.
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - L_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - model (Model): BNN model.
    - dataloader (DataLoader): data loader.

    returns:
    - test_loss (dict): test loss.
    """
    assert dataloader.replacement == False, "Data should be sampled without replacement"

    if model.likelihood == "Gaussian":
        # Read configuration
        ll_scale = model.ll_scale
        # Load test data    
        expected_ll, mse = 0., 0.
        for x, y in dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)
            # Prediction
            f = model.predict_f(mean_params, L_params, prior, inducing_points, x, key1, mc_samples)
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
            f = model.predict_f(mean_params, L_params, prior, inducing_points, x, key1, mc_samples)
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