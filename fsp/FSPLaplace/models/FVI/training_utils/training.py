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

from data_utils.dataset import Dataset 
from data_utils.utils import read_image_data
from data_utils.dataloader import DataLoader


from models.FVI.training_utils.objective import (
    kl_divergence_objective, 
    n_felbo_gaussian_objective, 
    n_felbo_categorical_objective
)

def fit_model(
    key, 
    mean_params, 
    rho_params,
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
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
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
    likelihood = config["fvi"]["likelihood"]["model"]
    
    # Fit BNN
    if likelihood == "Gaussian":
        mean_params, rho_params, loss_info = fit_gaussian_model(
            key, 
            mean_params, 
            rho_params,
            model, 
            config, 
            train_dataloader, 
            val_dataloader,
            prior
        )
    elif likelihood == "Categorical":
        mean_params, rho_params, loss_info = fit_categorical_model(
            key, 
            mean_params, 
            rho_params,
            model, 
            config, 
            train_dataloader, 
            val_dataloader,
            prior
        )

    return mean_params, rho_params, loss_info


def fit_gaussian_model(
    key, 
    mean_params, 
    rho_params,
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
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
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
    lr = config["fvi"]["training"]["lr"]
    dataset_name = config["data"]["name"]
    ll_scale = config["fvi"]["likelihood"]["scale"]
    nb_epochs = config["fvi"]["training"]["nb_epochs"]
    mc_samples = config["fvi"]["training"]["mc_samples"] 
    early_stopping_patience = config["fvi"]["training"]["patience"]
    n_context_points = config["fvi"]["training"]["n_context_points"]
    validation_freq = config["fvi"]["neural_net"]["validation_freq"]
    context_selection = config["fvi"]["training"]["context_selection"]
    context_points_minval = config["fvi"]["training"]["min_context_val"]
    context_points_maxval = config["fvi"]["training"]["max_context_val"]

    # Path for checkpoint
    dir_path = f"checkpoints/{dataset_name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    counter = 1
    checkpoint_path = dir_path + f"/fvi_{n_context_points}_{context_selection}_{counter}.pkl"
    while os.path.exists(checkpoint_path):
        checkpoint_path = dir_path + f"/fvi_{n_context_points}_{context_selection}_{counter}.pkl"
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

    # Initialize parameter optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    ll_rho = jnp.log(jnp.exp(ll_scale)-1)
    params_init = (mean_params, rho_params, ll_rho)
    opt_state = opt_init(params_init)

    # Early stopping initialization
    optimal_gelbo, no_improve_count, optimal_state = jnp.NINF, 0, None

    # Training loop 
    print("Previous training steps: ", model.training_steps, flush=True)
    step = model.training_steps
    for epoch in range(nb_epochs):
        # Train losses 
        train_expected_ll, train_felbo, train_kl_div = 0., 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1, key2, key3 = jax.random.split(key, num=4)

            # Get context points
            if context_selection in ["kmnist", "cifar100"]:
                try:
                    x_sampled_context, _ = next(context_loader)
                except StopIteration:
                    context_point_iter = iter(context_loader)
                    x_sampled_context, _ = next(context_point_iter)
            else:
                x_sampled_context = select_context_points(
                    n_context_points,
                    context_selection,
                    context_points_maxval,
                    context_points_minval,
                    x.shape[1:],
                    key1, 
                    x
                )

            # Update the model
            opt_state, loss_info = update_gaussian_nn(
                model,
                opt_state,
                get_params,
                opt_update,
                prior,
                mc_samples,
                n_context_points,
                x,
                y, 
                x_sampled_context, 
                key3, 
                step
            )
            train_expected_ll += loss_info["expected_ll"]
            train_felbo += loss_info["felbo"]
            train_kl_div += loss_info["kl_div"]
            step += 1
        
        # Log training loss
        wandb.log(
            {
                "Train/felbo": train_felbo,
                "Train/expected_ll": train_expected_ll, 
                "Train/kl_div": train_kl_div, 
            }
        )

        # Evaluation
        if epoch % validation_freq == 0 or epoch == nb_epochs-1:
            expected_ll, felbo, kl_div = 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1, key2 = jax.random.split(key, num=3)

                # Get context points
                if context_selection in ["kmnist", "cifar100"]:
                    try:
                        x_sampled_context, _ = next(context_loader)
                    except StopIteration:
                        context_point_iter = iter(context_loader)
                        x_sampled_context, _ = next(context_point_iter)
                else:
                    x_sampled_context = select_context_points(
                        n_context_points,
                        context_selection,
                        context_points_maxval,
                        context_points_minval,
                        x.shape[1:],
                        key1, 
                        x
                    )

                # Prediction
                mean_params, rho_params, ll_rho = get_params(opt_state)
                val_loss, val_info = n_felbo_gaussian_objective(
                    mean_params,
                    rho_params,
                    ll_rho,
                    model,
                    prior,
                    x,
                    y,
                    x_sampled_context,
                    key2,
                    mc_samples, 
                    n_context_points
                )
                expected_ll += val_info["expected_ll"]
                felbo += val_info["felbo"]
                kl_div += val_info["kl_div"]

            # Log validation loss 
            wandb.log(
                {
                    "Val/felbo": felbo,
                    "Val/expected_ll": expected_ll, 
                    "Val/kl_div": kl_div
                }
            )
            print(f"Epoch {epoch} - felbo: {felbo} - expected log-likelihood {expected_ll} - fKL {kl_div}", flush=True)

            # Early stopping
            if felbo > optimal_gelbo:
                optimal_gelbo = felbo
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

    # Update model
    mean_params, rho_params, ll_rho = get_params(opt_state)
    model.ll_scale = jax.nn.softplus(ll_rho)
    model.training_steps = step
    print("Likelihood scale:", model.ll_scale, flush=True)

    # Evaluate model
    key, key1 = jax.random.split(key)
    val_loss = evaluate_model(
        key1, 
        mean_params, 
        rho_params,
        model, 
        val_dataloader,
        mc_samples=100
    )

    return mean_params, rho_params, val_loss


def fit_categorical_model(
    key, 
    mean_params, 
    rho_params,
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
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
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
    lr = config["fvi"]["training"]["lr"]
    dataset_name = config["data"]["name"]
    nb_epochs = config["fvi"]["training"]["nb_epochs"]
    mc_samples = config["fvi"]["training"]["mc_samples"] 
    early_stopping_patience = config["fvi"]["training"]["patience"]
    validation_freq = config["fvi"]["neural_net"]["validation_freq"]
    n_context_points = config["fvi"]["training"]["n_context_points"]
    context_selection = config["fvi"]["training"]["context_selection"]
    context_points_minval = config["fvi"]["training"]["min_context_val"]
    context_points_maxval = config["fvi"]["training"]["max_context_val"]

    # Path for checkpoint
    dir_path = f"checkpoints/{dataset_name}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    counter = 1
    checkpoint_path = dir_path + f"/fvi_{n_context_points}_{context_selection}_{counter}.pkl"
    while os.path.exists(checkpoint_path):
        checkpoint_path = dir_path + f"/fvi_{n_context_points}_{context_selection}_{counter}.pkl"
        counter += 1

    # Initalize context point distribution
    if context_selection in ["kmnist", "cifar100"]:
        # Read data 
        X_train, X_test, y_train, y_test = read_image_data(context_selection)
        X_context = jnp.concatenate([X_train, X_test], axis=0)
        y_context = jnp.concatenate([y_train, y_test], axis=0)
        # Pre-process data
        mean = np.mean(X_context, axis=(0,), keepdims=True)
        std = np.std(X_context, axis=(0,), keepdims=True) + 1e-10
        X_context = (X_context - mean) / std    
        # Create iterator
        context_dataset = Dataset(X_context, y_context)
        context_loader = DataLoader(key, context_dataset, n_context_points, shuffle=True, replacement=False)
        context_point_iter = iter(context_loader)

    # Initialize parameter optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    params_init = (mean_params, rho_params)
    opt_state = opt_init(params_init)

    # Early stopping initialization
    optimal_gelbo, no_improve_count, optimal_state = jnp.NINF, 0, None

    # Training loop 
    print("Previous training steps: ", model.training_steps, flush=True)
    step = model.training_steps
    for epoch in range(nb_epochs):
        # Train losses 
        train_expected_ll, train_felbo, train_kl_div = 0., 0., 0.
        for x, y in train_dataloader:
            # Handle keys
            key, key1, key2, key3 = jax.random.split(key, num=4)

            # Get context points
            if context_selection in ["kmnist", "cifar100"]:
                try:
                    x_sampled_context, _ = next(context_loader)
                except StopIteration:
                    context_point_iter = iter(context_loader)
                    x_sampled_context, _ = next(context_point_iter)
            else:
                x_sampled_context = select_context_points(
                    n_context_points,
                    context_selection,
                    context_points_maxval,
                    context_points_minval,
                    x.shape[1:],
                    key1, 
                    x
                )

            # Update the model
            opt_state, loss_info = update_categorical_nn(
                model, 
                opt_state,
                get_params,
                opt_update,
                prior,
                mc_samples,
                n_context_points,
                x,
                y, 
                x_sampled_context, 
                key3, 
                step
            )
            train_expected_ll += loss_info["expected_ll"]
            train_felbo += loss_info["felbo"]
            train_kl_div += loss_info["kl_div"]
            step += 1
        
        # Log training loss
        wandb.log(
            {
                "Train/felbo": train_felbo,
                "Train/expected_ll": train_expected_ll, 
                "Train/kl_div": train_kl_div, 
            }
        )

        # Evaluation
        if epoch % validation_freq == 0 or epoch == nb_epochs-1:
            expected_ll, felbo, kl_div, val_acc = 0., 0., 0., 0.
            for x, y in val_dataloader:
                # Handle keys
                key, key1, key2, key3 = jax.random.split(key, num=4)

                # Get context points
                if context_selection in ["kmnist", "cifar100"]:
                    try:
                        x_sampled_context, _ = next(context_loader)
                    except StopIteration:
                        context_point_iter = iter(context_loader)
                        x_sampled_context, _ = next(context_point_iter)
                else:
                    x_sampled_context = select_context_points(
                        n_context_points,
                        context_selection,
                        context_points_maxval,
                        context_points_minval,
                        x.shape[1:],
                        key1, 
                        x
                    )

                # Prediction
                mean_params, rho_params = get_params(opt_state)
                val_loss, val_info = n_felbo_categorical_objective(
                    mean_params,
                    rho_params,
                    model,
                    prior,
                    x,
                    y,
                    x_sampled_context,
                    key2,
                    mc_samples, 
                    n_context_points
                )
                probs = model.predict_y(mean_params, rho_params, x, key3, mc_samples)
                val_acc += jnp.sum(jnp.argmax(probs.mean(0), axis=-1) == y.reshape(-1))
                expected_ll += val_info["expected_ll"]
                felbo += val_info["felbo"]
                kl_div += val_info["kl_div"]
            val_acc /= len(val_dataloader.dataset)

            # Log validation loss 
            wandb.log(
                {
                    "Val/felbo": felbo,
                    "Val/expected_ll": expected_ll, 
                    "Val/kl_div": kl_div,
                    "Val/acc": val_acc
                }
            )
            print(f"Epoch {epoch} - felbo: {felbo} - expected log-likelihood {expected_ll} - fKL {kl_div} - acc: {val_acc}", flush=True)
                
            # Early stopping
            if felbo > optimal_gelbo:
                optimal_gelbo = felbo
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

    # Update model
    mean_params, rho_params = get_params(opt_state)
    model.training_steps = step

    # Evaluate model
    key, key1 = jax.random.split(key)
    val_loss = evaluate_model(
        key1, 
        mean_params, 
        rho_params,
        model, 
        val_dataloader,
        mc_samples=100
    )

    return mean_params, rho_params, val_loss


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
    - n_sampled_context_points (int): number of context points to select.
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
        # key1, key2 = jax.random.split(key)
        # x1 = jax.random.uniform(
        #     key=key1,
        #     shape=(2,),
        #     minval=context_points_minval[0],
        #     maxval=context_points_maxval[0],
        # )
        # x2 = jax.random.uniform(
        #     key=key2,
        #     shape=(2,),
        #     minval=context_points_minval[1],
        #     maxval=context_points_maxval[1],
        # )
        x1 = jnp.linspace(context_points_minval[0], context_points_maxval[0], 10) # 10
        x2 = jnp.linspace(context_points_minval[1], context_points_maxval[1], 8) # 8
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


@partial(jax.jit, static_argnums=(0,2,3,4,5,6))
def update_gaussian_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    prior,
    mc_samples,
    n_context_points,
    x,
    y,
    x_sampled_context, 
    key, 
    step
):
    """
    Gradient update step on model params.

    params:
    - model (Model): BNN model.
    - opt_state (jax.tree_util.pytree): optimizer state.
    - get_params (jax.tree_util.pytree): function to get parameters.
    - opt_update (callable): optimizer update_gaussian_nn function.
    - ll_scale (float): scale of the likelihood model. 
    - prior (Prior): prior.
    - mc_samples (int): number of Monte Carlo samples.
    - n_context_points (int): total number of context points.
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - x_sampled_context (jnp.array): a batch of context points to 
        evaluate the regularized KL-divergence term in the ELBO objective.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.

    returns:
    - params (jax.tree_util.pytree): updated parameters.
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    mean_params, rho_params, ll_rho = get_params(opt_state)

    # Update mean parameters
    grads, other_info = jax.grad(n_felbo_gaussian_objective, argnums=(0,1,2), has_aux=True)(
        mean_params,
        rho_params,
        ll_rho,
        model,
        prior,
        x,
        y,
        x_sampled_context,
        key,
        mc_samples, 
        n_context_points
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


@partial(jax.jit, static_argnums=(0,2,3,4,5,6))
def update_categorical_nn(
    model,
    opt_state,
    get_params,
    opt_update,
    prior,
    mc_samples,
    n_context_points,
    x,
    y,
    x_sampled_context, 
    key, 
    step
):
    """
    Gradient update step on model params.

    params:
    - model (Model): BNN model.
    - opt_state (jax.tree_util.pytree): optimizer state.
    - get_params (jax.tree_util.pytree): function to get parameters.
    - opt_update (callable): optimizer update_gaussian_nn function.
    - prior (Prior): prior.
    - mc_samples (int): number of Monte Carlo samples.
    - n_context_points (int): total number of context points.
    - x (jnp.array): a batch of input.
    - y (jnp.array): a batch of labels.
    - x_sampled_context (jnp.array): a batch of context points to 
        evaluate the regularized KL-divergence term in the ELBO objective.
    - key (jax.random.PRNGKey): JAX random seed.
    - step (int): current step.

    returns:
    - params (jax.tree_util.pytree): updated parameters.
    - opt_state (jax.tree_util.pytree): updated optimizer state.
    - other_info (dict): other information.
    """
    # Get parameters
    mean_params, rho_params = get_params(opt_state)

    # Update parameters
    grads, other_info = jax.grad(n_felbo_categorical_objective, argnums=(0,1), has_aux=True)(
        mean_params,
        rho_params,
        model,
        prior,
        x,
        y,
        x_sampled_context,
        key,
        mc_samples, 
        n_context_points
    )
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


def evaluate_model(
    key, 
    mean_params, 
    rho_params,
    model, 
    dataloader,
    mc_samples=100
):
    """
    Evaluate the model.

    params:
    - key (jax.random.PRNGKey): JAX random seed.
    - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
    - rho_params (jax.tree_util.pytree): pre-activated variance parameters of the BNN.
    - model (Model): BNN model.
    - dataloader (DataLoader): data loader.

    returns:
    - test_loss (dict): test loss.
    """
    assert dataloader.replacement == False, "Data should be sampled without replacement"

    if model.likelihood == "Gaussian":
        # Get likelihood scale
        ll_scale = model.ll_scale
        # Load test data    
        expected_ll, mse = 0., 0.
        for x, y in dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)
            # Prediction
            f_hat = model.predict_f(mean_params, rho_params, x, key1, mc_samples) # (n_samples, n_batch, 1)
            expected_ll += jnp.mean(
                jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale),
                axis=0
            ).sum()
            mse += jnp.sum((f_hat.mean(0).reshape(-1) - y.reshape(-1))**2)
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
            f = model.predict_f(mean_params, rho_params, x, key1, mc_samples) # (n_samples, n_batch, n_classes)
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
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

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
