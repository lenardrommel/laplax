import jax 
import wandb

import jax.scipy as jsp

from functools import partial

def fit_model(
    key, 
    params, 
    model, 
    opt_update, 
    opt_state,
    get_params, 
    config, 
    dataloader, 
    prior
):
    """
    Fit the model.
    :param key (jax.random.PRNGKey): random key.
    :param params (jax.tree_util.pytree): parameters of the neural network.
    :param model (Model): neural network.
    :param opt_update (callable): optimizer update function.
    :param opt_state (jax.tree_util.pytree): optimizer state.
    :param get_params (jax.tree_util.pytree): function to get parameters.
    :param config (dict): configuration dictionary.
    :param dataloader (DataLoader): data loader.
    :param prior (callable): prior distribution.
    """
    # Read configuration
    likelihood_scale = config["FSPLaplace"]["likelihood_scale"]
    nb_epochs = config["FSPLaplace"]["training"]["nb_epochs"]
    
    # Training loop 
    step = 0
    for epoch in range(nb_epochs):
        for x, y in dataloader:
            # Handle keys
            key, key1 = jax.random.split(key)

            # Update the model
            params, opt_state, loss_info = update(
                neg_log_posterior, 
                params, 
                opt_state,
                get_params,
                x,
                y, 
                key1, 
                opt_update,
                model,
                likelihood_scale, 
                prior,
                step
            )
            step += 1
        
        # X = X_train.reshape(-1, X_train.shape[-1])
        # fwd = lambda p: model.apply_fn(p, key1, X)
        # jac = jax.jacfwd(fwd)(params)
        # leaves = jax.tree_util.tree_flatten(jac)[0]
        # jac = jnp.concatenate([i.reshape(X.shape[0], -1) for i in leaves], axis=-1)
        # jac_rank = jnp.linalg.matrix_rank(jac)
        
        # Log training loss
        wandb.log(
            {
                "Train/log_likelihood": loss_info["log_likelihood"],
                "Train/log_posterior": loss_info["log_posterior"],
                "Train/log_prior": loss_info["log_prior"],
                # "Train/train_data_jac_rank": jac_rank
            }, 
            step=epoch
        )

        # Evaluation
        if epoch % 100 == 0:
            log_likelihood, log_posterior, log_prior = 0., 0., 0.
            for x, y in zip(X_val, y_val):
                # Handle keys
                key, key1 = jax.random.split(key)

                # prediction
                val_loss, val_info = neg_log_posterior(
                    params,
                    model,
                    x,
                    y,
                    key1,
                    likelihood_scale, 
                    prior
                )
                log_likelihood += val_info["log_likelihood"]
                log_posterior += val_info["log_posterior"]
                log_prior += val_info["log_prior"]
            print(f"Epoch {epoch} - val log_posterior: {log_posterior} - val log_likelihood {log_likelihood}", flush=True)

            # Log validation loss 
            wandb.log(
                {
                    "Val/log_posterior": log_posterior,
                    "Val/log_likelihood": log_likelihood,
                    "Val/log_prior": log_prior,
                }, 
                step=epoch
            )

    return params


@partial(jax.jit, static_argnums=(0,3,7,8,9,10))
def update(
    loss, 
    params, 
    opt_state,
    get_params,
    x_batch,
    y_batch, 
    key, 
    opt_update,
    model,
    ll_scale, 
    prior,
    step
):
    """One step of gradient update.

    :param loss (callable): loss function.
    :param params (jax.tree_util.pytree): parameters of the BNN.
    :param opt_state (jax.tree_util.pytree): optimizer state.
    :param get_params (jax.tree_util.pytree): function to get parameters.
    :param x_batch (jax.numpy.ndarray): a batch of input images.
    :param y_batch (jax.numpy.ndarray): a batch of labels.
    :param key (jax.random.PRNGKey): JAX random seed.
    :param opt_update (callable): optimizer update function.
    :param model (Model): NN model.
    :param ll_scale (float): scale of the likelihood model. 
    :param prior (callable): prior distribution.
    :param step (int): current step.

    :returns params (jax.tree_util.pytree): updated parameters.
    :returns opt_state (jax.tree_util.pytree): updated optimizer state.
    :returns other_info (dict): other information.
    """
    grads, other_info = jax.grad(loss, argnums=0, has_aux=True)(
        params,
        model,
        x_batch,
        y_batch,
        key,
        ll_scale, 
        prior
    )

    opt_state = opt_update(step, grads, opt_state)
    params = get_params(opt_state)

    return params, opt_state, other_info


@partial(jax.jit, static_argnums=(1,5,6))
def neg_log_posterior(
    params,
    model,
    x,
    y,
    key,
    ll_scale, 
    prior
):
    """MAP estimation.
    
    :param params (jax.tree_util.pytree): parameters of the mlp.
    :param model (Model): neural network.
    :param x (jax.numpy.ndarray): feature batch.
    :param y (jax.numpy.ndarray): label batch.
    :param key (jax.random.PRNGKey): random key.
    :param ll_scale (float): scale of the likelihood model. 
    :param prior (callable): prior distribution.
    :returns:
        neg_log_posterior (float): posterior mean.
        log_likelihood (float): log likelihood.
    """
    key1, key2, key3 = jax.random.split(key, num=3)

    # Log-likelihood
    f_hat = model.apply_fn(params, key1, x)
    log_likelihood = jsp.stats.norm.logpdf(
        y, 
        loc=f_hat, 
        scale=ll_scale
    ).sum()

    # Log-prior
    x_context = jax.random.uniform(
        key2, 
        shape=(100, x.shape[-1]), 
        minval=-2, 
        maxval=2
    )
    f_hat = model.apply_fn(params, key3, x_context)
    prior_mean, prior_cov = prior(x_context)
    log_prior = jsp.stats.multivariate_normal.logpdf(
        f_hat, 
        mean=prior_mean, 
        cov=prior_cov
    ).sum()

    # Log-posterior
    log_posterior = log_likelihood + log_prior

    return (
        -log_posterior,
        {"log_likelihood": log_likelihood, "log_posterior": log_posterior, "log_prior": log_prior},
    )


def evaluate_model(
    key, 
    model, 
    dataloader, 
    config
):
    """
    Evaluate the model on the test set.

    :param key (jax.random.PRNGKey): JAX random seed.
    :param model (Model): BNN model.
    :param dataloader (DataLoader): data loader.
    :param config (dict): configuration.
    """
    # Read configuration
    likelihood_scale = config["FSPLaplace"]["likelihood_scale"]

    # Load test data    
    X_test, y_test = dataloader.get_data("test")

    expec_log_likelihood = 0.
    for x, y in zip(X_test, y_test):
        # Handle keys
        key, key1 = jax.random.split(key)

        # prediction
        mean, diag_cov = model.f_pred_diag_dist(x, key1)
        expec_log_likelihood += jsp.stats.norm.logpdf(
            y, 
            loc=mean.reshape(-1, 1), 
            scale=likelihood_scale
        ).sum() 
        expec_log_likelihood -= 0.5 * diag_cov.sum() / likelihood_scale**2

    wandb.log(
        {
            "Test/expec_log_likelihood": expec_log_likelihood
        }
    )

    print(f"Expected log-likelihood: {expec_log_likelihood}", flush=True)
