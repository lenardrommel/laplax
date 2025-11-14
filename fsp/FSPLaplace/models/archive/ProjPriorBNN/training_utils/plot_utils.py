import jax
import wandb
import jax.numpy as jnp

import matplotlib.pyplot as plt


def plot_function_samples(
    model, 
    key, 
    config, 
    dataloader
):
    """Plot function samples from the model.

    :param model (Model): model.
    :param params (jax.tree_util.pytree): model parameters.
    :param key (jax.random.PRNGKey): random key.
    :param config (dict): model configuration.
    :param dataloader (dataloader.Dataloader): dataloader.
    """
    # Get config 
    dataset = config["data"]["name"]

    # Keys
    key, sub_key = jax.random.split(key)

    # Sample functions
    x = jnp.arange(-2, 2, 0.1).reshape(-1, 1)
    f_samples = model.f_predict(
        x, 
        sub_key, 
        mc_samples=50, 
        is_training=False
    )
    
    # Format input 
    x = x.reshape(-1)
    f_samples = jnp.squeeze(f_samples)
    pred_mean = jnp.squeeze(f_samples.mean(0))
    pred_var = jnp.squeeze(f_samples.var(0))

    # Plot train data
    X_train, y_train = dataloader.get_data(data_split="train")
    plt.scatter(
        X_train.reshape(-1), 
        y_train.reshape(-1), 
        label="Train data"
    )

    # Plot predictive mean
    plt.plot(
        x, 
        pred_mean, 
        label="Predictive mean", 
        c="r", 
        linewidth=2
    )

    # Plot predictive std dev
    plt.fill_between(
        x, 
        pred_mean-pred_var**0.5, 
        pred_mean+pred_var**0.5, 
        color="r", 
        alpha=0.3, 
        label="Predictive 1-std-dev"
    )

    # Plot individual mean functions
    for i in range(f_samples.shape[0]):
        label = "Predictive sample" if not i else ""
        plt.plot(x, f_samples[i].reshape(-1), label=label, c="g", alpha=0.2)

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.ylim(-5, 5)
    plt.savefig(
        f"proj_gp_prior_{config['proj_prior_bnn']['prior']['kernel']}_{dataset}.pdf", 
        bbox_inches='tight'
    )
    wandb.log({"sample_functions": wandb.Image(plt)})
