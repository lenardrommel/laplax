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
    pred_std = jnp.squeeze(f_samples.std(0))

    # Plot train data
    X_train, y_train = dataloader.dataset.get_data()
    plt.scatter(
        X_train.reshape(-1), 
        y_train.reshape(-1), 
        marker="o",
        facecolors='lightgrey',
        edgecolors='dimgrey',
        linewidth=1,
        s=10
    )

    # Plot predictive mean
    plt.plot(
        x, 
        pred_mean, 
        c="#e41a1c", #"red",
        linewidth=2
    )

    # Plot predictive std dev
    plt.fill_between(
        x, 
        pred_mean-2*pred_std, 
        pred_mean+2*pred_std, 
        color="#2ca02c",
        alpha=0.2
    )

    # Plot individual mean functions
    for i in range(f_samples.shape[0]):
        plt.plot(
            x, 
            f_samples[i].reshape(-1),
            c="#2ca02c", #"forestgreen",
            alpha=0.5,
            linewidth=1
        )
    
    plt.ylim(-3, 3)
    cov_type = config["FSPLaplace"]["cov_type"]
    kernel = config['FSPLaplace']['prior']['kernel']
    dataset = config["data"]["name"]
    activation_fn = config["FSPLaplace"]["activation_fn"]
    context_points = config["FSPLaplace"]["training"]["n_context_points"]
    plt.savefig(
        f"flin_sampling_laplace_{cov_type}_{kernel}_{context_points}_{activation_fn}_{dataset}.pdf", 
        bbox_inches='tight'
    )
    wandb.log({"sample_functions": wandb.Image(plt)})
    plt.show()
    plt.close()

