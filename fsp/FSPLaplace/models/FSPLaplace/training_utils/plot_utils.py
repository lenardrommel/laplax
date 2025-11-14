import jax
import wandb
import pickle

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter


def _plot_after_training(
        model,
        params,
        state,
        key1,
        config,
        dataloader
):
    X_train, y_train = dataloader.dataset.get_data()
    f_hat, new_state = model.apply_fn(params, state, key1, X_train, training=False)
    plt.scatter(
            X_train.reshape(-1), 
            y_train.reshape(-1), 
            marker="o",
            facecolors='lightgrey',
            edgecolors='dimgrey',
            linewidth=1,
            s=10
        )
    plt.plot(
            X_train.reshape(-1), 
            f_hat.reshape(-1), 
            c="#e41a1c", #"red",
            linewidth=2
        )
    plt.show()

    



def plot_function_samples(
    model, 
    key, 
    config, 
    dataloader
):
    """
    Plot function samples from the model.

    params:
    - model (Model): model.
    - key (jax.random.PRNGKey): random key.
    - config (dict): model configuration.
    - dataloader (dataloader.Dataloader): dataloader.
    """
    # Keys
    key, sub_key = jax.random.split(key)

    # Get config 
    dataset = config["data"]["name"]
    kernel = config["fsplaplace"]['prior']['kernel']
    alpha_eps = config["fsplaplace"]["prior"]["alpha_eps"]
    activation_fn = config["fsplaplace"]["neural_net"]["activation_fn"]

    if model.likelihood == "Gaussian":
        # Sample functions
        x = jnp.arange(-3, 3, 0.05).reshape(-1, 1)
        f_samples = model.predict_f(
            x, 
            sub_key, 
            mc_samples=100
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
        
        plt.ylim(-2, 2)
        # file_name = f"flin_laplace_{dataset}_{kernel}_{activation_fn}.pkl"
        # with open(file_name, "wb") as f:
        #     pickle.dump(f_samples, f)
    elif model.likelihood == "Categorical":
        # Build grid
        grid_len = 25
        x1 = jnp.linspace(-4, 4, grid_len)
        x2 = jnp.linspace(-4, 4, grid_len)
        x = jnp.meshgrid(x1, x2, indexing='ij')
        x = jnp.stack(x, axis=-1).reshape(-1, 2)

        # Sample functions
        y_samples = model.predict_y(
            x, 
            sub_key, 
            mc_samples=200
        )[:,:,1] # (n_samples, n_batch, 1)
        y_mean = y_samples.mean(0).reshape(grid_len, grid_len).T # (grid_len, grid_len)
        y_std = 2*y_samples.std(0).reshape(grid_len, grid_len).T # (grid_len, grid_len)

        # Plot the training points
        fig, (ax1, ax2) = plt.subplots(2)        
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        X_train, y_train = dataloader.dataset.get_data()
        ax1.scatter(
            X_train[:,0], 
            X_train[:,1], 
            c=y_train.reshape(-1), 
            cmap=cm_bright, 
            s=10, 
            marker="o",
            zorder=2,
            edgecolors='dimgrey'
        )
        ax2.scatter(
            X_train[:,0], 
            X_train[:,1], 
            c=y_train.reshape(-1), 
            cmap=cm_bright, 
            marker="o",
            linewidth=1,
            s=10,
            zorder=2,
            edgecolors='dimgrey'
        )
        
        # Plot mean probs
        cmap = plt.cm.get_cmap('RdBu', 21)    
        levels = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        im1 = ax1.contourf(x1, x2, y_mean, levels=levels, alpha=1, cmap=cmap)
        fmt = lambda x, pos: '{:.2}'.format(x)
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], format=FuncFormatter(fmt))
        cbar1.set_label(r'$\mathbb{E}[y | \mathcal{D}, x]$', rotation=270, labelpad=15)

        # Plot std probs
        cmap = plt.cm.get_cmap('binary', 20) 
        levels = np.linspace(0, round(y_std.max(), 10), 20)
        im2 = ax2.contourf(x1, x2, y_std, levels=levels, alpha=1, cmap=cmap)
        cbar2 = plt.colorbar(im2, ax=ax2, format=FuncFormatter(fmt))
        cbar2.set_label(r'$2\mathbb{\sigma}[y | \mathcal{D}, x]$', rotation=270, labelpad=15)
        
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-4, 4)
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-4, 4)

        file_name = f"flin_laplace_{dataset}_{kernel}_{activation_fn}_{alpha_eps}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(y_samples, f)

    # plt.savefig(
    #     f"flin_laplace_{dataset}_{kernel}_{activation_fn}_{alpha_eps}.pdf", 
    #     bbox_inches='tight'
    # )
    # wandb.log({"sample_functions": wandb.Image(plt)})
    
    plt.show()
    plt.close()
