import gc
import jax 
import time
import wandb 
import pickle

import numpy as np 
import jax.numpy as jnp
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.model_selection import KFold
from data_utils.era5_data_utils import gcs_to_cartesian


from models.model import Model
from data_utils.era5_data_utils import (
    ERA5Dataset, 
    ERA5DataLoader, 
    ERA5PatchDataLoader
)


def era5_extrapolation_final_experiment(
    config, 
):
    """
    """
    # Define random key 
    key = jax.random.PRNGKey(0)

    # Load configuaration
    k_folds = config["data"]["k_folds"]
    batch_size = config["data"]["batch_size"]
    model_name = config["model"]["name"].lower()

    # Load ERA5 specific configuration
    ds_path = config["era5"]["ds_path"]
    t_idcs_max = config["era5"]["t_idcs_max"]
    subsample_rate = config["era5"]["subsample_rate"] # {"longitude": 4, "latitude": 4}
    n_test_time_steps = config["era5"]["n_test_time_steps"]

    # Initialize wandb
    init_wandb(config)
    
    # Update config
    if model_name != "gp":
        config[model_name]["neural_net"]["validation_freq"] = 1
    else:
        config["gp"]["training"]["validation_freq"] = 1
    config[model_name]["likelihood"]["model"] = "Gaussian"
    config[model_name]["training"]["patience"] = 10
    
    # Load data
    print("Loading ERA5 data...", flush=True)
    dataset = ERA5Dataset(
        ds_path=ds_path,
        t_idcs=slice(0,t_idcs_max, subsample_rate["time"]),
        step_long=subsample_rate["longitude"],
        step_lat=subsample_rate["latitude"]
    )
    train_loader = ERA5DataLoader(key, dataset, batch_size, shuffle=True, dataset_idx=jnp.arange(len(dataset)), normalize_labels=True)
    print("Length train_loader: ", len(train_loader), flush=True)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=1)

    # K-fold Cross Validation model evaluation
    results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold: {fold} / {k_folds}", flush=True)
        
        # Update config
        if model_name == "fsplaplace":
            config[model_name]["training"]["pretrained_weights_path"] = f"checkpoints/era5/flaplace_{fold+1}.pkl"
            config[model_name]["inference"]["covariance_path"] = f"checkpoints/era5/flaplace_low_rank_lanczos_cov_sq_root_{fold+1}.pkl"
        elif model_name == "laplace":
            cov_type = config["laplace"]["inference"]["cov_type"]
            config[model_name]["training"]["pretrained_weights_path"] = f"checkpoints/era5/laplace_{cov_type}_{fold+1}.pkl"
            config[model_name]["inference"]["covariance_path"] = f"checkpoints/era5/laplace_{cov_type}_cov_sq_root_{fold+1}.pkl"
        else:
            config["gp"]["training"]["model_path"] = f"checkpoints/era5/SVGP_ERA5Kernel_{fold+1}"
        
        # Load model
        print("Loading model...", flush=True)
        model = Model(key, config)

        # Fit model 
        print("Fitting the model...", flush=True)
        model.fit(train_loader, train_loader, train_loader)
        
        # Evaluate 
        print("Evaluating the model on [0, t_idcs_max]", flush=True)
        test_loss = model.evaluate(train_loader)

        # Plots 
        print("Plotting...", flush=True)
        #plot_era5_world(model, key3, test_loader, fold, model_name, time_idx=t_idcs_max+n_test_time_steps-1, save_fig=True)
        #plot_era5_time_series(model, key4, train_loader, test_loader, fold, config, save_fig=True)

        extrapolation_results = {0: test_loss}
        for t_extrapolation in range(24, n_test_time_steps+1, 24):
            print(f"Extrapolating {t_extrapolation // 24} days ahead", flush=True)
            test_dataset = ERA5Dataset(
                ds_path=ds_path,
                t_idcs=slice(t_idcs_max, t_idcs_max+t_extrapolation, subsample_rate["time"]),
                step_long=subsample_rate["longitude"],
                step_lat=subsample_rate["latitude"],
            )
            test_dataset.feature_stats = dataset.feature_stats
            test_loader = ERA5DataLoader(
                key, 
                test_dataset, 
                batch_size, 
                shuffle=False, 
                dataset_idx=jnp.arange(len(test_dataset)), 
                normalize_labels=True
            )
            test_loader.label_stats = train_loader.label_stats
            # Evaluate model on test set
            extrapolation_results[t_extrapolation] = model.evaluate(test_loader)
            del test_dataset, test_loader
            gc.collect()
            jax.clear_caches()

        # Save results
        results.append(extrapolation_results)
        
        del model
        gc.collect()
        jax.clear_caches()

    # Save results 
    save_model = model_name if model_name != "laplace" else "laplace" + "_" + config["laplace"]["inference"]["cov_type"]
    with open(f"era_results/{save_model}_era5_extrapolation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Close wandb
    wandb.finish()

def plot_era5_world(
    model,
    key,
    train_loader,
    it, 
    model_name, 
    time_idx, 
    save_fig=False
):
    # Build the figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"})    

    # Fetch data
    X = train_loader.dataset.X
    t2m = train_loader.dataset.y[:,:,time_idx] # long, lat, time_idx
    print("t2m.shape", t2m.shape)

    # Convert to cartesian coordinates
    xyzs = jax.vmap(lambda lon:
        jax.vmap(lambda lat: 
            gcs_to_cartesian(lon, lat, stats=train_loader.dataset.feature_stats)#.reshape(-1)
        )(X[1])
    )(X[0])

    # Plot ground truth
    print(t2m.min(), t2m.max())
    norm=Normalize(vmin=t2m.min(), vmax=t2m.max())
    my_col = cm.coolwarm(norm(t2m))
    surf = ax1.plot_surface(
        jnp.flip(jnp.concatenate([xyzs[:, :, 0], xyzs[0, :, 0].reshape(1, -1)], axis=0), 0), 
        jnp.flip(jnp.concatenate([xyzs[:, :, 1], xyzs[0, :, 1].reshape(1, -1)], axis=0), 0),
        jnp.flip(jnp.concatenate([xyzs[:, :, 2], xyzs[0, :, 2].reshape(1, -1)], axis=0), 0),
        facecolors=my_col,
        vmin=t2m.min(),
        vmax=t2m.max(),
        cmap=cm.coolwarm,
        linewidth=0, 
        antialiased=False
    )
    ax1.title.set_text('Ground truth')

    # Add a color bar which maps values to colors.
    ticks = np.linspace(round(t2m.min(), 2), round(t2m.max(), 2), 10)
    c1 = fig.colorbar(surf, shrink=0.25, aspect=20, ax=ax1)
    c1.set_ticks(
        ticks=ticks, 
        labels=[f"{s:.0f}" for s in ticks.tolist()]
    )

    # Predict
    time = (X[2][time_idx] - train_loader.dataset.feature_stats["t"]["mean"]) / train_loader.dataset.feature_stats["t"]["std"]
    x = jax.vmap(lambda lon:
        jax.vmap(lambda lat: jnp.concatenate([lon.reshape(-1, 1), lat.reshape(-1, 1), time.reshape(-1, 1)]))(X[1]) # long, lat
    )(X[0]).reshape(X[0].shape[0], X[1].shape[0], 3)
    print("x.shape", x.shape)

    if model_name == "gp":
        pred = jnp.concatenate([
            model.predict_f(_x, key, mc_samples=100) # (n_samples, n_points, 1)
            for _x in x
        ], axis=1)
    else:
        pred = model.predict_f(x.reshape(-1, 3), key, mc_samples=100)
    print("pred.shape", pred.shape)

    # Rescale the predictions
    pred = pred * train_loader.label_stats["std"] + train_loader.label_stats["mean"]

    # Plot mean
    mean_pred = jnp.mean(pred, axis=0).reshape(xyzs.shape[0], xyzs.shape[1])
    print("mean_pred.shape", mean_pred.shape)
    print(mean_pred.min(), mean_pred.max())
    #norm=Normalize(vmin=mean_pred.min(), vmax=mean_pred.max())
    my_col = cm.coolwarm(norm(mean_pred))

    surf = ax2.plot_surface(
        jnp.flip(jnp.concatenate([xyzs[:, :, 0], xyzs[0, :, 0].reshape(1, -1)], axis=0), 0), 
        jnp.flip(jnp.concatenate([xyzs[:, :, 1], xyzs[0, :, 1].reshape(1, -1)], axis=0), 0),
        jnp.flip(jnp.concatenate([xyzs[:, :, 2], xyzs[0, :, 2].reshape(1, -1)], axis=0), 0),
        facecolors=my_col,
        vmin=mean_pred.min(),
        vmax=mean_pred.max(),
        cmap=cm.coolwarm,
        linewidth=0, 
        antialiased=False
    )
    ax2.title.set_text('Mean prediction')

    # Add a color bar which maps values to colors.
    ticks = np.linspace(round(t2m.min(), 2), round(t2m.max(), 2), 10)
    c2 = fig.colorbar(surf, shrink=0.25, aspect=20, ax=ax2)
    c2.set_ticks(
        ticks=ticks, 
        labels=[f"{s:.0f}" for s in ticks.tolist()]
    )

    # Plot the std-dev
    std_pred = jnp.std(pred, axis=0).reshape(xyzs.shape[0], xyzs.shape[1])
    print("std_pred.shape", std_pred.shape)
    print(std_pred.min(), std_pred.max())
    norm=Normalize(vmin=std_pred.min(), vmax=std_pred.max())
    my_col = cm.Greys(norm(std_pred))

    surf = ax3.plot_surface(
        jnp.flip(jnp.concatenate([xyzs[:, :, 0], xyzs[0, :, 0].reshape(1, -1)], axis=0), 0), 
        jnp.flip(jnp.concatenate([xyzs[:, :, 1], xyzs[0, :, 1].reshape(1, -1)], axis=0), 0),
        jnp.flip(jnp.concatenate([xyzs[:, :, 2], xyzs[0, :, 2].reshape(1, -1)], axis=0), 0),
        facecolors=my_col,
        vmin=std_pred.min(),
        vmax=std_pred.max(),
        cmap=cm.Greys,
        linewidth=0, 
        antialiased=False
    )
    ax3.title.set_text('Standard deviation prediction')
    ticks = np.linspace(0, round(std_pred.max(), 2), 10)
    c3 = fig.colorbar(surf, shrink=0.25, aspect=20, ax=ax3)
    c3.set_ticks(
        ticks=ticks, 
        labels=[f"{s:.2f}" for s in ticks.tolist()]
    )

    plt.show()
    if save_fig:
        plt.savefig(f"{model_name}_era5_extrapolation_{it}.png")



def plot_era5_time_series(
    model,
    key,
    train_loader,
    test_loader,
    it, 
    config,
    save_fig=False,
):
    model_name = config["model"]["name"].lower()

    # Build the figure
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 6))    

    for i, _coord in enumerate([[0,0], [0,1], [0,2], [0,3], [0,4]]):
        # Fetch data
        long, lat = _coord
        
        # Fetch temperature data
        train_t2m = train_loader.dataset.y[long,lat,:].reshape(-1, 1)
        test_t2m = test_loader.dataset.y[long,lat,:].reshape(-1, 1)
        t2m = jnp.concatenate([train_t2m, test_t2m], axis=0)
        print("t2m.shape", t2m.shape)

        # Fetch time data
        time = jnp.concatenate([train_loader.dataset.X[2], test_loader.dataset.X[2]], axis=0)
        time = (time - train_loader.dataset.feature_stats["t"]["mean"]) / train_loader.dataset.feature_stats["t"]["std"]

        # Fetch features
        longitude = train_loader.dataset.X[0][long].reshape(-1, 1)
        latitude = train_loader.dataset.X[1][lat].reshape(-1, 1)        
        x = jnp.concatenate(
            (
                jnp.broadcast_to(longitude, (time.shape[0], 1)),
                jnp.broadcast_to(latitude, (time.shape[0], 1)),
                time.reshape(time.shape[0], 1)
            ),
            axis=-1,
        )

        # Predict
        f_samples = model.predict_f(x.reshape(-1, 3), key, mc_samples=50)
        print("f_samples.shape", f_samples.shape)

        # Rescale the predictions
        f_samples = f_samples*train_loader.label_stats["std"] + train_loader.label_stats["mean"]

        f_samples = jnp.squeeze(f_samples)
        pred_mean = jnp.squeeze(f_samples.mean(0))
        pred_std = jnp.squeeze(f_samples.std(0))

        for j in range(0, f_samples.shape[0], 10):
            axs[i].plot(f_samples[j,:], c="blue", alpha=0.3)

        # Plot mean
        print("pred_mean.shape", pred_mean.shape)
        axs[i].plot(t2m, label=f"T2m")
        axs[i].plot(pred_mean, label=f"Mean prediction", c="#e41a1c")

        # Plot predictive std dev
        axs[i].fill_between(
            jnp.arange(t2m.shape[0]), 
            pred_mean-2*pred_std, 
            pred_mean+2*pred_std, 
            color="#2ca02c",
            alpha=0.2
        )

        _, _, time_idx = jnp.unravel_index(train_loader.dataset_idx, train_loader.dataset.y.shape)
        x = train_loader.dataset.X[2][time_idx]
        y = train_loader.dataset.y[long, lat, time_idx]
        axs[i].scatter(x.reshape(-1), y.reshape(-1), c="black", s=5, label="Training data")
        axs[i].set_xticks(ticks=jnp.arange(0, len(t2m), 24), labels=[f"{i // (24)}" for i in jnp.arange(0, len(t2m), 24)])
        axs[i].set_xlabel("Time (days)")
        axs[i].set_ylabel("Temperature (°C)")
        axs[i].set_title(f"Temperature at coordinate [{long}, {lat}]")
        axs[i].set_xlim(0, len(t2m))

    plt.legend()

    if save_fig:
        plt.savefig(f"{model_name}_t2m_extrapolation_{long}_{lat}_it_{it}.pdf")
    
    plt.show()



def init_wandb(
    config
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            grp_name = f"era5_extra_{config['data']['name']}_{config['model']['name']}"
            mode = "online" if config["experiment"]["name"] == "hpo_era5_extrapolation" else "offline"
            wandb.init(
                project="flaplace",
                config=config, 
                group=grp_name,
                mode=mode
            )
            wandb_init = True
        except wandb.errors.UsageError:
            time.sleep(5)
        except wandb.errors.CommError:
            time.sleep(5)
        except wandb.errors.WaitTimeoutError:
            time.sleep(5)            
        except Exception as e:
            print(e, flush=True)
            time.sleep(30)

