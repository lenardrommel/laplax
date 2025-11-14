import jax 
import time
import wandb
import datetime

import numpy as np
import pandas as pd
import jax.scipy as jsp

from models.model import Model
from data_utils.dataset import Dataset
from data_utils.dataloader import DataLoader

import matplotlib.pyplot as plt




def mona_loa_experiments(
    config 
):
    """
    Run toy regression.

    params:
    - config (dict): configuration dictionary.
    """
    # Define random key 
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, num=4)

    batch_size = config["data"]["batch_size"]

    # Initialize wandb
    init_wandb(config)
    
    # Load data
    print("Loading data...", flush=True)

    co2 = pd.read_csv("../Data/co2_mona_loa.csv", sep=",",header=35, names=["year","month","day","decimal","average","ndays","1 year ago","10 years ago","increase since 1800"]) #fetch_openml(data_id=41187, as_frame=True)
    
    co2_data = co2# .frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "average"]].set_index("date")

    co2_data = co2_data[co2_data["average"] != -999.99]
    
    try:
        co2_data_resampled_monthly = co2_data.resample("ME")
    except ValueError:
        # pandas < 2.2 uses M instead of ME
        co2_data_resampled_monthly = co2_data.resample("M")

    co2_data = co2_data_resampled_monthly.mean().dropna(axis="index", how="any")
    X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["average"].to_numpy().reshape(-1, 1)
    print(f"X shape: {X.shape}, y shape: {y.shape}", flush=True)

    # Split data
    X_train, y_train = X[:int(0.7 * len(X))], y[:int(0.7 * len(y))]
    X_test, y_test = X[int(0.7 * len(X)):], y[int(0.7 * len(y)):]

    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    print(f"X_mean: {X_mean}, X_std: {X_std}, y_mean: {y_mean}, y_std: {y_std}", flush=True)

    # Build dataloaders
    lls, mses = [], []
    for i in range(config["data"]["k_folds"]):
        if config["model"]["name"] != "GP":
            key, key1, key2, key3 = jax.random.split(key, num=4)
            # Load model
            print("Loading model...", flush=True)
            model = Model(key1, config)

            # Parition data
            idx = np.random.permutation(len(X_train))
            train_idx, val_idx = idx[:int(0.9 * len(X_train))], idx[int(0.9 * len(X_train)):]
            _X_train, _y_train = X_train[train_idx], y_train[train_idx]
            _X_val, _y_val = X_train[val_idx], y_train[val_idx]

            # Build datasets
            train_dataset = Dataset((_X_train - X_mean) / X_std, (_y_train - y_mean) / y_std)
            val_dataset = Dataset((_X_val - X_mean) / X_std, (_y_val - y_mean) / y_std)

            train_loader = DataLoader(key2, train_dataset, batch_size, shuffle=True, replacement=False)
            val_loader = DataLoader(key3, val_dataset, batch_size, shuffle=True, replacement=False)
            
            # Fit model 
            print("Fitting the model...", flush=True)
            start_time = time.time()
            model.fit(train_loader, val_loader, train_loader)
            print(f"Training time: {time.time() - start_time}", flush=True)

            # Evaluate 
            print("Evaluating the model...", flush=True)
            if config["model"]["name"] == "FVI":
                f_hat = model.predict_f((X_test - X_mean) / X_std, key, mc_samples=100)
                f_hat = f_hat * y_std + y_mean
                mse = np.mean((f_hat.mean(0).reshape(-1) - y_test.reshape(-1)) ** 2)
                mses.append(mse)
                print(f"mse: {mse}", flush=True)

                ll_scale = model.model.ll_scale
                expected_ll = jsp.stats.norm.logpdf(
                    y_test.reshape(-1), 
                    loc=f_hat.reshape(100, -1), 
                    scale=ll_scale
                ).mean(0).sum()
                lls.append(expected_ll / np.prod(y_test.shape))
                print(f"expected_ll: {expected_ll /  np.prod(y_test.shape)}", flush=True)
            else:
                mean_y_pred, var_y_pred = model.f_distribution_mean_var((X_test - X_mean) / X_std, key, mc_samples=1)
                mean_y_pred = mean_y_pred.reshape(-1)*y_std + y_mean
                mses.append(np.mean((mean_y_pred - y_test.reshape(-1))**2))
                print(f"MSE: {np.mean((mean_y_pred - y_test.reshape(-1))**2)}", flush=True)

                expected_ll = jsp.stats.norm.logpdf(
                    y_test.reshape(-1), 
                    loc=mean_y_pred.reshape(-1), 
                    scale=model.model.ll_scale
                ).sum() 
                expected_ll -= 0.5 * var_y_pred.sum() / model.model.ll_scale**2
                lls.append(expected_ll / np.prod(y_test.shape))
                print(f"expected_ll: {expected_ll / np.prod(y_test.shape)}", flush=True)

            # Plot 
            print("Plot functions...", flush=True)
            today = datetime.datetime.now()
            current_month = today.year + today.month / 12
            X_plot = np.linspace(start=1974, stop=current_month, num=500).reshape(-1, 1)
            mean_y_pred, var_y_pred = model.f_distribution_mean_var((X_plot - X_mean) / X_std, key, mc_samples=1) 
            mean_y_pred = mean_y_pred.reshape(-1) * y_std + y_mean
            std_y_pred = var_y_pred.reshape(-1)**0.5 * y_std

            plt.plot(X_train, y_train, color="black", linestyle="dashed", label="Train measurements")
            plt.plot(X_test, y_test, color="red", linestyle="dashed", label="Test measurements")
            plt.plot(X_plot, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
            plt.fill_between(
                X_plot.ravel(),
                mean_y_pred - std_y_pred,
                mean_y_pred + std_y_pred,
                color="tab:blue",
                alpha=0.2,
            )
            plt.legend()
            plt.xlabel("Year")
            plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
            _ = plt.title(
                "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
            )
            plt.ylim(300, 460)
            plt.show()
            del model
            jax.clear_caches()

        else:
            from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, WhiteKernel
            from sklearn.gaussian_process import GaussianProcessRegressor

            co2_kernel = 66**2 * RBF(length_scale=67) + 2.4**2 * RBF(length_scale=90) * ExpSineSquared(length_scale=1.3, periodicity=1) + 0.66**2 * RationalQuadratic(alpha=0.78, length_scale=1.2) + 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.0361)
            
            gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False, n_restarts_optimizer=1)
            gaussian_process.fit(X_train, y_train  - y_mean)

            # Evaluate
            print("Evaluating the model...", flush=True)
            mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
            mean_y_pred = mean_y_pred.reshape(-1) + y_mean
            mses.append(np.mean((mean_y_pred - y_test.reshape(-1))**2))
            print(f"MSE: {np.mean((mean_y_pred - y_test.reshape(-1))**2)}", flush=True)

            ll_scale = gaussian_process.get_params()["kernel__k2__noise_leveld"]**0.5
            expected_ll = jsp.stats.norm.logpdf(
                y_test.reshape(-1), 
                loc=mean_y_pred.reshape(-1), 
                scale=ll_scale
            ).sum() 
            expected_ll -= 0.5 * (std_y_pred**2).sum() / ll_scale**2
            lls.append(expected_ll / np.prod(y_test.shape))
            print(f"expected_ll: {expected_ll / np.prod(y_test.shape)}", flush=True)

            # Plot 
            print("Plot functions...", flush=True)
            today = datetime.datetime.now()
            current_month = today.year + today.month / 12
            X_plot = np.linspace(start=1974, stop=current_month, num=500).reshape(-1, 1)
            mean_y_pred, std_y_pred = gaussian_process.predict(X_plot, return_std=True)
            mean_y_pred = mean_y_pred.reshape(-1) + y_mean

            plt.plot(X_train, y_train, color="black", linestyle="dashed", label="Train measurements")
            plt.plot(X_test, y_test, color="red", linestyle="dashed", label="Test measurements")
            plt.plot(X_plot, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
            plt.fill_between(
                X_plot.ravel(),
                mean_y_pred - std_y_pred,
                mean_y_pred + std_y_pred,
                color="tab:blue",
                alpha=0.2,
            )
            plt.legend()
            plt.xlabel("Year")
            plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
            _ = plt.title(
                "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
            )
            plt.ylim(300, 460)
            plt.show()

    print(f"Mean MSE: {np.mean(mses)} +/- {np.std(mses)}", flush=True)
    print(f"Mean expected_ll: {np.mean(lls)} +/- {np.std(lls)}", flush=True)

    plt.plot(X_train, y_train, color="black", linestyle="dashed", label="Train measurements")
    plt.plot(X_test, y_test, color="red", linestyle="dashed", label="Test measurements")
    plt.plot(X_plot, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
    plt.fill_between(
        X_plot.ravel(),
        mean_y_pred - std_y_pred,
        mean_y_pred + std_y_pred,
        color="tab:blue",
        alpha=0.2,
    )
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
    _ = plt.title(
        "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
    )
    plt.ylim(300, 460)
    plt.savefig("mauna_loa.pdf")
    plt.show()

    # Save results
    model_name = config["model"]["name"]
    np.savez(f"mauna_loa_{model_name}.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_plot=X_plot, mean_y_pred=mean_y_pred, std_y_pred=std_y_pred)

    # Close wandb
    wandb.finish()


def init_wandb(
    config
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            wandb.init(
                project="flaplace",
                config=config, 
                name=f'{config["model"]["name"]}_{config["data"]["name"]}', 
                mode="offline"
            )
            wandb_init = True
        except:
            time.sleep(10)