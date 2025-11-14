import pickle 
from jax import config
import time
import optax
import gpjax as gpx
from dataclasses import dataclass, field
import jax.scipy as jsp
import jax 
import wandb
from jax import config
import jax.numpy as jnp
import numpy as np

from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_probability as tfp

from data_utils.dataset import Dataset
from data_utils.dataloader import DataLoader
from models.model import Model

key = jax.random.key(123)
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
colors = rcParams["axes.prop_cycle"].by_key()["color"]

config.update("jax_enable_x64", True)

# function to place data from csv into correct array shape
def prepare_data(df):
    pos = jnp.array([df["lon"], df["lat"]])
    vel = jnp.array([df["ubar"], df["vbar"]])
    # extract shape stored as 'metadata' in the test data
    try:
        shape = (int(df["shape"][1]), int(df["shape"][0]))  # shape = (34,16)
        return pos, vel, shape
    except KeyError:
        return pos, vel

def plot_data(
    pos_train, 
    vel_train,
    pos_test,
    vel_test    
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.quiver(
        pos_test[0],
        pos_test[1],
        vel_test[0],
        vel_test[1],
        color=colors[0],
        label="Ocean Current - Test",
        angles="xy",
        scale=10,
    )
    ax.quiver(
        pos_train[0],
        pos_train[1],
        vel_train[0],
        vel_train[1],
        color=colors[1],
        alpha=0.7,
        label="Drifter - Train",
        angles="xy",
        scale=10,
    )

    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
    )
    ax.legend(
        framealpha=0.0,
        ncols=2,
        fontsize="medium",
        bbox_to_anchor=(0.5, -0.3),
        loc="lower center",
    )
    plt.show()


def plot_vector_field(ax, dataset, **kwargs):
    ax.quiver(
        dataset.X[::2][:, 0],
        dataset.X[::2][:, 1],
        dataset.y[::2],
        dataset.y[1::2],
        **kwargs,
    )


def prepare_ax(ax, X, Y, title, **kwargs):
    ax.set(
        xlim=[X.min() - 0.1, X.max() + 0.1],
        ylim=[Y.min() + 0.1, Y.max() + 0.1],
        aspect="equal",
        title=title,
        ylabel="latitude",
        **kwargs,
    )


def residuals(dataset_latent, test_dataset):
    return jnp.sqrt(
        (dataset_latent.y[::2] - test_dataset.y[::2]) ** 2
        + (dataset_latent.y[1::2] - test_dataset.y[1::2]) ** 2
    )

def plot_fields(
    test_dataset, dataset_trajectory, dataset_latent, shape, name, scale=10
):
    X = test_dataset.X[:, 0][::2]
    Y = test_dataset.X[:, 1][::2]
    # make figure
    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.0), sharey=True)

    # ground truth
    plot_vector_field(
        ax[0],
        test_dataset,
        color=colors[0],
        label="Ocean Current",
        angles="xy",
        scale=scale,
    )
    plot_vector_field(
        ax[0],
        dataset_trajectory,
        color=colors[1],
        label="Drifter",
        angles="xy",
        scale=scale,
    )
    prepare_ax(ax[0], X, Y, "Ground Truth", xlabel="Longitude")

    # Latent estimate of vector field F
    plot_vector_field(ax[1], dataset_latent, color=colors[0], angles="xy", scale=scale)
    plot_vector_field(
        ax[1], dataset_trajectory, color=colors[1], angles="xy", scale=scale
    )
    prepare_ax(ax[1], X, Y, "GP Estimate", xlabel="Longitude")

    # residuals
    residuals_vel = jnp.flip(
        residuals(dataset_latent, test_dataset).reshape(shape), axis=0
    )
    im = ax[2].imshow(
        residuals_vel,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap="jet",
        vmin=0,
        vmax=1.0,
        interpolation="spline36",
    )
    plot_vector_field(
        ax[2], dataset_trajectory, color=colors[1], angles="xy", scale=scale
    )
    prepare_ax(ax[2], X, Y, "Residuals", xlabel="Longitude")
    fig.colorbar(im, fraction=0.027, pad=0.04, orientation="vertical")

    fig.legend(
        framealpha=0.0,
        ncols=2,
        fontsize="medium",
        bbox_to_anchor=(0.5, -0.03),
        loc="lower center",
    )
    plt.savefig(f"{name}_ocean_current.pdf")
    plt.show()


def compute_lpd(mean, std, vel_test):
    vel_query = jnp.column_stack((vel_test[0], vel_test[1])).flatten()
    normal = tfp.substrates.jax.distributions.Normal(loc=mean, scale=std)
    return jnp.sum(normal.log_prob(vel_query))
    

@dataclass
class HelmholtzKernel(gpx.kernels.AbstractKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    potential_kernel: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])
    )
    stream_kernel: gpx.kernels.AbstractKernel = field(
        default_factory=lambda: gpx.kernels.RBF(active_dims=[0, 1])
    )

    def __call__(self, X, Xp):
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        sign = (-1) ** (z + zp)

        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        potential_dvtve = -jnp.array(
            jax.hessian(self.potential_kernel)(X, Xp), dtype=jnp.float64
        )[z][zp]
        stream_dvtve = -jnp.array(
            jax.hessian(self.stream_kernel)(X, Xp), dtype=jnp.float64
        )[1 - z][1 - zp]

        return potential_dvtve + sign * stream_dvtve



def ocean_current_modeling(
    config
):
    # Load configuration
    batch_size = config["data"]["batch_size"]
    folds = config["data"]["k_folds"]

    # Set random key
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Loading in data
    gulf_data_train = pd.read_csv("../Data/gulfdata/gulfdata_train.csv")
    gulf_data_test = pd.read_csv("../Data/gulfdata/gulfdata_test.csv")

    pos_test, vel_test, shape = prepare_data(gulf_data_test)
    pos_train, vel_train = prepare_data(gulf_data_train)

    print("pos_test.shape", pos_test.shape)
    print("vel_test.shape", vel_test.shape)
    print("pos_train.shape", pos_train.shape)
    print("vel_train.shape", vel_train.shape)
    #plot_data(pos_train, vel_train, pos_test, vel_test)

    # Initialize wandb
    init_wandb(config)

    # ## Problem Setting
    # Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
    def label_position(data):
        # introduce alternating z label
        n_points = len(data[0])
        label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
        return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T

    # change vectors y -> Y by reshaping the velocity measurements
    def stack_velocity(data):
        return data.T.flatten().reshape(-1, 1)
    
    # Normalize data 
    x_mean = pos_train.mean(-1).reshape(2, -1)
    x_std = pos_train.std(-1).reshape(2, -1)
    y_mean = stack_velocity(vel_test).mean(0)
    y_std = stack_velocity(vel_test).std(0)
    print("x_mean", x_mean)
    print("x_std", x_std)
    print("y_mean", y_mean)
    print("y_std", y_std)

    ####### GP ################
    # # #  # label and place the training data into a Dataset object to be used by GPJax
    # train_dataset = Dataset(label_position(pos_train), stack_velocity(vel_train))

    # # we also require the testing data to be relabelled for later use, such that we can query the 2Nx2N GP at the test points
    # test_dataset = Dataset(label_position(pos_test), stack_velocity(vel_test))

    # mses, lls = [], []
    # for i in range(folds):
    #     key1, key2 = jax.random.split(key1)

    #     # Define the GP model
    #     prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=HelmholtzKernel())
    #     likelihood = gpx.likelihoods.Gaussian(
    #         num_datapoints=len(train_dataset), obs_noise=jnp.array([1.0e-3], dtype=jnp.float64)
    #     )
    #     helmholtz_posterior = prior * likelihood
        
    #     # Fit model
    #     objective = gpx.objectives.ConjugateMLL(negative=True)
    #     train_data = gpx.Dataset(label_position(pos_train), stack_velocity(vel_train))
    #     # Optimise to minimise the MLL
    #     opt_helmholtz_posterior, history = gpx.fit(
    #         model=helmholtz_posterior,
    #         objective=objective,
    #         train_data=train_data,
    #         optim=optax.adam(learning_rate=0.1),
    #         num_iters=1000, 
    #         key=key2
    #     )

    #     # Log
    #     for i, n_mll in enumerate(history):
    #         wandb.log({f"Train/n_mll": n_mll})
    #         if i % 100 == 0 or i == len(history) - 1:
    #             print(f"Train/n_mll: {n_mll}", flush=True)     

    #     # Evaluate
    #     latent = opt_helmholtz_posterior.predict(test_dataset.X, train_data=train_data)
    #     mean, std = latent.mean(), latent.stddev()
    #     dataset_latent_velocity = gpx.Dataset(label_position(pos_test), stack_velocity(mean))
    #     #plot_fields(test_dataset, train_dataset, dataset_latent_velocity, shape, name="gp")

    #     mse = jnp.mean((mean.reshape(-1) - stack_velocity(vel_test).reshape(-1)) ** 2)
    #     mses.append(mse)
    #     print(f"mse: {mse}", flush=True)

    #     ll_scale = opt_helmholtz_posterior.likelihood.obs_noise**0.5
    #     expected_ll = jsp.stats.norm.logpdf(
    #         stack_velocity(vel_test).reshape(-1), 
    #         loc=mean.reshape(-1), 
    #         scale=ll_scale
    #     ).sum() 
    #     expected_ll -= 0.5 * (std**2).sum() / ll_scale**2
    #     lls.append(expected_ll / np.prod(vel_test.shape))
    #     print(f"expected_ll: {expected_ll / np.prod(vel_test.shape)}", flush=True)
    #     del prior, likelihood, helmholtz_posterior, opt_helmholtz_posterior
    
    # print(f"mse: {np.mean(mses)} +/- {np.std(mses)}")
    # print(f"expected_ll: {np.mean(lls)} +/- {np.std(lls)}")

    # with open("GP_ocean_current.pkl", "wb") as f:
    #     pickle.dump({
    #         "X_test":test_dataset.X,
    #         "y_test":test_dataset.y, 
    #         "X_train":train_dataset.X,
    #         "y_train":train_dataset.y,
    #         "mean_velocity": stack_velocity(mean), 
    #         "std_dev_velocity": stack_velocity(std), 
    #         "shape":shape
    #     }, f)
    
    # ####### FSPLaplace ################
    #label and place the training data into a Dataset object to be used by GPJax
    config["model"]["name"] = "FSPLaplace"
    norm_train_dataset = Dataset(label_position((pos_train-x_mean)/x_std), (stack_velocity(vel_train) - y_mean) / y_std)
    norm_train_dataloader = DataLoader(key1, norm_train_dataset, batch_size, shuffle=True, replacement=False)

    mses, lls = [], []
    for i in range(folds):
        key1, key2, key3 = jax.random.split(key1, 3)

        # Load model
        print("Loading Functional Laplace model...", flush=True)
        model = Model(key2, config)

        # Fit model 
        print("Fitting the model...", flush=True)
        model.fit(norm_train_dataloader, norm_train_dataloader, norm_train_dataloader)

        # Plot the results
        print("Plotting the results...", flush=True)
        mean, var = model.f_distribution_mean_var(label_position((pos_test-x_mean)/x_std), key3, mc_samples=100)
        mean = mean * y_std + y_mean
        var = y_std**2 * var
        # PLot the fields
        train_dataset = Dataset(label_position(pos_train), stack_velocity(vel_train))
        test_dataset = Dataset(label_position(pos_test),  stack_velocity(vel_test))
        dataset_latent_velocity = Dataset(label_position(pos_test), stack_velocity(mean))
        plot_fields(test_dataset, train_dataset, dataset_latent_velocity, shape, name="fsp_laplace")

        mse = jnp.mean((mean.reshape(-1) - stack_velocity(vel_test).reshape(-1)) ** 2)
        mses.append(mse)
        print(f"mse: {mse}", flush=True)

        ll_scale = model.model.ll_scale
        expected_ll = jsp.stats.norm.logpdf(
            stack_velocity(vel_test).reshape(-1), 
            loc=mean.reshape(-1), 
            scale=ll_scale
        ).sum() 
        expected_ll -= 0.5 * var.sum() / ll_scale**2
        lls.append(expected_ll / np.prod(vel_test.shape))
        print(f"expected_ll: {expected_ll / np.prod(vel_test.shape)}", flush=True)

        del model
        jax.clear_caches()

    print(f"mse: {np.mean(mses)} +/- {np.std(mses)}")
    print(f"expected_ll: {np.mean(lls)} +/- {np.std(lls)}")

    with open("flaplace_ocean_current.pkl", "wb") as f:
        pickle.dump({
            "X_test":test_dataset.X,
            "y_test":test_dataset.y, 
            "X_train":train_dataset.X,
            "y_train":train_dataset.y,
            "mean_velocity": stack_velocity(mean), 
            "std_dev_velocity": stack_velocity(var**0.5), 
            "shape":shape
        }, f)

    # # Test 
    # f_samples = model.predict_f(label_position((pos_test-x_mean)/x_std), key4, mc_samples=100) # (100, 1088, 1)
    # f_samples = f_samples * y_std + y_mean
    # print("f_samples.shape", f_samples.shape)   
    # f_mean = f_samples.mean(0).reshape(-1)
    # f_std = (f_samples.var(0)**0.5).reshape(-1)
    # plt.plot(jnp.arange(0, 1088), f_mean, label="mean")
    # plt.plot(jnp.arange(0, 1088), stack_velocity(vel_test).reshape(-1), label="true")
    # plt.fill_between(jnp.arange(0, 1088), f_mean - 2 * f_std, f_mean + 2 * f_std, alpha=0.2, label="std")
    # plt.legend()
    # plt.savefig("flaplace_ocean_current_plt.pdf")
    # plt.show()

    ####### FVI ################
    # # label and place the training data into a Dataset object to be used by GPJax
    # config["model"]["name"] = "FVI"
    # norm_train_dataset = Dataset(label_position((pos_train-x_mean)/x_std), (stack_velocity(vel_train) - y_mean) / y_std)
    # norm_train_dataloader = DataLoader(key1, norm_train_dataset, batch_size, shuffle=True, replacement=False)

    # mses, lls = [], []
    # for i in range(folds):
    #     key1, key2, key3 = jax.random.split(key1, 3)

    #     # Load model
    #     print("Loading FVI model...", flush=True)
    #     model = Model(key2, config)

    #     # Fit model 
    #     print("Fitting the model...", flush=True)
    #     model.fit(norm_train_dataloader, norm_train_dataloader, norm_train_dataloader)

    #     # Plot the results
    #     print("Plotting the results...", flush=True)
    #     mean, var = model.f_distribution_mean_var(label_position((pos_test-x_mean)/x_std), key3, mc_samples=100)
    #     mean = mean * y_std + y_mean
    #     var = y_std**2 * var
    #     # PLot the fields
    #     train_dataset = Dataset(label_position(pos_train), stack_velocity(vel_train))
    #     test_dataset = Dataset(label_position(pos_test),  stack_velocity(vel_test))
    #     dataset_latent_velocity = Dataset(label_position(pos_test), stack_velocity(mean))
    #     plot_fields(test_dataset, train_dataset, dataset_latent_velocity, shape, name="FVI")

    #     f_hat = model.predict_f(label_position((pos_test-x_mean)/x_std), key3, mc_samples=100)
    #     f_hat = f_hat * y_std + y_mean
    #     mse = jnp.mean((f_hat.mean(0).reshape(-1) - stack_velocity(vel_test).reshape(-1)) ** 2)
    #     mses.append(mse)
    #     print(f"mse: {mse}", flush=True)

    #     ll_scale = model.model.ll_scale
    #     expected_ll = jsp.stats.norm.logpdf(
    #         stack_velocity(vel_test).reshape(-1), 
    #         loc=f_hat.reshape(100, -1), 
    #         scale=ll_scale
    #     ).mean(0).sum()
    #     lls.append(expected_ll / np.prod(vel_test.shape))
    #     print(f"expected_ll: {expected_ll / np.prod(vel_test.shape)}", flush=True)

    #     del model
    #     jax.clear_caches()

    # print(f"mse: {np.mean(mses)} +/- {np.std(mses)}")
    # print(f"expected_ll: {np.mean(lls)} +/- {np.std(lls)}")

    # with open("fvi_ocean_current.pkl", "wb") as f:
    #     pickle.dump({
    #         "X_test":test_dataset.X,
    #         "y_test":test_dataset.y, 
    #         "X_train":train_dataset.X,
    #         "y_train":train_dataset.y,
    #         "mean_velocity": stack_velocity(mean), 
    #         "std_dev_velocity": stack_velocity(var**0.5), 
    #         "shape":shape
    #     }, f)

    # # Test 
    # f_samples = model.predict_f(label_position((pos_test-x_mean)/x_std), key4, mc_samples=100) # (100, 1088, 1)
    # f_samples = f_samples * y_std + y_mean
    # print("f_samples.shape", f_samples.shape)   
    # f_mean = f_samples.mean(0).reshape(-1)
    # f_std = (f_samples.var(0)**0.5).reshape(-1)
    # plt.plot(jnp.arange(0, 1088), f_mean, label="mean")
    # plt.plot(jnp.arange(0, 1088), stack_velocity(vel_test).reshape(-1), label="true")
    # plt.fill_between(jnp.arange(0, 1088), f_mean - 2 * f_std, f_mean + 2 * f_std, alpha=0.2, label="std")
    # plt.legend()
    # plt.savefig("fvi_ocean_current_plt.pdf")
    # plt.show()

    # ###### Laplace approximation BNN ################
    # config["model"]["name"] = "Laplace"
    
    # norm_train_dataset = Dataset(label_position((pos_train-x_mean)/x_std), (stack_velocity(vel_train) - y_mean) / y_std)
    # norm_train_dataloader = DataLoader(key1, norm_train_dataset, batch_size, shuffle=True, replacement=False)
    
    # mses, lls = [], []
    # for i in range(folds):
    #     key1, key2, key3 = jax.random.split(key1, 3)

    #     config["laplace"]["training"]["pretrained_weights_path"] = f"checkpoints/ocean_current/laplace_kfac_{i+1}.pkl"
    #     config["laplace"]["inference"]["covariance_path"] = f"checkpoints/ocean_current/laplace_full_cov_sq_root_{i+1}.pkl"

    #     # Load model
    #     print("Loading Laplace model...", flush=True)
    #     model = Model(key2, config)

    #     # Fit model 
    #     print("Fitting the model...", flush=True)
    #     model.fit(norm_train_dataloader, norm_train_dataloader, norm_train_dataloader)

    #     try:
    #         # Plot the results
    #         print("Plotting the results...", flush=True)
    #         x = label_position((pos_test-x_mean)/x_std).reshape(64, 17, 3)
    #         mean, var = [], []
    #         for _x in x:
    #             _mean, _var = model.f_distribution_mean_var(_x, key3, mc_samples=100)
    #             _mean = _mean * y_std + y_mean
    #             _var = y_std**2 * _var
    #             mean.append(_mean)
    #             var.append(_var)
    #         mean = jnp.array(mean).reshape(-1, 1)
    #         var = jnp.array(var).reshape(-1, 1)
    #         # PLot the fields
    #         train_dataset = Dataset(label_position(pos_train), stack_velocity(vel_train))
    #         test_dataset = Dataset(label_position(pos_test),  stack_velocity(vel_test))
    #         dataset_latent_velocity = Dataset(label_position(pos_test), stack_velocity(mean))
    #         #plot_fields(test_dataset, train_dataset, dataset_latent_velocity, shape, name="laplace")

    #         mse = jnp.mean((mean.reshape(-1) - stack_velocity(vel_test).reshape(-1)) ** 2)
    #         mses.append(mse)
    #         print(f"mse: {mse}", flush=True)

    #         ll_scale = model.model.ll_scale
    #         expected_ll = jsp.stats.norm.logpdf(
    #             stack_velocity(vel_test).reshape(-1), 
    #             loc=mean.reshape(-1), 
    #             scale=ll_scale
    #         ).sum() 
    #         expected_ll -= 0.5 * var.sum() / ll_scale**2
    #         lls.append(expected_ll / np.prod(vel_test.shape))
    #         print(f"expected_ll: {expected_ll / np.prod(vel_test.shape)}", flush=True)
    #     except Exception as e:  
    #         print(e)
    #         pass
        
    #     del model
    #     jax.clear_caches()

    # print(f"mse: {np.mean(mses)} +/- {np.std(mses)}")
    # print(f"expected_ll: {np.mean(lls)} +/- {np.std(lls)}")
    # exit()


    # with open("laplace_ocean_current.pkl", "wb") as f:
    #     pickle.dump({
    #         "X_test":test_dataset.X,
    #         "y_test":test_dataset.y, 
    #         "X_train":train_dataset.X,
    #         "y_train":train_dataset.y,
    #         "mean_velocity": stack_velocity(mean), 
    #         "std_dev_velocity": stack_velocity(var**0.5), 
    #         "shape":shape
    #     }, f)

  
    # # Test 
    # f_samples = model.predict_f(label_position((pos_test-x_mean)/x_std), key4, mc_samples=100) # (100, 1088, 1)
    # f_samples = f_samples * y_std + y_mean
    # print("f_samples.shape", f_samples.shape)   
    # f_mean = f_samples.mean(0).reshape(-1)
    # f_std = (f_samples.var(0)**0.5).reshape(-1)
    # plt.plot(jnp.arange(0, 1088), f_mean, label="mean")
    # plt.plot(jnp.arange(0, 1088), stack_velocity(vel_test).reshape(-1), label="true")
    # plt.fill_between(jnp.arange(0, 1088), f_mean - 2 * f_std, f_mean + 2 * f_std, alpha=0.2, label="std")
    # plt.legend()
    # plt.savefig("laplace_ocean_current_plt.pdf")
    # plt.show()


def init_wandb(
    config
):
    """
    """
    # Initialize wandb
    wandb_init = False
    while not wandb_init:
        try:
            grp_name = f"ocean_current_{config['data']['name']}_{config['model']['name']}"
            wandb.init(
                project="flaplace",
                config=config, 
                group=grp_name,
                mode="offline"
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