import os 
import jax
import copy
import wandb
import torch
import gpytorch

import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp

from pathlib import Path

from gpytorch.constraints import Positive
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.likelihoods import (
    GaussianLikelihood,
    SoftmaxLikelihood
)

from models.GP.plot_utils import plot_function_samples

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, y, likelihood, mean_fn, kernel):
        super(ExactGPModel, self).__init__(train_x, y, likelihood)
        self.mean_module = mean_fn
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_fn, kernel):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = mean_fn
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



class SVGPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, n_outputs, mean_fn, kernel):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([n_outputs]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=n_outputs,
        )
        super(SVGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = mean_fn
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    

class ERA5Kernel(gpytorch.kernels.Kernel):
    is_stationary = False
    def __init__(self, batch_shape=torch.Size(), **kwargs):
        super().__init__(**kwargs)
        self.period = 24.
        # Define parameters of the temporal kernel 1
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1))
        )
        # register the constraint
        self.register_constraint("raw_length", Positive())

        # Kernel for equator
        self.kernel_equator = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=1, 
            batch_shape=batch_shape, 
            ard_num_dims=1
        )

        # Define temporal kernel 2 
        self.k_temporal1 = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=[2],
            batch_shape=batch_shape, 
            ard_num_dims=1
        )

        self.k_temporal2 = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=[2],
            batch_shape=batch_shape, 
            ard_num_dims=1
        )

        # Define spatial kernel 
        self.k_spatial1 = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=[0,1], 
            batch_shape=batch_shape,
            ard_num_dims=2
        )

        self.k_spatial2 = gpytorch.kernels.MaternKernel(
            nu=2.5,
            active_dims=[0,1], 
            batch_shape=batch_shape,
            ard_num_dims=2
        )

    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    # this is the kernel function
    def forward(self, x1, x2, diag, **params):
        # 
        if diag:
            delta_t = torch.zeros_like(x1[..., 2]).reshape(-1).to(x1)
            ones = torch.ones((x1.shape[0])).to(x1)
        else:
            delta_t = x1[..., 2].reshape(-1, 1) - x2[..., 2]
            ones = torch.ones((x1.shape[0], x2.shape[0])).to(x1)

        # Kernel for equator - periodic signal 
        k1 = 0.5 * torch.exp(
            -0.5 * torch.sin(torch.pi * torch.abs(delta_t) / self.period)**2 / self.length**2
        ) * self.k_temporal1(
            x1, x2, diag, **params
        ) * self.k_spatial1(
              x1, x2, diag, **params
        ) * self.kernel_equator(
             torch.zeros_like(x1).to(x1), x2, diag, **params
        ) * self.kernel_equator(
             x1, torch.zeros_like(x2).to(x2), diag, **params
        ) 

        # Kernel for poles - non-periodic signal
        k2 = 0.5 * self.k_temporal2(
            x1, x2, diag, **params
        ) * self.k_spatial2(
            x1, x2, diag, **params
        ) * (
            ones - self.kernel_equator(torch.zeros_like(x1).to(x1), x2, diag, **params)
        ) * (
            ones -self.kernel_equator(x1, torch.zeros_like(x2).to(x2), diag, **params)
        )
        
        return k1 + k2
    

class GP:
    """
    Placeholder for the GP model.
    """
    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the GP model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        self.key = key
        self.config = copy.deepcopy(config)
        torch.set_default_dtype(torch.float64)

        

        # Likelihood model
        self.likelihood_model = self.config["gp"]["likelihood"]["model"]
        if self.likelihood_model == "Gaussian":
            self.n_outputs = 1
            self.scale = self.config["gp"]["likelihood"]["scale"]
        else:
            self.n_outputs = self.config["gp"]["likelihood"]["n_classes"]
        
        # Posterior parameters
        self.posterior_model = self.config["gp"]["posterior"]["type"]
        self.n_inducing_pts = self.config["gp"]["posterior"]["n_inducing_pts"]
        if self.posterior_model == "GP" and self.likelihood_model == "Categorical":
            raise Exception("GP posterior is not compatible with categorical likelihood.")

        # Kernel parameters
        self.kernel_name = self.config["gp"]["prior"]["kernel"]
        self.kernel_params = self.config["gp"]["prior"]["params"]

        # Optimization parameters
        self.lr = self.config["gp"]["training"]["lr"]
        self.n_epochs = self.config["gp"]["training"]["nb_epochs"]
        self.model_path = self.config["gp"]["training"]["model_path"]
        self.validation_freq = self.config["gp"]["training"]["validation_freq"]
        self.early_stopping_patience = self.config["gp"]["training"]["early_stopping_patience"]

        # Save model
        self.save_model = self.config["gp"]["training"]["save_model"]
        if self.save_model:
            counter = 1
            dir_path = f"checkpoints/{self.config['data']['name']}"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = dir_path + f"/{self.posterior_model}_{self.kernel_name}_{counter}"
            while os.path.exists(self.checkpoint_path):
                self.checkpoint_path = dir_path + f"/{self.posterior_model}_{self.kernel_name}_{counter}"
                counter += 1
            print(f"Checkpoint path for GP: {self.checkpoint_path}", flush=True)

        
     
    def fit(
        self, 
        train_dataloader, 
        val_dataloader, 
        prior_dataloader
    ):
        """
        Fit the model.
        
        params:
        - train_dataloader (DataLoader): train dataloader.
        - val_dataloader (DataLoader): validation dataloader.

        returns:
        - loss (dict): validation loss.
        """
        # Define the GP kernel
        batch_shape = None if self.likelihood_model == "Gaussian" else torch.Size([self.n_outputs])
        if self.kernel_name == "RBF":
            self.kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            )
        elif self.kernel_name == "Matern12":
            self.kernel = gpytorch.kernels.MaternKernel(
                nu=0.5,
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            )
        elif self.kernel_name == "Matern32":
            self.kernel = gpytorch.kernels.MaternKernel(
                nu=1.5,
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            )
        elif self.kernel_name == "Matern52":
            self.kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            )
        elif self.kernel_name == "RationalQuadratic":
            self.kernel = gpytorch.kernels.RQKernel(
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            )
        elif self.kernel_name == "ERA5Kernel":
            self.kernel = ERA5Kernel(
                batch_shape=batch_shape
            )
        elif self.kernel_name == "Periodic":
            self.kernel = gpytorch.kernels.PeriodicKernel(
                ard_num_dims=train_dataloader.feature_dim, 
                batch_shape=batch_shape
            ) * gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_dataloader.feature_dim,
                batch_shape=batch_shape
            )
        elif self.kernel_name == "MonaLoaKernel":
            # long-term seasonal trend
            k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=1, 
                batch_shape=batch_shape
            ))
            # Seasonal kernel 
            _k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(
                ard_num_dims=1,
                batch_shape=batch_shape
            ))
            _k2 = gpytorch.kernels.RBFKernel(
                ard_num_dims=1,
                batch_shape=batch_shape
            )
            k2 = _k1 * _k2
            # Irregular trend
            k3 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(
                ard_num_dims=1,
                batch_shape=batch_shape
            ))
            # Noise kernel
            k4 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=1,
                batch_shape=batch_shape)
            )
            # Sum kernel
            self.kernel = k1 + k2 + k3 + k4
        else:
            raise Exception("Unknown kernel")
        
        self.kernel = gpytorch.kernels.ScaleKernel(
            self.kernel, 
            batch_shape=batch_shape
        )
    
        # Set kernel hyperparameters
        if self.kernel_params:
            if self.kernel_name == "ERA5Kernel":
                time_std = train_dataloader.dataset.time_std.item()
                print(f"time_std: {time_std}", flush=True)
                ##### New
                # Equator kernel
                self.kernel.base_kernel.period = 24 / time_std
                self.kernel.base_kernel.length = 24*6 / time_std
                self.kernel.base_kernel.k_temporal1.lengthscale =  24 * 6 / time_std
                self.kernel.base_kernel.kernel_equator.lengthscale = 0.25 * 5 
                self.kernel.base_kernel.k_spatial1.lengthscale = torch.FloatTensor([0.25, 0.25])
                # Pole kernel
                self.kernel.base_kernel.k_temporal2.lengthscale = 12 / time_std
                self.kernel.base_kernel.k_spatial2.lengthscale = torch.FloatTensor([0.25, 0.25])
                self.kernel.outputscale = 1.
            elif self.kernel_name == "MonaLoaKernel":
                X_std = 12.585720395510304 # dataloader.dataset.X.std()
                y_std = 17.05232264183159 #dataloader.dataset.y.std()
                # Long-term seasonal trend
                self.kernel.base_kernel.kernels[0].base_kernel.lengthscale = 50.0 / X_std
                self.kernel.base_kernel.kernels[0].outputscale = 50.0 / y_std
                # Seasonal kernel
                # Periodic kernel
                self.kernel.base_kernel.kernels[1].kernels[0].base_kernel.lengthscale = 1.0 / X_std
                self.kernel.base_kernel.kernels[1].kernels[0].outputscale = 2.0  / y_std
                self.kernel.base_kernel.kernels[1].kernels[0].base_kernel.period = 1.0 / X_std
                # RBF kernel
                self.kernel.base_kernel.kernels[1].kernels[1].lengthscale = 100.0 / X_std
                # Irregular trend
                self.kernel.base_kernel.kernels[2].base_kernel.lengthscale  = 1.0 / X_std
                self.kernel.base_kernel.kernels[2].base_kernel.alpha = 1.0
                self.kernel.base_kernel.kernels[2].outputscale = 0.5 / y_std
                # Noise kernel
                self.kernel.base_kernel.kernels[3].outputscale = 0.1 / y_std
                self.kernel.base_kernel.kernels[3].base_kernel.lengthscale = 0.1 / X_std
                self.kernel.base_kernel.outputscale = 1.0
                self.scale = 0.01 / y_std
            elif self.kernel_name == "Periodic":
                self.kernel.base_kernel.kernels[0].period = 1
                self.kernel.base_kernel.kernels[1].lengthscale = 4
            else:
                self.kernel.base_kernel.lengthscale = self.kernel_params["lengthscale"] #torch.FloatTensor([10, 10, 10, 3/24])
                self.kernel.outputscale = self.kernel_params["variance"]**0.5 #100.


        # Define the GP mean function
        batch_shape = torch.Size() if self.likelihood_model == "Gaussian" else torch.Size([self.n_outputs])
        self.mean_fn = gpytorch.means.ZeroMean(batch_shape=batch_shape)

        if self.posterior_model == "GP":
            # Only compatible with Gaussian likelihood
            self.likelihood = GaussianLikelihood()
            
            # Get data
            if self.config["experiment"]["name"] in ["era5_interpolation", "hpo_era5_interpolation", "hpo_era5_extrapolation", "era5_extrapolation"]:
                X, y = train_dataloader.dataset[train_dataloader.dataset_idx]
                y = (y - train_dataloader.label_stats["mean"].item()) / train_dataloader.label_stats["std"].item()
                self.likelihood.noise = self.scale / train_dataloader.label_stats["std"].item()
            else:   
                n_train = len(train_dataloader.dataset)
                X, y = train_dataloader.dataset[jnp.arange(n_train)]
                self.likelihood.noise = self.scale

            X = self._jax_array_to_torch_tensor(X)
            y = self._jax_array_to_torch_tensor(y).reshape(-1)

            # Define GP model
            self.model = ExactGPModel(
                None, #X, 
                None, #y, 
                self.likelihood, 
                self.mean_fn, 
                self.kernel
            )

            if self.model_path:
                print(f"Model loaded from {self.model_path}", flush=True)
                state_dict = torch.load(self.model_path)
                self.model.load_state_dict(state_dict)

            ############################
            # self.model.eval()
            # import matplotlib.pyplot as plt
            # x = torch.linspace(-2, 2, 400).reshape(-1, 1)
            # output = self.model(x).sample(sample_shape=torch.Size((20,)))
            # for i in range(20):
            #     plt.plot(x.numpy(), output[i, :].numpy())
            # plt.show()          
            # import pickle 
            # with open(f"{self.kernel_name}.pkl", "wb") as f:
            #     pickle.dump(output, f)

            # exit()
            #######################


            # Fit model 
            self.model.train()
            self.likelihood.train()

            # Use GPU if available
            if torch.cuda.is_available():
                print("GPU is available", flush=True)
                X, y = X.cuda(), y.cuda()
                self.model = self.model.cuda()
                self.likelihood = self.likelihood.cuda()

            # Define objective
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, 
                self.model
            )

            # Define optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

            # Training loop
            optimal_mll, no_improve_count = jnp.inf, 0
            for i in range(self.n_epochs):
                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()
                if i % self.validation_freq == 0:
                    print(f'Epoch {i} - Loss: {loss.item():.3f}')
                # Early stopping
                if loss.item() < optimal_mll:
                    optimal_mll = loss.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.early_stopping_patience:
                        print("Early stopping.", flush=True)
                        break

            # import matplotlib.pyplot as plt
            # def autocorr(x):
            #     x = x - x.mean()
            #     result = np.correlate(x, x, mode='full')
            #     return result[result.size // 2:]
            # # Kernel 
            # time = (train_dataloader.dataset.X[2][:, None] - train_dataloader.dataset.feature_stats["t"]["mean"]) / train_dataloader.dataset.feature_stats["t"]["std"]
            # longitude = train_dataloader.dataset.X[0][2].reshape(-1, 1)
            # latitude = train_dataloader.dataset.X[1][2].reshape(-1, 1)        
            # x = jnp.concatenate(
            #     (
            #         jnp.broadcast_to(longitude, (time.shape[0], 1)),
            #         jnp.broadcast_to(latitude, (time.shape[0], 1)),
            #         time.reshape(time.shape[0], 1)
            #     ),
            #     axis=-1,
            # ).reshape(-1, 3)
            
            # y = (train_dataloader.dataset.y[2, 2,:] - train_dataloader.label_stats["mean"].item()) / train_dataloader.label_stats["std"].item()
            # x = self._jax_array_to_torch_tensor(x).reshape(-1, 3)
            # acorr = autocorr(y.reshape(-1))
            # plt.plot(np.arange(0, acorr.shape[0]), self.kernel.forward(x1=x[0,:].reshape(-1, 3), x2=x).detach().numpy().reshape(-1), label="Kernel")
            # # Ground truth
            # plt.plot(np.arange(0, acorr.shape[0]), acorr, label="GT")
            # plt.legend()
            # plt.show()

            
        elif self.posterior_model == "SVGP":
            n_inducing_pts = min(
                self.n_inducing_pts, 
                len(train_dataloader.dataset)
            )
            # Initialize inducing points
            if self.config["experiment"]["name"] in ["era5_interpolation", "hpo_era5_interpolation", "hpo_era5_extrapolation", "era5_extrapolation"]:
                indices = jax.random.choice(self.key, train_dataloader.dataset_idx, (n_inducing_pts,), replace=False)
            else:
                indices = jax.random.choice(self.key, jnp.arange(len(train_dataloader.dataset)), (n_inducing_pts,), replace=False)
            inducing_points, _ = train_dataloader.dataset[indices]
            inducing_points = self._jax_array_to_torch_tensor(inducing_points).reshape(n_inducing_pts, -1)

            #inducing_points = X[indices, :]
            if self.likelihood_model == "Categorical":
                inducing_points = inducing_points.reshape(1, n_inducing_pts, -1).repeat(self.n_outputs, 1, 1)
            print("inducing_points.size()", inducing_points.size())

            if self.likelihood_model == "Gaussian":
                self.likelihood = GaussianLikelihood()
                self.likelihood.noise = self.scale
                self.model = SVGPRegressionModel(inducing_points, self.mean_fn, self.kernel)
            elif self.likelihood_model == "Categorical":
                self.likelihood = SoftmaxLikelihood(mixing_weights=None, num_classes=self.n_outputs)
                self.model = SVGPClassificationModel(
                    inducing_points,
                    self.n_outputs,
                    self.mean_fn,
                    self.kernel
                )

            if self.model_path:
                print(f"Model loaded from {self.model_path}", flush=True)
                state_dict = torch.load(self.model_path)
                self.model.load_state_dict(state_dict)

            # Fit model 
            self.model.train()
            self.likelihood.train()
            
            # Use GPU if available
            if torch.cuda.is_available():
                print("GPU is available", flush=True)
                self.model = self.model.cuda()
                self.likelihood = self.likelihood.cuda()
            
            # Define loss 
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(train_dataloader.dataset))

            # Define optimizer
            optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.likelihood.parameters()), 
                self.lr
            )    
            
            optimal_mll, no_improve_count = jnp.inf, 0
            for i in range(self.n_epochs):
                train_loss = 0.
                for x, y in train_dataloader:
                    x = self._jax_array_to_torch_tensor(x).reshape(x.shape[0], -1)
                    y = self._jax_array_to_torch_tensor(y).reshape(-1)
                    optimizer.zero_grad()
                    output = self.model(x)
                    loss = -mll(output, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                if i % self.validation_freq == 0:
                    print(f'Epoch {i} - Loss: {train_loss:.3f}')
                # Early stopping
                if train_loss < optimal_mll:
                    optimal_mll = train_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.early_stopping_patience:
                        print("Early stopping.", flush=True)
                        break

        if self.save_model:
            torch.save(self.model.state_dict(), self.checkpoint_path)
            print(f"Model saved at {self.checkpoint_path}", flush=True)
        # for name, param in self.model.named_parameters():
        #     print(name, torch.nn.Softplus()(param.data))
            
        # Evaluate the model
        out = self.evaluate(val_dataloader)

        return out
    
    def evaluate(
        self, 
        dataloader, 
        mc_samples=100
    ):
        """
        Evaluate the model.

        params:
        - dataloader (DataLoader): dataloader.

        returns:
        - test_loss (dict): test loss.
        """
        assert dataloader.replacement == False, "Data should be sampled without replacement"

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        if self.likelihood_model == "Gaussian":
            # Get ll scale
            ll_scale = self.likelihood.noise.item()**0.5
            # Evaluate the model
            expected_ll, mse = 0, 0
            for x, y in dataloader:
                # Function distribution
                f_preds = self.predict_f(x, key=None, mc_samples=mc_samples)
                expected_ll += jsp.stats.norm.logpdf(
                    y, 
                    loc=f_preds, 
                    scale=ll_scale
                ).mean(0).sum() 
                # Mean squared error
                mse += jnp.sum((f_preds.mean(0).reshape(-1) - y.reshape(-1))**2)  
            mse /= len(dataloader.dataset)
            expected_ll /= len(dataloader.dataset)    
            out = {"expected_ll": expected_ll, "mse": mse}
            wandb.log({"Test/expected_ll": expected_ll, "Test/mse": mse})
            print(f"Expected log-likelihood: {expected_ll} - MSE: {mse}", flush=True)
        elif self.likelihood_model == "Categorical":
            # Load test data  
            y_one_hot_list, probs_list = [], []  
            expected_ll, acc = 0., 0.
            for x, y in dataloader:
                # Prediction
                f = self.predict_f(x, key=None, mc_samples=mc_samples) # (n_samples, n_batch, n_classes)
                probs = jax.nn.softmax(f, axis=-1)
                one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=probs.shape[-1])
                expected_ll += jnp.mean(
                    jnp.sum(
                        one_hot_y * jnp.log(probs), # (n_samples, n_batch, self.n_outputs)
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
    

    def predict_f(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from function distribution.

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - f_samples (jnp.ndarray): function samples.
        """
        # Reshape data
        batch_size = x.shape[0]
        
        # Convert to torch tensors
        x = self._jax_array_to_torch_tensor(x).reshape(batch_size, -1)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Sample from function distribution
        with torch.no_grad():
            f_dist = self.model(x)
            f_sample = f_dist.sample(sample_shape=torch.Size((mc_samples,))).cpu()
        
        f_sample = self._torch_tensor_to_jax_array(f_sample)
            
        return f_sample.reshape(mc_samples, batch_size, -1)

    
    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from predictive distribution.

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y_samples (jnp.ndarray): function samples.
        """
        # Reshape data
        batch_size = x.shape[0]
        
        # Convert to torch tensors
        x = self._jax_array_to_torch_tensor(x).reshape(batch_size, -1)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Sample from function distribution
        with torch.no_grad():
            if self.likelihood == "Gaussian":
                y_dist = self.likelihood(self.model(x))
                y = y_dist.sample(sample_shape=torch.Size(mc_samples,)).cpu()
            else:
                f_dist = self.model(x)
                f_sample = f_dist.sample(sample_shape=torch.Size((mc_samples,)))
                y = torch.nn.functional.softmax(f_sample, dim=-1)

        y = self._torch_tensor_to_jax_array(y)

        return y.reshape(mc_samples, batch_size, -1)


    def f_distribution_mean_cov(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and covariance the function distribution. 

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): dummy argument.
        - mc_samples (int): dummy argument.
        
        returns:
        - mean (jnp.ndarray): mean of the function distribution.
        - cov (jnp.ndarray): covariance of the function distribution.
        """
        # Reshape data
        batch_size = x.shape[0]
        
        # Convert to torch tensors
        x = self._jax_array_to_torch_tensor(x).reshape(batch_size, -1)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Compute mean and covariance of function distribution
        with torch.no_grad():
            f_preds = self.model(x)
            f_mean = f_preds.mean.cpu()
            f_cov = f_preds.covariance_matrix.cpu()
            # Convert to JAX arrays
            f_mean = self._torch_tensor_to_jax_array(f_mean)
            f_cov = self._torch_tensor_to_jax_array(f_cov)

        return f_mean.reshape(batch_size, -1), f_cov.reshape(batch_size, batch_size, -1)
    

    def y_distribution_torch(
        self,
        x
    ):
        """
        Return the mean and covariance the function distribution. 

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): dummy argument.
        - mc_samples (int): dummy argument.
        
        returns:
        - mean (jnp.ndarray): mean of the function distribution.
        - cov (jnp.ndarray): covariance of the function distribution.
        """
        # Reshape data
        batch_size = x.shape[0]
        
        # Convert to torch tensors
        #x = self._jax_array_to_torch_tensor(x).reshape(batch_size, -1)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Compute mean and covariance of function distribution
        with torch.no_grad():
            f_preds = self.model(x)
            y_preds = self.likelihood(f_preds)
        
        return y_preds


    def f_distribution_mean_var(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and variance the functional distribution. 

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): dummy argument.
        - mc_samples (int): dummy argument.

        returns:
        - mean (jnp.ndarray): mean of the function distribution.
        - var (jnp.ndarray): variance of the function distribution.
        """
        # Reshape data
        batch_size = x.shape[0]

        # Convert to torch tensors
        x = self._jax_array_to_torch_tensor(x).reshape(batch_size, -1)

        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        # Compute mean and covariance of function distribution
        with torch.no_grad():
            f_preds = self.model(x)
            f_mean = f_preds.mean
            f_var = f_preds.variance
            # Convert to JAX arrays
            f_mean = self._torch_tensor_to_jax_array(f_mean)
            f_var = jnp.array(f_var.cpu().numpy()) #f_var = self._torch_tensor_to_jax_array(f_var)

        return f_mean.reshape(batch_size, -1), f_var.reshape(batch_size, -1)
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): wrapper for dataset
        """
        # Evaluation mode
        self.model.eval()
        self.likelihood.eval()

        plot_function_samples(
            self, 
            self.config, 
            dataloader
        )
        

    def _jax_array_to_torch_tensor(
        self, 
        x
    ):
        """
        Convert a JAX array to a PyTorch tensor.

        params:
        - x (jnp.ndarray): input data.

        returns:
        - x (torch.Tensor): input data.
        """
        if isinstance(x, jnp.ndarray):
            tensor = torch.from_dlpack(jax.dlpack.to_dlpack(x))
        elif isinstance(x, torch.Tensor):
            tensor = x
        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x)
        else:
            raise Exception("Unknown data type.")
        
        return tensor.cuda() if torch.cuda.is_available() else tensor.cpu()


    def _torch_tensor_to_jax_array(
        self, 
        x
    ):
        return jax.dlpack.from_dlpack(torch.to_dlpack(x.cpu()))




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