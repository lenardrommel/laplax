import jax 
# import cola
import copy
import wandb
import optax

import gpjax as gpx
import jax.numpy as jnp
import jax.scipy as jsp
import tensorflow_probability.substrates.jax as tfp

from functools import partial
#from gpjax.lower_cholesky import lower_cholesky


from models.GP.plot_utils import plot_function_samples


tfd = tfp.distributions


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

        # Define the GP kernel
        feature_dim = self.config["data"]["feature_dim"]
        kernel = self.config["gp"]["prior"]["kernel"]
        kernel_params = self.config["gp"]["prior"]["params"]
        variance = kernel_params["variance"]
        if kernel == "RBF":
            lengthscale = kernel_params["lengthscale"]
            self.kernel = gpx.kernels.RBF(
                active_dims=list(range(feature_dim)), 
                lengthscale=lengthscale, 
                variance=variance
            )
        elif kernel == "Matern12":
            lengthscale = kernel_params["lengthscale"]
            self.kernel = gpx.kernels.Matern12(
                active_dims=list(range(feature_dim)), 
                lengthscale=lengthscale, 
                variance=variance
            )
        elif kernel == "Matern32":
            lengthscale = kernel_params["lengthscale"]
            self.kernel = gpx.kernels.Matern32(
                active_dims=list(range(feature_dim)),
                lengthscale=1, # lengthscale, 
                variance=10. #variance
            )
        elif kernel == "Matern52":
            lengthscale = kernel_params["lengthscale"]
            self.kernel = gpx.kernels.Matern52(
                active_dims=list(range(feature_dim)), 
                lengthscale=lengthscale, 
                variance=variance
            )
        elif kernel == "Linear":
            self.kernel = gpx.kernels.Linear(
                active_dims=list(range(feature_dim)), 
                variance=variance
            )
        elif kernel == "RationalQuadratic":
            lengthscale = kernel_params["lengthscale"]
            alpha = kernel_params["alpha"]
            self.kernel = gpx.kernels.RationalQuadratic(
                active_dims=list(range(feature_dim)), 
                alpha=alpha,
                lengthscale=lengthscale, 
                variance=variance
            )
        elif kernel == "BanditKernel":
            lengthscale = kernel_params["lengthscale"]
            variance = kernel_params["variance"]
            k1 = gpx.kernels.Matern32(
                active_dims=list(range(feature_dim)), 
                lengthscale=lengthscale, 
                variance=variance
            )
            k2 = gpx.kernels.Linear(
                active_dims=list(range(feature_dim)), 
                variance=variance
            )
            self.kernel = gpx.kernels.SumKernel(kernels=[k1, k2])
        else:
            raise Exception("Unknown kernel")
        
        # Define the GP mean function
        meanf = gpx.mean_functions.Zero()

        # Define the GP prior
        self.prior = gpx.Prior(mean_function=meanf, kernel=self.kernel)

             
    def fit(
        self, 
        train_dataloader, 
        val_dataloader,
        asldnkcl
    ):
        """
        Fit the model.
        
        params:
        - train_dataloader (DataLoader): train dataloader.
        - val_dataloader (DataLoader): validation dataloader.

        returns:
        - loss (dict): validation loss.
        """
        # Get config
        # sparse = self.config["gp"]["sparse"]
        sparse = False
        # n_inducing_pts = min(self.config["gp"]["n_inducing_pts"], len(train_dataloader.dataset))
        likelihood_model = self.config["gp"]["likelihood"]["model"]

        # Split keys
        key1, key2 = jax.random.split(self.key, 2)

        # Get the data
        X, y = train_dataloader.dataset[train_dataloader.dataset_idx]
        #X, y = train_dataloader.dataset.get_data()
        X, y = jnp.float64(X), jnp.float64(y)
        # alpha_eps = 0.1
        # y = jnp.where(
        #     y==1, 
        #     jnp.log(1+alpha_eps)-0.5*jnp.log(1/(1+alpha_eps)+1), 
        #     jnp.log(alpha_eps)-0.5*jnp.log(1/alpha_eps+1)
        # )
        self.D = gpx.Dataset(X=X, y=y)
        
        # Define the GP likelihood
        if likelihood_model == "Gaussian":
            self.likelihood = gpx.Gaussian(num_datapoints=self.D.n, obs_noise=jnp.array(self.config["gp"]["likelihood"]["scale"]))
        elif likelihood_model == "Bernoulli":
            self.likelihood = gpx.likelihoods.Bernoulli(num_datapoints=self.D.n)
        else:
            raise Exception("Unknown likelihood")

        # Define the GP posterior
        self.posterior = self.prior * self.likelihood
        objective = jax.jit(gpx.objectives.ConjugateMLL(negative=True))
        # if likelihood_model == "Gaussian":
        #     if sparse:
        #         # Model 
        #         inducing_inputs = jax.random.choice(key1, X, shape=(n_inducing_pts,), replace=False)
        #         self.posterior = gpx.CollapsedVariationalGaussian(posterior=self.posterior, inducing_inputs=inducing_inputs)
        #         self.posterior.posterior.likelihood.replace_trainable(obs_noise=False)
        #         # Objective
        #         objective = jax.jit(gpx.CollapsedELBO(negative=True))
        #     else:
        #         # Objective
        #         objective = jax.jit(gpx.objectives.ConjugateMLL(negative=True))
        # elif likelihood_model == "Bernoulli":
        #     objective = jax.jit(gpx.objectives.LogPosteriorDensity(negative=True))

        # Optimizer
        optimiser = optax.adam(
            learning_rate=self.config["gp"]["training"]["lr"]
        )

        # Fit the model
        self.posterior, history = gpx.fit(
            model=self.posterior,
            objective=objective,
            #batch_size=self.config["data"]["batch_size"],
            train_data=self.D,
            optim=optimiser,
            num_iters=self.config["gp"]["training"]["nb_epochs"], 
            key=key2
        )

        # Log
        for i, n_mll in enumerate(history):
            wandb.log({f"Train/n_mll": n_mll})
            if i % 100 == 0 or i == len(history) - 1:
                print(f"Train/n_mll: {n_mll}", flush=True)                    
        
        # Evaluate on validation set
        X, y = val_dataloader.dataset.get_data()
        X, y = jnp.float64(X), jnp.float64(y)
        val_data = gpx.Dataset(X=X, y=y)
        val_mll = -objective(self.posterior.constrain(), val_data).sum()
        print("Val mll:", val_mll, flush=True)

        return {"mll": 0.}
    

    def fit_to_prior(self):
        """
        Fit the model without any data.
        Used for bandits.
        """       
        self.D = None

        # Define the GP likelihood
        self.likelihood = gpx.Gaussian(num_datapoints=0)

        # Define the GP posterior
        self.posterior = self.prior 


    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.

        params:
        - dataloader (DataLoader): dataloader.

        returns:
        - test_loss (dict): test loss.
        """
        assert dataloader.replacement == False, "Data should be sampled without replacement"

        # Evaluate the model
        expected_ll, mse = 0, 0
        for x, y in dataloader:
            # Function distribution
            f_mean, f_var = self.f_distribution_mean_var(x, key=None, mc_samples=None)
            y, f_mean, f_var = y.reshape(-1, 1), f_mean.reshape(-1, 1), f_var.reshape(-1, 1)
            if self.config["gp"]["sparse"]:
                expected_ll += self.posterior.posterior.likelihood.expected_log_likelihood(y, f_mean, f_var).sum()
            else:
                expected_ll += self.posterior.likelihood.expected_log_likelihood(y, f_mean, f_var).sum()
            # Mean squared error
            mse += jnp.sum((f_mean.reshape(-1) - y.reshape(-1))**2)  
        mse /= len(dataloader.dataset)
        expected_ll /= len(dataloader.dataset)    

        # Log
        wandb.log({"Test/expected_ll": expected_ll, "Test/mse": mse})
        print(f"Expected log-likelihood: {expected_ll} - MSE: {mse}", flush=True)

        return {"expected_ll": expected_ll, "mse": mse}

        
    @partial(jax.jit, static_argnums=(0,3))
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
        x = x.reshape(-1, 4)

        # Function distribution
        if self.D is None:
            f_dist = self.posterior(x)
        else:
            f_dist = self.posterior(x, train_data=self.D)

        # Sample from function distribution
        samples = f_dist.sample(seed=key, sample_shape=(mc_samples,)) # (mc_samples, batch_size, 1)

        return samples.reshape(mc_samples, batch_size, 1)

    
    @partial(jax.jit, static_argnums=(0,3))
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
        x = x.reshape(-1, self.config["data"]["feature_dim"])
        
        # Predictive distribution
        if self.config["gp"]["likelihood"]["model"] == "Gaussian":
            # Function distribution
            if self.D is None:
                f_dist = self.posterior(x)
            else:
                f_dist = self.posterior(x, train_data=self.D)
            # Predictive distribution
            if self.config["gp"]["sparse"]:
                predictive_dist = self.posterior.posterior.likelihood(f_dist)
            else:
                predictive_dist = self.posterior.likelihood(f_dist)
        elif self.config["gp"]["likelihood"]["model"] == "Bernoulli":
            laplace_latent_dist = self.construct_laplace(x)
            predictive_dist = self.posterior.likelihood(laplace_latent_dist)

        # Sample from the predictive distribution
        samples = predictive_dist.sample(seed=key, sample_shape=(mc_samples,))

        return samples.reshape(mc_samples, batch_size, 1)
    
        
    def construct_laplace(
        self, 
        x
    ):
        jitter = 1e-6

        Kxt = self.kernel.cross_covariance(self.D.X, x)
        Kxx = self.kernel.gram(self.D.X) + jnp.eye(self.D.n) * jitter

        # Negative Hessian,  H = -∇²p_tilde(y|f):
        objective = gpx.objectives.LogPosteriorDensity(negative=True)
        H = jax.jacfwd(jax.jacrev(objective))(self.posterior, self.D).latent.latent[:, 0, :, 0]

        L = jnp.linalg.cholesky(H + jnp.eye(self.D.n) * jitter)

        # H⁻¹ = H⁻¹ I = (LLᵀ)⁻¹ I = L⁻ᵀL⁻¹ I
        L_inv = jsp.linalg.solve_triangular(L, jnp.eye(self.D.n), lower=True)
        H_inv = jsp.linalg.solve_triangular(L.T, L_inv, lower=False)

        map_latent_dist = self.posterior.predict(x, train_data=self.D)

        # Kxx⁻¹ Kxt
        Kxx_inv_Kxt = jnp.linalg.solve(Kxx.to_dense(), Kxt)

        # Ktx Kxx⁻¹[ H⁻¹ ] Kxx⁻¹ Kxt
        laplace_cov_term = jnp.matmul(jnp.matmul(Kxx_inv_Kxt.T, H_inv), Kxx_inv_Kxt)

        mean = map_latent_dist.mean()
        covariance = map_latent_dist.covariance() + laplace_cov_term
        L = jnp.linalg.cholesky(covariance)
        return tfd.MultivariateNormalTriL(jnp.atleast_1d(mean.squeeze()), L)



    @partial(jax.jit, static_argnums=(0,3))
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
        x = x.reshape(-1, self.config["data"]["feature_dim"])

        # Predictive distribution
        if self.D is None:
            f_dist = self.posterior(x)
        else:
            f_dist = self.posterior(x, train_data=self.D)
        
        # Mean and covariance
        f_mean = f_dist.mean()
        f_cov = f_dist.covariance()
    
        return f_mean, f_cov


    @partial(jax.jit, static_argnums=(0,3))
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
        x = x.reshape(-1, self.config["data"]["feature_dim"])

        # Function distribution
        if self.D is None:
            f_dist = self.posterior(x)
        else:
            f_dist = self.posterior(x, train_data=self.D)
        
        # Mean and variance
        f_mean = f_dist.mean()
        f_var = f_dist.variance()
    
        return f_mean, f_var
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): wrapper for dataset
        """
        plot_function_samples(
            self, 
            self.config, 
            dataloader
        )
        


