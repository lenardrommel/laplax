import jax 
import copy
import pickle
import jax.flatten_util

import haiku as hk
import jax.numpy as jnp

from jax import eval_shape
from functools import partial
from jax.example_libraries.optimizers import adam 
from haiku.data_structures import partition, merge

from data_utils.era5_data_utils import gcs_to_cartesian
from models.FSPLaplace.model_utils.mlp import MLP
from models.FSPLaplace.model_utils.prior import Prior
from models.FSPLaplace.model_utils.cnn import CNN1, CNN2
from models.FSPLaplace.training_utils.plot_utils import plot_function_samples, _plot_after_training
from models.FSPLaplace.training_utils.training import fit_model, evaluate_model
from models.FSPLaplace.training_utils.inference import (
    LanczosLowRankFunctionalLaplacePosterior
)


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}

class FSPLaplace:
    """
    Placeholder for the functional FSPLaplace model.
    """

    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the FSPLaplace model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        # General configuration
        self.key = key
        self.config = copy.deepcopy(config)

        # Neural network configuration
        nn_config = self.config["fsplaplace"]["neural_net"]
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]
        self.periodic_features = nn_config["periodic_features"]

        # Inference configuration
        inference_config = self.config["fsplaplace"]["inference"]
        self.cov_type = inference_config["cov_type"]
        
        # Likelihood configuration
        likelihood_config = self.config["fsplaplace"]["likelihood"]
        self.likelihood = likelihood_config["model"]
        if self.likelihood == "Categorical":
            self.n_classes = likelihood_config["n_classes"]
            self.architecture += [self.n_classes]
        elif self.likelihood == "Gaussian": 
            self.ll_scale = likelihood_config["scale"]
            self.architecture += [1]
        else:
            raise ValueError("Invalid likelihood model.")
        
        # Initialize model
        self.forward = hk.transform_with_state(
            self.make_forward_fn(
                self.architecture,
                self.activation_fn
            )
        )


    @property
    def apply_fn(
        self
    ):
        """
        Apply function of the MLP.

        returns:
        - apply (function): apply function.
        """
        return self.forward.apply
    

    def make_forward_fn(
        self, 
        architecture, 
        activation_fn
    ):
        """
        Build foward function.

        params:
        - architecture (list): architecture of the MLP.
        - activation_fn (str): activation function.
        
        returns:
        - forward_fn (function): forward function.
        """
        def forward_fn(x, training):

            if "era5" in self.config["experiment"]["name"]:
                x = jnp.concatenate([gcs_to_cartesian(x[:,0], x[:,1]).reshape(-1, 3), x[:,2].reshape(-1, 1)], axis=-1)

            if self.periodic_features:
                x4 = jnp.sin(2 * jnp.pi * x[:,-1] / self.periodic_features).reshape(-1, 1)
                x5 = jnp.cos(2 * jnp.pi * x[:,-1] / self.periodic_features).reshape(-1, 1)
                x = jnp.concatenate([x, x4, x5], axis=-1)

            if self.model_type == "MLP":
                return MLP(
                    architecture,
                    activation_fn
                )(x)
            elif self.model_type == "HelmholtzMLP":
                x, z = x[:, :-1].reshape(-1,1,2), x[:, -1].reshape(-1, 1, 1)
                z_int = jnp.array(z, dtype=int).reshape(-1)
                potential_MLP = lambda _x, _z: MLP(  # noqa: E731
                    architecture,
                    activation_fn
                )(jnp.concatenate([_x, _z], axis=-1))#(x, z)
                stream_MLP = lambda _x, _z: MLP(  # noqa: E731
                    architecture,
                    activation_fn
                )(jnp.concatenate([_x, _z], axis=-1))
                grad_potential = jax.vmap(jax.jacfwd(potential_MLP, argnums=0))(x, z).reshape(x.shape[0],2)[jnp.arange(x.shape[0]),z_int].reshape(-1,1)   
                grad_stream = jax.vmap(jax.jacfwd(stream_MLP, argnums=0))(x, z).reshape(x.shape[0],2)[jnp.arange(x.shape[0]),1-z_int].reshape(-1,1)
                out = grad_potential + (-1)**z_int.reshape(-1, 1) * grad_stream
                return out
            elif self.model_type == "CNN1":
                return CNN1(
                    output_dim=self.architecture[-1],
                    activation_fn=self.activation_fn
                )(x)
            elif self.model_type == "CNN2":
                return CNN2(
                    output_dim=self.architecture[-1],
                    activation_fn=self.activation_fn
                )(x)
            else:
                raise NotImplementedError()

        return forward_fn
    

    def fit(
        self, 
        train_dataloader, 
        val_dataloader, 
        prior_dataloader
    ):
        """
        Fit the model.

        params:
        - train_dataloader (DataLoader): training dataloader. 
        - val_dataloader (DataLoader): validation dataloader.

        returns:
        - val_loss (dict): validation loss.
        """
        # Set model configuration   
        x_min = train_dataloader.dataset.x_min
        x_max = train_dataloader.dataset.x_max
        if "era5" in self.config["experiment"]["name"]:
            self.config["fsplaplace"]["inference"]["min_context_val"] = x_max #x_min #x_min - 0.5 * (x_max - x_min) 
            self.config["fsplaplace"]["inference"]["max_context_val"] = x_max + 0.6 * (x_max - x_min) # 0.5 * (x_max - x_min) 
        elif self.config["experiment"]["name"] == "mona_loa_experiments":
            self.config["fsplaplace"]["inference"]["min_context_val"] = x_min
            self.config["fsplaplace"]["inference"]["max_context_val"] = x_max + 0.7 * (x_max - x_min)
        elif self.config["experiment"]["name"] == "ocean_current_modeling":
            self.config["fsplaplace"]["inference"]["min_context_val"] = jnp.array([-1.70981871, -1.70385883])
            self.config["fsplaplace"]["inference"]["max_context_val"] = jnp.array([2.01992571, 2.40336468])
        else:
            self.config["fsplaplace"]["inference"]["min_context_val"] = x_min - 0.5 * (x_max - x_min) 
            self.config["fsplaplace"]["inference"]["max_context_val"] = x_max + 0.5 * (x_max - x_min) 
            print(self.config["fsplaplace"]["inference"]["min_context_val"], self.config["fsplaplace"]["inference"]["max_context_val"])
        # print("min", self.config["fsplaplace"]["inference"]["min_context_val"])
        # print("max", self.config["fsplaplace"]["inference"]["max_context_val"])
        
        # Initialize model
        if not hasattr(self, "mean_params"):
            x_init, _ = next(iter(train_dataloader))
            self.mean_params, self.state = self.initialize_model(x_init)
        
        # Load pre-trained weights
        likelihood = self.config["fsplaplace"]["likelihood"]["model"]
        pretrained_weights_path = self.config["fsplaplace"]["training"]["pretrained_weights_path"]
        if pretrained_weights_path:
            print("Loading pre-trained weights...", flush=True)
            with open(pretrained_weights_path, "rb") as f:
                data = pickle.load(f)
            self.training_steps = data["step"]
            _, _, get_params = adam(step_size=0.1)
            if likelihood == "Categorical":
                self.mean_params = get_params(data["opt_state"])
            else:
                self.mean_params, ll_rho = get_params(data["opt_state"])
                self.ll_scale = jax.nn.softplus(ll_rho)

        # Load Prior
        self.prior = Prior(
            self.key,
            prior_dataloader,
            self.config
        )
        
        # Posterior mode 
        print("Fitting mean parameters of FSPLaplace approximation...", flush=True)
        self.mean_params, self.state = fit_model(
            self.key, 
            self.mean_params,
            self.state,
            self, # model
            self.config, 
            train_dataloader,
            val_dataloader, 
            self.prior
        )

       
        # Parition parameters
        self.mean_params, self.other_params = partition(
            lambda m, n, p: "batch_norm" not in m, self.mean_params
        )
        self.unravel_params = jax.flatten_util.ravel_pytree(self.mean_params)[1]

        ############ TMP ############
        print("Clearing JAX caches...", flush=True)
        jax.clear_caches()
        ############ TMP ############

        # Posterior covariance approx. 
        print("Fitting covariance parameters of FSPLaplace approximation...", flush=True)
        
        
        laplace_posterior = LanczosLowRankFunctionalLaplacePosterior(self).fit(train_dataloader, val_dataloader)
        self.posterior_covariance_sq_root = laplace_posterior.posterior_covariance_sq_root()

        # Evaluate model
        val_loss = self.evaluate(val_dataloader)

        return val_loss
    

    def initialize_model(
        self, 
        x_init
    ):
        """
        Initialize the BNN model parameters.
        
        params:
        - x_init (jnp.ndarray): dummy input data.

        returns:
        - params (jax.tree_util.pytree): parameters of the BNN.
        """
        # Handle random key
        self.key, key1 = jax.random.split(self.key)

        # Initialize model
        init_fn, apply_fn = self.forward
        params, state = init_fn(key1, x_init, training=True)

        # Reset training steps
        self.training_steps = 0

        print(f'Number of parameters: {hk.data_structures.tree_size(params)}', flush=True)

        return params, state


    @partial(jax.jit, static_argnums=(0,3))
    def predict_f(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the function distribution.
        
        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.
        - is_training (bool): dummy argument.
        - stochastic (bool): dummy argument.

        returns 
        - f_lnn (jnp.array): function samples.
        """
        # Split keys
        key1, key2 = jax.random.split(key)

        if self.cov_type in ["low_rank", "low_rank_sketch", "low_rank_lanczos"]:
            params_sample = jnp.einsum(
                "ij,mj->mi",
                self.posterior_covariance_sq_root, # (n_weights, n_weights)
                jax.random.normal(key1, shape=(mc_samples, self.posterior_covariance_sq_root.shape[1]))
            )
        else: # if cov is diagonal
            params_sample = self.posterior_covariance_sq_root * jax.random.normal(
                key=key1, 
                shape=(mc_samples, self.posterior_covariance_sq_root.shape[0])
            )
        params_sample = jax.vmap(self.unravel_params)(params_sample)

        # Compute Jacobian 
        f = lambda p: self.apply_fn(merge(p, self.other_params), self.state, key2, x, training=False)[0]  
        f_m, f_res = jax.vmap(lambda p: jax.jvp(f, (self.mean_params,), (p,)))(params_sample)

        return f_m + f_res
    
    
    @partial(jax.jit, static_argnums=(0,3))
    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from predictive distribution.
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.
        - is_training (bool): dummy argument.
        - stochastic (bool): dummy argument.

        returns:
        - y (jnp.array): function samples.
        """
        key1, key2 = jax.random.split(key)
        
        # Sample from function distribution
        f = self.predict_f(x, key1, mc_samples)

        # Sample from likelihood
        if self.likelihood == "Gaussian":
            y = f + self.ll_scale * jax.random.normal(key2, shape=f.shape) 
        elif self.likelihood == "Categorical":
            y = jax.nn.softmax(f, axis=-1)

        return y


    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_cov(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and covariance the linearized functional distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy argument.
        
        returns:
        - mean (jnp.array): mean of the functional distribution.
        - cov (jnp.array): covariance of the functional distribution.
        """
        assert self.mean_params, "Model is not trained."

        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean 
        mean = self.apply_fn(merge(self.mean_params, self.other_params), self.state, key1, x, training=False)[0]

        # Covariance
        cov = self.f_distribution_kernel(x, x, key2)

        return mean, cov
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_var(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and diagonalized covariance the linearized functional distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy variable.
        
        returns:
        - mean (jnp.array): mean of the distribution.
        - diag_cov (jnp.array): diagonal covariance of the distribution.
        """
        assert self.mean_params, "Model is not trained."

        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean
        mean = self.apply_fn(merge(self.mean_params, self.other_params), self.state, key1, x, training=False)[0]
        
        # Covariance
        x = jnp.expand_dims(x, axis=1)
        kernel_fn = lambda z: self.f_distribution_kernel(z, z, key2)
        diag_cov = jax.vmap(kernel_fn, in_axes=0)(x).reshape(x.shape[0], -1)
        
        return mean, diag_cov
    
    
    @partial(jax.jit, static_argnums=(0,))
    def f_distribution_kernel(
        self, 
        x1, 
        x2, 
        key
    ):
        """
        Compute the kernel induced by the linearized BNN.

        params:
        - x1 (jnp.array): input data.
        - x2 (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.

        return:
        - kernel (jnp.array): kernel matrix.
        """

        @jax.jit
        def classwise_kernel(k):
            # Define predict function
            f1 = lambda p: self.forward.apply(merge(p, self.other_params), self.state, key, x1, training=False)[0][:,k].reshape(-1) 
            f2 = lambda p: self.forward.apply(merge(p, self.other_params), self.state, key, x2, training=False)[0][:,k].reshape(-1) 

            # Compute kernel - jac_f1 @ cov @ jac_f2.T
            if self.cov_type in ["low_rank", "low_rank_sketch", "low_rank_lanczos"]:
                j1_sq = jax.vmap(
                    lambda a: jax.jvp(f1, (self.mean_params,), (self.unravel_params(a),))[1], 
                    in_axes=-1, out_axes=-1
                )(self.posterior_covariance_sq_root)
                j2_sq = jax.vmap(
                    lambda a: jax.jvp(f2, (self.mean_params,), (self.unravel_params(a),))[1], 
                    in_axes=-1, out_axes=-1
                )(self.posterior_covariance_sq_root)
                kernel = jax.tree_map(lambda x1, x2: x1 @ x2.T, j1_sq, j2_sq)
            else: # if cov is diagonal
                @jax.jit
                def delta_vjp_jvp(delta):
                    delta_vjp = jax.vjp(f2, self.mean_params)[1](delta)[0]
                    vj_prod = jax.tree_map(
                        lambda x1, x2: x1 * x2, self.unravel_params(self.posterior_covariance_sq_root**2), delta_vjp
                    )
                    return jax.jvp(f1, (self.mean_params,), (vj_prod,))[1]
                
                # Compute the kernel
                fx2 = eval_shape(f2, self.mean_params)
                eye = jnp.eye(x1.shape[0])
                kernel = jax.vmap(jax.linear_transpose(delta_vjp_jvp, fx2))(eye)[0].T

            return kernel
        
        # Compute kernel for each class
        kernel = jax.vmap(classwise_kernel)(jnp.arange(self.architecture[-1])).T  

        return kernel
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        assert self.mean_params, "Model is not trained."

        # Plot function samples
        plot_function_samples(
            self, 
            jax.random.PRNGKey(0), 
            self.config, 
            dataloader
        )

    def plot_after_training(
        self, 
        dataloader
    ):
        _plot_after_training(
            self,
            self.mean_params,
            self.state,
            jax.random.PRNGKey(0), 
            self.config, 
            dataloader
        )


    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.

        params:
        - test_dataloader (DataLoader): dataloader.

        returns:
        - loss (dict): loss.
        """
        assert self.mean_params, "Model is not trained."

        # Evaluate model
        loss = evaluate_model(
            self.key, 
            self, # model
            dataloader
        )

        return loss
