import jax 
import copy
import pickle 

import numpy as np
import haiku as hk
import jax.numpy as jnp

from functools import partial
from jax.example_libraries.optimizers import adam 

from models.FVI.model_utils.bnn import BNN
from models.FVI.model_utils.prior import Prior
from models.FVI.training_utils.plot_utils import plot_function_samples
from models.FVI.training_utils.training import fit_model, evaluate_model


class FVI:
    """
    Placeholder for the FVI model.
    Code adapted to JAX from https://github.com/ssydasheng/FBNN/tree/master.
    """

    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the FVI model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        self.key = key
        self.config = copy.deepcopy(config)
    
        # Build model
        self.model = BNN(
            self.config["fvi"]["neural_net"],
            self.config["fvi"]["likelihood"]
        )

    @property
    def ll_scale(self):
        return self.model.ll_scale


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
        - val_loss (dict): validation loss.
        """ 
        # Set model configuration   
        x_min = train_dataloader.dataset.x_min
        x_max = train_dataloader.dataset.x_max
        if "era5" in self.config["experiment"]["name"]:
            self.config["fvi"]["training"]["min_context_val"] = x_max #x_min #x_min - 0.5 * (x_max - x_min) 
            self.config["fvi"]["training"]["max_context_val"] = x_max + 0.6 * (x_max - x_min) # 0.5 * (x_max - x_min) 
        elif self.config["experiment"]["name"] == "mona_loa_experiments":
            self.config["fvi"]["training"]["min_context_val"] = x_min
            self.config["fvi"]["training"]["max_context_val"] = x_max + 0.7 * (x_max - x_min) 
        elif self.config["experiment"]["name"] == "ocean_current_modeling":
            self.config["fvi"]["training"]["min_context_val"] = jnp.array([-1.70981871, -1.70385883])
            self.config["fvi"]["training"]["max_context_val"] = jnp.array([2.01992571, 2.40336468])
        else:
            self.config["fvi"]["training"]["min_context_val"] = x_min - 0.5 * (x_max - x_min) 
            self.config["fvi"]["training"]["max_context_val"] = x_max + 0.5 * (x_max - x_min) 

        print("min", self.config["fvi"]["training"]["min_context_val"])
        print("max", self.config["fvi"]["training"]["max_context_val"])

        # Initialize model
        if not hasattr(self, "mean_params") or not hasattr(self, "rho_params"):
            x_init = jnp.expand_dims(train_dataloader.dataset.X[0], axis=0)
            self.mean_params, self.rho_params = self.initialize_model(x_init)
            print(f'Number of mean parameters: {hk.data_structures.tree_size(self.mean_params)}')
            print(f'Number of rho parameters: {hk.data_structures.tree_size(self.rho_params)}')

        # Load Prior
        self.prior = Prior(
            self.key,
            prior_dataloader,
            self.config
        )

        # Load pre-trained weights
        likelihood = self.config["fvi"]["likelihood"]["model"]
        pretrained_weights_path = self.config["fvi"]["training"]["pretrained_weights_path"]
        if pretrained_weights_path:
            print("Loading pre-trained weights...", flush=True)
            with open(pretrained_weights_path, "rb") as f:
                data = pickle.load(f)
            self.prior.params, self.model.training_steps = data["prior_params"], data["step"]
            _, _, get_params = adam(step_size=0.1)
            if likelihood == "Categorical":
                self.mean_params, self.rho_params = get_params(data["opt_state"])
            else:
                self.mean_params, self.rho_params, ll_rho = get_params(data["opt_state"])
                self.model.ll_scale = jax.nn.softplus(ll_rho)

        # Set batch size
        train_dataloader.batch_size = self.config["data"]["batch_size"]

        # Fit the model
        self.mean_params, self.rho_params, val_loss = fit_model(
            self.key, 
            self.mean_params, 
            self.rho_params,
            self.model, 
            self.config, 
            train_dataloader, 
            val_dataloader,
            self.prior
        )
        
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
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        """
        # Handle random key
        self.key, key1, key2 = jax.random.split(self.key, 3)

        # Read config 
        last_layer_vi = self.config["fvi"]["neural_net"]["last_layer_vi"]

        # Initialize model
        init_fn, apply_fn = self.model.forward
        
        # Initialize mean parameters
        mean_params = init_fn(key1, x_init)
        
        # Initialize rho parameters
        rho_params = init_fn(key2, x_init)
        rho_params = jax.tree_util.tree_map(lambda x: x*0 - 5, rho_params)
        if last_layer_vi:
            last_layer_key = list(rho_params.keys())[-1]
            rho_params = hk.data_structures.to_immutable_dict(
                {last_layer_key: rho_params[last_layer_key]}
            )
            rho_params = jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(rho_params), 
                jax.tree_util.tree_leaves(rho_params)
            )
        
        # Reset training steps
        self.model.training_steps = 0

        return mean_params, rho_params


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
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - f_lin_nn (jnp.ndarray): function samples.
        """
        assert self.mean_params and self.rho_params, "Model is not trained."

        return self.model.predict_f(
            self.mean_params, 
            self.rho_params,
            x, 
            key, 
            mc_samples
        )

    
    @partial(jax.jit, static_argnums=(0,3))
    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the predictive distribution.
        
        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.
        
        returns:
        - y (jnp.ndarray): function samples.
        """
        assert self.mean_params and self.rho_params, "Model is not trained."

        return self.model.predict_y(
            self.mean_params, 
            self.rho_params,
            x, 
            key, 
            mc_samples
        )


    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_cov(
        self,
        x, 
        key, 
        mc_samples
    ):
        """
        Estimate the mean and variance the functional distribution
        from samples as there is no closed form density over functions.

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy variable for compatibility.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution.
        - cov (jnp.ndarray): covariance of the distribution.
        """
        # Sample from the functional distribution
        f = self.predict_f(x, key, mc_samples) # (mc_samples, n_batch, n_classes)

        # Compute mean and covariance
        mean = jnp.mean(f, axis=0)
        cov = jax.vmap(lambda f: jnp.cov(f, rowvar=False), in_axes=-1)(f)

        return mean, cov
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_var(
        self,
        x, 
        key, 
        mc_samples
    ):
        """
        Estimate the mean and variance the functional distribution
        from samples as there is no closed form density over functions.
        
        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy variable for compatibility.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution.
        - diag_cov (jnp.ndarray): diagonal covariance of the distribution.
        """
        # Sample from the functional distribution
        f = self.predict_f(x, key, mc_samples) # (mc_samples, n_batch, n_classes)

        # Compute mean and covariance
        mean = jnp.mean(f, axis=0)
        var = jnp.var(f, axis=0)

        return mean, var
    

    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.
        
        params:
        - dataloader (DataLoader): test dataloader.

        returns:
        - test_loss (dict): test loss.
        """
        assert self.mean_params and self.rho_params, "Model is not trained."

        test_loss = evaluate_model(
            self.key, 
            self.mean_params, 
            self.rho_params,
            self.model, 
            dataloader
        )

        return test_loss
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        assert self.mean_params and self.rho_params, "Model is not trained."

        plot_function_samples(
            self.model, 
            self.mean_params, 
            self.rho_params,
            jax.random.PRNGKey(0), 
            self.config, 
            dataloader
        )
