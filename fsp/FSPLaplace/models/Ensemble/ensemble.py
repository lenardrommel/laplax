import gc
import jax 
import copy

import haiku as hk
import jax.flatten_util
import jax.numpy as jnp

from functools import partial

from models.Ensemble.model_utils.mlp import MLP
from models.Ensemble.model_utils.cnn import CNN, CNN2
from models.Ensemble.training_utils.plot_utils import plot_function_samples
from models.Ensemble.training_utils.training import fit_model, evaluate_model


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}


class Ensemble:
    """
    Placeholder for the Ensemble BNN.
    """
    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the Ensemble BNN model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        # General configuration
        self.key = key
        self.config = copy.deepcopy(config)
        
        # Training configuration
        training_config = self.config["ensemble"]["training"]
        self.n_neural_nets = training_config["n_neural_nets"]

        # Neural network configuration
        nn_config = self.config["ensemble"]["neural_net"]
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]
        self.periodic_features = False # (self.config["experiment"]["name"] == "era5_interpolation")        

        # Likelihood configuration
        likelihood_config = self.config["ensemble"]["likelihood"]
        self.likelihood = likelihood_config["model"]
        self.ll_scale = [likelihood_config["scale_init"] for _ in range(self.n_neural_nets)]
        if self.likelihood == "Categorical":
            self.n_outputs = likelihood_config["n_classes"]
        elif self.likelihood == "Gaussian": 
            self.n_outputs = 1
        else:
            raise ValueError("Invalid likelihood model.")
        self.architecture += [self.n_outputs]
        
        # Initialize model
        self.forward = hk.transform_with_state(
            self.make_forward_fn(
                self.architecture,
                self.activation_fn
            )
        )

    
    @property
    def apply_fn(self):
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
            if self.periodic_features:
                x4 = jnp.sin(2 * jnp.pi * x[:,-1] / 24).reshape(-1, 1)
                x5 = jnp.cos(2 * jnp.pi * x[:,-1] / 24).reshape(-1, 1)
                x = jnp.concatenate([x, x4, x5], axis=-1)

            if self.model_type == "MLP":
                return MLP(
                    architecture,
                    activation_fn
                )(x)
            elif self.model_type == "CNN1":
                return CNN(
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
        self.params, self.states = [], []
        for i in range(self.n_neural_nets):
            # Initialize model
            x_init = next(iter(train_dataloader))[0]
            _params, _state = self.initialize_model(x_init)
            print("Previous training steps: ", self.training_steps, flush=True)
        
            # Shuffle training data
            self.key, key1 = jax.random.split(self.key)
            train_dataloader.key = key1

            # Train model
            print(f"Training neural network {i+1}/{self.n_neural_nets}...", flush=True)
            _params, _ll_scale, _state = fit_model(
                self, # model
                _params,
                _state,
                self.ll_scale[i],
                train_dataloader,
                val_dataloader
            )
            self.params.append(_params)
            self.states.append(_state)
            self.ll_scale[i] = _ll_scale
            jax.clear_caches()
            gc.collect()
            print("Likelihood scale:", self.ll_scale[i], flush=True)

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

        # Set training steps
        self.training_steps = 0

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

        returns:
        - f_lnn (jnp.array): function samples.
        """
        f_samples = jnp.stack([
                self.apply_fn(self.params[i], self.states[i], key, x, training=False)[0]
                for i in range(self.n_neural_nets)
            ], 
            axis=0
        )

        return f_samples

    
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
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y (jnp.array): function samples.
        """
        # Split keys
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
        Return the mean and covariance the linearized function distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.array): mean of the functional distribution.
        - cov (jnp.array): covariance of the functional distribution.
        """
        # Compute function distribution
        f_samples = self.predict_f(x, key, mc_samples)

        # Mean 
        mean = f_samples.mean(0)

        # Covariance
        cov = jnp.cov(f_samples, rowvar=False) 

        return mean, cov
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_var(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and variance of the linearized function distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.array): mean of the distribution.
        - variance (jnp.array): variance of the distribution.
        """
        # Compute function distribution
        f_samples = self.predict_f(x, key, mc_samples)

        # Mean 
        mean = f_samples.mean(0)

        # Variance
        variance = f_samples.var(0)

        return mean, variance
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        # Plot function samples
        plot_function_samples(
            self, 
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
        # Evaluate model
        loss = evaluate_model(
            self.key, 
            self, # model
            dataloader
        )

        return loss
