import jax 
import pickle

import haiku as hk
import jax.numpy as jnp

from jax.example_libraries.optimizers import adam 


from models.FunctionalLaplaceSampling.model_utils.mlp import MLP
from models.FunctionalLaplaceSampling.model_utils.prior import Prior
from models.FunctionalLaplaceSampling.model_utils.cnn import CNN1, CNN2
from models.FunctionalLaplaceSampling.training_utils.plot_utils import plot_function_samples
from models.FunctionalLaplaceSampling.training_utils.training import fit_model, evaluate_model
from models.FunctionalLaplaceSampling.training_utils.inference import sample_laplace_posterior


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}

class FunctionalLaplaceSampling:
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
        self.config = config

        # Neural network configuration
        nn_config = config["flaplace_sampling"]["neural_net"]
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.last_layer_laplace = nn_config["last_layer_laplace"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]
        
        # Likelihood configuration
        likelihood_config = config["flaplace_sampling"]["likelihood"]
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
        self.forward = hk.transform(
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
        def forward_fn(x):
            if self.model_type == "MLP":
                return MLP(
                    architecture,
                    activation_fn
                )(x)
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
        val_dataloader
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
        x_min = float(train_dataloader.dataset.X.min())
        x_max = float(train_dataloader.dataset.X.max())
        self.config["flaplace_sampling"]["training"]["min_context_val"] = x_min - 0.5 * (x_max - x_min)
        self.config["flaplace_sampling"]["training"]["max_context_val"] = x_max + 0.5 * (x_max - x_min)
        self.train_dataloader = train_dataloader
        
        # Initialize model
        x_init = jnp.expand_dims(train_dataloader.dataset.X[0], axis=0)
        self.mean_params = self.initialize_model(x_init)

        # Load pre-trained weights
        likelihood = self.config["flaplace_sampling"]["likelihood"]["model"]
        pretrained_weights_path = self.config["flaplace_sampling"]["training"]["pretrained_weights_path"]
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
            train_dataloader,
            self.config
        )
        
        # Posterior mode 
        print("Fitting mean parameters of FSPLaplace approximation...", flush=True)
        self.mean_params = fit_model(
            self.key, 
            self.mean_params, 
            self, # model
            self.config, 
            train_dataloader,
            val_dataloader, 
            self.prior
        )
        
        # Evaluate model
        # val_loss = self.evaluate(val_dataloader)
        val_loss = 0.

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
        params = init_fn(key1, x_init)

        # Reset training steps
        self.training_steps = 0

        print(f'Number of parameters: {hk.data_structures.tree_size(params)}', flush=True)

        return params


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

        # Split parameters
        sto_params, det_params = self.partition_parameters(self.mean_params)

        # Sample from posterior
        sample_params = sample_laplace_posterior(
            self, 
            self.mean_params, 
            self.train_dataloader, 
            self.prior,
            self.config,
            key1, 
            mc_samples, 
            x
        )

        # GLM 
        fwd = lambda _sto_params: self.apply_fn(self.merge_parameters(_sto_params, det_params), key2, x)        
        f_m, f_lnn = jax.vmap(jax.jvp, in_axes=(None, None, 0))(fwd, (sto_params,), (sample_params,))
        
        return f_m + f_lnn
    
    
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
        mean = self.apply_fn(self.mean_params, key1, x)

        # Covariance
        f = self.predict_f(x, key2, mc_samples) # (mc_samples, n_classes, n_samples)
        cov = jax.vmap(lambda f: jnp.cov(f, rowvar=False), in_axes=-1)(f)

        return mean, cov
    

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
        mean = self.apply_fn(self.mean_params, key1, x)
        
        # Covariance
        f = self.predict_f(x, key2, mc_samples) # (mc_samples, n_classes, n_samples)
        diag_cov = jnp.var(f, axis=0)

        return mean, diag_cov
    

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
    

    def partition_parameters(
        self, 
        params
    ):
        """
        Split parameters into stochastic and non-stochastic.
        
        params:
        - params (jax.tree_util.pytree): parameters of the BNN.

        returns:
        - sto_params (jax.tree_util.pytree): stochastic parameters of the BNN.
        - det_params (jax.tree_util.pytree): deterministic parameters of the BNN.
        """
        if self.last_layer_laplace:
            params = hk.data_structures.to_mutable_dict(params)
            last_layer_key = list(params.keys())[-1]
            sto_params = hk.data_structures.to_immutable_dict(
                {last_layer_key: params[last_layer_key]}
            )
            det_params = hk.data_structures.to_immutable_dict(
                {k:v for k,v in params.items() if k != last_layer_key}
            )
        else:
            sto_params, det_params = params, hk.data_structures.to_immutable_dict({})

        return sto_params, det_params
        

    def merge_parameters(
        self,
        sto_params, 
        det_params
    ):
        """
        Join stochastic and non-stochastic parameters.
        
        params:
        - sto_params (jax.tree_util.pytree): stochastic parameters.
        - det_params (jax.tree_util.pytree): deterministic parameters.

        returns:
        - params (jax.tree_util.pytree): parameters of the BNN.
        """
        return hk.data_structures.merge(sto_params, det_params)