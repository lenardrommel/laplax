import jax 
import copy
import pickle
import jax.flatten_util

import haiku as hk
import jax.numpy as jnp

from functools import partial
from jax.example_libraries.optimizers import adam 

from models.SNGP.model_utils.mlp import MLP
from models.SNGP.model_utils.cnn import CNN1, CNN2
from models.SNGP.training_utils.plot_utils import plot_function_samples
from models.SNGP.training_utils.training import fit_model, evaluate_model


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}

class SNGP:
    """
    Placeholder for the functional Sngp model.
    """

    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the Sngp model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        # General configuration
        self.key = key
        self.config = copy.deepcopy(config)

        # Neural network configuration
        nn_config = self.config["sngp"]["neural_net"]
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]

        # Inference configuration
        inference_config = self.config["sngp"]["inference"]
        self.n_rff = inference_config["n_rff"]
        self.kernel = inference_config["kernel"]

        # Likelihood configuration
        likelihood_config = self.config["sngp"]["likelihood"]
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
        self.forward = hk.transform_with_state(self.make_forward_fn())


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
    ):
        """
        Build foward function.

        returns:
        - forward_fn (function): forward function.
        """
        def forward_fn(x, training):
            if self.model_type == "MLP":
                return MLP(
                    n_rff=self.n_rff, 
                    kernel=self.kernel,
                    architecture=self.architecture,
                    activation_fn=self.activation_fn, 
                    key=self.key
                )(x)
            elif self.model_type == "CNN1":
                return CNN1(
                    n_rff=self.n_rff,
                    output_dim=self.architecture[-1],
                    kernel=self.kernel,
                    activation_fn=self.activation_fn,
                    key=self.key
                )(x)
            elif self.model_type == "CNN2":
                return CNN2(
                    n_rff=self.n_rff,
                    output_dim=self.architecture[-1],
                    kernel=self.kernel,
                    activation_fn=self.activation_fn,
                    key=self.key
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
        # Initialize model
        x_init = jnp.expand_dims(train_dataloader.dataset.X[0], axis=0)
        self.params, self.state = self.initialize_model(x_init)

        # Load pre-trained weights
        likelihood = self.config["sngp"]["likelihood"]["model"]
        pretrained_weights_path = self.config["sngp"]["training"]["pretrained_weights_path"]
        if pretrained_weights_path:
            print("Loading pre-trained weights...", flush=True)
            with open(pretrained_weights_path, "rb") as f:
                data = pickle.load(f)
            self.training_steps = data["step"]
            _, _, get_params = adam(step_size=0.1)
            if likelihood == "Categorical":
                self.params = get_params(data["opt_state"])
            else:
                self.params, ll_rho = get_params(data["opt_state"])
                self.ll_scale = jax.nn.softplus(ll_rho)

        spectral_norm_fn = lambda p: hk.SNParamsTree()(p)
        self.fwd_sn = hk.transform_with_state(spectral_norm_fn)
        from haiku.data_structures import partition
        nn_params, _= partition(lambda m, n, p: "RandomFourierFeatures" not in m and "b" not in n, self.params)
        self.sn_params, self.sn_state = self.fwd_sn.init(self.key, nn_params)
        self.sn_params, self.sn_state = self.fwd_sn.apply(self.sn_params, self.sn_state, self.key, nn_params)

        # Posterior mode 
        print("Fitting mean parameters of SNGP...", flush=True)
        self.params, self.state, self.cov = fit_model(
            self.key, 
            self.params,
            self.state,
            self, # model
            self.config, 
            train_dataloader,
            val_dataloader
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
        # Split keys
        key1, key2 = jax.random.split(key)

        # Compute mean and covariance of the functional distribution
        f_mean, f_cov = self.f_distribution_mean_cov(x, key1, mc_samples)

        # Sample from function distribution
        #jax.debug.print("{q}", q=jnp.linalg.eigvalsh(f_cov[:,:,0]))
        sample_fn = lambda _mean, _cov: jax.random.multivariate_normal(key2, _mean, _cov, shape=(mc_samples,))
        f_samples = jax.vmap(sample_fn, in_axes=(-1,-1), out_axes=-1)(f_mean, f_cov)

        return f_samples
    
    
    @partial(jax.jit, static_argnums=(0,3))
    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
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
        f_mean, phi = self.apply_fn(self.params, self.state, key, x, training=False)[0]
        f_cov = jnp.einsum("ni,ijk,jb->nbk", phi, self.cov, phi.T) 
        f_cov += 1e-4 * jax.vmap(jnp.diag, in_axes=-1, out_axes=-1)(jnp.ones(f_cov.shape[1:]))

        return f_mean, f_cov
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_var(
        self,
        x, 
        key, 
        mc_samples
    ):
        f_mean, phi = self.apply_fn(self.params, self.state, key, x, training=False)[0]
        f_var = jnp.einsum("ni,ijk,jn->nk", phi, self.cov, phi.T)
        
        return f_mean, f_var
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        assert self.params, "Model is not trained."

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
        assert self.params, "Model is not trained."

        # Evaluate model
        loss = evaluate_model(
            self.key, 
            self, # model
            dataloader
        )

        return loss
