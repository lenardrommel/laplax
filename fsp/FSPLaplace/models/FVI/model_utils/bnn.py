import jax 

import haiku as hk
import jax.numpy as jnp

from functools import partial

from models.FVI.model_utils.mlp import MLP
from models.FVI.model_utils.cnn import CNN1, CNN2


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}

class BNN:
    """
    Bayesian neural network (BNN) model.
    """
    def __init__(
        self, 
        nn_config,
        likelihood_config
    ):
        """
        Initialize the BNN.

        params:
        - nn_config (dict): configuration dictionary for the neural network.
        - likelihood_config (dict): configuration dictionary for the likelihood model.
        """
        # Neural network configuration
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.last_layer_vi = nn_config["last_layer_vi"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]
        self.periodic_features = nn_config["periodic_features"]

        # Likelihood configuration
        self.likelihood = likelihood_config["model"]
        if self.likelihood == "Categorical":
            self.n_classes = likelihood_config["n_classes"]
            self.architecture += [self.n_classes]
            self.ll_scale = 1.0
        elif self.likelihood == "Gaussian": 
            self.ll_scale = likelihood_config["scale"]
            self.architecture += [1]
        else:
            raise ValueError("Invalid likelihood model.")

        # Initialize model
        self.forward = hk.transform(self.make_forward_fn())
        self.training_steps = 0


    @property
    def apply_fn(
        self
    ):
        """
        Build vectorized apply function.

        returns:
        - apply (callable): vectorized apply function.
        """
        return jax.vmap(
            self.forward.apply, in_axes=(0, 0, None)
        )


    def make_forward_fn(
        self
    ):
        """
        Build forward function.

        returns:
        - forward (callable): forward function.
        """
        def forward_fn(x):
            if self.periodic_features:
                x4 = jnp.sin(2 * jnp.pi * x[:,-1] / self.periodic_features).reshape(-1, 1)
                x5 = jnp.cos(2 * jnp.pi * x[:,-1] / self.periodic_features).reshape(-1, 1)
                x = jnp.concatenate([x, x4, x5], axis=-1)

            if self.model_type == "MLP":
                return MLP(
                    architecture=self.architecture,
                    activation_fn=self.activation_fn
                )(x)
            elif self.model_type == "HelmholtzMLP":
                x, z = x[:, :-1].reshape(-1,1,2), x[:, -1].reshape(-1, 1, 1)
                z_int = jnp.array(z, dtype=int).reshape(-1)
                potential_MLP = lambda _x, _z: MLP(
                    self.architecture,
                    self.activation_fn
                )(jnp.concatenate([_x, _z], axis=-1))#(x, z)
                stream_MLP = lambda _x, _z: MLP(
                    self.architecture,
                    self.activation_fn
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
                NotImplementedError()

        return forward_fn


    @partial(jax.jit, static_argnums=(0,5))
    def predict_f(
        self,
        mean_params, 
        rho_params,
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the linearized function distribution.

        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - f (jnp.ndarray): function samples.
        """
        # Handle random keys
        keys = jax.random.split(key, num=mc_samples+1)
        key1, keys = keys[0], keys[1:]

        # Partition parameters
        params_sto, params_det = self.partition_stochastic_params(mean_params)
        params_sig = jax.tree_util.tree_map(lambda p: jax.nn.softplus(p), rho_params)

        # Sample parameters
        eps = jax.tree_util.tree_map(lambda x: jax.random.normal(key1, (mc_samples,) + x.shape), params_sig)
        params_sample = jax.tree_util.tree_map(lambda m, e, s: m + e * s, params_sto, eps, params_sig)
        if self.last_layer_vi:
            params_sample = hk.data_structures.merge(
                params_sample, 
                jax.tree_util.tree_map(
                    lambda x: jnp.stack([x for _ in range(mc_samples)], axis=0), params_det
                )
            )

        # Neural network prediction
        f = self.apply_fn(params_sample, keys, x)
        
        return f
        
        
    @partial(jax.jit, static_argnums=(0,5))
    def predict_y(
        self, 
        mean_params, 
        rho_params,
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the linearized predictive distribution.
        
        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y (jnp.ndarray): function samples.
        """    
        key1, key2 = jax.random.split(key)
        
        # Sample from function distribution
        f = self.predict_f(mean_params, rho_params, x, key1, mc_samples)

        # Sample from likelihood distribution
        if self.likelihood == "Gaussian":
            y = f + self.ll_scale*jax.random.normal(key2, shape=f.shape) 
        elif self.likelihood == "Categorical":
            y = jax.nn.softmax(f, axis=-1)

        return y


    @partial(jax.jit, static_argnums=(0,))
    def partition_stochastic_params(
        self, 
        params
    ):
        """
        Split parameters into stochastic and non-stochastic.
        
        params:
        - params (jax.tree_util.pytree): parameters of the BNN.

        returns:
        - stochastic_params (jax.tree_util.pytree): stochastic parameters of the BNN.
        - non_stochastic_params (jax.tree_util.pytree): non-stochastic parameters of the BNN.
        """
        if self.last_layer_vi:
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