import jax 

import haiku as hk
import jax.numpy as jnp

from functools import partial

from models.GWI.model_utils.mlp import MLP
from models.GWI.model_utils.cnn import CNN1, CNN2


ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu
}

class BNN:
    """
    Bayesian neural network (BNN) model.
    """
    def __init__(
        self, 
        nn_config,
        inference_config,
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
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]

        # Inference configuration
        self.n_inducing_points = inference_config["n_inducing_points"]

        # Likelihood configuration
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
            self.forward.apply, in_axes=(None, 0, None)
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
            if self.model_type == "MLP":
                return MLP(
                    architecture=self.architecture,
                    activation_fn=self.activation_fn
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


    @partial(jax.jit, static_argnums=(0,3,7))
    def predict_f(
        self,
        mean_params,
        L_params,
        prior,
        inducing_points,
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
        - f_nn (jnp.ndarray): function samples.
        """
        # Split the random key
        key1, key2 = jax.random.split(key)

        # Get mean and covariance of the function distribution
        mean, cov = self.f_distribution(mean_params, L_params, prior, inducing_points, x, key1)
        cov += 1e-10 * jax.vmap(jnp.diag, in_axes=-1, out_axes=-1)(jnp.ones(cov.shape[1:]))

        # Sample 
        sample_fn = lambda _mean, _cov: jax.random.multivariate_normal(key2, _mean, _cov, shape=(mc_samples,))
        f = jax.vmap(sample_fn, in_axes=(-1,-1), out_axes=-1)(mean, cov)
    
        return f
        
        
    @partial(jax.jit, static_argnums=(0,3,7))
    def predict_y(
        self, 
        mean_params, 
        L_params,
        prior, 
        inducing_points,
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
        
        # Sample from the linearized function distribution
        f = self.predict_f(mean_params, L_params, prior, inducing_points, x, key1, mc_samples)

        # Sample from the likelihood distribution
        if self.likelihood == "Gaussian":
            y = f + self.ll_scale * jax.random.normal(key2, shape=f.shape) 
        elif self.likelihood == "Categorical":
            y = jax.nn.softmax(f, axis=-1)
        
        return y
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution(
        self, 
        mean_params, 
        L_params,
        prior, 
        inducing_points,
        x,
        key
    ):
        """
        Return the mean and covariance the linearized functional distribution.

        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution (n_batch, n_classes).
        - cov (jnp.ndarray): covariance of the distribution (n_batch, n_batch, n_classes).
        """
        # Neural network predict function 
        mean = self.forward.apply(mean_params, key, x)

        # Compute the kernel
        cov = self.f_distribution_kernel(L_params, prior, inducing_points, x, x)
        
        return mean, cov
    

    @partial(jax.jit, static_argnums=(0,3))
    def f_diag_distribution(
        self, 
        mean_params, 
        L_params, 
        prior, 
        inducing_points,
        x, 
        key
    ):
        """
        Return the mean and diagonalized covariance the linearized functional distribution.

        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution (n_batch, n_classes).
        - diag_cov (jnp.ndarray): diagonal covariance of the distribution (n_batch, n_classes).
        """
        # Compute the mean of the distribution 
        mean = self.forward.apply(mean_params, key, x) 

        # Compute the marginal variance of the distribution 
        x = jnp.expand_dims(x, axis=1)
        kernel_fn = lambda z: self.f_distribution_kernel(
            L_params, prior, inducing_points, z, z
        )
        diag_cov = jax.vmap(kernel_fn, in_axes=0)(x).reshape(x.shape[0], -1)

        return mean, diag_cov


    @partial(jax.jit, static_argnums=(0,2))
    def f_distribution_kernel(
        self, 
        L_params,
        prior, 
        inducing_points,
        x1, 
        x2
    ):
        """
        Compute the kernel induced by the linearized BNN.

        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x1 (jnp.ndarray): input data.
        - x2 (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.

        returns
        - kernel (jnp.ndarray): kernel matrix (n_batch, n_batch, n_classes).
        """
        # ensure_pos_diag
        L = jax.vmap(jnp.tril, in_axes=-1, out_axes=-1)(L_params) + 1e-10 * jax.vmap(jnp.diag, in_axes=-1, out_axes=-1)(jnp.ones(L_params.shape[1:]))
        L = jax.vmap(self.ensure_pos_diag, in_axes=-1, out_axes=-1)(L)

        inducing_points_x = inducing_points[0]
        # Parition parameters 
        K_x1x2 = prior.cross_covariance(x1, x2) # (n_batch, n_batch, n_classes)
        K_ZZ = prior(inducing_points_x)[1] + 1e-10 * jax.vmap(jnp.diag, in_axes=-1, out_axes=-1)(jnp.ones(L_params.shape[1:]))
        K_x1Z = prior.cross_covariance(x1, inducing_points_x) # (x_batch, inducing_points, n_classes)
        K_x2Z = prior.cross_covariance(x2, inducing_points_x) # (x_batch, inducing_points, n_classes)
        cov_fn = lambda _K_x1x2, _K_ZZ, _K_x1Z, _K_x2Z, _L: _K_x1x2 - _K_x1Z @ jnp.linalg.solve(_K_ZZ, _K_x2Z.T) + _K_x1Z @ _L @ _L.T @ _K_x2Z.T
        kernel = jax.vmap(cov_fn, in_axes=(-1,-1,-1,-1,-1), out_axes=-1)(K_x1x2, K_ZZ, K_x1Z, K_x2Z, L)

        return kernel
    

    @partial(jax.jit, static_argnums=(0,))
    def ensure_pos_diag(self, L):
        v = jnp.diagonal(L)
        v = jnp.clip(v, a_min=1e-6)
        mask = jnp.diag(jnp.ones_like(v))
        L = mask * jnp.diag(v) + (1. - mask) * L
        return L


    @partial(jax.jit, static_argnums=(0,))
    def mean(
        self, 
        mean_params,
        x,
        key
    ):
        """
        Return the mean and covariance the linearized functional distribution.

        params:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        - rho_params (jax.tree_util.pytree): pre-activated scale parameters of the BNN.
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution (n_batch, n_classes).
        """
        # Neural network predict function 
        mean = self.forward.apply(mean_params, key, x)
        
        return mean