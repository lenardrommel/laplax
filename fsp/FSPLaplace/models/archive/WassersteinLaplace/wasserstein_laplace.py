import jax 

import haiku as hk
import jax.numpy as jnp

from functools import partial
from jax.example_libraries.optimizers import adam 

from models.WassersteinLaplace.model_utils.mlp import MLP
from models.WassersteinLaplace.model_utils.prior import Prior
from models.WassersteinLaplace.training_utils.inference import wasserstein_laplace_inference
from models.WassersteinLaplace.training_utils.plot_utils import plot_function_samples
from models.WassersteinLaplace.training_utils.training import fit_model, evaluate_model


ACTIVATION_DICT = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}


class WassersteinLaplace:
    """
    Placeholder for the Laplace model.
    """

    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the Laplace model.
        :param key (jax.random.PRNGKey): random key.
        :param config (dict): configuration dictionary.
        """
        self.key = key
        self.key, key1 = jax.random.split(key, num=2)
        self.config = config
        self.stochastic_layers = config["FSPLaplace"]["stochastic_layers"]
    
        # Initialize model
        self.forward = hk.transform(
            self.make_forward_fn(
                config["FSPLaplace"]["architecture"], 
                config["FSPLaplace"]["activation_fn"]
            )
        )

        init_fn, apply_fn = self.forward
        x_init = jnp.ones(
            (config["data"]["batch_size"], config["data"]["feature_dim"])
        )
        self.mean_params = init_fn(key1, x_init)

        # Load Prior
        self.prior = Prior(
            key=key1,
            kernel=config["FSPLaplace"]["prior"]["kernel"], 
            kernel_params=config["FSPLaplace"]["prior"]["parameters"], 
        )

        # Initialize optimizer
        opt_init, self.opt_update, self.get_params = adam(
            config["FSPLaplace"]["training"]["lr"],
            config["FSPLaplace"]["training"]["b1"],
            config["FSPLaplace"]["training"]["b2"],
            config["FSPLaplace"]["training"]["eps"]
        )
        self.opt_state = opt_init(self.mean_params)

    
    @property
    def apply_fn(self):
        """
        Apply function of the MLP.
        """
        return self.forward.apply
    

    def make_forward_fn(
        self, 
        architecture, 
        activation_fn
    ):
        """
        Build foward function.
        :param architecture (list): architecture of the MLP.
        :param activation_fn (str): activation function.
        :returns (function): forward function.
        """
        def forward_fn(x):
            _forward_fn = MLP(
                architecture=architecture,
                activation_fn=ACTIVATION_DICT[activation_fn],
            )
            return _forward_fn(x)

        return forward_fn
        

    def fit(
        self, 
        train_dataloader, 
        val_dataloader
    ):
        """
        Fit the model.
        :param dataloader (DataLoader): data loader.
        """
        # Find posterior mode 
        self.mean_params = fit_model(
            self.key, 
            self.mean_params, 
            self, # model
            self.opt_update, 
            self.opt_state, 
            self.get_params, 
            self.config, 
            train_dataloader, 
            self.prior
        )
        # Inference 
        self.cov = wasserstein_laplace_inference(
            self, # model
            self.mean_params, 
            train_dataloader, 
            self.prior,
            self.key, 
            self.config, 
        )


    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.
        :param dataloader (DataLoader): data loader.
        """
        evaluate_model(
            self.key, 
            self, # model
            dataloader, 
            self.config
        )


    @partial(jax.jit, static_argnums=(0,3,4))
    def f_predict(
        self, 
        x, 
        key, 
        mc_samples, 
        is_training
    ):
        """
        Sample from function distribution induced by the linearized BNN.
        :param x (jax.numpy.ndarray): input data.
        :param key (jax.random.PRNGKey): random key.
        :param mc_samples (int): number of Monte Carlo samples.
        :param is_training (bool): whether the model is in training mode.
        :returns (jax.numpy.ndarray): function samples.
        """
        # Get configuration 
        cov_type = self.config["FSPLaplace"]["cov_type"]

        # Split keys
        key1, key2 = jax.random.split(key)

        # Split parameters
        stochastic_params, static_params = self._split_parameters()

        # Vectorize parameters
        vec_mean_params, params_unravel = jax.flatten_util.ravel_pytree(stochastic_params)

        # Sample parameters
        if cov_type == "full":
            vec_sample_params = jax.random.multivariate_normal(
                key1, 
                mean=jnp.zeros_like(vec_mean_params), 
                cov=self.cov, 
                shape=(mc_samples,)
            )
        elif cov_type == "diag":
            vec_sample_params = self.cov**0.5 * jax.random.normal(
                key1, 
                vec_mean_params.shape
            )

        # Combine inference and static parameters
        sample_params = jax.vmap(params_unravel)(vec_sample_params)
        sample_params = jax.vmap(hk.data_structures.merge, in_axes=(0, None))(sample_params, static_params)

        # GLM 
        fwd = lambda p: self.apply_fn(p, key2, x)        
        logits = fwd(self.mean_params)       

        return logits + jax.vmap(jax.jvp, in_axes=(None, None, 0))(fwd, (self.mean_params,), (sample_params,))[1]

    
    @partial(jax.jit, static_argnums=(0,3,4))
    def y_predict(
        self, 
        x, 
        key, 
        mc_samples, 
        is_training 
    ):
        """
        Sample from predictive distribution induced by the linearized BNN.
        :param x (jax.numpy.ndarray): input data.
        :param key (jax.random.PRNGKey): random key.
        :param mc_samples (int): number of Monte Carlo samples.
        :param is_training (bool): whether the model is in training mode.
        :returns (jax.numpy.ndarray): function samples.
        """
        key1, key2 = jax.random.split(key)
        
        f = self.f_predict(
            x, 
            key1, 
            mc_samples, 
            is_training
        ) 

        y = f + self.likelihood_scale*jax.random.normal(key2, shape=f.shape) 

        return y


    @partial(jax.jit, static_argnums=(0,))
    def f_pred_dist(
        self,
        x, 
        key
    ):
        """
        Return the mean and covariance the functional distribution induced by 
        the linearized BNN.
        :param x (jax.numpy.ndarray): input data.
        :param key (jax.random.PRNGKey): random key.
        :returns (jax.numpy.ndarray): mean of the distribution.
        :returns (jax.numpy.ndarray): covariance of the distribution.
        """
        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean 
        mean = self.apply_fn(self.mean_params, key1, x)
        # Covariance
        cov = self.f_distribution_kernel(x, x, key2)

        return mean, cov
    

    def f_pred_diag_dist(
        self,
        x, 
        key
    ):
        """
        Return the mean and diagonalized covariance the functional distribution 
        induced by the linearized BNN.
        :param params (jax.tree_util.pytree): parameters of the BNN.
        :param x (jax.numpy.ndarray): input data.
        :param key (jax.random.PRNGKey): random key.
        :returns (jax.numpy.ndarray): mean of the distribution.
        :returns (jax.numpy.ndarray): covariance of the distribution.
        """
        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean
        mean = self.apply_fn(self.mean_params, key1, x)
        # Covariance
        x = jnp.expand_dims(x, axis=1)
        kernel_fn = lambda z: self.f_distribution_kernel(z, z, key2)
        diag_cov = jax.vmap(kernel_fn, in_axes=0)(x).reshape(-1)

        return mean, diag_cov
    
    
    @partial(jax.jit, static_argnums=(0,))
    def f_distribution_kernel(
        self, 
        x1, 
        x2, 
        key
    ):
        """
        Evaluate the kernel induced by the linearized BNN.
        :params params (jax.tree_util.pytree): parameters of the BNN.
        :params x1 (jax.numpy.ndarray): input data.
        :params x2 (jax.numpy.ndarray): input data.
        :params key (jax.random.PRNGKey): random key.
        :returns (jax.numpy.ndarray): kernel matrix.
        """
        # Parition parameters 
        params_stochastic, static_params = self._split_parameters()
        
        predict_fn = lambda p, x: self.apply_fn(self._join_parameters(p, static_params), key, x)
        f1 = lambda p: predict_fn(p, x1)
        f2 = lambda p: predict_fn(p, x2)

        # Covariance Jacobian product
        unravel = jax.flatten_util.ravel_pytree(params_stochastic)[1]
        pytree_cov = jax.vmap(unravel)(self.cov)
        SJ = jax.vmap(jax.jvp, in_axes=(None, None, 0))(f1, (params_stochastic,), (pytree_cov,))[1] # (p,n,1)
        leaves = jax.tree_util.tree_flatten(SJ)[0]
        JtS = jnp.concatenate([i.reshape(self.cov.shape[0], -1) for i in leaves], axis=-1).T # (n,p)
        pytree_JtS = jax.vmap(unravel)(JtS)

        # Jacobian covariance Jacobian product
        kernel = jax.vmap(jax.jvp, in_axes=(None, None, 0))(f2, (params_stochastic,), (pytree_JtS,))[1] # (n,n)
        kernel = kernel.reshape(x1.shape[0], x2.shape[0])

        return kernel
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.
        :param dataloader (DataLoader): data loader.
        """
        plot_function_samples(
            self, 
            jax.random.PRNGKey(0), 
            self.config, 
            dataloader
        )
        
    
    def _split_parameters(self):
        """
        Split the model parameters into two sets: 
        -Parameters on which to perform inference
        -Parameters kept as MAP estimates
        """
        stochastic_params, static_params = hk.data_structures.partition(
            lambda m, n, p: self.stochastic_layers[int(m[23:]) if m[23:] else 0], self.mean_params
        )

        return stochastic_params, static_params
    

    def _join_parameters(
        self,
        stochastic_params, 
        static_params
    ):
        """
        Join inference and static parameters into full set of parameters.
        :params stochastic_params: parameters on which to perform inference.
        :params static_params: parameters left as MAP estimates.
        """
        return hk.data_structures.merge(stochastic_params, static_params)