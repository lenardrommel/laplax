import gc
import jax 
import copy
import pickle 

import haiku as hk
import jax.flatten_util
import numpy as np
import jax.numpy as jnp

from jax import eval_shape
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.example_libraries.optimizers import adam 
from haiku.data_structures import merge


from models.Laplace.model_utils.mlp import MLP
from models.Laplace.model_utils.cnn import CNN, CNN2
from models.Laplace.training_utils.plot_utils import plot_function_samples
from models.Laplace.training_utils.training import fit_model, evaluate_model
from data_utils.era5_data_utils import gcs_to_cartesian
from models.Laplace.training_utils.laplace_posterior import (
    FullLaplacePosterior,
    KFACLaplacePosterior, 
    DiagLaplacePosterior,
    MAPLaplacePosterior,
)

ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "lrelu": jax.nn.leaky_relu,
    "elu": jax.nn.elu,
}


class LaplaceBNN:
    """
    Placeholder for the Laplace BNN.
    """
    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the Laplace BNN model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        # General configuration
        self.key = key
        self.config = copy.deepcopy(config)
        
        # Training configuration
        training_config = self.config["laplace"]["training"]
        self.pretrained_weights_path = training_config["pretrained_weights_path"]

        # Neural network configuration
        nn_config = self.config["laplace"]["neural_net"]
        self.model_type = nn_config["type"]
        self.architecture = nn_config["architecture"]
        self.activation_fn = ACTIVATIONS[nn_config["activation_fn"]]
        self.periodic_features = nn_config["periodic_features"]

        # Inference configuration
        inference_config = self.config["laplace"]["inference"]
        self.cov_type = inference_config["cov_type"]

        # Prior configuration
        prior_config = self.config["laplace"]["prior"]
        self.prior_structure = prior_config["structure"]
        self.prior_scale_init = prior_config["scale_init"]

        # Likelihood configuration
        likelihood_config = self.config["laplace"]["likelihood"]
        self.likelihood = likelihood_config["model"]
        self.ll_scale = likelihood_config["scale_init"]
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

            if (self.config["experiment"]["name"] in ["era5_interpolation", "era5_extrapolation", "hpo_era5_interpolation", "hpo_era5_extrapolation", "era5_extrapolation_final_experiment"]):
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
                potential_MLP = lambda _x, _z: MLP(
                    architecture,
                    activation_fn
                )(jnp.concatenate([_x, _z], axis=-1))#(x, z)
                stream_MLP = lambda _x, _z: MLP(
                    architecture,
                    activation_fn
                )(jnp.concatenate([_x, _z], axis=-1))
                grad_potential = jax.vmap(jax.jacfwd(potential_MLP, argnums=0))(x, z).reshape(x.shape[0],2)[jnp.arange(x.shape[0]),z_int].reshape(-1,1)   
                grad_stream = jax.vmap(jax.jacfwd(stream_MLP, argnums=0))(x, z).reshape(x.shape[0],2)[jnp.arange(x.shape[0]),1-z_int].reshape(-1,1)
                out = grad_potential + (-1)**z_int.reshape(-1, 1) * grad_stream
                return out
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
        # Initialize model and load pre-trained weights
        if not hasattr(self, "sto_params"):
            x_init = next(iter(train_dataloader))[0]
            self.sto_params, self.det_params, self.state = self.initialize_model(x_init)
            print("Previous training steps: ", self.training_steps, flush=True)
            
        # Create helper function
        self.unravel_params = ravel_pytree(self.sto_params)[1]

        # Build prior
        if not hasattr(self, "prior_scale_params"):
            print("Building prior...", flush=True)
            self.prior_scale_params = self.build_prior_scale_params()

        # Posterior mode 
        print("Fitting mean and prior parameters of Laplace approximation...", flush=True)
        self.sto_params, self.det_params, self.prior_scale_params, self.ll_scale, self.state = fit_model(
            self, # model
            train_dataloader,
            val_dataloader
        )
        print("Likelihood scale:", self.ll_scale, flush=True)

        # Build and fit the Laplace approximation
        print("Fitting covariance parameters of Laplace approximation...", flush=True)
        prior_scale, _ = self.expand_prior(self.prior_scale_params)
        laplace_args = (self, prior_scale, False)
        if self.cov_type in ["full", "last_layer"]:
            laplace_posterior = FullLaplacePosterior(*laplace_args).fit(train_dataloader)
        elif self.cov_type == "diag":
            laplace_posterior = DiagLaplacePosterior(*laplace_args).fit(train_dataloader)
        elif self.cov_type == "map":
            laplace_posterior = MAPLaplacePosterior(*laplace_args).fit(train_dataloader)
        elif self.cov_type == "kfac":
            laplace_posterior = KFACLaplacePosterior(*laplace_args).fit(train_dataloader)
        else:
            raise ValueError("Invalid covariance type.")
        
        # Compute posterior covariance
        self.posterior_covariance_sq_root = laplace_posterior.posterior_covariance_sq_root()
        self.log_marginal_likelihood = 0 #laplace_posterior.log_marginal_likelihood()
        del laplace_posterior
        gc.collect()
        jax.clear_caches()

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
        print(f'Number of parameters: {hk.data_structures.tree_size(params)}', flush=True)    

        # Set training steps
        self.training_steps = 0

        # Load pre-trained weights 
        if self.pretrained_weights_path:
            with open(self.pretrained_weights_path, "rb") as f:
                data = pickle.load(f)
            self.training_steps = data["lp_step"]
            get_params = adam(step_size=0.)[2]
            self.prior_scale_params = jax.nn.softplus(get_params(data["mll_opt_state"]))
            if self.likelihood == "Gaussian":
                params, ll_rho = get_params(data["lp_opt_state"])
                self.ll_scale = jax.nn.softplus(ll_rho)
            elif self.likelihood == "Categorical":
                params = get_params(data["lp_opt_state"])

        # Partion MAP parameters and batch norm parameters
        sto_params, det_params  = self.partition_inference_parameters(params)

        return sto_params, det_params, state
    

    def build_prior_scale_params(
      self
    ):
        """
        Build the prior distribution's scale parameter.

        returns:
        - prior (jnp.array): prior distribution scale parameter.
        """
        if self.prior_structure == "parameterwise":
            dim = hk.data_structures.tree_size(self.sto_params)
        elif self.prior_structure == "global":
            dim = 1
        elif self.prior_structure == "layerwise":
            dim = sum(
                len(self.sto_params[layer])
                for layer in hk.data_structures.to_mutable_dict(self.sto_params)
            )
        
        return jnp.ones(dim) * self.prior_scale_init


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
        assert self.sto_params, "Model is not trained."

        # Split keys
        key1, key2 = jax.random.split(key)

        # Sample parameters
        if self.cov_type in ["full", "last_layer"]:
            print(self.posterior_covariance_sq_root.shape)
            params_sample = jnp.einsum(
                "ij,mj->mi",
                self.posterior_covariance_sq_root, # (n_weights, k)
                jax.random.normal(key1, shape=(mc_samples, self.posterior_covariance_sq_root.shape[1]))
            )  
        elif self.cov_type == "kfac":
            params_sample = []
            for i in range(len(self.posterior_covariance_sq_root)):
                params_sample += [jnp.einsum(
                    "ij,mj->mi",
                    self.posterior_covariance_sq_root[i], # (n_weights, k)
                    jax.random.normal(key1, shape=(mc_samples, self.posterior_covariance_sq_root[i].shape[1]))
                )]
            params_sample = jnp.concatenate(params_sample, axis=1)
        else: # if cov is diagonal i.e "map" or "diag"
            params_sample = self.posterior_covariance_sq_root * jax.random.normal(
                key=key1, 
                shape=(mc_samples, self.posterior_covariance_sq_root.shape[0])
            )
        params_sample = jax.vmap(self.unravel_params)(params_sample)

        # Compute Jacobian 
        f = lambda p: self.apply_fn(merge(p, self.det_params), self.state, key2, x, training=False)[0]  
        f_m, f_res = jax.vmap(lambda p: jax.jvp(f, (self.sto_params,), (p,)))(params_sample)

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

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y (jnp.array): function samples.
        """
        assert self.sto_params, "Model is not trained."
        
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
        assert self.sto_params, "Model is not trained."

        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean 
        mean = self.apply_fn(merge(self.sto_params, self.det_params), self.state, key1, x, training=False)[0]

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
        Return the mean and variance of the linearized function distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        
        returns:
        - mean (jnp.array): mean of the distribution.
        - variance (jnp.array): variance of the distribution.
        """
        assert self.sto_params, "Model is not trained."

        # Split keys
        key1, key2 = jax.random.split(key)

        # Mean
        mean = self.apply_fn(merge(self.sto_params, self.det_params), self.state, key1, x, training=False)[0]
        
        # Variance
        x = jnp.expand_dims(x, axis=1)
        kernel_fn = lambda z: self.f_distribution_kernel(z, z, key2)
        variance = jax.vmap(kernel_fn, in_axes=0)(x).reshape(x.shape[0], -1)

        return mean, variance
    
    
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
        assert self.sto_params, "Model is not trained."

        @jax.jit
        def classwise_kernel(k):
            # Define predict function
            f1 = lambda p: self.forward.apply(merge(p, self.det_params), self.state, key, x1, training=False)[0][:,k].reshape(-1) 
            f2 = lambda p: self.forward.apply(merge(p, self.det_params), self.state, key, x2, training=False)[0][:,k].reshape(-1) 

            # Compute kernel - jac_f1 @ cov @ jac_f2.T
            if self.cov_type in ["full", "last_layer"]:
                j1_sq = jax.vmap(
                    lambda a: jax.jvp(f1, (self.sto_params,), (self.unravel_params(a),))[1], 
                    in_axes=-1, out_axes=-1
                )(self.posterior_covariance_sq_root)
                j2_sq = jax.vmap(
                    lambda a: jax.jvp(f2, (self.sto_params,), (self.unravel_params(a),))[1], 
                    in_axes=-1, out_axes=-1
                )(self.posterior_covariance_sq_root)
                kernel = jax.tree_map(lambda x1, x2: x1 @ x2.T, j1_sq, j2_sq)
            elif self.cov_type == "kfac":
                kernel = jnp.zeros((x1.shape[0], x2.shape[0]))
                for layer_module, cov_sq in zip(self.sto_params, self.posterior_covariance_sq_root):
                    # Split parameters
                    sto_params, det_params = hk.data_structures.partition(lambda m, n, p: m == layer_module , self.sto_params)
                    det_params = hk.data_structures.merge(det_params, self.det_params)
                    unravel_params = jax.flatten_util.ravel_pytree(sto_params)[1]
                    # Define predict function
                    f1 = lambda p: self.forward.apply(merge(p, det_params), self.state, key, x1, training=False)[0][:,k].reshape(-1) 
                    f2 = lambda p: self.forward.apply(merge(p, det_params), self.state, key, x2, training=False)[0][:,k].reshape(-1) 
                    # Compute kernel - jac_f1 @ cov @ jac_f2.T
                    j1_sq = jax.vmap(
                        lambda a: jax.jvp(f1, (sto_params,), (unravel_params(a),))[1], 
                        in_axes=-1, out_axes=-1
                    )(cov_sq)
                    j2_sq = jax.vmap(
                        lambda a: jax.jvp(f2, (sto_params,), (unravel_params(a),))[1], 
                        in_axes=-1, out_axes=-1
                    )(cov_sq)
                    kernel += jax.tree_map(lambda x1, x2: x1 @ x2.T, j1_sq, j2_sq)
            else: # if cov is diagonal
                @jax.jit
                def delta_vjp_jvp(delta):
                    delta_vjp = jax.vjp(f2, self.sto_params)[1](delta)[0]
                    vj_prod = jax.tree_map(
                        lambda x1, x2: x1 * x2, self.unravel_params(self.posterior_covariance_sq_root**2), delta_vjp
                    )
                    return jax.jvp(f1, (self.sto_params,), (vj_prod,))[1]
                
                # Compute the kernel
                fx2 = eval_shape(f2, self.sto_params)
                eye = jnp.eye(x1.shape[0])
                kernel = jax.vmap(jax.linear_transpose(delta_vjp_jvp, fx2))(eye)[0].T

            return kernel
        
        # Compute kernel for each class
        kernel = jax.vmap(classwise_kernel)(jnp.arange(self.architecture[-1])).T  

        return kernel
    

    @partial(jax.jit, static_argnums=(0,))
    def expand_prior(
        self,
        v_prior_scale_params
    ):
        """
        Expand prior to the full parameter space.

        params:
        - v_prior_scale_params (jnp.array): prior scale parameters.

        returns:
        - prior_cov (jnp.array): prior covariance.
        - prior_cov_pytree (jax.tree_util.pytree): prior covariance.
        """
        assert self.sto_params, "Model is not trained."

        # Expand prior to the full parameter space
        if self.prior_structure == "parameterwise":
            prior_cov = v_prior_scale_params
        elif self.prior_structure == "global":
            prior_cov = v_prior_scale_params * jnp.ones(hk.data_structures.tree_size(self.sto_params))
        elif self.prior_structure == "layerwise":
            v, i = [], 0
            for layer in hk.data_structures.to_mutable_dict(self.sto_params):
                for m in self.sto_params[layer]:
                    v += [v_prior_scale_params[i] * jnp.ones(np.prod(self.sto_params[layer][m].shape))]
                    i += 1
            prior_cov = jnp.concatenate(v)
        
        # Convert to pytree
        prior_cov_pytree = self.unravel_params(prior_cov)

        return prior_cov, prior_cov_pytree
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        assert self.sto_params, "Model is not trained."

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
        assert self.sto_params, "Model is not trained."

        # Evaluate model
        loss = evaluate_model(
            self.key, 
            self, # model
            dataloader
        )

        return loss
    

    def partition_inference_parameters(
        self, 
        params
    ):
        """
        Split parameters into MAP and Laplace.
        
        params:
        - params (jax.tree_util.pytree): parameters of the BNN.
        
        returns:
        - sto_params (jax.tree_util.pytree): MAP parameters of the BNN.
        - det_params (jax.tree_util.pytree): batch norm parameters of the BNN.
        """
        # Partition last layer parameters
        if self.cov_type == "last_layer":
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

        # Partition batch norm parameters
        sto_params, _det_params = hk.data_structures.partition(
            lambda m, n, p: "batch_norm" not in m, sto_params
        )

        # Merge deterministic parameters
        det_params = merge(det_params, _det_params)

        return sto_params, det_params

    
        
