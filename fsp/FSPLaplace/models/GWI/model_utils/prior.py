import jax 
import copy
import wandb

import numpy as np
import jaxkern as jk
import jax.numpy as jnp
import jax.scipy as jsp

from functools import partial
from jax.example_libraries.optimizers import adam 


KERNELS = {
    "RBF": jk.RBF, 
    "Matern12": jk.Matern12, 
    "Matern32": jk.Matern32, 
    "Matern52": jk.Matern52,
    "Linear": jk.Linear, 
    "RationalQuadratic": jk.RationalQuadratic
}

class Prior:
    """
    Gaussian process prior.
    """
    def __init__(
        self, 
        key, 
        inducing_points,
        config
    ):
        """
        Initialize the prior.

        params:
        - key (jax.random.PRNGKey): a random key.
        - dataloader (dataloader): wrapper for the dataset.
        - config (dict): configuration dictionary.
        """
        # Attributs
        self.key = key 
        self.config = config
        self.parameter_tuning = config["gwi"]["prior"]["parameter_tuning"]
        self.feature_dim = np.prod(inducing_points[0].shape[1:])
        
        # Likelihood parameters
        self.likelihood = config["gwi"]["likelihood"]["model"]

        # Kernel parameters
        self.kernel_name = config["gwi"]["prior"]["kernel"]
        self.kernel_params = config["gwi"]["prior"]["parameters"]
        if self.likelihood == "Gaussian":
            self.n_priors = 1
        elif self.likelihood == "Categorical":
            self.n_priors = config["gwi"]["likelihood"]["n_classes"]

        # Build prior
        self._build_standard_prior(config, inducing_points)


    def __call__(
        self, 
        x=None
    ):
        """
        Compute the prior mean and covariance of the prior.

        params:
        - x (jnp.ndarray): the input data.
        
        returns:
        - prior_mean (jnp.ndarray): the prior mean.
        - prior_cov (jnp.ndarray): the prior covariance.
        """
        x = x.reshape(x.shape[0], -1)
        prior_mean = jnp.zeros((x.shape[0], self.n_priors))
        prior_cov = jnp.stack(
            [
                self.kernel.gram(self.params[i], x).to_dense()
                for i in range(self.n_priors)
            ], 
            axis=-1
        )
        
        return prior_mean, prior_cov
    

    def mean_var(
        self, 
        x=None
    ):
        """
        Compute the prior mean and covariance of the prior.

        params:
        - x (jnp.ndarray): the input data.
        
        returns:
        - prior_mean (jnp.ndarray): the prior mean.
        - prior_cov (jnp.ndarray): the prior covariance.
        """
        x = x.reshape(x.shape[0], -1)
        prior_mean = jnp.zeros((x.shape[0], self.n_priors))
        prior_var = jnp.stack(
            [
                jax.vmap(lambda _x: self.kernel.gram(self.params[i], _x).to_dense())(x.reshape(-1, 1, x.shape[-1]))
                for i in range(self.n_priors)
            ], 
            axis=-1
        )
        
        return prior_mean, prior_var
    

    def cross_covariance(
        self,
        x1, 
        x2
    ):
        """
        Compute the prior mean and covariance of the prior.

        params:
        - x1 (jnp.ndarray): the input data.
        - x2 (jnp.ndarray): the input data.
        
        returns:
        - cross_covariance (jnp.ndarray): the prior cross covariance.
        """
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        prior_cross_covariance = jnp.stack(
            [
                self.kernel.cross_covariance(self.params[i], x1, x2)
                for i in range(self.n_priors)
            ], 
            axis=-1
        )
    
        return prior_cross_covariance
    

    def mean(
        self,
        x
    ):
        """
        Compute the prior mean and covariance of the prior.

        params:
        - x1 (jnp.ndarray): the input data.
        - x2 (jnp.ndarray): the input data.
        
        returns:
        - cross_covariance (jnp.ndarray): the prior cross covariance.
        """
        x = x.reshape(x.shape[0], -1)
        prior_mean = jnp.zeros((x.shape[0], self.n_priors))
    
        return prior_mean
    

    def _build_standard_prior(
        self, 
        config, 
        inducing_points
    ):
        """
        Build the kernel for standard tasks.

        params:
        - config (dict): configuration dictionary.
        - dataloader (dataloader): wrapper for the dataset.

        returns:
        - kernel (jaxkern.kernel): the kernel.
        """
        # Build kernel
        self.kernel = KERNELS[self.kernel_name](
            active_dims=list(range(self.feature_dim))
        )

        # Initialize parameters
        keys = jax.random.split(self.key, self.n_priors+1)
        self.key, keys = keys[0], keys[1:]
        self.params = [
            self.kernel.init_params(key) for key in keys
        ]

        # Set prior parameters
        if self.parameter_tuning:
            self._tune_parameters(config, inducing_points)
        else:
            for i in range(self.n_priors):
                self.params[i]["variance"] = self.kernel_params["variance"]
                if self.kernel_name in ["RBF", "Matern12", "Matern32", "Matern52"]:
                    self.params[i]["lengthscale"] = self.kernel_params["lengthscale"]*jnp.ones(self.feature_dim)
                elif self.kernel_name == "RationalQuadratic":
                    self.params[i]["lengthscale"] = self.kernel_params["lengthscale"]*jnp.ones(self.feature_dim)
                    self.params[i]["alpha"] = self.kernel_params["alpha"]


    def _tune_parameters(
        self,   
        config, 
        dataloader 
    ):
        """
        Select prior parameters via maximum marginal likelihood maximization with SGD.

        params:
        - config (dict): configuration dictionary.
        - dataloader (dataloader): wrapper for the dataset.
        """
        # Find optimal parameters
        if self.likelihood == "Gaussian":
            self._tune_parameters_gaussian_likelihood(config, dataloader)
            print(f"Optimal parameters: {self.params}", flush=True)
        elif self.likelihood == "Categorical":
            self._tune_parameters_categorical_likelihood(config, dataloader)


    def _tune_parameters_gaussian_likelihood(
        self,   
        config, 
        inducing_points 
    ):
        """
        Select prior parameters via maximum marginal likelihood maximization with SGD.

        params:
        - config (dict): configuration dictionary.
        - dataloader (dataloader): wrapper for the dataset.
        """
        # Get configuration
        lr = config["gwi"]["prior"]["lr"]
        ll_scale = config["gwi"]["likelihood"]["scale"]
        nb_epochs = config["gwi"]["prior"]["nb_epochs"]
        validation_freq = config["gwi"]["neural_net"]["validation_freq"]
        early_stopping_patience = config["gwi"]["training"]["patience"]

        # Initialize optimizers
        opt_init, opt_update, get_params = adam(step_size=lr)
        z_params = self._to_unconstrained(copy.deepcopy(self.params))
        ll_rho = jnp.log(jnp.exp(ll_scale)-1)
        params_init = (z_params, ll_rho)
        opt_state = opt_init(params_init)
        print(f"Initial parameters: {self.params}", flush=True)

        # Early stopping initialization
        optimal_n_mll, no_improve_count, optimal_state = jnp.inf, 0, None

        # Training loop
        step = 0
        x, y = inducing_points
        for epoch in range(nb_epochs):
            # Update kernel parameters
            opt_state, n_mll = self._update_n_mll_gaussian(
                opt_state,
                opt_update,
                get_params,
                x, 
                y,
                step
            )
            step += 1

            # Log negative marginal likelihood
            wandb.log({"prior/n_mll": n_mll})            
            if epoch % validation_freq == 0 or epoch == nb_epochs - 1:    
                print(f"Epoch {epoch} - n_mll: {n_mll}", flush=True)
            
            # Early stopping
            if n_mll < optimal_n_mll:
                optimal_n_mll = n_mll
                optimal_state = opt_state
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping_patience:
                    opt_state = optimal_state
                    print("Early stopping.", flush=True)
                    break
        
        # Set optimal parameters
        z_params, ll_rho = get_params(opt_state)
        self.ll_scale = jax.nn.softplus(ll_rho)
        print("ll_scale", self.ll_scale, flush=True)

        # Enforce parameter positivity constraints
        self.params = self._to_constrained(z_params)

    
    @partial(jax.jit, static_argnums=(0,2,3))
    def _update_n_mll_gaussian(
        self,
        opt_state,
        opt_update,
        get_params,
        x, 
        y,
        step
    ):
        """
        Update step on the mll loss.

        params:
        - opt_state (jax.tree_util.PyTree): optimizer state.
        - opt_update (jax.tree_util.PyTree): optimizer update function.
        - get_params (jax.tree_util.PyTree): optimizer get parameters function.
        - x (jnp.ndarray): input features.
        - y (jnp.ndarray): targets.
        - step (int): current step.

        returns:
        - opt_state (jax.tree_util.PyTree): optimizer state.
        - loss (float): negative marginal likelihood.
        """
        # Get parameters
        z_params, ll_rho = get_params(opt_state)

        # Update parameters
        loss, grads = jax.value_and_grad(self._n_mll_gaussian, argnums=(0,1))(
            z_params,
            ll_rho,
            x,
            y
        )
        opt_state = opt_update(step, grads, opt_state)

        return opt_state, loss

    
    @partial(jax.jit, static_argnums=(0,))
    def _n_mll_gaussian(
        self, 
        z_params,
        ll_rho,   
        x, 
        y
    ):
        """
        Compute the negative marginal likelihood of the batch.

        params: 
        - z_params (dict): unconstained prior parameters.
        - ll_rho (float): pre-activated likelihood scale.
        - x (jnp.ndarray): input features.
        - y (jnp.ndarray): targets.
        
        returns:
        - mll (float): the negative marginal likelihood of the batch.
        """
        # Flatten input
        x = x.reshape(x.shape[0], -1)

        # Enforce parameter positivity constraints
        ll_scale = jax.nn.softplus(ll_rho)
        params = self._to_constrained(z_params)

        # Compute the prior mean and covariance
        prior_mean = jnp.zeros(x.shape[0])
        prior_cov = self.kernel.gram(params[0], x).to_dense() + 1e-6 * jnp.eye(x.shape[0])

        # Compute the marginal likelihood mean and covariance
        ml_mean = prior_mean
        ml_cov = prior_cov + ll_scale**2 * jnp.eye(prior_cov.shape[0])

        # Compute the marginal likelihood
        mll = jsp.stats.multivariate_normal.logpdf(y.reshape(-1), mean=ml_mean, cov=ml_cov)

        return -mll.sum() 
    

    def _tune_parameters_categorical_likelihood(
        self,   
        config, 
        inducing_points 
    ):
        """
        Select prior parameters via maximum marginal likelihood maximization with SGD.

        params:
        - config (dict): configuration dictionary.
        - dataloader (dataloader): wrapper for the dataset.
        """
        # Get configuration
        lr = config["gwi"]["prior"]["lr"]
        alpha_eps = config["gwi"]["prior"]["alpha_eps"]
        nb_epochs = config["gwi"]["prior"]["nb_epochs"]
        validation_freq = config["gwi"]["neural_net"]["validation_freq"]
        early_stopping_patience = config["gwi"]["training"]["patience"]

        # Initialize optimizer
        opt_init, opt_update, get_params = adam(step_size=lr)
        z_params = self._to_unconstrained(copy.deepcopy(self.params))
        ll_scale = config["gwi"]["likelihood"]["scale"] * jnp.ones((self.n_priors,))
        ll_rho = jnp.log(jnp.exp(ll_scale)-1)
        params_init = (z_params, ll_rho)
        opt_state = opt_init(params_init)

        # Early stopping initialization
        optimal_n_mll, no_improve_count, optimal_state = jnp.inf, 0, None 

        # Training
        step = 0
        x, y = inducing_points
        for epoch in range(nb_epochs):
            # Update parameters
            opt_state, n_mll = self._update_n_mll_categorical(
                opt_state,
                opt_update,
                get_params,
                alpha_eps,
                x, 
                y, 
                step
            )
            step += 1
            # Log negative marginal likelihood
            wandb.log({"prior/n_mll": n_mll})            
            if epoch % validation_freq == 0:  
                print(f"Epoch {epoch} - n_mll: {n_mll}", flush=True)
            
            # Early stopping
            if n_mll < optimal_n_mll:
                optimal_n_mll = n_mll
                optimal_state = opt_state
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stopping_patience:
                    opt_state = optimal_state
                    print("Early stopping.", flush=True)
                    break
        
        # Set optimal parameters
        z_params, ll_rho = get_params(opt_state)
        self.ll_scale = jax.nn.softplus(ll_rho)
        self.params = self._to_constrained(z_params)
        
        print("ll_scale", self.ll_scale, flush=True)
        

    @partial(jax.jit, static_argnums=(0,2,3,4))
    def _update_n_mll_categorical(
        self,
        opt_state,
        opt_update,
        get_params,
        alpha_eps,
        x, 
        y, 
        step
    ):
        """
        Update step on the mll loss.

        params:
        - opt_state (jax.tree_util.PyTree): optimizer state.
        - opt_update (jax.tree_util.PyTree): optimizer update function.
        - get_params (jax.tree_util.PyTree): optimizer get parameters function.
        - alpha_eps (float): alpha + eps.
        - x (jnp.ndarray): input features.
        - y (jnp.ndarray): targets.
        - step (int): current step.

        returns:
        - opt_state (jax.tree_util.PyTree): optimizer state.
        - loss (float): negative marginal likelihood.
        """
        # Get parameters
        z_params, ll_rho = get_params(opt_state)
        
        # Update parameters
        loss, grads = jax.value_and_grad(self._n_mll_categorical, argnums=(0,1))(
            z_params, 
            ll_rho, 
            alpha_eps,
            x, 
            y
        )
        opt_state = opt_update(step, grads, opt_state)

        return opt_state, loss
    

    @partial(jax.jit, static_argnums=(0,3))
    def _n_mll_categorical(
        self, 
        z_params,
        ll_rho,
        alpha_eps,
        x, 
        y
    ):
        """
        Regress on labels. 

        params:
        - z_params (dict): unconstained prior parameters.
        - ll_rho (float): pre-activated likelihood scale.
        - alpha_eps (float): alpha + eps.
        - x (jnp.ndarray): input features.
        - y (jnp.ndarray): targets.

        returns:
        - mll (float): the negative marginal likelihood of the batch.
        """
        # Flatten input
        x = x.reshape(x.shape[0], -1)

        # Enforce parameter positivity constraints
        ll_scale = jax.nn.softplus(ll_rho)
        params = self._to_constrained(z_params)

        # Compute labels
        labels = jax.nn.one_hot(y.reshape(-1), num_classes=self.n_priors) # (n_batch, n_classes)
        labels = jnp.where(
            labels==1, 
            jnp.log(1+alpha_eps)-0.5*jnp.log(1/(1+alpha_eps)+1), 
            jnp.log(alpha_eps)-0.5*jnp.log(1/alpha_eps+1)
        )
        ll_var = jnp.where(
            labels==1, 
            jnp.log(1/(1+alpha_eps)+1), 
            jnp.log(1/alpha_eps+1)
        ) # (n_batch, n_classes)
        ll_var = jax.vmap(jnp.diag, in_axes=1, out_axes=2)(ll_var)

        # Compute the prior mean and covariance
        ml_mean = jnp.zeros((x.shape[0], self.n_priors))
        ml_cov = jnp.stack(
            [
                self.kernel.gram(params[i], x).to_dense() + (1e-6 + ll_scale[i]**2) * jnp.eye(x.shape[0])
                for i in range(self.n_priors)
            ], 
            axis=-1
        )
        ml_cov += ll_var

        # Compute the marginal likelihood
        mll = jsp.stats.multivariate_normal.logpdf(labels.T, mean=ml_mean.T, cov=ml_cov.T).sum()

        return -mll.sum() 


    def _to_constrained(
        self, 
        params
    ):
        """
        Map parameters to the unconstrained space.

        params:
        - params (dict): unconstained prior parameters.

        returns:
        - params (dict): constrained prior parameters.
        """

        for i in range(self.n_priors):
            for k in params[i].keys():
                params[i][k] = jax.nn.softplus(params[i][k])
        
        return params
    

    def _to_unconstrained(
        self, 
        params
    ):
        """
        Map parameters to the constrained space.

        params:
        - params (dict): unconstained prior parameters.

        returns:
        - params (dict): constrained prior parameters.
        """

        for i in range(self.n_priors):
            for k in params[i].keys():
                params[i][k] = jnp.log(jnp.exp(jnp.ones_like(params[i][k]))-1)
        
        return params
    
