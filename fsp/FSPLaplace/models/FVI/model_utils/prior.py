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
    "RationalQuadratic": jk.RationalQuadratic, 
    "Periodic": jk.Periodic, 
    "White": jk.White
}


class HelmholtzKernel(jk.base.AbstractKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    
    def __init__(self):
        self.potential_kernel =jk.RBF(active_dims=[0, 1])
        self.stream_kernel =jk.RBF(active_dims=[0, 1])
        
        super().__init__()


    def __call__(self, params, X, Xp):
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        sign = (-1) ** (z + zp)

        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        kernel_fn = lambda x1, x2: self.potential_kernel(params["potential_kernel"], x1, x2)
        potential_dvtve = -jnp.array(
            jax.hessian(kernel_fn)(X, Xp), dtype=jnp.float64
        )[z][zp]
        kernel_fn = lambda x1, x2: self.stream_kernel(params["stream_kernel"], x1, x2)
        stream_dvtve = -jnp.array(
            jax.hessian(kernel_fn)(X, Xp), dtype=jnp.float64
        )[1 - z][1 - zp]

        return potential_dvtve + sign * stream_dvtve
    
    def init_params(self, key):
        potential_params = self.potential_kernel.init_params(key)
        stream_params = self.stream_kernel.init_params(key)
        return {"potential_kernel": potential_params, "stream_kernel": stream_params}
    

class Prior:
    """
    Gaussian process prior.
    """
    def __init__(
        self, 
        key, 
        dataloader,
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
        self.parameter_tuning = config["fvi"]["prior"]["parameter_tuning"]
        if dataloader:
            dataloader.batch_size = config["fvi"]["prior"]["batch_size"]
            self.feature_dim = dataloader.feature_dim
        else:
            self.feature_dim = config["data"]["feature_dim"]
        
        # Likelihood parameters
        self.likelihood = config["fvi"]["likelihood"]["model"]

        # Kernel parameters
        self.kernel_name = config["fvi"]["prior"]["kernel"]
        self.kernel_params = config["fvi"]["prior"]["parameters"]
        if self.likelihood == "Gaussian":
            self.n_priors = 1
        elif self.likelihood == "Categorical":
            self.n_priors = config["fvi"]["likelihood"]["n_classes"]

        # Build prior
        self._build_standard_prior(config, dataloader)


    def __call__(
        self,
        x
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
                self.kernel.gram(self.params[i], x).to_dense() + 1e-10 * jnp.eye(x.shape[0])
                for i in range(self.n_priors)
            ], 
            axis=-1
        )
    
        return prior_mean, prior_cov
    

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
    

    def cross_covariance_k(
        self,
        x1, 
        x2, 
        k_idx
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
        prior_cross_covariance = self.kernel.cross_covariance(
            self.params[k_idx], x1, x2
        )
        
        return prior_cross_covariance
    
    def covariance_trace(
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
        x = x.reshape(x.shape[0], 1, -1)

        prior_trace = jnp.stack(
            [jax.vmap(lambda _x: self.kernel.gram(self.params[i], _x).to_dense())(x).sum() for i in range(self.n_priors)], 
            axis=-1
        )
        
        return prior_trace


    def _build_standard_prior(
        self, 
        config, 
        dataloader
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
        if self.kernel_name == "Periodic":
            # self.kernel = KERNELS["Periodic"](active_dims=list(range(self.feature_dim)))
            k1 = KERNELS["Periodic"](active_dims=list(range(self.feature_dim)))
            k2 = KERNELS["Matern52"](active_dims=list(range(self.feature_dim)))
            k_time = jk.ProductKernel(
                kernel_set=[k1,k2], active_dims=list(range(self.feature_dim))
            )
            k_space = KERNELS["Matern12"](list(range(self.feature_dim)))
            self.kernel = jk.SumKernel(
                kernel_set=[k_time,k_space], active_dims=list(range(self.feature_dim))
            )
        elif self.kernel_name == "Helmholtz":
            self.kernel = HelmholtzKernel()
        elif self.kernel_name == "MonaLoaKernel":
            # long-term seasonal trend
            k1 = KERNELS["RBF"](active_dims=[0])
            # Seasonal kernel 
            _k1 = KERNELS["Periodic"](active_dims=[0])
            _k2 = KERNELS["RBF"](active_dims=[0])
            k2 = jk.ProductKernel(kernel_set=[_k1,_k2], active_dims=[0])
            # Irregular trend
            k3 = KERNELS["RationalQuadratic"](active_dims=[0])
            # Noise kernel
            _k1 = KERNELS["RBF"](active_dims=[0])
            _k2 = KERNELS["White"](active_dims=[0])
            k4 = jk.SumKernel(kernel_set=[_k1,_k2], active_dims=[0])
            # Sum kernel
            self.kernel = jk.SumKernel(
                kernel_set=[k1,k2,k3,k4], active_dims=[0]
            )
        else:
            self.kernel = KERNELS[self.kernel_name](list(range(self.feature_dim)))

        # Initialize parameters
        keys = jax.random.split(self.key, self.n_priors+1)
        self.key, keys = keys[0], keys[1:]
        self.params = [self.kernel.init_params(key) for key in keys]
        if self.kernel_name == "MonaLoaKernel":
            # X_std = 8.667453442321532 # dataloader.dataset.X.std()
            # y_std = 13.72968828243573 #dataloader.dataset.y.std()
            x_std = 10.109848173957218
            y_std = 16.62166076148791
            # Long-term seasonal trend
            self.params[0][0]["lengthscale"] = 67 / x_std # 50.0
            self.params[0][0]["variance"] = 66**2 / y_std**2 # 50.0
            # Seasonal kernel
            # Periodic kernel
            self.params[0][1][0]["lengthscale"] = 1.3 / x_std # 1.0
            self.params[0][1][0]["variance"] = 2.4**2  / y_std**2 # 2.0
            self.params[0][1][0]["period"] = 1.0 / x_std
            # RBF kernel
            self.params[0][1][1]["lengthscale"] = 90 / x_std # 100.0
            # Irregular trend
            self.params[0][2]["lengthscale"] = 1.2 / x_std # 1.0
            self.params[0][2]["alpha"] = 0.78 # 1.0
            self.params[0][2]["variance"] = 0.66**2 / y_std**2 # 0.5
            # Noise kernel
            self.params[0][3]["variance"] = 0.18**2 / y_std**2 # 0.1
            self.params[0][3]["lengthscale"] = 0.134 / x_std #  0.1
            self.params[0][4]["variance"] = 0.0361**2 / y_std**2 # 0.1
        elif self.kernel_name == "Periodic": 
            # self.params[0]["period"] = 1.
            # self.params[0]["variance"] = 1.
            # self.params[0]["lengthscale"] = 2.
            self.params[0][0][0]["period"] = 1.
            self.params[0][0][1]["variance"] = 1.
            self.params[0][0][1]["lengthscale"] = 4. #2
            self.params[0][1]["lengthscale"] = 0.25
            self.params[0][1]["variance"] = 0.
        elif self.kernel_name == "Helmholtz": 
            x_std = jnp.array([1.87364991, 0.84454863])
            y_std = 0.17750018
            self.params[0]["potential_kernel"]["lengthscale"] = 7.91214829 / x_std
            self.params[0]["potential_kernel"]["variance"] = 0.25026575 / y_std**2 # 0.0342
            self.params[0]["stream_kernel"]["lengthscale"] = 1.13632106 / x_std
            self.params[0]["stream_kernel"]["variance"] = 0.02812761 / y_std**2 # 0.8884
        else:
            self.params[0]["lengthscale"] = self.kernel_params["lengthscale"]
            self.params[0]["variance"] = self.kernel_params["variance"]

        # Set prior parameters
        if self.parameter_tuning:
            self._tune_parameters(config, dataloader)
            
        # else:
        #     for i in range(self.n_priors):
        #         self.params[i]["variance"] = self.kernel_params["variance"]
        #         if self.kernel_name in ["RBF", "Matern12", "Matern32", "Matern52"]:
        #             self.params[i]["lengthscale"] = self.kernel_params["lengthscale"]*jnp.ones(self.feature_dim)
        #         elif self.kernel_name == "RationalQuadratic":
        #             self.params[i]["lengthscale"] = self.kernel_params["lengthscale"]*jnp.ones(self.feature_dim)
        #             self.params[i]["alpha"] = self.kernel_params["alpha"]


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
        # Set the dataloader to sample dataset with replacement
        dataloader.set_replacement_mode(replacement=True) 

        # Get configuration
        lr = config["fvi"]["prior"]["lr"]
        ll_scale = config["fvi"]["likelihood"]["scale"]
        nb_epochs = config["fvi"]["prior"]["nb_epochs"]
        alpha_eps = config["fvi"]["prior"]["alpha_eps"]
        validation_freq = config["fvi"]["neural_net"]["validation_freq"]
        early_stopping_patience = config["fvi"]["training"]["patience"]

        # Initialize optimizers
        opt_init, opt_update, get_params = adam(step_size=lr)
        z_params = self._to_unconstrained(copy.deepcopy(self.params))
        ll_rho = jnp.log(jnp.exp(ll_scale)-1)
        if self.likelihood == "Gaussian":
            params_init = (z_params, ll_rho)
            print(f"Initial parameters: {self.params}", flush=True)
        elif self.likelihood == "Categorical":
            params_init = (z_params, ll_rho*jnp.ones((self.n_priors,)))
        opt_state = opt_init(params_init)
        
        # Early stopping initialization
        optimal_n_mll, no_improve_count, optimal_state = jnp.inf, 0, None

        # Training loop
        step = 0
        for epoch in range(nb_epochs):
            n_mll = 0
            for x, y in dataloader:
                # Update kernel parameters
                try:
                    if self.likelihood == "Gaussian":
                        opt_state, loss = self._update_n_mll_gaussian(
                            opt_state,
                            opt_update,
                            get_params,
                            x, 
                            y,
                            step
                        )
                    elif self.likelihood == "Categorical":
                        opt_state, loss = self._update_n_mll_categorical(
                            opt_state,
                            opt_update,
                            get_params,
                            alpha_eps,
                            x, 
                            y, 
                            step
                        )
                except Exception as e:
                    z_params, ll_rho = get_params(opt_state)
                    params = self._to_constrained(z_params)
                    ll_scale = jax.nn.softplus(ll_rho) + 1e-3
                    print("params", params, flush=True)  
                    print("ll_scale", ll_scale, flush=True)
                    raise e 
                n_mll += loss
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
        ll_scale = jax.nn.softplus(ll_rho) + 1e-3
        print("ll_scale", ll_scale, flush=True)

        # Enforce parameter positivity constraints
        self.params = self._to_constrained(z_params)

        # Set the dataloader to sample data without replacement
        dataloader.set_replacement_mode(replacement=False)

        if self.likelihood == "Gaussian":
            print(f"Optimal parameters: {self.params}", flush=True)

    
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
        ll_scale = jax.nn.softplus(ll_rho) + 1e-3
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
        ll_scale = jax.nn.softplus(ll_rho) + 1e-3
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
                self.kernel.gram(params[i], x).to_dense() + ((1e-3 + ll_scale[i])**2) * jnp.eye(x.shape[0])
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
        if self.kernel_name == "ERA5Kernel":
            # Kernel temporal 
            params[0]["length"] = jax.nn.softplus(params[0]["length"])
            params[0]["k_temporal1"]["lengthscale"] = jax.nn.softplus(params[0]["k_temporal1"]["lengthscale"])
            params[0]["k_temporal1"]["variance"] = jax.nn.softplus(params[0]["k_temporal1"]["variance"])
            params[0]["k_temporal2"]["lengthscale"] = jax.nn.softplus(params[0]["k_temporal2"]["lengthscale"])
            params[0]["k_temporal2"]["variance"] = jax.nn.softplus(params[0]["k_temporal2"]["variance"])
            params[0]["kernel_equator"]["lengthscale"] = jax.nn.softplus(params[0]["kernel_equator"]["lengthscale"])
            params[0]["kernel_equator"]["variance"] = jax.nn.softplus(params[0]["kernel_equator"]["variance"])
            params[0]["k_spatial1"]["lengthscale"] = jax.nn.softplus(params[0]["k_spatial1"]["lengthscale"])
            params[0]["k_spatial1"]["variance"] = jax.nn.softplus(params[0]["k_spatial1"]["variance"])
            params[0]["k_spatial2"]["lengthscale"] = jax.nn.softplus(params[0]["k_spatial2"]["lengthscale"])
            params[0]["k_spatial2"]["variance"] = jax.nn.softplus(params[0]["k_spatial2"]["variance"])
        elif self.kernel_name == "Periodic": 
            # params[0]["period"] = jax.nn.softplus(params[0]["period"])
            # params[0]["variance"] = jax.nn.softplus(params[0]["variance"])
            # params[0]["lengthscale"] = jax.nn.softplus(params[0]["lengthscale"])
            # # Kernel temporal 
            params[0][0][0]["lengthscale"] = jax.nn.softplus(params[0][0][0]["lengthscale"])
            params[0][0][0]["variance"] = jax.nn.softplus(params[0][0][0]["variance"])
            params[0][0][0]["period"] = jax.nn.softplus(params[0][0][0]["period"])
            params[0][0][1]["lengthscale"] = jax.nn.softplus(params[0][0][1]["lengthscale"])
            params[0][0][1]["variance"] = jax.nn.softplus(params[0][0][1]["variance"])
            # Kernel spatial
            params[0][1]["lengthscale"] = jax.nn.softplus(params[0][1]["lengthscale"])
            params[0][1]["variance"] = jax.nn.softplus(params[0][1]["variance"])
        elif self.kernel_name == "Helmholtz":
            params[0]["potential_kernel"]["lengthscale"] = jax.nn.softplus(params[0]["potential_kernel"]["lengthscale"])
            params[0]["potential_kernel"]["variance"] = jax.nn.softplus(params[0]["potential_kernel"]["variance"])
            params[0]["stream_kernel"]["lengthscale"] = jax.nn.softplus(params[0]["stream_kernel"]["lengthscale"])
            params[0]["stream_kernel"]["variance"] = jax.nn.softplus(params[0]["stream_kernel"]["variance"])
        else:
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
        if self.kernel_name == "ERA5Kernel":
            # Kernel temporal 
            params[0]["length"] = jnp.log(jnp.exp(params[0]["length"])-1)
            params[0]["k_temporal1"]["lengthscale"] = jnp.log(jnp.exp(params[0]["k_temporal1"]["lengthscale"])-1)
            params[0]["k_temporal1"]["variance"] = jnp.log(jnp.exp(params[0]["k_temporal1"]["variance"])-1)
            params[0]["k_temporal2"]["lengthscale"] = jnp.log(jnp.exp(params[0]["k_temporal2"]["lengthscale"])-1)
            params[0]["k_temporal2"]["variance"] = jnp.log(jnp.exp(params[0]["k_temporal2"]["variance"])-1)
            params[0]["kernel_equator"]["lengthscale"] = jnp.log(jnp.exp(params[0]["kernel_equator"]["lengthscale"])-1)
            params[0]["kernel_equator"]["variance"] = jnp.log(jnp.exp(params[0]["kernel_equator"]["variance"])-1)
            params[0]["k_spatial1"]["lengthscale"] = jnp.log(jnp.exp(params[0]["k_spatial1"]["lengthscale"])-1)
            params[0]["k_spatial1"]["variance"] = jnp.log(jnp.exp(params[0]["k_spatial1"]["variance"])-1)
            params[0]["k_spatial2"]["lengthscale"] = jnp.log(jnp.exp(params[0]["k_spatial2"]["lengthscale"])-1)
            params[0]["k_spatial2"]["variance"] = jnp.log(jnp.exp(params[0]["k_spatial2"]["variance"])-1)
        elif self.kernel_name == "Periodic": 
            # params[0]["period"] = jnp.log(jnp.exp(params[0]["period"])-1)
            # params[0]["variance"] = jnp.log(jnp.exp(params[0]["variance"])-1)
            # params[0]["lengthscale"] = jnp.log(jnp.exp(params[0]["lengthscale"])-1)
            # Kernel temporal 
            params[0][0][0]["lengthscale"] = jnp.log(jnp.exp(params[0][0][0]["lengthscale"])-1)
            params[0][0][0]["variance"] = jnp.log(jnp.exp(params[0][0][0]["variance"])-1)
            params[0][0][0]["period"] = jnp.log(jnp.exp(params[0][0][0]["period"])-1)
            params[0][0][1]["lengthscale"] = jnp.log(jnp.exp(params[0][0][1]["lengthscale"])-1)
            params[0][0][1]["variance"] = jnp.log(jnp.exp(params[0][0][1]["variance"])-1)
            # Kernel spatial
            params[0][1]["lengthscale"] = jnp.log(jnp.exp(params[0][1]["lengthscale"])-1)
            params[0][1]["variance"] = jnp.log(jnp.exp(params[0][1]["variance"])-1)
        elif self.kernel_name == "Helmholtz":
            params[0]["potential_kernel"]["lengthscale"] = jnp.log(jnp.exp(params[0]["potential_kernel"]["lengthscale"])-1)
            params[0]["potential_kernel"]["variance"] = jnp.log(jnp.exp(params[0]["potential_kernel"]["variance"])-1)
            params[0]["stream_kernel"]["lengthscale"] = jnp.log(jnp.exp(params[0]["stream_kernel"]["lengthscale"])-1)
            params[0]["stream_kernel"]["variance"] = jnp.log(jnp.exp(params[0]["stream_kernel"]["variance"])-1)
        else:
            for i in range(self.n_priors):
                for k in params[i].keys():
                    params[i][k] = jnp.log(jnp.exp(jnp.ones_like(params[i][k]))-1)
        
        return params



