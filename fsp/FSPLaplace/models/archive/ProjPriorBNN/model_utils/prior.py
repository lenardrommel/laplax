import jax 

import jaxkern as jk
import jax.numpy as jnp

class Prior:
    """Functional prior."""

    def __init__(
            self, 
            kernel, 
            key, 
            kernel_params=None, 
            data=None,
            eps=1e-10
        ):
        """
        Initialize the prior.
        :params kernel (string): the kernel to use.
        :params key (jax.random.PRNGKey): a random key.
        :params kernel_params (dictionary): the parameters of the kernel.
        :params data (jax.numpy.ndarray): data used to compute kernel.
        :params eps (float): the epsilon value for numerical stability.
        """
        self.key = key 
        self.eps = eps
        self.kernel = kernel
        if self.kernel == "RBF":
            self.prior = jk.RBF()
            self.params = self.prior.init_params(self.key)
            self.params["lengthscale"] = kernel_params["lengthscale"]
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Matern12":
            self.prior = jk.Matern12()
            self.params = self.prior.init_params(self.key)
            self.params["lengthscale"] = kernel_params["lengthscale"]
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Matern32":
            self.prior = jk.Matern32()
            self.params = self.prior.init_params(self.key)
            self.params["lengthscale"] = kernel_params["lengthscale"]
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Matern52":
            self.prior = jk.Matern52()
            self.params = self.prior.init_params(self.key)
            self.params["lengthscale"] = kernel_params["lengthscale"]
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Polynomial":
            self.prior = jk.Polynomial(degree=kernel_params["degree"])
            self.params = self.prior.init_params(self.key)
            self.params["lengthscale"] = kernel_params["lengthscale"]
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Linear":
            self.prior = jk.Linear()
            self.params = self.prior.init_params(self.key)
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "White":
            self.prior = jk.White()
            self.params = self.prior.init_params(self.key)
            self.params["variance"] = kernel_params["variance"]
        elif self.kernel == "Empirical": 
            X, y = data
            self.mean = jnp.mean(y, axis=0).reshape(-1)
            self.cov = jnp.cov(y.reshape(y.shape[0], -1), rowvar=False)
        else:
            raise Exception("Unknown kernel")
        
        
    def __call__(
        self, 
        x=None
    ):
        """
        Compute the prior mean and covariance of the prior.
        :params x (jax.numpy.ndarray): the input data.
        :returns prior_mean (jax.numpy.ndarray): the prior mean.
        :returns prior_cov (jax.numpy.ndarray): the prior covariance.
        """
        if self.kernel == "Empirical":
            prior_mean = self.mean
            prior_cov = self.cov
        else:
            prior_mean = jnp.ones(x.shape[0]) * 0
            prior_cov = self.prior.gram(self.params, x).to_dense()
        
        # Add jitter for numerical stability
        prior_cov += self.eps * jnp.eye(prior_cov.shape[0])
        
        return prior_mean, prior_cov
    

    def cross_covariance(
        self, 
        x, 
        y
    ):
        """
        Compute the cross covariance between x and y.
        :params x (jax.numpy.ndarray): the input data.
        :params y (jax.numpy.ndarray): the input data.
        :returns cross_cov (jax.numpy.ndarray): the cross covariance.
        """
        return self.prior.cross_covariance(self.params, x, y)
