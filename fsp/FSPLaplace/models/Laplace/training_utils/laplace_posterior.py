import os 
import jax
import torch
import pickle
import kfac_jax

import jax.numpy as jnp
import jax.scipy as jsp

from pathlib import Path
from functools import partial
from jax.flatten_util import ravel_pytree
from haiku.data_structures import tree_size, merge


class BaseLaplacePosterior:
    """
    Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model,
        prior_scale, 
        training
    ):
        # Configuration
        self.key = model.key
        self.config = model.config
        self.training = training

        # Model parameters
        self.model = model
        self.state = model.state
        self.sto_params = model.sto_params
        self.det_params = model.det_params
        self.dim = tree_size(self.sto_params)
        
        
        # Likelihood parameters
        self.likelihood = model.likelihood
        if self.likelihood == "Gaussian":
            self.ll_scale = model.ll_scale
            self.n_outputs = 1
        elif self.likelihood == "Categorical":
            self.n_outputs = model.architecture[-1]
        
        # Prior parameters
        self.prior_scale = prior_scale

        # Laplace configuration
        if self.training: # Training : fit prior parameters using the marginal likelihood
            laplace_config = self.config["laplace"]["training"]["mll"]
            self.covariance_path = "" # do not load covariance square root
            self.save_covariance = False
            self.cov_type = laplace_config["cov_type"]
        else: # Inference : fit Laplace approximation to the neural network posterior
            laplace_config = self.config["laplace"]["inference"]
            self.save_covariance = laplace_config["save_covariance"]
            self.covariance_path = laplace_config["covariance_path"]
            self.cov_type = laplace_config["cov_type"]
            
        # Define save path
        if self.save_covariance:
            counter = 1
            dir_path = f"checkpoints/{self.config['data']['name']}"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = dir_path + f"/laplace_{self.cov_type}_cov_sq_root_{counter}.pkl"
            while os.path.exists(self.checkpoint_path):
                self.checkpoint_path = dir_path + f"/laplace_{self.cov_type}_cov_sq_root_{counter}.pkl"
                counter += 1
            print(f"Checkpoint path for covariance: {self.checkpoint_path}", flush=True)


    def fit(
        self,
        train_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """
        raise NotImplementedError

    
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        raise NotImplementedError
        

    def log_marginal_likelihood(
        self,
    ):
        """
        Marginal likelihood.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - lmll (float): log marginal likelihood.
        """
        raise NotImplementedError


    @partial(jax.jit, static_argnums=(0,))
    def negative_log_marginal_likelihood_objective(
        self,
        prior_rho
    ):
        """
        Marginal likelihood objective.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - neg_lmll (float): negative log marginal likelihood.
        """
        raise NotImplementedError



class FullLaplacePosterior(BaseLaplacePosterior):
    """
    Full Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model,
        prior_scale, 
        training
    ):
        super().__init__(
            model,
            prior_scale, 
            training
        )
        self.cov_type = "full"


    def fit(
        self,
        train_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """
        # If path was specified, precision and likelihood are loaded from memory
        # in posterior_covariance_sq_root()
        if self.covariance_path:
            return self

        # Initialize curvature
        self.curvature = jnp.diag(jnp.zeros((self.dim,))) 

        # Compute curvature matrix
        self.log_likelihood_loss = 0.
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1, key2 = jax.random.split(self.key, 3)
            # Update precision
            self.curvature += self._update_curvature(x, key1)
            # Update likelihood
            self.log_likelihood_loss += self._log_likelihood(x, y, key2)

        return self
    

    def _log_likelihood(
        self,
        p,
        x,
        y, 
    ):
        """
        Compute the log-likelihood.

        params:
        - x (jnp.array): input data.
        - y (jnp.array): target data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - log_likelihood (float): log-likelihood.
        """
        f = self.model.apply_fn(merge(p, self.det_params), self.state, self.key, x, training=False)[0]
        if self.likelihood == "Gaussian":
            kfac_jax.register_squared_error_loss(f, y, weight=0.5*x.shape[0]/self.ll_scale**2)
            log_likelihood = jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=self.ll_scale
            ).sum()
        elif self.likelihood == "Categorical":
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
            kfac_jax.register_softmax_cross_entropy_loss(f, y.reshape(-1), weight=x.shape[0])
            log_likelihood = jnp.sum(
                one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
                axis=-1
            ).sum()
        
        return log_likelihood 
    

    @partial(jax.jit, static_argnums=(0,))
    def _update_curvature(
        self,
        x,
        key
    ):
        """
        Update curvature matrix.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - curvature_update (jnp.array): curvature update.
        """
        # Compute Jacobian
        f = lambda p: self.model.apply_fn(merge(p, self.det_params), self.state, key, x, training=False)[0]
        jacobian = jax.jacrev(f)(self.sto_params)

        # Vectorize Jacobian
        v_jacobian = jnp.concatenate(
            [l.reshape(x.shape[0],self.n_outputs,-1) for l in jax.tree_util.tree_leaves(jacobian)],
            axis=-1
        )

        # Compute likelihood hessian 
        if self.likelihood == "Gaussian":
            ll_hessian = -1 / self.ll_scale**2
        elif self.likelihood == "Categorical":
            probs = jax.nn.softmax(f(self.sto_params), axis=-1)  # (n_batch, n_classes)
            ll_hessian = -jax.vmap(jnp.diag)(probs) + jnp.einsum('bk,bc->bck', probs, probs) # (n_batch, n_classes, n_classes)
        n_likelihood_hessian = -ll_hessian

        # Update precision
        if self.likelihood == "Gaussian":
            curvature_update = n_likelihood_hessian * jnp.einsum('bcp,bcq->pq', v_jacobian, v_jacobian)
        else:
            curvature_update = jnp.einsum('bcp,bck,bkq->pq', v_jacobian, n_likelihood_hessian, v_jacobian)

        return curvature_update
    

    @partial(jax.jit, static_argnums=(0,))
    def _log_likelihood(
        self,
        x, 
        y,
        key
    ):
        """
        Compute the log-likelihood.

        params:
        - x (jnp.array): input data.
        - y (jnp.array): target data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - log_likelihood (float): log-likelihood.
        """
        f = self.model.apply_fn(merge(self.sto_params, self.det_params), self.state, key, x, training=False)[0]
        if self.likelihood == "Gaussian":
            log_likelihood = jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=self.ll_scale
            ).sum()
        elif self.likelihood == "Categorical":
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
            log_likelihood = jnp.sum(
                one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
                axis=-1
            ).sum()
        
        return log_likelihood
    
    
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        # If path was not specified, factors are computed from the curvature
        if not self.covariance_path:
            # Prior precision 
            prior_precision = jnp.diag(1 / self.prior_scale**2)
            
            # Compute posterior precision
            posterior_precision = prior_precision + self.curvature

            # Compute square root factorization of the covariance
            try:
                Lf = jnp.linalg.cholesky(posterior_precision)
                Id = jnp.eye(posterior_precision.shape[-1])
                sq_root_cov = jsp.linalg.solve_triangular(Lf, Id, lower=True).T
            except Exception as e:
                print(e)
                spectrum = jnp.linalg.eigvalsh(posterior_precision)
                print("precision spectrum", spectrum)
                posterior_precision += jnp.eye(posterior_precision.shape[-1]) * (jnp.abs(spectrum).min() + 1e-6)
                spectrum = jnp.linalg.eigvalsh(posterior_precision)
                print("corrected precision spectrum", spectrum)
                Lf = jnp.linalg.cholesky(posterior_precision)
                Id = jnp.eye(posterior_precision.shape[-1])
                sq_root_cov = jsp.linalg.solve_triangular(Lf, Id, lower=True).T
        else:
            with open(self.covariance_path, "rb") as file:
                data = pickle.load(file)
                sq_root_cov = data["sq_root_cov"]
                self.log_likelihood_loss = data["log_likelihood_loss"]
                posterior_precision = data["posterior_precision"]
            
        if self.save_covariance:
            with open(self.checkpoint_path, "wb") as file:
                pickle.dump(
                    {
                        "posterior_precision": posterior_precision,
                        "sq_root_cov": sq_root_cov, 
                        "log_likelihood_loss":self.log_likelihood_loss
                    }, 
                    file
                )
        
        return sq_root_cov
        

    def log_marginal_likelihood(
        self,
    ):
        """
        Marginal likelihood.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - lmll (float): log marginal likelihood.
        """
        assert self.training == False, "Model must be in inference mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        sto_params_v = ravel_pytree(self.sto_params)[0]
        prior_scale_v = self.prior_scale
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5*jnp.linalg.slogdet(jnp.diag(1/prior_scale_v**2) + self.curvature)[1]

        return lmll


    @partial(jax.jit, static_argnums=(0,))
    def negative_log_marginal_likelihood_objective(
        self,
        prior_rho
    ):
        """
        Marginal likelihood objective.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - neg_lmll (float): negative log marginal likelihood.
        """
        assert self.training == True, "Model must be in training mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        prior_scale_v, _ = self.model.expand_prior(jax.nn.softplus(prior_rho))
        sto_params_v = ravel_pytree(self.sto_params)[0]
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5*jnp.linalg.slogdet(jnp.diag(1/prior_scale_v**2) + self.curvature)[1]

        return -lmll


class DiagLaplacePosterior(BaseLaplacePosterior):
    """
    Diagonal Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model,
        prior_scale, 
        training
    ):
        super().__init__(
            model,
            prior_scale, 
            training
        )
        self.cov_type = "diag"


    def fit(
        self,
        train_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """
        # If path was specified, precision and likelihood are loaded from memory
        # in posterior_covariance_sq_root()
        if self.covariance_path:
            return self

        # Initialize curvature
        self.curvature = jnp.zeros((self.dim,))
        
        # Compute curvature matrix
        self.log_likelihood_loss = 0.
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1, key2 = jax.random.split(self.key, 3)
            # Update precision
            self.curvature += self._update_curvature(x, key1)
            # Update likelihood
            self.log_likelihood_loss += self._log_likelihood(x, y, key2)

        return self


    @partial(jax.jit, static_argnums=(0,))
    def _update_curvature(
        self,
        x,
        key
    ):
        """
        Update curvature matrix.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - curvature_update (jnp.array): curvature update.
        """
        # Compute Jacobian
        f = lambda p: self.model.apply_fn(merge(p, self.det_params), self.state, key, x, training=False)[0]
        jacobian = jax.jacrev(f)(self.sto_params)

        # Vectorize Jacobian
        v_jacobian = jnp.concatenate(
            [l.reshape(x.shape[0],self.n_outputs,-1) for l in jax.tree_util.tree_leaves(jacobian)],
            axis=-1
        )

        # Compute likelihood hessian 
        if self.likelihood == "Gaussian":
            ll_hessian = -1 / self.ll_scale**2
        elif self.likelihood == "Categorical":
            probs = jax.nn.softmax(f(self.sto_params), axis=-1)  # (n_batch, n_classes)
            ll_hessian = -jax.vmap(jnp.diag)(probs) + jnp.einsum('bk,bc->bck', probs, probs) # (n_batch, n_classes, n_classes)
        n_likelihood_hessian = -ll_hessian

        # Update precision
        if self.likelihood == "Gaussian":
            curvature_update = n_likelihood_hessian * jnp.einsum('bcp,bcp->p', v_jacobian, v_jacobian)
        else:
            curvature_update = jnp.einsum('bcp,bck,bkp->p', v_jacobian, n_likelihood_hessian, v_jacobian)
    
        return curvature_update
    

    @partial(jax.jit, static_argnums=(0,))
    def _log_likelihood(
        self,
        x, 
        y,
        key
    ):
        """
        Compute the log-likelihood.

        params:
        - x (jnp.array): input data.
        - y (jnp.array): target data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - log_likelihood (float): log-likelihood.
        """
        f = self.model.apply_fn(merge(self.sto_params, self.det_params), self.state, key, x, training=False)[0]
        if self.likelihood == "Gaussian":
            log_likelihood = jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=self.ll_scale
            ).sum()
        elif self.likelihood == "Categorical":
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
            log_likelihood = jnp.sum(
                one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
                axis=-1
            ).sum()
        
        return log_likelihood

    
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        # If path was not specified, factors are computed from the curvature
        if not self.covariance_path:
            # Prior precision 
            prior_precision = 1 / self.prior_scale**2
            
            # Compute posterior precision
            posterior_precision = prior_precision + self.curvature

            # Compute square root factorization of the covariance
            sq_root_cov = 1 / jnp.sqrt(posterior_precision)
        else:
            with open(self.covariance_path, "rb") as file:
                data = pickle.load(file)
                sq_root_cov = data["sq_root_cov"]
                self.log_likelihood_loss = data["log_likelihood_loss"]
                posterior_precision = data["posterior_precision"]
            
        if self.save_covariance:
            with open(self.checkpoint_path, "wb") as file:
                pickle.dump(
                    {
                        "posterior_precision": posterior_precision,
                        "sq_root_cov": sq_root_cov, 
                        "log_likelihood_loss":self.log_likelihood_loss
                    }, 
                    file
                )
        
        return sq_root_cov
        

    def log_marginal_likelihood(
        self,
    ):
        """
        Marginal likelihood.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - lmll (float): log marginal likelihood.
        """
        assert self.training == False, "Model must be in inference mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        sto_params_v = ravel_pytree(self.sto_params)[0]
        prior_scale_v = self.prior_scale
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5*jnp.log(1/prior_scale_v**2 + self.curvature).sum()

        return lmll


    @partial(jax.jit, static_argnums=(0,))
    def negative_log_marginal_likelihood_objective(
        self,
        prior_rho
    ):
        """
        Marginal likelihood objective.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - neg_lmll (float): negative log marginal likelihood.
        """
        assert self.training == True, "Model must be in training mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        prior_scale_v, _ = self.model.expand_prior(jax.nn.softplus(prior_rho))
        sto_params_v = ravel_pytree(self.sto_params)[0]
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5*jnp.log(1/prior_scale_v**2 + self.curvature).sum()

        return -lmll




class MAPLaplacePosterior(BaseLaplacePosterior):
    """
    Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model,
        prior_scale, 
        training
    ):
        super().__init__(
            model,
            prior_scale, 
            training
        )
        self.cov_type = "map"


    def fit(
        self,
        train_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """
        # If covariance type is MAP, do not fit the Laplace approximation
        self.log_likelihood_loss = 0 
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1 = jax.random.split(self.key)
            # Update likelihood
            f = self.model.apply_fn(merge(self.sto_params, self.det_params), self.state, key1, x, training=False)[0]
            self.log_likelihood_loss += self._log_likelihood(f, y)
                
        return self


    @partial(jax.jit, static_argnums=(0,))
    def _log_likelihood(
        self,
        f, 
        y
    ):
        """
        Compute the log-likelihood.

        params:
        - x (jnp.array): input data.
        - y (jnp.array): target data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - log_likelihood (float): log-likelihood.
        """
        if self.likelihood == "Gaussian":
            log_likelihood = jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=self.ll_scale
            ).sum()
        elif self.likelihood == "Categorical":
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
            log_likelihood = jnp.sum(
                one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
                axis=-1
            ).sum()
            
        return log_likelihood    

    
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        return 1e-10 * jnp.ones((self.dim,))
    

    def log_marginal_likelihood(
        self,
    ):
        """
        Marginal likelihood.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - lmll (float): log marginal likelihood.
        """
        assert self.training == False, "Model must be in inference mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        sto_params_v = ravel_pytree(self.sto_params)[0]
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / self.prior_scale**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(self.prior_scale**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5 * jnp.log(1e-10 * jnp.ones((self.dim,)) + 1 / self.prior_scale**2).sum()

        return lmll


class KFACLaplacePosterior(BaseLaplacePosterior):
    """
    Full Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model,
        prior_scale, 
        training
    ):
        super().__init__(
            model,
            prior_scale, 
            training
        )
        self.cov_type = "kfac"


    def fit(
        self,
        train_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """
        # If path was specified, precision and likelihood are loaded from memory
        # in posterior_covariance_sq_root()
        if self.covariance_path:
            return self

        # Initialize curvature
        curvature = kfac_jax.BlockDiagonalCurvature(
            func=lambda _p, x, y: self._log_likelihood(_p, x, y),
            params_index=0, 
            default_estimation_mode="ggn_exact", 
            layer_tag_to_block_ctor={"dense_tag": kfac_jax.DenseTwoKroneckerFactored, "conv2d_tag": kfac_jax.Conv2DTwoKroneckerFactored}, 
            index_to_block_ctor=None, 
            auto_register_tags=True, 
            distributed_multiplies=True, 
            distributed_cache_updates=True, 
            num_samples=1, 
            should_vmap_samples=False
        )
        
        # Initialize curvature state - does not compute anything
        x, y = next(iter(train_dataloader))
        curv_state = curvature.init(
            self.key, 
            func_args=(self.sto_params, x, y),
            exact_powers_to_cache=None, 
            approx_powers_to_cache=None
        )

        # Compute curvature matrix
        self.log_likelihood_loss = 0.
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1 = jax.random.split(self.key, )
            # Update precision
            curv_state = curvature.update_curvature_matrix_estimate(
                curv_state, 
                ema_old=1,
                ema_new=1,
                batch_size=x.shape[0],
                rng=key1,
                func_args=(self.sto_params, x, y),
                estimation_mode="ggn_exact"
            )
            # Update likelihood
            self.log_likelihood_loss += curvature.func(self.sto_params, x, y)

        # Compute block-eigendecomposition of the curvature
        self.eigvals_curv, self.eigvecs_curv, i = [], [], 0
        for b, bs in zip(curvature.blocks, curv_state.blocks_states):
            try:
                if b.has_bias and b.parameters_canonical_order[0] != 0:
                    q = kfac_jax.utils.block_permuted(
                        bs.factors[0].value * x.shape[0],
                        block_sizes=[bs.factors[0].raw_value.shape[0] - 1, 1],
                        block_order=(1, 0),
                    )
                else:
                    q = bs.factors[0].value * x.shape[0]
            except:
                q = 1
            w = bs.factors[1].value * b.scale(bs, use_cache=False) * x.shape[0]
            # Eigenvalue decomposition of the factors
            _eigval_q, _eigvec_q = jnp.linalg.eigh(q)
            _eigval_w, _eigvec_w = jnp.linalg.eigh(w)
            # Compute the layer eigenvectors
            self.eigvecs_curv.append(jnp.kron(_eigvec_q, _eigvec_w))
            # Compute the layer eigenvalues
            _eigvals = jnp.concatenate([_eigval_q[i] * _eigval_w for i in range(_eigval_q.shape[0])], axis=0)
            self.eigvals_curv.append(jnp.where(_eigvals > 0, _eigvals, 0))
            # Compute the square root of the layer covariance
            i += b.dim

        return self
    

    def _log_likelihood(
        self,
        p,
        x,
        y, 
    ):
        """
        Compute the log-likelihood.

        params:
        - x (jnp.array): input data.
        - y (jnp.array): target data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - log_likelihood (float): log-likelihood.
        """
        f = self.model.apply_fn(merge(p, self.det_params), self.state, self.key, x, training=False)[0]
        if self.likelihood == "Gaussian":
            #kfac_jax.register_squared_error_loss(f, y, weight=0.5*x.shape[0]/self.ll_scale**2)
            kfac_jax.register_normal_predictive_distribution(f, y, self.ll_scale**2, weight=x.shape[0])
            log_likelihood = jsp.stats.norm.logpdf(
                y, 
                loc=f, 
                scale=self.ll_scale
            ).sum()
        elif self.likelihood == "Categorical":
            one_hot_y = jax.nn.one_hot(y.reshape(-1), num_classes=f.shape[-1])
            kfac_jax.register_softmax_cross_entropy_loss(f, y.reshape(-1), weight=x.shape[0])
            log_likelihood = jnp.sum(
                one_hot_y * jax.nn.log_softmax(f, axis=-1), # (n_batch, n_classes)
                axis=-1
            ).sum()
        
        return log_likelihood 
    
    
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        # If path was not specified, factors are computed from the curvature
        if not self.covariance_path:
            sq_root_cov = []
            i = 0
            for eigvals, eigvecs in zip(self.eigvals_curv, self.eigvecs_curv):
                eigval_prec = jnp.where(eigvals > 0, eigvals, jnp.inf) + 1/self.prior_scale[i:eigvals.shape[0]+i]**2
                # Compute the square root of the layer covariance
                sq_root_cov.append(eigvecs @ jnp.diag(eigval_prec**(-0.5)))
                print("sq_root_cov", sq_root_cov[-1].shape, flush=True)
                i += eigvecs.shape[0]
        else:
            with open(self.covariance_path, "rb") as file:
                data = pickle.load(file)
                sq_root_cov = data["sq_root_cov"]
                self.log_likelihood_loss = data["log_likelihood_loss"]
            
        if self.save_covariance:
            with open(self.checkpoint_path, "wb") as file:
                pickle.dump(
                    {
                        "sq_root_cov": sq_root_cov, 
                        "log_likelihood_loss":self.log_likelihood_loss
                    }, 
                    file
                )
        
        return sq_root_cov
        

    def log_marginal_likelihood(
        self,
    ):
        """
        Marginal likelihood.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - lmll (float): log marginal likelihood.
        """
        assert self.training == False, "Model must be in inference mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        sto_params_v = ravel_pytree(self.sto_params)[0]
        prior_scale_v = self.prior_scale
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5 * jnp.log(jnp.concatenate(self.eigvals_curv, axis=0) + 1/prior_scale_v**2).sum()

        return lmll


    @partial(jax.jit, static_argnums=(0,))
    def negative_log_marginal_likelihood_objective(
        self,
        prior_rho
    ):
        """
        Marginal likelihood objective.

        params:
        - prior_rho (jnp.array): prior scale.

        returns:
        - neg_lmll (float): negative log marginal likelihood.
        """
        assert self.training == True, "Model must be in training mode."

        # Get configuration
        dim = tree_size(self.sto_params)

        # Compute log-prior
        prior_scale_v, _ = self.model.expand_prior(jax.nn.softplus(prior_rho))
        sto_params_v = ravel_pytree(self.sto_params)[0]
        log_prior = -0.5 * jnp.sum(sto_params_v**2 / prior_scale_v**2)
        log_prior -= 0.5 * jnp.sum(jnp.log(prior_scale_v**2)) 
        log_prior -= 0.5 * dim * jnp.log(2*jnp.pi)
        
        # Compute marginal likelihood
        lmll = self.log_likelihood_loss 
        lmll += log_prior 
        lmll += 0.5 * dim * jnp.log(2*jnp.pi)
        lmll -= 0.5 * jnp.log(jnp.concatenate(self.eigvals_curv, axis=0) + 1/prior_scale_v**2).sum()

        return -lmll