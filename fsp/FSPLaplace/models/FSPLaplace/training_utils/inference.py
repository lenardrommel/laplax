import gc
import os 
import jax
import copy
import torch 
import pickle 

import haiku as hk
import numpy as np
import scipy as scp
import jax.numpy as jnp
import jax.scipy as jsp


from pathlib import Path
from functools import partial
from jax.flatten_util import ravel_pytree
from haiku.data_structures import tree_size, merge
from skerch.decompositions import seigh, truncate_core
from models.FSPLaplace.training_utils.skerch_wrapper import TorchGGNLinearOperatorWrapper

from data_utils.utils import read_image_data
from models.FSPLaplace.training_utils.lanczos import (
    lanczos_compute_efficient,
    lanczos_compute_efficient_precision, 
    lanczos_memory_efficient,
    lanczos_memory_efficient_precision,
    cg,
    cg_precision
)
from models.FSPLaplace.training_utils.inference_utils import (
    PrecisionLinearOperator, 
    KernelLinearOperator,
    GGNLinearOperator
)

jax.config.update("jax_enable_x64", True)


class BaseFunctionalLaplacePosterior:
    """
    Laplace approximation to the neural network posterior.
    """
    def __init__(
        self,
        model
    ):
        # Configuration
        self.key = model.key
        self.config = copy.deepcopy(model.config)

        # Model parameters
        self.model = model
        self.state = model.state
        self.mean_params = model.mean_params
        self.other_params = model.other_params
        self.dim = tree_size(self.mean_params)

        # Likelihood parameters
        self.likelihood = model.likelihood
        if self.likelihood == "Gaussian":
            self.ll_scale = model.ll_scale
            self.n_outputs = 1
        elif self.likelihood == "Categorical":
            self.n_outputs = model.architecture[-1]
        
        # Prior parameters
        self.prior = model.prior

        # Laplace configuration
        laplace_config = self.config["fsplaplace"]["inference"]
        self.save_covariance = laplace_config["save_covariance"]
        self.covariance_path = laplace_config["covariance_path"]
        self.min_context_val = laplace_config["min_context_val"]
        self.max_context_val = laplace_config["max_context_val"]
        self.n_context_points = laplace_config["n_context_points"]
        self.context_selection = laplace_config["cov_context_selection"]
        self.cov_type = laplace_config["cov_type"]
            
        # Define save path
        if self.save_covariance:
            counter = 1
            dir_path = f"checkpoints/{self.config['data']['name']}"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.checkpoint_path = dir_path + f"/flaplace_{self.cov_type}_cov_sq_root_{counter}.pkl"
            while os.path.exists(self.checkpoint_path):
                self.checkpoint_path = dir_path + f"/flaplace_{self.cov_type}_cov_sq_root_{counter}.pkl"
                counter += 1
            print(f"Checkpoint path for covariance: {self.checkpoint_path}", flush=True)


    def fit(
        self,
        train_dataloader,
        val_dataloader
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
    

    def _select_context_points(
        self, 
        train_dataloader, 
        val_dataloader,
        key
    ):
        """
        Select context points.

        params:
        - datapoint_shape (tuple): shape of data.
        - key: random key.
        - x (jnp.array): input data.

        returns:
        - context points (jnp.array): context points.
        """
        dim = train_dataloader.feature_dim
        if self.config["experiment"]["name"] in ["era5_extrapolation", "era5_interpolation", "hpo_era5_extrapolation", "hpo_era5_interpolation"]:
            x_shape = (3,)
        else:
            x_shape = train_dataloader.dataset.X.shape[1:]
        if self.context_selection == "halton":
            context_points = scp.stats.qmc.Halton(d=dim).random(self.n_context_points).reshape(-1, *x_shape)
            context_points = context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
        elif self.context_selection == "latin_hypercube":
            context_points = scp.stats.qmc.LatinHypercube(d=dim).random(self.n_context_points).reshape(-1, *x_shape)
            context_points = context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
        elif self.context_selection == "train_val_latin":
            context_points = jnp.concatenate(
                [   
                    train_dataloader.dataset.X, 
                    val_dataloader.dataset.X,
                ],
                axis=0
            )[:self.n_context_points,...]
            if context_points.shape[0] < self.n_context_points:
                latin_context_points = scp.stats.qmc.LatinHypercube(d=dim).random(self.n_context_points - context_points.shape[0]).reshape(-1, *x_shape)
                latin_context_points = latin_context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
                context_points = jnp.concatenate(
                    [
                        context_points,
                        latin_context_points
                    ],
                    axis=0
                )
        elif self.context_selection == "train_val_halton":
            context_points = jnp.concatenate(
                [   
                    train_dataloader.dataset.X, 
                    val_dataloader.dataset.X,
                ],
                axis=0
            )[:self.n_context_points,...]
            if context_points.shape[0] < self.n_context_points:
                halton_context_points = scp.stats.qmc.Halton(d=dim).random(self.n_context_points - context_points.shape[0]).reshape(-1, *x_shape)
                halton_context_points = halton_context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
                context_points = jnp.concatenate(
                    [
                        context_points,
                        halton_context_points
                    ],
                    axis=0
                )
        elif self.context_selection == "val_latin":
            n_val = val_dataloader.dataset.X.shape[0]
            context_points = val_dataloader.dataset.X[:self.n_context_points,...]
            if n_val < self.n_context_points:
                latin_context_points = scp.stats.qmc.LatinHypercube(d=dim).random(self.n_context_points - n_val).reshape(-1, *x_shape)
                latin_context_points = latin_context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
                context_points = jnp.concatenate(
                    [
                        context_points, 
                        latin_context_points
                    ],
                    axis=0
                )
        elif self.context_selection == "val_halton":
            n_val = val_dataloader.dataset.X.shape[0]
            context_points = val_dataloader.dataset.X[:self.n_context_points,...]
            if n_val < self.n_context_points:
                halton_context_points = scp.stats.qmc.Halton(d=dim).random(self.n_context_points - n_val).reshape(-1, *x_shape)
                halton_context_points = halton_context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
                context_points = jnp.concatenate(
                    [
                        context_points, 
                        halton_context_points
                    ],
                    axis=0
                )
        elif self.context_selection == "random_monochrome":
            h, w, c = train_dataloader.dataset.X[0].shape
            X_reshaped = train_dataloader.dataset.X[:self.n_context_points,...].reshape(-1, h * w * c)
            random_indices = jax.random.randint(key, shape=(self.n_context_points, h, w, c), minval=0, maxval=self.n_context_points)
            context_points = X_reshaped[random_indices, jnp.arange(c)].reshape(self.n_context_points, h, w, c)
        elif self.context_selection == "grid":
            assert dim in [1,2,3,4], "Grid context selection only works for 1D or 2D features."
            if dim == 1:
                context_points = jnp.linspace(
                    self.min_context_val[0], 
                    self.max_context_val[0],
                    self.n_context_points
                ).reshape(-1, 1)
            elif dim == 2:
                n_dim = jnp.rint(self.n_context_points**(1/2)).astype(int)
                x1 = jnp.linspace(self.min_context_val[0], self.max_context_val[0], n_dim)
                x2 = jnp.linspace(self.min_context_val[1], self.max_context_val[1], n_dim)
                x = jnp.meshgrid(x1, x2, indexing='ij')
                context_points = jnp.stack(x, axis=-1).reshape(-1, 2)
            elif dim == 3:
                n_dim = jnp.rint(self.n_context_points**(1/3)).astype(int)
                print("n_dim", n_dim, self.n_context_points**(1/3))
                x1 = jnp.linspace(self.min_context_val[0], self.max_context_val[0], n_dim)
                x2 = jnp.linspace(self.min_context_val[1], self.max_context_val[1], n_dim)
                x3 = jnp.linspace(self.min_context_val[2], self.max_context_val[2], n_dim)
                context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, indexing='ij'), axis=-1).reshape(-1, 3)
            elif dim== 4:
                n_dim = jnp.rint(self.n_context_points**(1/4)).astype(int)
                x1 = jnp.linspace(self.min_context_val[0], self.max_context_val[0], n_dim)
                x2 = jnp.linspace(self.min_context_val[1], self.max_context_val[1], n_dim)
                x3 = jnp.linspace(self.min_context_val[2], self.max_context_val[2], n_dim)
                x4 = jnp.linspace(self.min_context_val[3], self.max_context_val[3], n_dim)
                context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, x4, indexing='ij'), axis=-1).reshape(-1, 4)
        elif self.context_selection in ["kmnist", "cifar100"]:
            X_train, X_test, _, _ = read_image_data(self.context_selection)
            context_points = jnp.concatenate([X_train, X_test], axis=0)[:self.n_context_points,...]
            # Pre-process data
            mean = jnp.mean(context_points, axis=(0,), keepdims=True)
            std = jnp.std(context_points, axis=(0,), keepdims=True) + 1e-10
            context_points = (context_points - mean) / std    
            # Add extra context points if needed
            if context_points.shape[0] < self.n_context_points:
                n_samples = self.n_context_points - context_points.shape[0]
                latin_context_points = scp.stats.qmc.LatinHypercube(d=dim).random(n_samples).reshape((n_samples,) + context_points.shape[1:]).reshape(-1, *x_shape)
                latin_context_points = latin_context_points * (self.max_context_val - self.min_context_val) + self.min_context_val
                context_points = jnp.concatenate(
                    [
                        context_points, 
                        latin_context_points
                    ],
                    axis=0
                )
        elif self.context_selection == "ocean_current_modeling":
            x1 = jnp.linspace(self.min_context_val[0], self.max_context_val[0], 34)
            x2 = jnp.linspace(self.min_context_val[1], self.max_context_val[1], 16)
            x3 = jnp.array([0., 1.])
            context_points = jnp.stack(jnp.meshgrid(x1, x2, x3, indexing='ij'), axis=-1).reshape(-1, 3)
        
        
        if self.config["data"]["name"] in ["mnist", "fashion_mnist"]:
            context_points = context_points.reshape(-1, 28, 28, 1)
        elif self.config["data"]["name"] in ["cifar10", "cifar100"]:
            context_points = context_points.reshape(-1, 32, 32, 3)

            
        
        return context_points
        

class SkerchLowRankFunctionalLaplacePosterior(BaseFunctionalLaplacePosterior):
    """Low rank Laplace approximation to the neural network posterior using SKERCH."""
    def __init__(
        self, 
        model
    ):
        super().__init__(model)
        self.cov_type = "low_rank"
        self.max_rank = self.config["fsplaplace"]["inference"]["rank"]
        self.max_posterior_precision = self.config["fsplaplace"]["inference"]["max_posterior_precision"]
    

    def fit(
        self,
        train_dataloader,
        val_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """        
        # Get configuration
        OUTER, INNER = 70, 140
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DTYPE = torch.float64

        # Get context points
        x_context = self._select_context_points(
            train_dataloader, 
            val_dataloader,
            self.key
        )
        
        # Define posterior precision linops
        posterior_precision_linops = TorchGGNLinearOperatorWrapper(
            shape=(self.dim, self.dim),
            fun=lambda p, x: self.model.apply_fn(merge(p, self.other_params), self.state, self.key, x, training=False)[0], 
            loss=lambda f, y: -self._log_likelihood(f, y), 
            params=self.mean_params, 
            prior=self.prior, #.covariance_linop(x_context), 
            x_c=x_context,
            dataloader=train_dataloader, 
            key=self.key
        )

        # Compute low rank approximation
        q_psd, u_psd, s_psd = seigh(
            posterior_precision_linops,
            op_device=DEVICE,
            op_dtype=DTYPE,
            outer_dim=OUTER,
            inner_dim=INNER,
        )

        # Truncate low rank approximation
        u_psd, s_psd = truncate_core(self.max_rank, u_psd, s_psd)
        
        # Convert back to jax arrays
        prec_eigvals = jax.dlpack.from_dlpack(torch.to_dlpack(s_psd))
        prec_eigvecs = jax.dlpack.from_dlpack(torch.to_dlpack(q_psd @ u_psd))

        # Sort eigenvalues-vectors in descending order
        idx = jnp.argsort(prec_eigvals)[::-1]
        prec_eigvals = prec_eigvals[idx]
        prec_eigvecs = prec_eigvecs[:,idx]
        
        # Filter out negative eigenvalues
        idx = jnp.sum(prec_eigvals > 0, axis=-1)
        prec_eigvals = prec_eigvals[:idx]
        prec_eigvecs = prec_eigvecs[:,:idx]

        # Filter out small eigenvalues
        tol = prec_eigvals.max() * self.dim * jnp.finfo(prec_eigvals.dtype).eps
        idx = jnp.sum(prec_eigvals > tol, axis=-1)
        # Heuristic when the number of context points is small 
        # if idx > self.n_context_points:
        idx = jnp.sum(prec_eigvals > self.max_posterior_precision, axis=-1)
        self.prec_eigvals = prec_eigvals[:idx]
        self.prec_eigvecs = prec_eigvecs[:,:idx]
        print(self.prec_eigvals)

        # from skerch.a_posteriori import (
        #     a_posteriori_error,
        #     a_posteriori_error_bounds,
        #     scree_bounds,
        # )
        # from skerch.linops import CompositeLinOp, DiagonalLinOp
        # import matplotlib.pyplot as plt
        # NUM_A_POSTERIORI = 50
        # appr_psd = CompositeLinOp(
        #     (
        #         ("Q", q_psd),
        #         ("U", u_psd),
        #         ("S", DiagonalLinOp(s_psd)),
        #         ("Ut", u_psd.T),
        #         ("Qt", q_psd.T),
        #     )
        # )
        # (f1_psd, f2_psd, frob_err_psd) = a_posteriori_error(
        #     posterior_precision_linops, appr_psd, NUM_A_POSTERIORI, dtype=DTYPE, device=DEVICE
        # )[0]
        # print("Estimated Frobenius Error (psd):", frob_err_psd**0.5)
        # print(a_posteriori_error_bounds(NUM_A_POSTERIORI, 0.5))
        # scree_lo, scree_hi = scree_bounds(s_psd, f1_psd**0.5, frob_err_psd**0.5)
        # plt.plot(scree_lo.cpu())
        # plt.plot(scree_hi.cpu())
        # plt.show()
        # plt.close()

        return self
    

    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        return self.prec_eigvecs * 1 / self.prec_eigvals**0.5 #jnp.diag(1 / self.prec_eigvals**0.5)


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
    


class LanczosLowRankFunctionalLaplacePosterior(BaseFunctionalLaplacePosterior):
    """Low rank Laplace approximation to the neural network posterior using SKERCH."""
    
    def __init__(
        self, 
        model
    ):
        super().__init__(model)
        self.max_rank = self.config["fsplaplace"]["inference"]["max_rank"]
        self.n_chunks = self.config["fsplaplace"]["inference"]["n_chunks"]
    

    def fit(
        self,
        train_dataloader,
        val_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """ 
        print("Load covariance from file", flush=True)
        if self.covariance_path:   
            with open(self.covariance_path, "rb") as f:
                self.cov_sqrt = pickle.load(f)
            return self  
        
        # Get context points
        print("Select context points", flush=True)
        x_context = self._select_context_points(
            train_dataloader, 
            val_dataloader,
            self.key
        )
        n_context_points = x_context.shape[0]

        # Divide the context points into batches
        print("Divide context points into chunks", flush=True)
        x_context = x_context.reshape(self.n_chunks, -1, *x_context.shape[1:])
        print("x_context", x_context.shape)

        # Define neural network function 
        f = lambda p, x: self.model.apply_fn(merge(p, self.other_params), self.state, self.key, x, training=False)[0]
        
        # GGN linear operator - can we compute the kernel for one class at a time?
        print("Compute GGN linear operator", flush=True)
        ggn_linop = GGNLinearOperator(
            shape=(self.dim, self.dim),
            fun=lambda p, x: f(p, x), 
            loss=lambda f, y: -self._log_likelihood(f, y),
            params=self.mean_params,
            dataloader=train_dataloader
        )

        # Machine precision
        eps = jnp.finfo(x_context.dtype).eps

        # Compute low rank approximation of the posterior precision
        M = []
        jvp_jit = jax.jit(lambda _x, _k, _v: jax.jvp(lambda _p: f(_p, _x)[:,_k], (self.mean_params,), (_v,))[1])
        vjp_jit = jax.jit(lambda _x, _k, _k_c: jax.vmap(jax.vjp(lambda _p: f(_p, _x)[:,_k], self.mean_params)[1], in_axes=-1, out_axes=-1)(_k_c)[0])
        ravel_jit = jax.jit(lambda M: jax.vmap(lambda p: ravel_pytree(p)[0], in_axes=-1, out_axes=-1)(M))
        for k in range(self.n_outputs):
            # Compute low rank approximation of the kernel precision 
            print(f"Compute low rank approximation of the kernel precision for class {k}", flush=True)
            kernel_linop = KernelLinearOperator(self.prior, k, x_context, self.n_chunks)
            ones_pytree = jax.tree_map(lambda x: x*0 + 1, self.mean_params)
            b = jnp.concatenate([jvp_jit(x_c, k, ones_pytree) for x_c in x_context], axis=0)
            k_inv_sqrt = lanczos_compute_efficient(kernel_linop, b, tol=eps**0.5, min_eta=1e-9, max_iter=self.max_rank)

            # Chunk x_context and k_inv_sqrt
            k_inv_sqrt = jnp.reshape(k_inv_sqrt, (self.n_chunks, -1, *k_inv_sqrt.shape[1:]))
            
            # Batch compute M = J^T @ k_inv_sqrt
            zeros_pytree = jax.tree_map(lambda x: x*0, self.mean_params)
            M_k = jax.vmap(lambda p: zeros_pytree, out_axes=-1)(jnp.arange(k_inv_sqrt.shape[-1]))
            for x_c, k_c in zip(x_context, k_inv_sqrt):
                _M = vjp_jit(x_c, k, k_c)
                M_k = jax.tree_map(lambda a,b: a+b, M_k, _M)
            M += [ravel_jit(M_k)] # (p, rk)
            
        M = jnp.concatenate(M, axis=-1) # (p, rk)
        print("M", M.shape)
        
        # SVD
        _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
        eps = jnp.finfo(M.dtype).eps
        tol = eps**0.5 #threshold for machine precision for torch.diag(s**2) + uT_ggn_u
        s = _s[_s > tol]
        u = _u[:, _s > tol]
        print("u", u.shape, "s", s.shape)
        
        del M
        gc.collect()

        # Compute u.T @ ggn @ u
        uT_ggn_u = u.T @ (ggn_linop @ u) # (rk, rk)
        
        # Compute A = u @ (lam + u.T @ ggn @ u)^-1/2
        _eigvals, _eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + uT_ggn_u)
        tol = 0 #eps * (_eigvals.max()**0.5) * s.shape[0]
        eigvals = _eigvals[_eigvals > tol] # for pseudo-inversion
        eigvecs = _eigvecs[:, _eigvals > tol]
        print("eigvals", eigvals.shape, "eigvecs", eigvecs.shape)

        # Add eigenvectors/eigenvalues starting by the largest eigenvalues
        eigvals = jnp.flip(eigvals, axis=0)
        eigvecs = jnp.flip(eigvecs, axis=1)
        
        # Marginal variance heuristic
        i = 0
        post_var = jnp.zeros((n_context_points, self.n_outputs))
        prior_var = self.model.prior.marginal_variance(x_context.reshape(np.prod(x_context.shape[:2]), -1)) # (n_context_points, n_outputs)
        cov_sqrt = []
        _f1 = lambda _e, _u, _v: _u @ (_v * (1 / _e**0.5))
        _f2 = lambda _x, _v: jax.jvp(lambda _p: f(_p, _x), (self.mean_params,), (_v,))[1] #(_lr_fac_i,))[1] # (n_batch, n_outputs)
        while jnp.all(post_var < prior_var) and i < eigvals.shape[0]:
            cov_sqrt += [jax.jit(_f1)(eigvals[i], u, eigvecs[:,i])] 
            lr_fac_i = jax.jit(self.model.unravel_params)(cov_sqrt[-1]) 
            post_var += jnp.concatenate(
                [jax.jit(_f2)(x_c, lr_fac_i)**2 for x_c in x_context],
                axis=0
            )
            print(f"{i} - post_tr={post_var.sum()} - prior_tr={prior_var.sum()}")
            i += 1
        
        # # Remove unused columns
        truncation_idx = i if i == eigvals.shape[0] else i-1
        self.cov_sqrt = jnp.stack(cov_sqrt[:truncation_idx], axis=-1) # (p, rk)

        return self
    
       
    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        return self.cov_sqrt


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
    

class LowRankFunctionalLaplacePosterior(BaseFunctionalLaplacePosterior):
    """Low rank Laplace approximation to the neural network posterior."""
    def __init__(
        self, 
        model
    ):
        super().__init__(model)
        self.cov_type = "low_rank"
        self.max_rank = self.config["fsplaplace"]["inference"]["rank"]
        self.max_posterior_precision = self.config["fsplaplace"]["inference"]["max_posterior_precision"]
    

    def fit(
        self,
        train_dataloader,
        val_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """        
        self.key, key1 = jax.random.split(self.key)

        # Get context points
        x_context = self._select_context_points(
            train_dataloader, 
            val_dataloader,
            key1
        )

        precision = jnp.zeros((self.dim, self.dim))
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1 = jax.random.split(self.key)
            # Update precision
            precision += self._update_curvature(x, key1)

        self.key, key1 = jax.random.split(self.key)
        fwd = lambda p: self.model.apply_fn(merge(p, self.other_params), self.state, key1, x_context, training=False)[0]
        J = jax.jacrev(fwd)(self.mean_params)
        leaves = jax.tree_util.tree_leaves(J)
        J = jnp.concatenate([l.reshape(self.n_context_points, self.n_outputs, -1) for l in leaves], axis=-1) # (n_batch, n_classes, n_params)
        prior_cov = self.prior(x_context)[1]
        precision += jax.vmap(
            lambda _K, _J: _J.T @ jnp.linalg.solve(_K, _J), 
            in_axes=(-1,1), 
            out_axes=0
        )(prior_cov, J).sum(0)

        # Eigenvalue decomposition
        prec_eigvals, prec_eigvecs = jnp.linalg.eigh(precision)
        prec_eigvals = prec_eigvals[::-1]
        prec_eigvecs = prec_eigvecs[:,::-1]

        # Filter out negative eigenvalues
        idx = jnp.sum(prec_eigvals > 0, axis=-1)
        prec_eigvals = prec_eigvals[:idx]
        prec_eigvecs = prec_eigvecs[:,:idx]

        # Filter out small eigenvalues
        tol = prec_eigvals.max() * self.dim * jnp.finfo(prec_eigvals.dtype).eps
        idx = jnp.sum(prec_eigvals > tol, axis=-1)
        # Heuristic when the number of context points is small 
        if idx > self.n_context_points:
            idx = jnp.sum(prec_eigvals > self.max_posterior_precision, axis=-1)
        self.prec_eigvals = prec_eigvals[:idx]
        self.prec_eigvecs = prec_eigvecs[:,:idx]
    
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
        f = lambda p: self.model.apply_fn(merge(p, self.other_params), self.state, key, x, training=False)[0]
        jacobian = jax.jacrev(f)(self.mean_params)

        # Vectorize Jacobian
        v_jacobian = jnp.concatenate(
            [l.reshape(x.shape[0],self.n_outputs,-1) for l in jax.tree_util.tree_leaves(jacobian)],
            axis=-1
        )

        # Compute likelihood hessian 
        if self.likelihood == "Gaussian":
            ll_hessian = -1 / self.ll_scale**2
        elif self.likelihood == "Categorical":
            probs = jax.nn.softmax(f(self.mean_params), axis=-1)  # (n_batch, n_classes)
            ll_hessian = -jax.vmap(jnp.diag)(probs) + jnp.einsum('bk,bc->bck', probs, probs) # (n_batch, n_classes, n_classes)
        n_likelihood_hessian = -ll_hessian

        # Update precision
        if self.likelihood == "Gaussian":
            curvature_update = n_likelihood_hessian * jnp.einsum('bcp,bcq->pq', v_jacobian, v_jacobian)
        else:
            curvature_update = jnp.einsum('bcp,bck,bkq->pq', v_jacobian, n_likelihood_hessian, v_jacobian)

        return curvature_update
    

    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        return self.prec_eigvecs * 1 / self.prec_eigvals**0.5


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
    
    
class DiagFunctionalLaplacePosterior(BaseFunctionalLaplacePosterior):

    def __init__(
        self, 
        model
    ):
        super().__init__(model)
        self.cov_type = "diag"
    

    def fit(
        self,
        train_dataloader,
        val_dataloader
    ):
        """
        Fit Laplace approximation to the neural network posterior.

        params:
        - train_dataloader (DataLoader): dataloader of training data.

        returns:
        - self (LaplacePosterior): Laplace approximation to the neural network posterior.
        """        
        self.key, key1 = jax.random.split(self.key)

        # Get context points
        x_context = self._select_context_points(
            train_dataloader, 
            val_dataloader,
            self.key
        )

        precision = jnp.zeros((self.dim,))
        for x, y in train_dataloader:
            # Split the keys
            self.key, key1 = jax.random.split(self.key)
            # Update precision
            precision += self._update_curvature(x, key1)

        # Contribution of the prior to the precision
        self.key, key1 = jax.random.split(self.key)
        fwd = lambda p: self.model.apply_fn(merge(p, self.other_params), self.state, key1, x_context, training=False)[0]
        J = jax.jacrev(fwd)(self.mean_params)
        leaves = jax.tree_util.tree_leaves(J)
        J = jnp.concatenate([l.reshape(self.n_context_points, self.n_outputs, -1) for l in leaves], axis=-1) # (n_batch, n_classes, n_params)
        prior_cov = self.prior(x_context)[1]
        precision += jnp.einsum(
            "bkp,bkp->p", 
            J, 
            jax.vmap(
                lambda _K, _J: jnp.linalg.solve(_K, _J), 
                in_axes=(-1,1), 
                out_axes=1
            )(prior_cov, J)
        )

        self.covariance = 1 / precision

        return self
    

    @partial(jax.jit, static_argnums=(0,))
    def _update_curvature(
        self,
        x,
        key
    ):
        """
        Update curvature matrix.

        params
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.

        returns:
        - curvature_update (jnp.array): curvature update.
        """
        # Compute Jacobian
        f = lambda p: self.model.apply_fn(merge(p, self.other_params), self.state, key, x, training=False)[0]
        jacobian = jax.jacrev(f)(self.mean_params)[0]

        # Vectorize Jacobian
        v_jacobian = jnp.concatenate(
            [l.reshape(x.shape[0],self.n_outputs,-1) for l in jax.tree_util.tree_leaves(jacobian)],
            axis=-1
        )

        # Compute likelihood hessian 
        if self.likelihood == "Gaussian":
            ll_hessian = -1 / self.ll_scale**2
        elif self.likelihood == "Categorical":
            probs = jax.nn.softmax(f(self.mean_params), axis=-1)  # (n_batch, n_classes)
            ll_hessian = -jax.vmap(jnp.diag)(probs) + jnp.einsum('bk,bc->bck', probs, probs) # (n_batch, n_classes, n_classes)
        n_likelihood_hessian = -ll_hessian

        # Update precision
        if self.likelihood == "Gaussian":
            curvature_update = n_likelihood_hessian * jnp.einsum('bcp,bcp->p', v_jacobian, v_jacobian)
        else:
            curvature_update = jnp.einsum('bcp,bck,bkp->p', v_jacobian, n_likelihood_hessian, v_jacobian)
    
        return curvature_update
    

    def posterior_covariance_sq_root(
        self,
    ):
        """
        Compute the factorization of the posterior covariance.

        returns:
        - sq_root_cov (jnp.array): factorization of the posterior covariance.
        """
        return self.covariance ** 0.5


