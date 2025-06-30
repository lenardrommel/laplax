import operator
import pickle
from functools import partial
import gc

import jax
from flax import nnx
from inference_utils import GGNLinearOperator, KernelLinearOperator
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from examples.pipeline.lanzcos import lanczos_compute_efficient

jax.config.update("jax_enable_x64", True)


class LanczosLowRankFunctionalLaplacePosterior:
    """Low rank Laplace approximation to the neural network posterior using SKERCH."""

    def __init__(self, model):
        super().__init__(model)
        self.max_rank = self.config["fsplaplace"]["inference"]["max_rank"]
        self.n_chunks = self.config["fsplaplace"]["inference"]["n_chunks"]

    def fit(self, train_dataloader, val_dataloader):
        print("Load covariance from file", flush=True)
        if self.covariance_path:
            with open(self.covariance_path, "rb") as f:
                self.cov_sqrt = pickle.load(f)
            return self

        # Get context points
        print("Select context points", flush=True)
        x_context = self._select_context_points(
            train_dataloader, val_dataloader, self.key
        )
        n_context_points = x_context.shape[0]

        # Divide the context points into batches
        print("Divide context points into chunks", flush=True)
        x_context = x_context.reshape(self.n_chunks, -1, *x_context.shape[1:])
        print("x_context", x_context.shape)

        # Define neural network function

        f = lambda p, x: self.model.apply_fn(
            merge(p, self.other_params), self.state, self.key, x, training=False
        )[0]

        # GGN linear operator - can we compute the kernel for one class at a time?
        print("Compute GGN linear operator", flush=True)
        ggn_linop = GGNLinearOperator(
            shape=(self.dim, self.dim),
            fun=lambda p, x: f(p, x),
            loss=lambda f, y: -self._log_likelihood(f, y),
            params=self.mean_params,
            dataloader=train_dataloader,
        )

        # Machine precision
        eps = jnp.finfo(x_context.dtype).eps

        # Compute low rank approximation of the posterior precision
        M = []
        jvp_jit = jax.jit(
            lambda _x, _k, _v: jax.jvp(
                lambda _p: f(_p, _x)[:, _k], (self.mean_params,), (_v,)
            )[1]
        )
        vjp_jit = jax.jit(
            lambda _x, _k, _k_c: jax.vmap(
                jax.vjp(lambda _p: f(_p, _x)[:, _k], self.mean_params)[1],
                in_axes=-1,
                out_axes=-1,
            )(_k_c)[0]
        )
        ravel_jit = jax.jit(
            lambda M: jax.vmap(lambda p: ravel_pytree(p)[0], in_axes=-1, out_axes=-1)(M)
        )
        for k in range(self.n_outputs):
            # Compute low rank approximation of the kernel precision
            print(
                f"Compute low rank approximation of the kernel precision for class {k}",
                flush=True,
            )
            kernel_linop = KernelLinearOperator(self.prior, k, x_context, self.n_chunks)
            ones_pytree = jax.tree_map(lambda x: x * 0 + 1, self.mean_params)
            b = jnp.concatenate(
                [jvp_jit(x_c, k, ones_pytree) for x_c in x_context], axis=0
            )
            k_inv_sqrt = lanczos_compute_efficient(
                kernel_linop, b, tol=eps**0.5, min_eta=1e-9, max_iter=self.max_rank
            )

            # Chunk x_context and k_inv_sqrt
            k_inv_sqrt = jnp.reshape(
                k_inv_sqrt, (self.n_chunks, -1, *k_inv_sqrt.shape[1:])
            )

            # Batch compute M = J^T @ k_inv_sqrt
            zeros_pytree = jax.tree_map(lambda x: x * 0, self.mean_params)
            M_k = jax.vmap(lambda p: zeros_pytree, out_axes=-1)(
                jnp.arange(k_inv_sqrt.shape[-1])
            )
            for x_c, k_c in zip(x_context, k_inv_sqrt):
                _M = vjp_jit(x_c, k, k_c)
                M_k = jax.tree_map(lambda a, b: a + b, M_k, _M)
            M += [ravel_jit(M_k)]  # (p, rk)

        M = jnp.concatenate(M, axis=-1)  # (p, rk)
        print("M", M.shape)

        # SVD
        _u, _s, _ = jnp.linalg.svd(M, full_matrices=False)
        eps = jnp.finfo(M.dtype).eps
        tol = (
            eps**0.5
        )  # threshold for machine precision for torch.diag(s**2) + uT_ggn_u
        s = _s[_s > tol]
        u = _u[:, _s > tol]
        print("u", u.shape, "s", s.shape)

        del M
        gc.collect()

        # Compute u.T @ ggn @ u
        uT_ggn_u = u.T @ (ggn_linop @ u)  # (rk, rk)

        # Compute A = u @ (lam + u.T @ ggn @ u)^-1/2
        _eigvals, _eigvecs = jnp.linalg.eigh(jnp.diag(s**2) + uT_ggn_u)
        tol = 0  # eps * (_eigvals.max()**0.5) * s.shape[0]
        eigvals = _eigvals[_eigvals > tol]  # for pseudo-inversion
        eigvecs = _eigvecs[:, _eigvals > tol]
        print("eigvals", eigvals.shape, "eigvecs", eigvecs.shape)

        # Add eigenvectors/eigenvalues starting by the largest eigenvalues
        eigvals = jnp.flip(eigvals, axis=0)
        eigvecs = jnp.flip(eigvecs, axis=1)

        # Marginal variance heuristic
        i = 0
        post_var = jnp.zeros((n_context_points, self.n_outputs))
        prior_var = self.model.prior.marginal_variance(
            x_context.reshape(np.prod(x_context.shape[:2]), -1)
        )  # (n_context_points, n_outputs)
        cov_sqrt = []
        _f1 = lambda _e, _u, _v: _u @ (_v * (1 / _e**0.5))
        _f2 = lambda _x, _v: jax.jvp(lambda _p: f(_p, _x), (self.mean_params,), (_v,))[
            1
        ]  # (_lr_fac_i,))[1] # (n_batch, n_outputs)
        while jnp.all(post_var < prior_var) and i < eigvals.shape[0]:
            cov_sqrt += [jax.jit(_f1)(eigvals[i], u, eigvecs[:, i])]
            lr_fac_i = jax.jit(self.model.unravel_params)(cov_sqrt[-1])
            post_var += jnp.concatenate(
                [jax.jit(_f2)(x_c, lr_fac_i) ** 2 for x_c in x_context], axis=0
            )
            print(f"{i} - post_tr={post_var.sum()} - prior_tr={prior_var.sum()}")
            i += 1

        # # Remove unused columns
        truncation_idx = i if i == eigvals.shape[0] else i - 1
        self.cov_sqrt = jnp.stack(cov_sqrt[:truncation_idx], axis=-1)  # (p, rk)

        return self
