import jax
import torch

import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from jax.flatten_util import ravel_pytree



class TorchGGNLinearOperatorWrapper:
    """Wraps the JAX GGN linear operator to be compatible with Skerch."""
    def __init__(
        self,
        shape,
        fun,
        loss,
        params,
        prior,
        x_c,
        dataloader, 
        key
    ):
        """
        params:
        - shape (tuple): shape of the operator.
        - fun (callable): function to evaluate the model - fun(params, x).
        - loss (callable): loss function - loss(f, y).
        - params (pytree): model parameters.
        - prior_prec (pytree): prior precision.
        - dataloader (DataLoader): dataloader.
        """
        self.shape = shape
        self.fun = fun
        self.loss = loss
        self.params = params
        self.prior_cov_linop = KernelLinearOperatorWrapper(prior, x_c)
        self.dataloader = dataloader
        self.x_c = x_c
        self.unravel_fn = ravel_pytree(self.params)[1]
        self.D = None
        self.key = key


    def _input_wrapper(
        self,
        x
    ):
        """
        Helper method to admit PyTorch tensors.

        params:
        - x (array): input array.

        returns:
        - x_pytree (pytree): input tensor as a pytree.
        - device (torch.device): device of the input tensor.
        """
        if isinstance(x, torch.Tensor):
            x_jnp = jax.dlpack.from_dlpack(torch.to_dlpack(x.cpu()))
            x_pytree = self.unravel_fn(x_jnp)
            return x_pytree, x.device
        else:
            return x, None


    def _output_wrapper(
        self,
        x,
        torch_device=None
    ):
        """
        Helper method to produce PyTorch tensors.

        params:
        - x (array): input array.
        - torch_device (torch.device): device of the output tensor.

        returns:
        - x_torch (torch.Tensor): output tensor.
        """
        x =  jax.flatten_util.ravel_pytree(x)[0]
        if torch_device is not None:
            x_torch = torch.from_dlpack(jax.dlpack.to_dlpack(x))
            return x_torch.to(torch_device)
        else:
            return torch.from_dlpack(jax.dlpack.to_dlpack(x))
        

    def __rmatmul__(
        self,
        x
    ):
        """
        Adjoint (left) matrix-vector multiplication ``x @ self``.
        The GGN is self-adjoint, so this is the same as ``self @ x``.

        params:
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        x, device = self._input_wrapper(x)
        result = self._output_wrapper(self.__matmul__(x), device)

        return result


    def __repr__(self):
        """
        Returns a string in the form TorchLinOpWrapper<LinOp ...>.
        """
        wrapper = self.__class__.__name__
        result = f"{wrapper}<{super().__repr__()}>"

        return result
    

    
    def __matmul__(
        self,
        x
    ):
        """
        Forward (right) GGN-vector multiplication ``self @ x``.

        params:
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        # Convert PyTorch tensor to JAX pytree
        x, device = self._input_wrapper(x)

        # Compute the GGN-vector product
        prec = self._update_prior_ggn_cg(x) if self.D == None else self._update_prior_ggn(x)
        for x_b, y_b in self.dataloader:
            prec = jax.tree_map(lambda a,b: a+b, prec, self._update_ll_ggn(x_b, y_b, x))

        # Convert the result to PyTorch tensor
        result = self._output_wrapper(prec, device)

        return result


    @partial(jax.jit, static_argnums=(0,))
    def _update_ll_ggn(
        self,
        x_b,
        y_b,
        x
    ):
        """
        Helper method to compute the GGN-vector product.

        params:
        - x_b (array): batch of inputs.
        - y_b (array): batch of outputs.
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        f = lambda p: self.fun(p, x_b)
        loss = lambda f: self.loss(f, y_b)
        f_b, jx = jax.jvp(f, (self.params,), (x,))
        hjx = jax.jvp(jax.grad(loss), (f_b,), (jx,))[1] # jax.hvp(loss, f_b, jx)
        _, jhjx_fn = jax.vjp(f, self.params)

        return jhjx_fn(hjx)[0]
    

    def _update_prior_ggn_cg(
        self,
        x
    ):
        """
        Helper method to compute the GGN-vector product.

        params:
        - x_b (array): batch of inputs.
        - y_b (array): batch of outputs.
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        # Prior term
        f = lambda p: self.fun(p, self.x_c)
        _, jx = jax.jvp(f, (self.params,), (x,))
        #K_inv_jx, self.D = self._cg(self.prior_cov_linop, jx, atol=1e-6, tol=1e-8, maxiter=1000)
        #K_inv_jx, self.D = self._lanczos_compute_efficient(self.prior_cov_linop, jx, atol=1e-6, tol=1e-8, maxiter=1000)
        b = jax.jvp(f, (self.params,), (jax.tree_map(lambda x: x*0 + 1, self.params),))[1].reshape(-1) / self.shape[0] # do this by chuncks
        self.D = self._lanczos_memory_efficient(self.prior_cov_linop, b, tol=1e-10, maxiter=1000)
        K_inv_jx = self.D @ (self.D.T @ jx)
        #K_inv_jx = jsp.sparse.linalg.cg(self.prior_cov_linop, jx, atol=1e-6, tol=1e-8, maxiter=1000)[0] # add preconditionner to speed up convergence
        _, jK_inv_jx_fn = jax.vjp(f, self.params)

        return jK_inv_jx_fn(K_inv_jx)[0]
    

    @partial(jax.jit, static_argnums=(0,))
    def _update_prior_ggn(
        self,
        x
    ):
        """
        Helper method to compute the GGN-vector product.

        params:
        - x_b (array): batch of inputs.
        - y_b (array): batch of outputs.
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        # Prior term
        f = lambda p: self.fun(p, self.x_c)
        _, jx = jax.jvp(f, (self.params,), (x,))
        K_inv_jx = self.D @ (self.D.T @ jx)
        _, jK_inv_jx_fn = jax.vjp(f, self.params)

        return jK_inv_jx_fn(K_inv_jx)[0]


    def _cg(
        self,
        A,
        b,
        atol=0,
        tol=1e-3,
        maxiter=1000
    ):
        """
        Conjugate gradient method to solve the linear system Ax = b.

        params:
        - A (callable): linear operator.
        - b (array): right-hand side.
        - x0 (array): initial guess.
        - atol (float): absolut tolerance.
        - tol (float): relative tolerance.
        - max_iter (int): maximum number of iterations.
        """
        @jax.jit
        def _step(values):
            x, r, _, D, i = values
            alpha = jnp.linalg.norm(r)**2
            A_r = A @ r
            # Double gram schmidt
            d = r - D @ (D.T @ A_r)
            d = d - D @ (D.T @ (A @ d))
            eta = d.T @ A_r
            jax.debug.print("eta: {a} - norm = {r}", a=eta, r=jnp.linalg.norm(r))
            x = x + alpha * d / eta
            D = jnp.concatenate([D, (d / jnp.sqrt(eta)).reshape(-1,1)], axis=-1)
            r = b - A @ x
            return x, r, eta, D, i + 1

        def _cond_fun(values):
            _, _, eta, _, i = values
            return (jnp.linalg.norm(r) > tol) & (i < maxiter) & (eta > 1e-20)
        
        # Initialization
        tol = jnp.maximum(tol * jnp.linalg.norm(b), atol)
        D = jnp.zeros((b.shape[0], 0))
        x = jnp.zeros_like(b)
        r = b

        # CG loop
        values = (x, r, jnp.inf, D, 0)
        while _cond_fun(values): values = _step(values)
        
        # Collect results
        x, _, _, D, _ = values

        return x.reshape(-1,1), D
    

    def _lanczos_memory_efficient(
        self,
        A,
        b=None,
        tol=1e-3,
        maxiter=1000
    ):
        """
        Conjugate gradient method to solve the linear system Ax = b.

        params:
        - A (callable): linear operator.
        - b (array): right-hand side.
        - x0 (array): initial guess.
        - atol (float): absolut tolerance.
        - tol (float): relative tolerance.
        - maxiter (int): maximum number of iterations.
        """
        @jax.jit
        def _step(values):
            x, r, _, r_norm_sq, D, i = values
            alpha = r_norm_sq
            Ar = A @ r
            d = r - D @ (D.T @ Ar)
            d = d - D @ (D.T @ (A @ d))
            eta = d @ Ar
            x = x + (alpha / eta) * d
            D = jnp.concatenate((D, d[:, None] / jnp.sqrt(eta)), axis=-1)
            r = b - A @ x
            r_norm_sq = r @ r
            jax.debug.print("eta: {a} - norm = {r}", a=eta, r=jnp.linalg.norm(r))
            return x, r, eta, r_norm_sq, D, i + 1

        def _cond_fun(values):
            _, _, eta, r_norm_sq, _, i = values
            return (i < maxiter) & (jnp.sqrt(r_norm_sq) > tol) & (eta > 1e-20) 
        
        # Initialization
        eta = jnp.inf

        if b is None:
            b = jax.random.normal(self.key, shape=(A.shape[0],))
            b /= jnp.linalg.norm(b, 2)

        x = jnp.zeros_like(b)
        D = jnp.zeros_like(b, shape=(b.size, 0))

        r = b
        r_norm_sq = r @ r
        
        # Lanczos iterations
        values = (x, r, eta, r_norm_sq, D, 0)
        while _cond_fun(values): values = _step(values)
        
        # Collect results
        _, _, _, _, D, _ = values

        return D
    

    def _lanczos_compute_efficient(
        self,
        A,
        b,
        atol=0,
        tol=1e-3,
        maxiter=1000
    ):
        """
        Conjugate gradient method to solve the linear system Ax = b.

        params:
        - A (callable): linear operator.
        - b (array): right-hand side.
        - x0 (array): initial guess.
        - atol (float): absolut tolerance.
        - tol (float): relative tolerance.
        - maxiter (int): maximum number of iterations.
        """
        @jax.jit
        def _step(values):
            x, e, D, R, alpha_prev, eta, i = values
            r = R[:,-1].reshape(-1,1)
            alpha = jnp.linalg.norm(r)**2
            d = r + alpha * e / alpha_prev
            e = A @ d
            eta = r.T @ e
            x = x + alpha * d / eta
            r = r - alpha * e / eta
            # Double gram schmidt
            r = r - R @ (R.T @ r)
            r = r - R @ (R.T @ r)
            D = jnp.concatenate([D, (d / jnp.sqrt(eta)).reshape(-1,1)], axis=-1)
            R = jnp.concatenate([R, r.reshape(-1,1)], axis=-1)
            jax.debug.print("eta: {a} - norm = {r}", a=eta, r=jnp.linalg.norm(r))
            return x, e, D, R, alpha, eta, i+1

        def _cond_fun(values):
            _, _, _, R, _, eta, i = values
            return (jnp.linalg.norm(R[:,-1]) > tol) & (i < maxiter) & (eta > 1e-20)
        
        # Initialization
        eps = jax.random.normal(self.key, b.shape)
        R = eps / jnp.linalg.norm(eps)
        D = jnp.zeros((b.shape[0], 0))
        x = jnp.zeros_like(b)
        e = jnp.zeros_like(b)
        tol = jnp.maximum(tol * jnp.linalg.norm(b), atol)
        alpha_prev = 1
        eta = jnp.inf
    
        # Lanczos iterations
        values = (x, e, D, R, alpha_prev, eta, 0)
        while _cond_fun(values): values = _step(values)
        
        # Collect results
        x, _, D, _, _, _, _ = values

        return x.reshape(-1,1), D

    


class KernelLinearOperatorWrapper:

    def __init__(
        self,
        kernel,
        x
    ):
        self.kernel = kernel
        self.x = x
        self.chunks = 50 # number of chunks to split the data - hyperparameter
        self.shape = (self.x.shape[0], self.x.shape[0])


    def _chunk(
        self, 
        x, 
        chunks
    ):
        return jnp.reshape(x, (chunks, -1, *x.shape[1:]))
    

    @partial(jax.jit, static_argnums=(0,))
    def _kernel_mvm(
        self, 
        xi, 
        xj, 
        vj
    ):
        # cc = jax.vmap(lambda _xj: self.kernel.cross_covariance(xi, _xj))(xj)
        # return jnp.einsum('bijc,bjc->bic', cc, vj)
        cc = jax.vmap(lambda _xj: self.kernel.cross_covariance(xi, _xj))(xj)
        return jnp.einsum('bijc,bj->bi', cc, vj)


    
    @partial(jax.jit, static_argnums=(0,))
    def __matmul__(
        self,
        v
    ):
        """
        Forward (right) GGN-vector multiplication ``self @ x``.

        params:
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        x_chunked = self._chunk(self.x, self.chunks)
        v_chunked = self._chunk(v, self.chunks)
        def scan_fun(i, x_i):
            return i + 1, jnp.sum(self._kernel_mvm(x_i, x_chunked, v_chunked), 0)
        val = jax.lax.scan(scan_fun, 0, x_chunked)[1]
        kv = jnp.reshape(val, (-1, *val.shape[2:]))

        return kv

