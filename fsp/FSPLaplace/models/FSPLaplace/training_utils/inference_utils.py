import jax

import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree



class PrecisionLinearOperator:

    def __init__(
        self,
        shape,
        fun,
        loss,
        params,
        kernel_sqrt_inv,
        x_c,
        dataloader
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
        self.kernel_sqrt_inv = kernel_sqrt_inv
        self.dataloader = dataloader
        self.x_c = x_c
        self.unravel_fn = ravel_pytree(self.params)[1]
        self.dtype = x_c.dtype


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
        if isinstance(x, jnp.ndarray):
            x = self.unravel_fn(x)
        return x


    def _output_wrapper(
        self,
        x
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
        return x
    

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
        # Convert array to JAX pytree
        x = self._input_wrapper(x)

        # Compute the prior GGN-vector product
        prec = jax.tree_map(lambda p: p*0., self.params)
        for k in range(len(self.kernel_sqrt_inv)):
            prec = jax.tree_map(lambda a,b: a+b, prec, self._update_prior_ggn(x, k))

        # Compute the likelihood GGN-vector product
        for x_b, y_b in self.dataloader:
            prec = jax.tree_map(lambda a,b: a+b, prec, self._update_likelihood_ggn(x_b, y_b, x))

        # Convert the result to array
        result = self._output_wrapper(prec)

        return result


    @partial(jax.jit, static_argnums=(0,))
    def _update_likelihood_ggn(
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
    

    @partial(jax.jit, static_argnums=(0,2))
    def _update_prior_ggn(
        self,
        x, 
        k
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
        f = lambda p: self.fun(p, self.x_c)[:,k]
        _, jx = jax.jvp(f, (self.params,), (x,)) 
        K_inv_jx = self.kernel_sqrt_inv[k] @ (self.kernel_sqrt_inv[k].T @ jx) # chunk computation
        _, jK_inv_jx_fn = jax.vjp(f, self.params)
        return jK_inv_jx_fn(K_inv_jx)[0]
    

class GGNLinearOperator:

    def __init__(
        self,
        shape,
        fun,
        loss,
        params,
        dataloader
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
        self.dataloader = dataloader
        self.unravel_fn = ravel_pytree(self.params)[1]
        self.dtype = dataloader.dataset.y.dtype
    

    def __matmul__(
        self,
        M
    ):
        """
        Forward (right) GGN-vector multiplication ``self @ x``.

        params:
        - x (array): input array.

        returns:
        - result (array): output array.
        """
        # Convert array to JAX pytree
        M = [self.unravel_fn(x) for x in M.T]

        # Compute the likelihood GGN-vector product
        prec = [jax.tree_map(lambda p: p*0., self.params) for _ in M]
        for x_b, y_b in self.dataloader:
            for i, x in enumerate(M): # Maybe vmap with chunk
                prec[i] = jax.tree_map(lambda a,b: a+b, prec[i], self._ggn_vector_product(x_b, y_b, x))

        # Convert the result to array
        result = jnp.stack(
            [jax.flatten_util.ravel_pytree(prec[i])[0] for i in range(len(M))],
            axis=-1
        )

        return result
    

    @partial(jax.jit, static_argnums=(0,))
    def _ggn_vector_product(
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
    

class KernelLinearOperator:

    def __init__(
        self,
        kernel,
        k_idx,
        x, 
        chunks
    ):
        self.kernel = kernel
        self.x = x
        self.chunks = chunks # number of chunks to split the data - hyperparameter
        self.shape = (self.x.shape[0], self.x.shape[0])
        self.dtype = x.dtype
        self.k_idx = k_idx


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
        cc = jax.vmap(lambda _xj: self.kernel.cross_covariance_k(xi, _xj, self.k_idx))(xj)
        return jnp.einsum('bij,bj->bi', cc, vj)

    
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
        v_chunked = self._chunk(v, self.chunks)
        def scan_fun(i, x_i):
            return i + 1, jnp.sum(self._kernel_mvm(x_i, self.x, v_chunked), 0)
        val = jax.lax.scan(scan_fun, 0, self.x)[1]
        kv = jnp.reshape(val, (-1, *val.shape[2:]))

        return kv
    

