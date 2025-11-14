import jax

import jax.scipy as jsp
import jax.numpy as jnp

from functools import partial

class SpectralScoreEstimator:
    """
    Spectral Stein gradient estimator.
    """
    def __init__(
        self, 
        n_eigen=None, 
        eta=0., 
        n_eigen_threshold=None
    ):
        """
        Build Spectral Stein gradient estimator.

        params:
        - n_eigen (int): number of eigen values and eigen vectors to pick.
        - eta (float): regularization parameter.
        - n_eigen_threshold (float): threshold to pick number of eigen values and eigen vectors.
        """
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold


    @partial(jax.jit, static_argnums=(0,))
    def rbf_kernel(
        self, 
        x1, 
        x2, 
        kernel_width
    ):
        """
        RBF kernel.

        inputs:
        - x1 (jnp.ndarray): input data of shape [..., n1, x_dim].
        - x2 (jnp.ndarray): input data of shape [..., n2, x_dim].
        - kernel_width (jnp.ndarray): kernel width of shape [..., 1, 1, x_dim].

        returns:
        - rbf_x1_x2 (jnp.ndarray): RBF kernel evaluated at x1 and x2.
        """
        return jnp.exp(
            -jnp.sum(
                jnp.square((x1 - x2) / kernel_width), 
                axis=-1
            ) / 2
        )


    @partial(jax.jit, static_argnums=(0,))
    def gram(
        self, 
        x1, 
        x2, 
        kernel_width
    ):
        """
        Compute Gram matrix.

        inputs:
        - x1 (jnp.ndarray): input data of shape [..., n1, x_dim].
        - x2 (jnp.ndarray): input data of shape [..., n2, x_dim].
        - kernel_width (jnp.ndarray): kernel width of shape [..., 1, 1, x_dim].

        returns:
        - G (jnp.ndarray): gram matrix of shape [..., n1, n2].
        """
        x_row = jnp.expand_dims(x1, -2)
        x_col = jnp.expand_dims(x2, -3)
        
        return self.rbf_kernel(x_row, x_col, kernel_width)


    @partial(jax.jit, static_argnums=(0,))
    def grad_gram(
        self, 
        x1, 
        x2, 
        kernel_width
    ):
        """
        Compute gradient of gram matrix wrt inputs.

        inputs:
        - x1 (jnp.ndarray): input data of shape [..., n1, x_dim].
        - x2 (jnp.ndarray): input data of shape [..., n2, x_dim].
        - kernel_width (jnp.ndarray): kernel width of shape [..., 1, 1, x_dim].

        returns:
        - G (jnp.ndarray): gram matrix of shape [..., n1, n2].
        - grad_x1 (jnp.ndarray): gradient of gram matrix wrt x1 of shape [..., n1, n2, x_dim].
        - grad_x2 (jnp.ndarray): gradient of gram matrix wrt x2 of shape [..., n1, n2, x_dim].
        """
        x_row = jnp.expand_dims(x1, -2)
        x_col = jnp.expand_dims(x2, -3)
        
        # Compute Gram matrix [..., n1, n2]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        
        # Compute gradient of Gram matrix wrt x1 and x2
        diff = (x_row - x_col) / (kernel_width**2) # [..., n1, n2, n_x]
        G_expand = jnp.expand_dims(G, axis=-1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        
        return G, grad_x1, grad_x2
    

    @partial(jax.jit, static_argnums=(0,))
    def heuristic_kernel_width(
        self, 
        x_samples, 
        x_basis
    ):
        """
        Heuristic to compute kernel width.

        inputs:
        - x_samples (jnp.ndarray): input data of shape [..., n_samples, x_dim].
        - x_basis (jnp.ndarray): input data of shape [..., n_basis, x_dim].

        returns:
        - kernel_width (jnp.ndarray): kernel width of shape [..., 1, 1, x_dim].
        """
        # Compute pairwise distance
        x_dim = x_samples.shape[-1]
        n_samples = x_samples.shape[-2]
        n_basis = x_basis.shape[-2]
        x_samples_expand = jnp.expand_dims(x_samples, -2)
        x_basis_expand = jnp.expand_dims(x_basis, -3)
        pairwise_dist = jnp.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.shape)
        reshape_dims = list(range(length-3)) + [length-1, length-3, length-2]
        pairwise_dist = jnp.transpose(pairwise_dist, reshape_dims)

        # Compute kernel width
        k = n_samples * n_basis // 2
        top_k_values = jax.lax.top_k(
            pairwise_dist.reshape([-1, x_dim, n_samples * n_basis]),
            k=k
        )[0]
        kernel_width = jnp.reshape(
            top_k_values[:, :, -1],
            x_samples.shape[:-2] + tuple([1, 1, x_dim])
        )
        kernel_width = kernel_width * (float(x_dim) ** 0.5)
        kernel_width = kernel_width + jnp.where(kernel_width < 1e-6, jnp.ones((1,)), jnp.zeros((1,))) * 1.

        return jax.lax.stop_gradient(kernel_width)
    

    @partial(jax.jit, static_argnums=(0,))
    def nystrom_ext(
        self, 
        samples, 
        x, 
        eigen_vectors, 
        eigen_values, 
        kernel_width
    ):
        """
        Compute Nystrom extension.

        inputs:
        - samples (jnp.ndarray): input data of shape [..., M, x_dim].
        - x (jnp.ndarray): input data of shape [..., N, x_dim].
        - eigen_vectors (jnp.ndarray): eigen vectors of shape [..., M, n_eigen].
        - eigen_values (jnp.ndarray): eigen values of shape [..., n_eigen].
        - kernel_width (jnp.ndarray): kernel width of shape [..., 1, 1, x_dim].

        returns:
        - eigen_ext (jnp.ndarray): Nystrom extension of shape [..., N, n_eigen], by default n_eigen=M.
        """
        M = samples.shape[-2]
        
        # Kxq: [..., N, M]
        Kxq = self.gram(x, samples, kernel_width)
        
        # Compute nyström extension
        ret = jnp.sqrt(float(M)) * jnp.matmul(Kxq, eigen_vectors) # ret: [..., N, n_eigen]
        ret *= 1. / jnp.expand_dims(eigen_values, axis=-2)

        return ret
    

    @partial(jax.jit, static_argnums=(0,))
    def pick_n_eigen(
        self, 
        eigen_values, 
        eigen_vectors, 
        n_eigen
    ):
        """
        Pick top n_eigen eigenvalues and eigenvectors.

        inputs:
        - eigen_values (jnp.ndarray): eigen values of shape [..., M].
        - eigen_vectors (jnp.ndarray): eigen vectors of shape [..., M, M].
        - n_eigen (int): number of eigen values and eigen vectors to pick.

        returns:
        - eigen_values (jnp.ndarray): eigen values of shape [..., n_eigen].
        - eigen_vectors (jnp.ndarray): eigen vectors of shape [..., M, n_eigen].
        """
        M = eigen_values.shape[-1]
        
        # eigen_values: [..., n_eigen]
        # top_k_indices: [..., n_eigen]
        eigen_values, top_k_indices = jax.lax.top_k(
            eigen_values,
            k=n_eigen
        )

        # eigen_vectors_flat: [... * M, M]
        eigen_vectors_flat = jnp.reshape(
            jnp.matrix_transpose(eigen_vectors), 
            [-1, M]
        )
        
        # indices_2d: [..., n_eigen]
        indices_2d = jnp.reshape(top_k_indices, [-1, n_eigen])        
        indices_2d += jnp.arange(indices_2d.shape[0])[..., None] * M

        # indices_flat: [... * n_eigen]
        indices_flat = jnp.reshape(indices_2d, [-1])

        # eigen_vectors_flat: [... * n_eigen, M]
        eigen_vectors_flat = eigen_vectors_flat[tuple(indices_flat[..., i] for i in range(indices_flat.shape[-1]))]
        #tf.gather(eigen_vectors_flat, indices_flat)
        eigen_vectors = jnp.matrix_transpose(
            jnp.reshape(eigen_vectors_flat, top_k_indices.shape + tuple([M]))
        )

        return eigen_values, eigen_vectors
        

    @partial(jax.jit, static_argnums=(0,))
    def compute_gradients(
        self, 
        samples
    ):
        """
        Compute gradients of log p(x) wrt x.

        inputs:
        - samples (jnp.ndarray): input data of shape [..., M, x_dim].

        returns:
        - grads (jnp.ndarray): gradients of log p(x) wrt x of shape [..., N, x_dim].
        """
        kernel_width = self.heuristic_kernel_width(samples, samples)
        x = samples
        
        M = samples.shape[-2]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * jnp.eye(M)
        
        # eigen_vectors: [..., M, M]
        # eigen_values: [..., M]
        eigen_values, eigen_vectors = jsp.linalg.eigh(Kq)

        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(
            samples, x, eigen_vectors, eigen_values, kernel_width
        )

        # grad_K1_avg = [..., M, x_dim]
        grad_K1_avg = jnp.mean(grad_K1, axis=-3)
        
        # beta: [..., n_eigen, x_dim]
        beta = -jnp.sqrt(float(M)) * jnp.matmul(
            eigen_vectors.T, grad_K1_avg
        ) / jnp.expand_dims(eigen_values, -1)
        
        # grads: [..., N, x_dim]
        grads = jnp.matmul(eigen_ext, beta)

        return grads
