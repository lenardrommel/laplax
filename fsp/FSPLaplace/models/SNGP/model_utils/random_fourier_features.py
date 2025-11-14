import jax

import haiku as hk
import jax.numpy as jnp 


class RandomFourierFeatures(hk.Module):
    """
   
    """
    def __init__(
        self,
        in_features,
        features_out,
        n_classes, 
        key,
        kernel,
        inner_dim=128
    ):
        super().__init__("RandomFourierFeatures")
        
        # Configuration
        self.features_out = features_out
        scale = jnp.sqrt(inner_dim / 2.0) if kernel == 'Gaussian' else 1.0
        self.scale = hk.get_parameter("scale", shape=[1,], init=hk.initializers.Constant(scale))
        self.beta = hk.get_parameter("beta", shape=[n_classes, features_out], init=hk.initializers.RandomNormal(1.0))

        # Layer norm 
        self.layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        # Model parameters
        key1, key2, key3 = jax.random.split(key, 3)
        self.w_proj = jax.random.normal(key1, shape=(inner_dim, in_features)) 
        if kernel == 'Gaussian':
            self.w = jax.random.normal(key2, shape=(features_out, inner_dim))
        elif kernel == 'Laplacian':
            self.w = jax.random.cauchy(key2, shape=(features_out, inner_dim))
        elif kernel == 'Cauchy':
            self.w = jax.random.laplace(key2, shape=(features_out, inner_dim))
        else:
            raise ValueError("Select kernel from one of ['Gaussian', 'Laplacian', 'Cauchy']")

        self.b = 2 * jnp.pi * jax.random.uniform(key3, shape=(features_out,))


    def __call__(
        self, 
        x
    ):
        """
        Forward pass on the RandomFourierFeatures.
        
        params:
        - x (torch.Tensor): input.
        
        returns:
        - x (torch.Tensor): output of the RandomFourierFeatures.
        """
        # Normalize input
        x = self.layer_norm(x)

        # Compute weight matrix
        weight = self.w / jax.nn.softplus(self.scale)
        
        # Compute features
        logits = jnp.einsum("oi,ij,bj->bo", weight, self.w_proj, x) + self.b
        phi = jnp.sqrt(2.0 / self.features_out) * jnp.cos(logits) # (batch, features_out)

        return phi @ self.beta.T, phi # (batch, n_classes), (batch, features_out)