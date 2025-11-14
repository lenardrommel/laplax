import haiku as hk
import jax.numpy as jnp


class Linear(hk.Module):
    """
    Linear layer.
    """
    def __init__(
        self, 
        output_dim, 
        with_bias=True
    ):
        """
        :params output_dim: Output size of the linear layer.
        :params with_bias: True to include bias in linear layer.
        """
        super().__init__('LinearStochastic')
        
        # Weight dimension
        self.output_dim = output_dim
        self.bias = with_bias
    

    def __call__(self, x):
        """
        Forward pass.
        :params x: input.
        """
        self.input_dim = x.shape[-1]

        # Posterior weight distribution 
        self.w = hk.get_parameter(
            name="w",
            shape=[self.input_dim, self.output_dim],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
            dtype=jnp.float64
        )        
        if self.bias:
            # Posterior bias distribution 
            self.b = hk.get_parameter(
                name="b",
                shape=[self.output_dim],
                init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                dtype=jnp.float64
            )

        logits = jnp.einsum("bi,io->bo", x, self.w)
        
        if self.bias:
            logits += self.b
            
        return logits