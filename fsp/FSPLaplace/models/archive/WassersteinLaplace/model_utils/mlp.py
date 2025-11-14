import haiku as hk

from models.WassersteinLaplace.model_utils.linear import Linear


class MLP(hk.Module):
    """
    Multi-layer perceptron.
    """
    def __init__(
        self, 
        architecture, 
        activation_fn
    ):
        """
        Build MLP.

        :param architecture (List[int]): number of layers and hidden 
            units for MLP. For example, `[10, 100, 100, 1]` means an 
            MLP of two layers of 100 hidden units each with input 
            dim 10 and output dim 1.
        :param activation_fn (Callable): activation function.
        """
        super().__init__("mlp")
        self.activation_fn = activation_fn
        self.architecture = architecture

        self.layers = []
        for unit in architecture:
            self.layers += [
                Linear(output_dim=unit, with_bias=True)
            ]


    def __call__(self, x):
        """Forward pass of the model.

        :params x (jax.np.array): inputs.
        :returns (jax.np.array): outputs.
        """
        out = x
        for l in range(len(self.layers) - 1):
            out = self.activation_fn(self.layers[l](out))

        # Final layer
        out = self.layers[-1](out)

        return out