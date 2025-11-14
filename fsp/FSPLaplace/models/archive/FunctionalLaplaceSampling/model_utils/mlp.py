import haiku as hk


class MLP(hk.Module):
    """
    Multilayer perceptron (MLP).
    """
    def __init__(
        self, 
        architecture,
        activation_fn
    ):
        """
        Initialize a MLP.

        params:
        - activation_fn (callable): activation function.
        - architecture (List[int]): number of layers and hidden units for MLP.
            For example, `[100, 100, 1]` means an MLP of two layers of 100 hidden
            units each with output dim 1.
        """
        super().__init__("MLP")
        # Model parameters
        self.activation_fn = activation_fn
        
        # Build MLP
        self.layers = [
            hk.Linear(
                output_size=architecture[i], 
                with_bias=True
            )
            for i in range(len(architecture))
        ]


    def __call__(
        self, 
        x
    ):
        """
        Forward pass on the MLP.

        params:
        - x (jnp.ndarray): input.
        
        returns: 
        - x (jnp.ndarray): output of the MLP.
        """
        # Flatten input
        x = x.reshape((x.shape[0], -1))
        
        # Forward pass on MLP layers (except output layer)
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            x = self.activation_fn(x)

        # Output layer
        x = self.layers[-1](x)

        return x


