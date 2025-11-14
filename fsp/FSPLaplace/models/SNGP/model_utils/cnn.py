import haiku as hk

from models.SNGP.model_utils.random_fourier_features import RandomFourierFeatures

class CNN1(hk.Module):
    """
    Convolutional neural network (CNN).
    """
    def __init__(
        self, 
        n_rff,
        output_dim,
        kernel,
        activation_fn,
        key
    ):
        """
        Initialize a CNN.

        params:
        - output_dim (int): output dimension.
        - activation_fn (callable): activation function.
        """
        super().__init__("CNN1")
        # Model parameters
        self.activation_fn = activation_fn
        
        # Define layers
        self.conv1 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), with_bias=True)
        self.conv2 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), with_bias=True)
        self.conv3 = hk.Conv2D(output_channels=64, kernel_shape=(3,3), with_bias=True)
        self.max_pool = hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding="VALID")
        self.dense1 = hk.Linear(output_size=128, with_bias=True)
        #self.dense2 = hk.Linear(output_size=output_dim, with_bias=True)

        # Random fourier features layer
        self.rff = RandomFourierFeatures(
            in_features=128,
            features_out=n_rff,
            n_classes=output_dim, 
            key=key,
            kernel=kernel
        )



    def __call__(
        self, 
        x
    ):
        """
        Forward pass on the CNN.

        params:
        - x (jnp.ndarray): input.
        
        returns: 
        - x (jnp.ndarray): output of the MLP.
        """
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        x = self.activation_fn(x)
        #x = self.dense2(x)
        x, phi = self.rff(x)

        return x, phi


class CNN2(hk.Module):
    """
    Convolutional neural network (CNN).
    """
    def __init__(
        self, 
        n_rff,
        output_dim,
        kernel,
        activation_fn,
        key
    ):
        """
        Initialize a CNN.

        params:
        - output_dim (int): output dimension.
        - activation_fn (callable): activation function.
        """
        super().__init__("CNN2")
        # Model parameters
        self.activation_fn = activation_fn
        
        # Define layers
        self.conv1 = hk.Conv2D(output_channels=16, kernel_shape=(3,3), with_bias=True)
        self.conv2 = hk.Conv2D(output_channels=32, kernel_shape=(3,3), with_bias=True)
        self.conv3 = hk.Conv2D(output_channels=64, kernel_shape=(3,3), with_bias=True)
        self.conv4 = hk.Conv2D(output_channels=64, kernel_shape=(3,3), with_bias=True)
        self.max_pool = hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding="VALID")
        self.dense1 = hk.Linear(output_size=128, with_bias=True)
        #self.dense2 = hk.Linear(output_size=output_dim, with_bias=True)

        # Random fourier features layer
        self.rff = RandomFourierFeatures(
            in_features=128,
            features_out=n_rff,
            n_classes=output_dim, 
            key=key,
            kernel=kernel
        )


    def __call__(
        self, 
        x
    ):
        """
        Forward pass on the CNN.

        params:
        - x (jnp.ndarray): input.
        
        returns: 
        - x (jnp.ndarray): output of the MLP.
        """
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = self.conv4(x)
        x = self.activation_fn(x)
        x = self.max_pool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.dense1(x)
        x = self.activation_fn(x)
        # x = self.dense2(x)
        x, phi = self.rff(x)

        return x, phi