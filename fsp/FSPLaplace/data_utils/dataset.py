import jax.numpy as jnp


class Dataset:
    """
    Dataset object.
    """
    def __init__(
        self, 
        X, 
        y
    ):
        """
        Build Dataset.
        """
        self.X, self.y = jnp.array(X), jnp.array(y)
        self.x_min, self.x_max = jnp.min(self.X, axis=0), jnp.max(self.X, axis=0)
        
        # Print dataset statistics
        print("Data: ", self.X.shape, self.y.shape, flush=True)


    def __len__(self):
        """
        Number of data points.

        returns:
        - len (int): number of data points.
        """
        return len(self.X)
    

    def __getitem__(
        self, 
        idx
    ):
        """
        Get data point.

        params:
        - idx (int): index of the data point.

        returns:
        - X (jnp.array): feature vector.
        - y (jnp.array): label.
        """
        return self.X[idx,:], self.y[idx,:]


    def get_data(self):
        """
        Return entire dataset.

        returns:
        - X (jnp.array): feature matrix.
        - y (jnp.array): label matrix.
        """
        return self.X, self.y
   