import jax
import math

import numpy as np

class DataLoader:
    """
    Object that samples from a dataset.
    """
    def __init__(
        self, 
        key, 
        dataset, 
        batch_size,
        shuffle=False, 
        replacement=False
    ):
        """
        Build DataLoader object.

        params:
        - key (jax.random.PRNGKey): random key.
        - dataset (Dataset): dataset to sample from.
        - batch_size (int): batch size.
        - shuffle (bool): shuffle the dataset at each epoch.
        - replacement (bool): sample dataset with replacement.
        """
        self.key = key
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.feature_dim = np.prod(dataset.X.shape[1:])
    
    
        # Shuffle dataset
        self.idxs = jax.random.choice(
            self.key, 
            len(self.dataset), 
            shape=(len(self.dataset),), 
            replace=False
        )

    
    def __len__(
        self
    ):
        """
        Length of the dataloader i.e. number of batches.

        returns:
        - len (int): number of batches.
        """
        return math.ceil(len(self.dataset) / self.batch_size)
    
    
    def __iter__(
        self
    ):
        """
        Build iterator.

        returns:
        - self (DataLoader): iterator.
        """
        # Reset iterator
        self.it = 0
        
        # Shuffle dataset
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key)
            self.idxs = jax.random.choice(
                subkey, 
                len(self.dataset), 
                shape=(len(self.dataset),), 
                replace=False
            )
        
        return self
    

    def __next__(
        self
    ):
        """
        Sample a batch from the dataset.

        returns:
        - out (jnp.array): batch of data.
        """
        # If end of array is reached
        if self.it * self.batch_size >= len(self.dataset):    
            raise StopIteration
        
        if self.replacement:
            # Sample the dataset uniformly with replacement
            self.key, subkey = jax.random.split(self.key)
            sample_size = min(self.batch_size, len(self.dataset))
            idxs = jax.random.choice(
                subkey, 
                len(self.dataset), 
                shape=(sample_size,), 
                replace=True
            )
            out = self.dataset[idxs]
        else:
            # Select start and end indices
            start = self.it * self.batch_size
            if (self.it+1) * self.batch_size > len(self.dataset):
                out = self.dataset[self.idxs[start:]]
            else:
                end = (self.it+1) * self.batch_size
                out = self.dataset[self.idxs[start:end]]

        # Update iterator
        self.it += 1

        return out
    

    def set_replacement_mode(
        self, 
        replacement
    ):
        """
        Set replacement attribute.

        params:
        - replacement (bool): sample dataset with replacement.
        """
        self.replacement = replacement
