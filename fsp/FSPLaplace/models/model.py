import jax

from models.GP.gp_pytorch import GP
# from models.GP.gp import GP
from models.GWI.gwi import GWI
from models.SNGP.sngp import SNGP
from models.FVI.fvi import FVI
from models.Laplace.laplace_bnn import LaplaceBNN
from models.SamplingLaplace.laplace_bnn import SamplingLaplaceBNN
from models.FSPLaplace.fsplaplace import FSPLaplace
from models.Ensemble.ensemble import Ensemble


class Model:
    """
    Abstract class for the model.
    """

    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        self.model_name = config["model"]["name"]
        if self.model_name == "FSPLaplace":
            self.model = FSPLaplace(key, config)
        elif self.model_name == "Laplace":
            self.model = LaplaceBNN(key, config)
        elif self.model_name == "SNGP":
            self.model = SNGP(key, config)
        elif self.model_name == "GWI":
            self.model = GWI(key, config)
        elif self.model_name == "SamplingLaplace":
            self.model = SamplingLaplaceBNN(key, config)
        elif self.model_name == "GP":
            self.model = GP(key, config)
        elif self.model_name == "Ensemble":
            self.model = Ensemble(key, config)
        elif self.model_name == "FVI":
            self.model = FVI(key, config)
        else:
            raise NotImplementedError


    def fit(
        self, 
        train_dataloader, 
        val_dataloader, 
        prior_dataloader
    ):
        """
        Fit the model.

        params:
        - train_dataloader (DataLoader): training dataloader.
        - val_dataloader (DataLoader): validation dataloader.

        returns:
        - val_loss (dict): validation loss.
        """
        return self.model.fit(train_dataloader, val_dataloader,  prior_dataloader)
        

    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.

        params:
        - dataloader (DataLoader): data.

        returns:
        - test_loss (dict): test loss.
        """
        return self.model.evaluate(dataloader)


    def predict_f(
        self, 
        x, 
        key, 
        mc_samples 
    ):
        """
        Sample from function distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - f_samples (jnp.array): function samples.
        """
        fn = lambda x, key: self.model.predict_f(x, key, mc_samples)
        if self.model_name in ["GP"]:
            return fn(x, key)
        else:
            return jax.jit(fn)(x, key)
            

    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from predictive distribution.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y_samples (jnp.array): function samples.
        """
        fn = lambda x, key: self.model.predict_y(x, key, mc_samples)
        if self.model_name in ["GP"]:
            return fn(x, key)
        else:
            return jax.jit(fn)(x, key)

    
    def f_distribution_mean_cov(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and covariance the functional distribution. 
        For MFVI, these values are estimated from samples as there is 
        no closed form density over functions.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.
        
        returns:
        - mean (jnp.array): mean of the distribution.
        - cov (jnp.array): covariance of the distribution.
        """
        fn = lambda x, key: self.model.f_distribution_mean_cov(x, key, mc_samples)
        if self.model_name in ["GP"]:
            return fn(x, key)
        else:
            return jax.jit(fn)(x, key)
    

    def f_distribution_mean_var(
        self,
        x, 
        key, 
        mc_samples
    ):
        """
        Return the mean and variance the functional distribution. 
        For MFVI, these values are estimated from samples as there is 
        no closed form density over functions.

        params:
        - x (jnp.array): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - mean (jnp.array): mean of the distribution.
        - var (jnp.array): variance of the distribution.
        """
        fn = lambda x, key: self.model.f_distribution_mean_var(x, key, mc_samples)
        if self.model_name in ["GP"]:
            return fn(x, key)
        else:
            return jax.jit(fn)(x, key)
        
    
    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        self.model.plot(
            dataloader
        )
        