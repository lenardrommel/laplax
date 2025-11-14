import jax 
import copy
import pickle 

import haiku as hk
import jax.numpy as jnp

from functools import partial
from jax.example_libraries.optimizers import adam 

from models.GWI.model_utils.bnn import BNN
from models.GWI.model_utils.prior import Prior
from models.GWI.training_utils.plot_utils import plot_function_samples
from models.GWI.training_utils.training import fit_model, evaluate_model


class GWI:
    """
    Placeholder for the GWI model.
    """
    def __init__(
        self, 
        key, 
        config
    ):
        """
        Initialize the GWI model.

        params:
        - key (jax.random.PRNGKey): random key.
        - config (dict): configuration dictionary.
        """
        self.key = key
        self.config = copy.deepcopy(config)
    
        # Build model
        self.model = BNN(
            self.config["gwi"]["neural_net"],
            self.config["gwi"]["inference"],
            self.config["gwi"]["likelihood"]
        )


    def fit(
        self, 
        train_dataloader, 
        val_dataloader, 
        prior_dataloader
    ):
        """
        Fit the model.

        params:
        - train_dataloader (DataLoader): train dataloader.
        - val_dataloader (DataLoader): validation dataloader.

        returns:
        - val_loss (dict): validation loss.
        """
        # Initialize model
        x_init = jnp.expand_dims(train_dataloader.dataset.X[0], axis=0)
        self.mean_params = self.initialize_model(x_init)
        print(f'Number of mean parameters: {hk.data_structures.tree_size(self.mean_params)}', flush=True)

        # Inducing points
        self.key, subkey = jax.random.split(self.key)
        likelihood = self.config["gwi"]["likelihood"]["model"]
        if likelihood == "Categorical":
            self.model.n_inducing_points = int(jnp.sqrt(len(train_dataloader.dataset)))
        else:
            self.model.n_inducing_points = 2*int(jnp.sqrt(len(train_dataloader.dataset)))
        idxs = jax.random.choice(
            subkey, 
            len(train_dataloader.dataset), 
            shape=(self.model.n_inducing_points,), 
            replace=False
        )
        inducing_points_x = train_dataloader.dataset.X[idxs,...]
        inducing_points_y = train_dataloader.dataset.y[idxs,...]
        self.inducing_points = (inducing_points_x, inducing_points_y)
        print("Inducing points shape:", inducing_points_x.shape, flush=True)

        # Load Prior
        self.prior = Prior(
            self.key,
            self.inducing_points,
            self.config
        )

        # Initialize L parameters
        batch_size = min(self.config["data"]["batch_size"], len(train_dataloader.dataset))
        n_samples = len(train_dataloader.dataset)
        _, k_ZZ = self.prior(inducing_points_x) # (n_inducing_points, n_inducing_points, n_classes)
        if len(train_dataloader.dataset) > self.model.n_inducing_points: # Sample some points
            self.key, subkey = jax.random.split(self.key)
            idxs = jax.random.choice(
                subkey, 
                len(train_dataloader.dataset), 
                shape=(batch_size,), 
                replace=False
            )
            X = train_dataloader.dataset.X[idxs,...]
        else: # Use all points
            X = train_dataloader.dataset.X
        ll_var = self.prior.ll_scale.reshape(1,-1)**2 # (1, n_classes)
        k_XZ = self.prior.cross_covariance(X, inducing_points_x) # (n_batch, n_inducing_points, n_classes)
        fn = lambda _k_ZZ, _k_XZ, _ll_var: jnp.linalg.solve(_k_ZZ + n_samples * _k_XZ.T @ _k_XZ / (batch_size*_ll_var) + 1e-10 * jnp.eye(_k_ZZ.shape[0]), jnp.eye(_k_ZZ.shape[0]))
        inv_r = jax.vmap(fn, in_axes=(-1,-1,-1), out_axes=-1)(k_ZZ, k_XZ, ll_var) # (n_inducing_points, n_inducing_points, n_classes)
        self.L_params = jax.vmap(jnp.linalg.cholesky, in_axes=-1, out_axes=-1)(inv_r) # (n_inducing_points, n_inducing_points, n_classes)

        # Load pre-trained weights
        pretrained_weights_path = self.config["gwi"]["training"]["pretrained_weights_path"]
        if pretrained_weights_path:
            print("Loading pre-trained weights...", flush=True)
            with open(pretrained_weights_path, "rb") as f:
                data = pickle.load(f)
            self.prior.params, self.model.training_steps = data["prior_params"], data["step"]
            _, _, get_params = adam(step_size=0.1)
            if likelihood == "Categorical":
                self.mean_params, self.L_params = get_params(data["opt_state"])
            else:
                self.mean_params, self.L_params, ll_rho = get_params(data["opt_state"])
                self.model.ll_scale = jax.nn.softplus(ll_rho)

        # Set batch size
        train_dataloader.batch_size = self.config["data"]["batch_size"]

        # Fit the model
        self.mean_params, self.L_params, val_loss = fit_model(
            self.key, 
            self.mean_params, 
            self.L_params,
            self.model, 
            self.config, 
            train_dataloader, 
            val_dataloader,
            self.inducing_points,
            self.prior
        )

        return val_loss
    

    def initialize_model(
        self, 
        x_init
    ):
        """
        Initialize the BNN model parameters.

        params:
        - x_init (jnp.ndarray): initial input data.
        
        returns:
        - mean_params (jax.tree_util.pytree): mean parameters of the BNN.
        """
        # Handle random key
        self.key, key1 = jax.random.split(self.key, 2)

        # Initialize model
        init_fn, apply_fn = self.model.forward
        
        # Initialize mean parameters
        mean_params = init_fn(key1, x_init)
        
        # Reset training steps
        self.model.training_steps = 0

        return mean_params


    @partial(jax.jit, static_argnums=(0,3))
    def predict_f(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the linearized function distribution.
        
        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - f_lin_nn (jnp.ndarray): function samples.
        """
        return self.model.predict_f(
            self.mean_params, 
            self.L_params,
            self.prior, 
            self.inducing_points,
            x, 
            key, 
            mc_samples
        )
    
    
    @partial(jax.jit, static_argnums=(0,3))
    def predict_y(
        self, 
        x, 
        key, 
        mc_samples
    ):
        """
        Sample from the linearized predictive distribution.
        
        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): number of Monte Carlo samples.

        returns:
        - y (jnp.ndarray): function samples.
        """
        return self.model.predict_y(
            self.mean_params, 
            self.L_params,
            self.prior, 
            self.inducing_points,
            x, 
            key, 
            mc_samples
        )


    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_cov(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and covariance the linearized functional distribution.

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy variable for compatibility.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution.
        - cov (jnp.ndarray): covariance of the distribution.
        """
        return self.model.f_distribution(
            self.mean_params, 
            self.L_params,
            self.prior, 
            self.inducing_points,
            x,
            key
        )


    @partial(jax.jit, static_argnums=(0,3))
    def f_distribution_mean_var(
        self,
        x, 
        key,
        mc_samples
    ):
        """
        Return the mean and diagonalized covariance the linearized functional distribution.

        params:
        - x (jnp.ndarray): input data.
        - key (jax.random.PRNGKey): random key.
        - mc_samples (int): dummy variable for compatibility.
        
        returns:
        - mean (jnp.ndarray): mean of the distribution.
        - diag_cov (jnp.ndarray): diagonal covariance of the distribution.
        """
        return self.model.f_diag_distribution(
            self.mean_params, 
            self.L_params,
            self.prior, 
            self.inducing_points,
            x,
            key
        )
    

    def plot(
        self, 
        dataloader
    ):
        """
        Plot function samples.

        params:
        - dataloader (DataLoader): dataloader.
        """
        plot_function_samples(
            self, 
            jax.random.PRNGKey(0), 
            self.config, 
            dataloader
        )


    def evaluate(
        self, 
        dataloader
    ):
        """
        Evaluate the model.
        
        params:
        - dataloader (DataLoader): test dataloader.

        returns:
        - test_loss (dict): test loss.
        """
        test_loss = evaluate_model(
            self.key, 
            self.mean_params, 
            self.L_params,
            self.model, 
            dataloader, 
            self.prior, 
            self.inducing_points
        )

        return test_loss

