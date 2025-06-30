import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
import gpjax as gpx
from flax import nnx
from functools import partial
from jax.example_libraries.optimizers import adam
from jax.flatten_util import ravel_pytree

# Set random seed for reproducibility
key = jax.random.key(42)

# Configure JAX for double precision
jax.config.update("jax_enable_x64", True)


# Define the Prior class for GP priors
class Prior:
    def __init__(self, data, kernel_name="pureperiodic"):
        self.n_priors = 1
        self._build_standard_prior(kernel_name)
        self.data = gpx.Dataset(
            data["input"].reshape(-1, 1), data["target"].reshape(-1, 1)
        )

    def __call__(self, x, jitter=1e-6):
        x = x.reshape(x.shape[0], -1)
        prior_mean = jnp.zeros((x.shape[0], 1))
        prior_cov = jnp.stack(
            [
                gpx.kernels.computations.DenseKernelComputation()
                .gram(self.kernel, x)
                .to_dense()
                + jitter * jnp.eye(x.shape[0])
                for i in range(1)
            ],
            axis=-1,
        )
        return prior_mean, prior_cov

    def _build_standard_prior(self, kernel_name):
        if kernel_name == "periodic":
            k1 = gpx.kernels.Periodic()
            k2 = gpx.kernels.Matern52()
            kt = gpx.kernels.ProductKernel(kernels=[k1, k2])
            ks = gpx.kernels.Matern12()
            self.kernel = gpx.kernels.SumKernel(kernels=[kt, ks])
        elif kernel_name == "pureperiodic":
            self.kernel = gpx.kernels.Periodic(lengthscale=0.1, period=1.0)
        elif kernel_name == "rbf":
            self.kernel = gpx.kernels.RBF()
        else:
            self.kernel = gpx.kernels.RBF()

    def _tune_hyperparameters(self):
        prior = gpx.gps.Prior(
            mean_function=gpx.mean_functions.Zero(), kernel=self.kernel
        )
        posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=self.data.n)
        opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            # we use the negative mll as we are minimising
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.data,
        )
        # latent_dist = opt_posterior.predict(self.xtest, train_data=self.data)
        # predictive_dist = opt_posterior.likelihood(latent_dist)
        self.kernel = opt_posterior.prior.kernel
        return opt_posterior  # .prior


class Model(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


def create_model(config=None):
    if config is None:
        config = {
            "in_channels": 1,
            "hidden_channels": 64,
            "out_channels": 1,
            "rngs": nnx.Rngs(jax.random.key(0)),
            "param": jnp.log(0.1),
            "data_size": 150,
            "dtype": jnp.float64,
        }
    in_channels = config.get("in_channels", 1)
    hidden_channels = config.get("hidden_channels", 64)
    out_channels = config.get("out_channels", 1)
    rngs = config.get("rngs", nnx.Rngs(0))
    param = config.get("param", None)
    data_size = config.get("data_size", None)
    dtype = config.get("dtype", jnp.float64)

    model = Model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        rngs=rngs,
    )

    graph_def, _ = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    return model, model_fn, graph_def


def model_fn(input, params):
    return nnx.call((graph_def, params))(input)[0]


# Gaussian log posterior objective function
# @partial(jax.jit, static_argnums=(3, 7, 8))
def n_gaussian_log_posterior_objective(
    model, ll_rho, x, y, x_context, key, prior, n_samples, training
):
    key1, key2 = jax.random.split(key)

    # Log-likelihood
    ll_scale = jax.nn.softplus(ll_rho)
    graph_def, params = nnx.split(model)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0]

    f_hat = model_fn(x, params)
    log_likelihood = (
        jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).mean() * n_samples
    )

    # Squared RKHS norm of the neural network
    f_hat_context = model_fn(x_context, params)
    prior_mean, prior_cov = prior(x_context, jitter=1e-10)
    sq_rkhs_norm = (f_hat_context[:, 0] - prior_mean[:, 0]).T @ jnp.linalg.solve(
        prior_cov[:, :, 0], f_hat_context[:, 0] - prior_mean[:, 0]
    )

    # Log-posterior
    log_posterior = log_likelihood - 0.5 * sq_rkhs_norm

    return (
        -log_posterior,
        {
            "log_likelihood": log_likelihood,
            "log_posterior": log_posterior,
            "sq_rkhs_norm": sq_rkhs_norm,
        },
    )


# Generate synthetic data: sin(2πx) + ε
def generate_data(key, num_samples, noise_level, intervals=[(0.0, 1.0), (3.0, 4.0)]):
    n_half = num_samples // 2
    xs = []
    for i, (a, b) in enumerate(intervals):
        k, key = jax.random.split(key)
        size = n_half if i == 0 else num_samples - n_half
        xs.append(jax.random.uniform(k, (size, 1), minval=a, maxval=b))
    x = jnp.vstack(xs)
    y = jnp.sin(2 * jnp.pi * x).flatten() + noise_level * jax.random.normal(
        key, (num_samples,)
    )
    return x, y, key


# Select context points
def select_context_points(key, n_context_points, minval=0.0, maxval=4.0):
    return jax.random.uniform(
        key=key,
        shape=(n_context_points, 1),
        minval=minval,
        maxval=maxval,
    )


# Update function for Gaussian model
# @partial(jax.jit, static_argnums=(3, 4, 5))
def update_gaussian_nn(
    model,
    ll_rho,
    opt_state,
    get_params,
    opt_update,
    prior,
    n_samples,
    x,
    y,
    x_context,
    key,
    step,
):
    # Get parameters
    params = get_params(opt_state)
    model_params, ll_rho_param = params

    # Define loss function for current batch
    def loss_fn(params):
        model_params, ll_rho_param = params
        graph_def, _ = nnx.split(model)
        model_full = nnx.merge(graph_def, model_params)

        # Compute loss
        loss, info = n_gaussian_log_posterior_objective(
            model_full,
            ll_rho_param,
            x,
            y,
            x_context,
            key,
            prior,
            n_samples,
            training=True,
        )
        return loss, info

    # Compute gradients
    (_, other_info), grads = jax.value_and_grad(loss_fn, has_aux=True)((
        model_params,
        ll_rho_param,
    ))

    # Update parameters
    opt_state = opt_update(step, grads, opt_state)

    return opt_state, other_info


# Main training function
def train_model(
    key,
    model,
    x_train,
    y_train,
    prior,
    initial_ll_scale=0.1,
    n_epochs=10,
    batch_size=64,
    lr=0.01,
    n_context_points=100,
):
    # Initialize optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)

    # Get initial parameters
    graph_def, model_params = nnx.split(model)
    ll_rho = jnp.log(jnp.exp(initial_ll_scale) - 1)  # Softplus inverse of initial scale

    # Initialize optimizer state
    opt_state = opt_init((model_params, ll_rho))

    # Number of training samples
    n_train_samples = x_train.shape[0]
    n_batches = max(1, n_train_samples // batch_size)

    # Training loop
    step = 0
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data
        key, subkey = jax.random.split(key)
        x_shuffled = x_train
        y_shuffled = y_train

        epoch_loss = 0.0

        for batch in range(n_batches):
            # Get batch
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_train_samples)
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Get context points
            key, subkey = jax.random.split(key)
            x_context = select_context_points(subkey, n_context_points)

            # Update model
            opt_state, loss_info = update_gaussian_nn(
                model,
                ll_rho,
                opt_state,
                get_params,
                opt_update,
                prior,
                n_train_samples,
                x_batch,
                y_batch,
                x_context,
                key,
                step,
            )

            epoch_loss += loss_info["log_posterior"]
            step += 1

        # Record average loss for this epoch
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Log posterior: {avg_loss:.4f}")

    # Get final parameters and update model
    final_params = get_params(opt_state)
    model_params, ll_rho = final_params

    # Reconstruct final model
    final_model = nnx.merge(graph_def, model_params)

    # Calculate final likelihood scale
    ll_scale = jax.nn.softplus(ll_rho)

    return final_model, ll_scale, losses


# Function to make predictions with the model
def predict(model, x, include_noise=False, ll_scale=None):
    graph_def, params = nnx.split(model)
    model_fn = lambda x: nnx.call((graph_def, params))(x)[0]
    f_mean = model_fn(x)

    if include_noise and ll_scale is not None:
        # Return predictive distribution (mean, variance)
        predictive_var = ll_scale**2 * jnp.ones_like(f_mean)
        return f_mean, predictive_var
    else:
        # Return just the function values
        return f_mean


# Plotting function
def plot_results(x_train, y_train, model, ll_scale):
    # Generate a grid of x values for plotting
    x_grid = jnp.linspace(0, 4, 500).reshape(-1, 1)

    # True function
    y_true = jnp.sin(2 * jnp.pi * x_grid)

    # Model predictions (mean and variance)
    f_mean, f_var = predict(model, x_grid, include_noise=True, ll_scale=ll_scale)

    # Standard deviation
    f_std = jnp.sqrt(f_var)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, s=20, color="blue", alpha=0.4, label="Training data")
    plt.plot(x_grid, y_true, "g--", linewidth=2, label="True function $\sin(2\pi x)$")
    plt.plot(x_grid, f_mean, "r-", linewidth=2, label="Predicted mean")

    # Plot uncertainty
    plt.fill_between(
        x_grid.flatten(),
        (f_mean - 2 * f_std).flatten(),
        (f_mean + 2 * f_std).flatten(),
        color="red",
        alpha=0.2,
        label="95% confidence interval",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network Model with Function-Space Prior (nnx implementation)")
    plt.legend()
    plt.grid(True)
    plt.show()


class SimpleLaplacePosterior:
    """Simplified Laplace approximation to neural network posterior."""

    def __init__(self, model_fn, graph_def, params, ll_scale, prior, n_samples=100):
        self.model_fn = model_fn
        self.graph_def = graph_def
        self.params = params
        self.ll_scale = ll_scale
        self.prior = prior
        self.n_samples = n_samples

        # Flattened parameter dimension
        self.flat_params, self.unravel_fn = ravel_pytree(params)
        self.dim = len(self.flat_params)

        # Will store Laplace approximation parameters
        self.precision = None  # Precision matrix (or approximation)
        self.samples = None  # Posterior samples

    def fit(self, x_train, y_train):
        """Fit a simple diagonal Laplace approximation."""
        # Use a very simplified approach with diagonal precision matrix
        # This is just a placeholder for a more sophisticated implementation

        # Compute Hessian diagonal via finite differences
        epsilon = 1e-5
        hessian_diag = jnp.zeros(self.dim)

        # Define loss function
        def loss_fn(params_flat):
            params = self.unravel_fn(params_flat)
            pred = self.model_fn(x_train, params)
            # Negative log likelihood
            nll = -jsp.stats.norm.logpdf(y_train, loc=pred, scale=self.ll_scale).sum()
            # Prior term (simplified)
            prior_term = 0.0  # In a full implementation, use prior properly
            return nll + prior_term

        # Compute diagonal Hessian approximation
        for i in range(self.dim):
            params_plus = self.flat_params.at[i].add(epsilon)
            params_minus = self.flat_params.at[i].add(-epsilon)
            loss_plus = loss_fn(params_plus)
            loss_minus = loss_fn(params_minus)
            loss_center = loss_fn(self.flat_params)
            hessian_diag = hessian_diag.at[i].set(
                (loss_plus + loss_minus - 2 * loss_center) / (epsilon**2)
            )

        # Ensure positive definiteness
        hessian_diag = jnp.maximum(hessian_diag, 1e-6)

        # Store precision matrix (diagonal)
        self.precision = hessian_diag

        # Generate samples
        self.generate_samples()

        return self

    def generate_samples(self, key=None):
        """Generate samples from the Laplace posterior approximation."""
        if key is None:
            key = jax.random.key(0)

        # Generate samples using diagonal precision
        std_dev = 1.0 / jnp.sqrt(self.precision + 1e-8)
        samples_flat = []

        for i in range(self.n_samples):
            # Sample from N(0, precision^-1)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (self.dim,)) * std_dev
            sample_flat = self.flat_params + noise
            samples_flat.append(sample_flat)

        self.samples_flat = jnp.stack(samples_flat)
        return self.samples_flat

    def predict(self, x_new):
        """Make predictions with uncertainty estimates."""
        # Use samples to estimate predictive distribution
        preds = []

        for sample_flat in self.samples_flat:
            params = self.unravel_fn(sample_flat)
            pred = self.model_fn(x_new, params)
            preds.append(pred)

        # Compute mean and variance of predictions
        preds = jnp.stack(preds)
        pred_mean = jnp.mean(preds, axis=0)
        pred_var = jnp.var(preds, axis=0) + self.ll_scale**2  # Add likelihood variance

        return pred_mean, pred_var


def plot_with_laplace(model_fn, params, laplace, x_train, y_train, n_periods=3):
    # Generate points on a wider domain
    x_wide = jnp.linspace(-1, n_periods, 1000).reshape(-1, 1)

    # True function
    y_true = jnp.sin(2 * jnp.pi * x_wide)

    # Predict with Laplace approximation
    pred_mean, pred_var = laplace.predict(x_wide)
    pred_std = jnp.sqrt(pred_var)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.scatter(x_train, y_train, s=20, color="blue", alpha=0.4, label="Training data")
    plt.plot(x_wide, y_true, "g--", linewidth=2, label="True function $\sin(2\pi x)$")
    plt.plot(x_wide, pred_mean, "r-", linewidth=2, label="Predicted mean")

    # Plot uncertainty
    plt.fill_between(
        x_wide.flatten(),
        (pred_mean - 2 * pred_std).flatten(),
        (pred_mean + 2 * pred_std).flatten(),
        color="red",
        alpha=0.2,
        label="95% confidence interval",
    )

    # Add vertical lines to indicate training domain
    plt.axvspan(0, 1, alpha=0.1, color="blue", label="Training domain")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Predictions with Laplace Approximation")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Generate data
    noise_level = 0.1
    num_samples = 200
    x_train, y_train, key = generate_data(key, num_samples, noise_level)
    data = {"input": x_train, "target": y_train}
    # Create model
    key, subkey = jax.random.split(key)
    model, model_fn, graph_def = create_model()

    # Create prior
    # Use "pureperiodic" for a prior that already encodes periodic structure
    prior = Prior(data, kernel_name="pureperiodic")
    prior._tune_hyperparameters()

    # Train model
    print("Training model...")
    final_model, ll_scale, losses = train_model(
        key,
        model,
        x_train,
        y_train,
        prior,
        initial_ll_scale=0.1,
        n_epochs=100,
        batch_size=20,
        n_context_points=100,
    )

    # Plot results
    plot_results(x_train, y_train, final_model, ll_scale)

    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Log posterior")
    plt.title("Training loss")
    plt.grid(True)
    plt.show()

    print(f"Final likelihood scale (noise estimate): {ll_scale:.4f}")
    print(f"True noise level: {noise_level:.4f}")

    graph_def, final_params = nnx.split(final_model)

    laplace = SimpleLaplacePosterior(
        model_fn=model_fn,
        graph_def=graph_def,
        params=final_params,
        ll_scale=ll_scale,
        prior=prior,
        n_samples=100,
    )
    laplace.fit(x_train, y_train)

    # Plot with Laplace approximation
    print("\nPlotting predictions with Laplace approximation...")
    plot_with_laplace(model_fn, final_params, laplace, x_train, y_train, n_periods=3)
