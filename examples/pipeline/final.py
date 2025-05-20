import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import matplotlib.pyplot as plt
import gpjax as gpx
from flax import nnx
import optax
from lanzcos import lanczos_compute_efficient
from laplax.util import tree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def plot_sinusoid_task(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    X_pred=None,
    y_pred=None,
    y_std=None,
    title=None,
):
    """Plot the training and test data for the sinusoid task with optional predictions.

    Args:
        X_train: Training input data (shape: n_samples, 1)
        y_train: Training target data (shape: n_samples, 1)
        X_test: Test input data (shape: n_samples, 1), optional
        y_test: Test target data (shape: n_samples, 1), optional
        X_pred: Prediction input data (shape: n_samples, 1), optional
        y_pred: Prediction mean (shape: n_samples, 1), optional
        y_std: Prediction standard deviation (shape: n_samples, 1), optional
        title: Plot title, optional

    Returns:
        The matplotlib figure object
    """
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays if they are JAX arrays
    if hasattr(X_train, "device_buffer"):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if X_test is not None:
            X_test = np.array(X_test)
            y_test = np.array(y_test)
        if X_pred is not None and y_pred is not None:
            X_pred = np.array(X_pred)
            y_pred = np.array(y_pred)
            if y_std is not None:
                y_std = np.array(y_std)

    # Plot training data
    plt.scatter(X_train, y_train, color="blue", alpha=0.6, label="Training data")

    # Plot test data if provided
    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test, color="green", alpha=0.6, label="Test data")

    # Plot prediction with uncertainty if provided
    if X_pred is not None and y_pred is not None:
        # Sort X for proper line plotting
        sort_idx = np.argsort(X_pred.flatten())
        X_pred_sorted = X_pred[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        plt.plot(X_pred_sorted, y_pred_sorted, color="red", label="Prediction")

        # Plot uncertainty if provided
        if y_std is not None:
            y_std_sorted = y_std[sort_idx]
            plt.fill_between(
                X_pred_sorted.flatten(),
                (y_pred_sorted - 2 * y_std_sorted).flatten(),
                (y_pred_sorted + 2 * y_std_sorted).flatten(),
                color="red",
                alpha=0.2,
                label="95% confidence interval",
            )

    # Plot true function
    x_true = np.linspace(0, 8, 1000).reshape(-1, 1)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, color="black", linestyle="--", label="True function")

    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    if title:
        plt.title(title)
    else:
        plt.title("Sinusoid Task Data")

    # Add legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, context_points, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.context_points = context_points
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        return (
            self.X[batch_indices],
            self.y[batch_indices],
            self.context_points[batch_indices],
        )

    def select_context_points(
        self,
        n_context_points,
        context_points_maxval,
        context_points_minval,
    ):
        return jnp.linspace(
            context_points_minval[0], context_points_maxval[0], n_context_points
        ).reshape(-1, 1)


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


def prior(kernel, x, jitter=1e-4):
    K = gpx.kernels.computations.DenseKernelComputation().gram(kernel, x).to_dense()
    return K + jitter * jnp.eye(x.shape[0])


def create_prior_fn(kernel):
    return lambda x: prior(kernel, x)


def loss_fn(model_fn, prior_fn, params, ll_rho, data):
    ll_scale = jax.nn.softplus(ll_rho)
    f_hat = model_fn(data["input"], params)
    log_likelihood = (
        jsp.stats.norm.logpdf(data["target"], loc=f_hat, scale=ll_scale).mean()
        * data["input"].shape[0]
    )
    log_likelihood = -jnp.sum((f_hat - data["target"]) ** 2)

    # f_hat = model_fn(data["context"], params)
    # prior_cov = prior_fn(data["context"])
    # sq_rkhs_norm = f_hat.T @ jnp.linalg.solve(prior_cov, f_hat)
    # log_posterior = log_likelihood - 0.0 * sq_rkhs_norm
    # return -log_posterior, {
    #     "log_likelihood": log_likelihood,
    #     "log_posterior": log_posterior,
    #     "sq_rkhs_norm": sq_rkhs_norm,
    # }
    return ((f_hat - data["target"]) ** 2).mean(), {}


def select_context_points(
    n_context_points,
    context_points_maxval,
    context_points_minval,
):
    return jnp.linspace(
        context_points_minval[0], context_points_maxval[0], n_context_points
    ).reshape(-1, 1)


# @nnx.jit
def train_step(model, prior_fn, ll_rho, optimizer, x, y, c):
    def loss_fn(model):
        f_hat = model(x)  # Call methods directly
        ll_scale = jax.nn.softplus(ll_rho)
        log_likelihood = (
            100
            * (
                jsp.stats.norm.logpdf(y, loc=f_hat, scale=ll_scale).mean() * x.shape[0]
            ).mean()
        )
        f_hat = model(c)
        prior_cov = prior_fn(c)
        sq_rkhs_norm = f_hat.T @ jnp.linalg.solve(prior_cov, f_hat)
        return -log_likelihood + 0.5 * sq_rkhs_norm.reshape(-1).squeeze(0)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates

    return loss


def train_model(model, ll_rho, prior_fn, n_epochs, lr=1e-3, train_loader=None):
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adamw(lr))  # Reference sharing

    # Train epoch
    for epoch in range(n_epochs):
        for x_tr, y_tr, e_tr in train_loader:
            loss = train_step(model, prior_fn, ll_rho, optimizer, x_tr, y_tr, e_tr)

        if epoch % 100 == 0:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")

    print(f"Final loss: {loss:.4f}")
    return model


def visualize_results(model_fn, params, data, title="Model Predictions"):
    x = data["input"]
    y = data["target"]

    # Generate predictions
    predictions = model_fn(x, params)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Data", alpha=0.6)
    plt.plot(x, predictions, "r-", label="Predictions", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def lanczos_jacobian_initialization(
    model_fn,
    params,
    data,
    has_batch_dim: bool = True,
    *,
    lanczos_initialization_batch_size: int = 32,
):
    # Define model Jacobian vector product
    if has_batch_dim:
        model_jvp = jax.vmap(
            lambda x: jax.jvp(
                lambda w: model_fn(x, params=w),
                (params,),
                (tree.ones_like(params),),
            )[1],
            in_axes=0,
            out_axes=0,
        )

        initial_vec = jax.lax.map(
            model_jvp,
            data["input"],
            batch_size=lanczos_initialization_batch_size,
        ).reshape(-1)
    else:
        initial_vec = jax.jvp(
            lambda w: model_fn(data["input"], params=w),
            (params,),
            (tree.ones_like(params),),
        )[1]

    # Normalize
    initial_vec = initial_vec / jnp.linalg.norm(initial_vec, 2)

    return initial_vec.squeeze(-1)


def fit(model, kernel, x, y, context_points, ll_rho):
    v = 0


def main():
    rngs = nnx.Rngs(0)
    key = jax.random.key(0)
    model = Model(in_channels=1, hidden_channels=64, out_channels=1, rngs=rngs)
    x = jnp.linspace(-5, 5, 100).reshape(-1, 1)
    y = jnp.sin(x) + 0.1 * jax.random.normal(key, x.shape)
    context_points = select_context_points(
        n_context_points=100,
        context_points_maxval=(5,),
        context_points_minval=(-5,),
    )

    train_loader = DataLoader(
        x,
        y,
        context_points,
        batch_size=32,
        shuffle=True,
    )
    graph_def, params = nnx.split(model)
    kernel = gpx.kernels.Periodic(lengthscale=0.5, period=0.8)
    prior_fn = create_prior_fn(kernel)
    ll_rho = jnp.log(0.1)

    def model_fn(input, params):
        return nnx.call((graph_def, params))(input)[0].reshape(-1)

    model = train_model(
        model,
        ll_rho,
        prior_fn,
        n_epochs=1000,
        lr=1e-3,
        train_loader=train_loader,
    )

    X_pred = jnp.linspace(0.0, 8.0, 200).reshape(200, 1)

    y_pred = jax.vmap(model)(X_pred)

    _, trained_params = nnx.split(model)
    data = {
        "input": x,
        "target": y,
        "context": context_points,
        "test_input": X_pred,
    }

    visualize_results(model_fn, trained_params, data, title="Trained Model Predictions")


if __name__ == "__main__":
    main()
