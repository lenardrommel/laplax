import jax
import jax.numpy as jnp
import jax.scipy as jsp
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import gpjax as gpx
from functools import partial

# Configure JAX for double precision
jax.config.update("jax_enable_x64", True)


class GPrior:
    """Gaussian process prior with configurable kernel."""

    def __init__(self, kernel_name="rbf", jitter=1e-6):
        self.jitter = jitter
        if kernel_name == "pureperiodic":
            self.kernel = gpx.kernels.Periodic(lengthscale=0.01, period=2 * jnp.pi)
        elif kernel_name == "periodic":
            k1 = gpx.kernels.Periodic()
            k2 = gpx.kernels.Matern52()
            kt = gpx.kernels.ProductKernel(kernels=[k1, k2])
            ks = gpx.kernels.Matern12()
            self.kernel = gpx.kernels.SumKernel(kernels=[kt, ks])
        else:
            self.kernel = gpx.kernels.RBF()
        self.comp = gpx.kernels.computations.DenseKernelComputation()

    def __call__(self, x):
        # x: [N, D]
        mean = jnp.zeros((x.shape[0],))
        K = self.comp.gram(self.kernel, x).to_dense()
        K = K + self.jitter * jnp.eye(x.shape[0])
        return mean, K


class MLP(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


@partial(jax.jit, static_argnames=["n_context"])
def compute_sq_rkhs(f_vals, x_context, prior, n_context):
    mean, K = prior(x_context)
    diff = f_vals - mean
    return diff.T @ jnp.linalg.solve(K, diff)


@partial(jax.jit, static_argnums=(0,))
def loss_and_stats(params, model, rho, x_batch, y_batch, x_context, prior, n_samples):
    # Predict
    preds = model.apply(params, x_batch)
    scale = jax.nn.softplus(rho)
    # Log-likelihood
    ll = jsp.stats.norm.logpdf(y_batch, loc=preds, scale=scale).mean() * n_samples
    # RKHS penalty
    f_ctx = model.apply(params, x_context)
    sq_norm = compute_sq_rkhs(f_ctx, x_context, prior, x_context.shape[0])
    # Negative log-posterior
    obj = -(ll - 0.5 * sq_norm)
    return obj, {"log_likelihood": ll, "sq_rkhs_norm": sq_norm}


@partial(jax.jit, static_argnums=(0,))
def update(
    params, opt_state, rho, x_batch, y_batch, x_context, prior, n_samples, optimizer
):
    (obj, stats), grads = jax.value_and_grad(loss_and_stats, has_aux=True)(
        params, optimizer.target, rho, x_batch, y_batch, x_context, prior, n_samples
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, stats


# Data generation for two intervals
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


# Training routine
def train_gp_nn(
    key,
    num_samples=200,
    noise_level=0.1,
    n_epochs=1000,
    batch_size=32,
    lr=1e-2,
    n_context=100,
    kernel_name="pureperiodic",
):
    # Prepare data
    x_train, y_train, key = generate_data(key, num_samples, noise_level)
    # Model + params
    model = MLP()
    params = model.init(key, x_train[:batch_size])
    # Latent log-scale parameter
    rho = jnp.log(jnp.exp(0.1) - 1.0)
    # Optimizer
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale(-lr))
    opt_state = optimizer.init((params, rho))
    prior = GPrior(kernel_name)
    losses = []

    for epoch in range(n_epochs):
        # Shuffle
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, num_samples)
        x_sh, y_sh = x_train[perm], y_train[perm]
        for i in range(0, num_samples, batch_size):
            xb = x_sh[i : i + batch_size]
            yb = y_sh[i : i + batch_size]
            # Sample context grid once per epoch
            rng, key = jax.random.split(key)
            x_ctx = jnp.linspace(0, 4, n_context).reshape(-1, 1)
            # Update
            (params, rho), opt_state, stats = update(
                (params, rho),
                opt_state,
                rho,
                xb,
                yb,
                x_ctx,
                prior,
                num_samples,
                optimizer,
            )
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}, ll={stats["log_likelihood"]:.3f}, rkhs={stats["sq_rkhs_norm"]:.3f}"
            )
        losses.append(stats["log_likelihood"])

    return params, rho, model, prior, losses, x_train, y_train


# Prediction + plotting


def predict(params, model, rho, x, include_noise=True):
    mu = model.apply(params, x)
    if include_noise:
        sigma2 = jax.nn.softplus(rho) ** 2
        return mu, sigma2 * jnp.ones_like(mu)
    return mu, None


def main():
    key = jax.random.PRNGKey(42)
    params, rho, model, prior, losses, x_train, y_train = train_gp_nn(key)

    # Test on grid including gap
    x_test = jnp.linspace(0, 4, 500).reshape(-1, 1)
    mu, var = predict(params, model, rho, x_test)
    std = jnp.sqrt(var)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=20, alpha=0.3, label="Train")
    plt.plot(x_test, jnp.sin(2 * jnp.pi * x_test), "g--", label="True")
    plt.plot(x_test, mu, "r-", label="Pred")
    plt.fill_between(
        x_test.flatten(), (mu - 2 * std).flatten(), (mu + 2 * std).flatten(), alpha=0.2
    )
    plt.title("GP-informed NN with two-interval data")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
