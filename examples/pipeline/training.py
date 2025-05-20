import jax
from jax import numpy as jnp
from flax import nnx
import optax
from objective import n_gaussian_log_posterior_objective


def train_model(
    model,
    n_epochs,
    train_loader,
    model_fn,
    prior_cov_kernel,
    ll_scale,
    n_context_points,
    context_selection,
    context_points_maxval,
    context_points_minval,
    lr=1e-3,
    key=jax.random.PRNGKey(0),
):
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    train_step_fn = create_train_step(
        model_fn,
        prior_cov_kernel,
        ll_scale,
        n_context_points,
        context_selection,
        context_points_maxval,
        context_points_minval,
    )

    for epoch in range(n_epochs):
        train_ll, train_rkhs, train_lpost = 0.0, 0.0, 0.0
        batch_count = 0

        for x_tr, y_tr in train_loader:
            data = {"input": x_tr, "target": y_tr}
            loss, key, ll, rkhs_norm, log_posterior = train_step_fn(
                model, optimizer, data, key
            )

            # Accumulate metrics
            train_ll += ll
            train_rkhs += rkhs_norm
            train_lpost += log_posterior
            batch_count += 1

        # Average metrics over batches
        train_ll /= batch_count
        train_rkhs /= batch_count
        train_lpost /= batch_count

        if epoch % 100 == 0:
            print(
                f"[epoch {epoch}]: loss: {loss:.4f}, log_likelihood: {train_ll:.4f}, "
                f"rkhs_norm: {train_rkhs:.4f}, log_posterior: {train_lpost:.4f}"
            )

    print(f"Final loss: {loss:.4f}")
    return model
