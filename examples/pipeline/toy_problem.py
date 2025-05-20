import jax
import jax.numpy as jnp
from helper import DataLoader
from sklearn.model_selection import train_test_split


from prior import Prior
from fsplaplace import fit_model


def _truncated_sine(key, n_samples, feature_dim):
    key1, key2, key3 = jax.random.split(key, num=3)

    # Features
    X1 = jax.random.uniform(
        key1, minval=-1, maxval=-0.5, shape=(n_samples // 2, feature_dim)
    )
    X2 = jax.random.uniform(
        key2, minval=0.5, maxval=1, shape=(n_samples // 2, feature_dim)
    )
    X = jnp.concatenate([X1, X2], axis=0)

    # Targets
    eps = 0.1 * jax.random.normal(key3, shape=(n_samples,))
    y = jnp.sin(2 * jnp.pi * X.mean(axis=-1)) + eps

    # Format
    X = X.reshape(-1, feature_dim)
    y = y.reshape(-1, 1)

    return X, y


def run_experiment():
    X, y = _truncated_sine(jax.random.PRNGKey(0), 300, 1)
    dataset_size = X.shape[0] // 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=dataset_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=dataset_size, random_state=42
    )
    train_loader = DataLoader(X_train, y_train, batch_size=20, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=20, shuffle=False)
    test_loader = DataLoader(X_test, y_test, batch_size=20, shuffle=False)
    fit(train_loader, val_loader)
