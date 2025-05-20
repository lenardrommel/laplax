import jax

from prior import Prior
from model import Model, create_model_
from training import fit_model


def fit(train_loader, val_dataloader, prior):
    x_min = train_loader.dataset.x_min
    x_max = train_loader.dataset.x_max
    x_min = x_min - 0.5 * (x_max - x_min)
    x_max = x_max + 0.5 * (x_max - x_min)

    key = jax.random.PRNGKey(0)
    prior = Prior(key, prior)
    model = create_model_()

    fit_model(key, model, train_loader, val_dataloader, prior)


def main():
    # Create the data loaders
    train_loader = None  # Replace with actual DataLoader
    val_dataloader = None  # Replace with actual DataLoader
    prior_dataloder = None  # Replace with actual DataLoader

    # Fit the model
    fit(train_loader, val_dataloader, prior_dataloder)


if __name__ == "__main__":
    main()
