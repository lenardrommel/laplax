import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from laplax.util.context_points import (
    _apply_grid_stride,
    _make_grid_from_loader,
)


class _SimpleSpatialDataset(Dataset):
    """Minimal dataset to exercise grid construction from loaders."""

    def __init__(self, spatial_shape: tuple[int, int, int], channels: int = 2):
        self._x = torch.zeros((spatial_shape[0],), dtype=torch.float32)
        self._y = torch.zeros((*spatial_shape, channels), dtype=torch.float32)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx):
        # DataLoader will stack batch dimension for us; return copies for clarity.
        return self._x.clone(), self._y.clone()


def test_make_grid_from_loader_matches_spatial_shape():
    dataset = _SimpleSpatialDataset((4, 5, 3))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    grid, dx = _make_grid_from_loader(dataloader)

    assert grid.shape == (5, 3, 2, 3)
    assert dx == pytest.approx(2 * np.pi / 5)


def test_apply_grid_stride_2d_tuple_stride():
    grid = jnp.zeros((6, 8, 2))
    context_x = jnp.zeros((5, 8, 6, 4, 1))

    strided_grid, strided_context = _apply_grid_stride(grid, context_x, (2, 3))

    assert strided_grid.shape == (2, 4, 2)
    assert strided_context.shape == (5, 4, 2, 4, 1)


def test_apply_grid_stride_3d_tuple_stride():
    grid = jnp.zeros((6, 8, 10, 3))
    context_x = jnp.zeros((3, 6, 8, 10, 2, 1))

    strided_grid, strided_context = _apply_grid_stride(grid, context_x, (2, 3, 5))

    assert strided_grid.shape == (3, 3, 2, 3)
    assert strided_context.shape == (3, 3, 3, 2, 2, 1)


def test_apply_grid_stride_invalid_grid_dimension():
    grid = jnp.zeros((6, 8))
    context_x = jnp.zeros((2, 6, 8, 1))

    with pytest.raises(ValueError, match="Unsupported grid dimension"):
        _apply_grid_stride(grid, context_x, 2)

