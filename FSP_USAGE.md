# FSP Inference Usage Guide

## Quick Start

The FSP inference code has been reorganized into a clean structure:

### New Import Structure

```python
# FSP posterior functions (Laplace-specific)
from laplax.curv import (
    create_fsp_posterior,           # Main dispatcher
    create_fsp_posterior_kronecker, # For Kronecker-structured priors
    create_fsp_posterior_none,      # For unstructured priors
    KernelStructure,                # Enum: KRONECKER, NONE
    create_ggn_pytree_mv,           # Pytree-based GGN operations
)

# Utility functions (general purpose)
from laplax.util.context_points import select_context_points
from laplax.util.lanczos import lanczos_invert_sqrt, lanczos_jacobian_initialization

# Types
from laplax.types import Kernel
```

## Example Usage

### 1. Kronecker Structured Prior

For separable spatial/function kernels:

```python
from laplax.curv import create_fsp_posterior, KernelStructure

# Define your spatial and function kernels
spatial_kernels = [kernel_x, kernel_y]  # e.g., RBF kernels for each spatial dim
function_kernels = [kernel_f]           # e.g., Linear kernel for functions

# Create FSP posterior
posterior = create_fsp_posterior(
    model_fn=my_model,
    params=trained_params,
    x_context=context_points,
    kernel_structure=KernelStructure.KRONECKER,
    spatial_kernels=spatial_kernels,
    function_kernels=function_kernels,
    prior_variance=prior_var,
    n_chunks=4,
    spatial_max_iters=[8, 8]  # Max Lanczos iterations per spatial dim
)
```

### 2. Unstructured Prior (Full Covariance)

For general kernel functions:

```python
from laplax.curv import create_fsp_posterior, KernelStructure

# Define your full kernel (can be a function or matrix)
def my_kernel(x):
    # Returns K @ x where K is the full covariance matrix
    return kernel_matrix @ x

# Or provide a kernel matrix directly
kernel_matrix = build_full_kernel(context_points)

# Create FSP posterior
posterior = create_fsp_posterior(
    model_fn=my_model,
    params=trained_params,
    x_context=context_points,
    kernel_structure=KernelStructure.NONE,
    kernel=my_kernel,  # or kernel_matrix
    prior_variance=prior_var,
    n_chunks=4,
    max_iter=100  # Max Lanczos iterations
)
```

### 3. Context Point Selection

```python
from laplax.util.context_points import select_context_points

# Select context points using various strategies
x_context, y_context, grid = select_context_points(
    dataloader=train_loader,
    context_selection="sobol",  # or "halton", "random", "latin_hypercube"
    n_context_points=50,
    n_pca_components=None,      # Auto-select based on variance
    pca_variance_threshold=0.95,
    seed=42,
    grid_stride=2               # Downsample spatial grid
)
```

### 4. Custom GGN Matrix-Vector Product

```python
from laplax.curv import create_ggn_pytree_mv

# Create GGN operator that works with pytrees
ggn_mv = create_ggn_pytree_mv(
    model_fn=my_model,
    params=trained_params,
    x_context=context_points,
    hessian_diag=True  # Assumes diagonal Hessian (identity for regression)
)

# Use it with a pytree parameter
ggn_u = ggn_mv(u_pytree)  # Returns u^T @ GGN @ u
```

## Key Features

### 1. No External Dependencies
- Pure JAX implementation
- No linox or other linear algebra libraries required
- Compatible with any kernel function (including GPJax kernels)

### 2. Modular Design
- Easy to extend with new kernel structures
- Clear separation: utilities in `util/`, Laplace code in `curv/`
- Each kernel structure has its own implementation

### 3. Kernel Interface
Kernels can be:
- **Callable**: A function that computes `K @ x`
- **Array**: A dense kernel matrix
- **Compatible with external libraries**: Just wrap them in a callable

Example with GPJax:
```python
import gpjax as gpx

# Create GPJax kernel
kernel = gpx.kernels.RBF()

# Wrap for FSP
def kernel_fn(x):
    K = kernel.gram(x_context)
    return K @ x

posterior = create_fsp_posterior(
    ...,
    kernel=kernel_fn,
    ...
)
```

## File Locations

- **Main FSP inference**: `laplax/curv/fsp.py`
- **GGN operations**: `laplax/curv/ggn.py`
- **Context point selection**: `laplax/util/context_points.py`
- **Lanczos algorithm**: `laplax/util/lanczos.py`

## Backwards Compatibility

Old imports still work but show deprecation warnings:

```python
# This still works but is deprecated
from laplax.extra import create_fsp_posterior

# You'll see:
# DeprecationWarning: laplax.extra is deprecated and will be removed.
# Use laplax.curv for FSP/GGN functions and laplax.util for utilities.
```

## Branch Information

- **Branch**: `claude/refactor-fsp-inference-laplax-011CV2BWYVyo2U1hjV8W3trf`
- **Key commits**:
  - `2601095`: Reorganize FSP code from extra/ to curv/ and util/
  - `2f88b9b`: Add modular FSP inference system with kernel structure support
