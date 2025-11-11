"""DEPRECATED: This module will be removed. Use laplax.curv and laplax.util instead.

The FSP inference functionality has been reorganized:
- FSP posterior functions: laplax.curv.fsp
- GGN functions: laplax.curv.ggn
- Context points: laplax.util.context_points
- Lanczos: laplax.util.lanczos
"""

import warnings

# Backwards compatibility imports
from laplax.curv.fsp import (
    KernelStructure,
    create_fsp_posterior,
    create_fsp_posterior_kronecker,
    create_fsp_posterior_none,
)
from laplax.curv.ggn import create_ggn_pytree_mv
from laplax.util.context_points import select_context_points
from laplax.util.lanczos import lanczos_invert_sqrt, lanczos_jacobian_initialization

# Keep old extra.fsp imports for backwards compatibility
try:
    from laplax.extra.fsp import (
        compute_curvature_fn,
        compute_matrix_jacobian_product,
        create_fsp_ggn_mv,
        create_fsp_objective,
        fsp_laplace,
    )
except ImportError:
    # These may not be available after cleanup
    pass

warnings.warn(
    "laplax.extra is deprecated and will be removed. "
    "Use laplax.curv for FSP/GGN functions and laplax.util for utilities.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "KernelStructure",
    "compute_curvature_fn",
    "compute_matrix_jacobian_product",
    "create_fsp_ggn_mv",
    "create_fsp_objective",
    "create_fsp_posterior",
    "create_fsp_posterior_kronecker",
    "create_fsp_posterior_none",
    "create_ggn_pytree_mv",
    "fsp_laplace",
    "lanczos_invert_sqrt",
    "lanczos_jacobian_initialization",
    "select_context_points",
]
