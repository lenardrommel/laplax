from laplax.extra.fsp.curv import compute_curvature_fn

from .fsp import compute_matrix_jacobian_product, fsp_laplace
from .ggn import create_fsp_ggn_mv
from .inference import fsp_inference, fsp_operator_inference
from .kernels import (
    AdditiveKernel,
    GPJaxKernelAdapter,
    Kernel,
    KroneckerKernel,
    Matern52Kernel,
    ProductKernel,
    RBFKernel,
    build_gram_matrix,
    kernel_variance,
    wrap_kernel_fn,
)
from .lanczos_isqrt import (
    lanczos_hosvd_initialization,
    lanczos_invert_sqrt,
    lanczos_jacobian_initialization,
)
from .objective import create_fsp_objective, select_context_points
from .operator import (
    compute_M_batch,
    compute_M_batch_chunked,
    hosvd_lanczos_init,
)

__all__ = [
    # Core inference
    "fsp_inference",
    "fsp_laplace",
    "fsp_operator_inference",
    # Kernels
    "AdditiveKernel",
    "GPJaxKernelAdapter",
    "Kernel",
    "KroneckerKernel",
    "Matern52Kernel",
    "ProductKernel",
    "RBFKernel",
    "build_gram_matrix",
    "kernel_variance",
    "wrap_kernel_fn",
    # GGN and curvature
    "compute_curvature_fn",
    "create_fsp_ggn_mv",
    # Lanczos
    "lanczos_hosvd_initialization",
    "lanczos_invert_sqrt",
    "lanczos_jacobian_initialization",
    # Operator learning
    "compute_M_batch",
    "compute_M_batch_chunked",
    "hosvd_lanczos_init",
    # Utilities
    "compute_matrix_jacobian_product",
    "create_fsp_objective",
    "select_context_points",
]
