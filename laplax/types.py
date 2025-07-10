"""All types defined in one place."""

from collections.abc import Callable, Iterable, Mapping  # noqa: F401
from typing import Any

import jax
from jaxtyping import Array, Float, Int, Num, PRNGKeyArray, PyTree  # noqa: F401

from laplax.enums import CurvApprox

# Basic JAX types
KeyType = PRNGKeyArray
DType = jax.typing.DTypeLike
ShapeType = tuple[int, ...]
PyTreeDef = jax.tree_util.PyTreeDef

# Array types
InputArray = Num[Array, "..."]
PredArray = Num[Array, "..."]
TargetArray = Num[Array, "..."]
FlatParams = Num[Array, "P"]

# Parameter and model types
Params = PyTree[Num[Array, "..."]]
ModelFn = Callable[..., Params]  # [InputArray, Params]
CurvatureMV = Callable[[Params], Params]

# Data structures
Data = Mapping[str, Num[Array, "..."]]  # {"input": ..., "target": ...}
Layout = PyTree | int
PriorArguments = Mapping[str, Array | float]
PosteriorState = PyTree[Num[Array, "..."]]

# Pushforward types
DistState = dict[str, ...]  # type: ignore  # noqa: PGH003
# This contains the following types:
# - Posterior
# - int
# - JVPType: Callable[[InputArray, Params], PredArray]
# - VJPType: Callable[[InputArray, PredArray], Params]
# - Callable[[int], Params]
# - None

# Curvature types
CurvatureKeyType = CurvApprox | str | None

# Utility types
Kwargs = Any

# Declare all types as api
__all__ = [
    "KeyType",
    "DType",
    "ShapeType",
    "PyTreeDef",
    "InputArray",
    "PredArray",
    "TargetArray",
    "FlatParams",
    "Params",
    "ModelFn",
    "CurvatureMV",
    "Data",
    "Layout",
    "PriorArguments",
    "PosteriorState",
    "DistState",
    "CurvatureKeyType",
    "Kwargs",
    "Array",
    "Float",
    "Int",
    "Num",
    "PyTree",
    "Callable",
    "Iterable",
    "Mapping",
    "Any",
    "CurvApprox",
    "PRNGKeyArray",
]
