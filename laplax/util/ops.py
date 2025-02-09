"""Contains operations for flexible/adaptive compute."""

import operator
import os

import jax
import jax.numpy as jnp

from laplax.types import Callable, DType, Iterable

# -------------------------------------------------------------------------
# Default values
# -------------------------------------------------------------------------

DEFAULT_PARALLELISM = None
DEFAULT_DTYPE = "float32"
DEFAULT_PRECOMPUTE_LIST = True

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    """Convert a string representation of a boolean to a boolean value.

    Args:
        value: A string representation of a boolean ("True" or "False").

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the string does not represent a valid boolean value.
    """
    valid_values = {"True": True, "False": False}
    if value not in valid_values:
        msg = "invalid string representation of a boolean value"
        raise ValueError(msg)
    return valid_values[value]


def get_env_value(key: str, default: int | str | None = None) -> str | None:
    """Fetch the value of an environment variable or return a default value.

    Args:
        key: The name of the environment variable.
        default: The default value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default.
    """
    if default is not None:
        default = str(default)
    return os.getenv(key, default)


def get_env_int(key: str, default: int | None = None) -> int | None:
    """Fetch the value of an environment variable as an integer.

    Args:
        key: The name of the environment variable.
        default: The default integer value to return if the variable is not set.

    Returns:
        int: The value of the environment variable as an integer.
    """
    val = get_env_value(key, default)
    if val is not None:
        val = int(val)
    return val


def get_env_bool(key: str, default: str | None = None) -> bool | None:
    """Fetch the value of an environment variable as a boolean.

    Args:
        key: The name of the environment variable.
        default: The default string value ("True" or "False") if the variable is not
            set.

    Returns:
        bool: The value of the environment variable as a boolean.

    Raises:
        ValueError: If the default string is not a valid boolean representation.
    """
    val = get_env_value(key, default)
    if val is not None:
        val = str_to_bool(val)
    return val


# -------------------------------------------------------------------------
# Adaptive operations
# -------------------------------------------------------------------------


def laplax_dtype() -> DType:
    """Get the data type (dtype) used by the library.

    This function retrieves the dtype specified by the "LAPLAX_DTYPE" environment
    variable or returns the default dtype.

    Returns:
        DType: The JAX-compatible dtype to use.
    """
    dtype = get_env_value("LAPLAX_DTYPE", DEFAULT_DTYPE)
    return jnp.dtype(dtype)


def precompute_list(
    func: Callable, items: Iterable, precompute: bool | None = None, **kwargs
) -> Callable:
    """Precompute results for a list of items or return the original function.

    If `option` is enabled, this function applies `func` to all items in `items`
    and stores the results for later retrieval. Otherwise, it returns `func` as-is.

    Args:
        func: The function to apply to each item in the list.
        items: An iterable of items to process.
        precompute: Determines whether to precompute results:
            - None: Use the default precompute setting.
            - bool: Specify directly whether to precompute.
        **kwargs: Additional keyword arguments, including:
            - precompute_list_batch_size: Batch size for precomputing results.

    Returns:
        Callable: A function to retrieve precomputed elements by index, or the original
        `func` if precomputation is disabled.
    """
    if precompute is None:
        precompute = DEFAULT_PRECOMPUTE_LIST

    if precompute:
        precomputed = jax.lax.map(
            func, items, batch_size=kwargs.get("precompute_list_batch_size")
        )

        def get_element(index: int):
            return jax.tree.map(operator.itemgetter(index), precomputed)

        return get_element

    return func
