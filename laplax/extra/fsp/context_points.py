"""Backward-compatibility shim for context_points.

The implementation has moved to `laplax.util.context_points`.
This module re-exports all public names to avoid breaking imports.
"""

from laplax.util.context_points import *  # noqa: F401,F403

