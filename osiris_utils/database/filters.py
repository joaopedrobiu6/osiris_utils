"""Backwards-compatible alias for :mod:`osiris_utils.filters`.

The spatial filters used to live here, next to the database creators that
apply them.  They are now shared with the lazy diagnostic pipeline
(:mod:`osiris_utils.postprocessing.filtering`), which this package cannot be
imported from without a cycle (``database`` imports ``ar``, ``ar`` imports
``postprocessing``), so the module moved one level up.  Importing from
``osiris_utils.database.filters`` keeps working.
"""

from __future__ import annotations

from ..filters import (
    FilterChain,
    GaussianFilter,
    NoFilter,
    SavitzkyGolayFilter,
    SpatialFilter,
    as_filter,
    fd_derivative,
)

__all__ = [
    "FilterChain",
    "GaussianFilter",
    "NoFilter",
    "SavitzkyGolayFilter",
    "SpatialFilter",
    "as_filter",
    "fd_derivative",
]
