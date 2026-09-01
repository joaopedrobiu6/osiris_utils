"""Lazy, per-frame spatial filtering of diagnostics.

The database creators smooth every raw 2-D frame with a
:class:`~osiris_utils.filters.SpatialFilter` *before* any physics is
computed (see :func:`osiris_utils.database.database._load_filtered_fields`).
This module applies the very same filter objects inside the lazy
:class:`~osiris_utils.data.diagnostic.Diagnostic` pipeline: nothing is
smoothed until a timestep is actually asked for, and then only that frame
is smoothed.

    >>> from osiris_utils import SavitzkyGolayFilter, Simulation
    >>> from osiris_utils.postprocessing.filtering import Filtered_Simulation
    >>> sim = Simulation("run/os.2d")
    >>> smooth = Filtered_Simulation(sim, SavitzkyGolayFilter(9, 4))
    >>> smooth["electrons"]["vfl1"][12]  # frame 12, smoothed on the way out

Derivatives are *not* handled here: a smoothing filter also carries its own
single-pass derivative scheme, which
:class:`~osiris_utils.postprocessing.derivative.Derivative_Diagnostic`
applies when it is given ``filter=``.  The database pipeline smooths each
field once and then takes derivatives of the smoothed field, so mirroring
it means wrapping the leaves with :class:`Filtered_Diagnostic` *and*
passing the same filter to every spatial derivative built on top.

Boundaries follow the database convention: the averaging (transverse) axis
is periodic, every other axis is not.  Pass ``periodic_axes`` to override.

Scope
-----
A filtered view smooths **raw OSIRIS quantities** only.  Anything derived
(a product, a derivative, e_vlasov) is already built from smoothed leaves,
so smoothing it again would apply the kernel twice; such diagnostics are
registered on the view with :meth:`Filtered_Simulation.add_diagnostic` and
handed back untouched.  Asking a filtered view for a derived diagnostic
that lives on the *wrapped* simulation raises :class:`KeyError` rather than
silently double-smoothing it — and, since those diagnostics carry the
filter of whoever built them, rather than silently mixing two filters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.diagnostic import Diagnostic
from ..filters import NoFilter, SpatialFilter, as_filter
from .postprocess import PostProcess

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "Filtered_Diagnostic",
    "Filtered_Simulation",
    "Filtered_Species_Handler",
]

#: Metadata carried over from the wrapped diagnostic — smoothing changes the
#: values, never the grid.
_METADATA_ATTRS = ["_dt", "_dx", "_ndump", "_axis", "_nx", "_x", "_grid", "_dim", "_maxiter", "_tunits", "_type", "_iterations"]


def _periodic_flags(ndim: int, periodic_axes: Sequence[int]) -> tuple[bool, ...]:
    """Per-axis boundary flags for a frame with *ndim* axes.

    *periodic_axes* holds OSIRIS 1-indexed axes (``2`` = x2), matching
    ``mft_axis`` everywhere else in the package; the returned tuple is
    0-indexed, as :meth:`SpatialFilter.smooth` expects.  Axes beyond *ndim*
    are ignored, so the same view works for 1-D and 2-D diagnostics.
    """
    axes = {int(a) - 1 for a in periodic_axes}
    return tuple(ax in axes for ax in range(ndim))


class Filtered_Diagnostic(Diagnostic):
    """A diagnostic whose frames are smoothed on the way out.

    Parameters
    ----------
    diagnostic :
        The diagnostic to smooth.
    filter :
        The :class:`~osiris_utils.filters.SpatialFilter` to apply (or a
        sequence of them, chained in order).
    periodic_axes :
        OSIRIS 1-indexed axes with periodic boundaries. Default: ``(2,)``,
        the database convention (longitudinal x1 open, transverse x2
        periodic).

    Notes
    -----
    Spatial slicing (``diag[t, :, 100:200]``) filters the *slice*, not the
    full frame, so the kernel sees the slice edges as domain boundaries —
    the same caveat that already applies to sliced derivatives.
    """

    def __init__(
        self,
        diagnostic: Diagnostic,
        filter: SpatialFilter | Sequence[SpatialFilter] | None,  # noqa: A002
        periodic_axes: Sequence[int] = (2,),
    ) -> None:
        if hasattr(diagnostic, "_species"):
            super().__init__(
                simulation_folder=getattr(diagnostic, "_simulation_folder", None),
                species=getattr(diagnostic, "_species", None),
            )
        else:
            super().__init__(None)

        self.postprocess_name = "FILTER"

        self._filter = as_filter(filter)
        self._periodic_axes = tuple(int(a) for a in periodic_axes)
        self._diag = diagnostic
        self._name = f"Filtered[{getattr(diagnostic, '_name', '?')}, {self._filter!r}]"
        self._data = None
        self._all_loaded = False

        for attr in _METADATA_ATTRS:
            if hasattr(diagnostic, attr):
                setattr(self, attr, getattr(diagnostic, attr))

    @property
    def filter(self) -> SpatialFilter:
        return self._filter

    def _smooth(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        return self._filter.smooth(f, periodic=_periodic_flags(f.ndim, self._periodic_axes))

    def _frame(self, index: int, data_slice: tuple | None = None) -> np.ndarray:
        return self._smooth(self._diag._frame(index, data_slice=data_slice))

    def load_all(self) -> np.ndarray:
        """Smooth every timestep eagerly.

        Frames are smoothed one at a time: the loaded array carries time on
        axis 0, and a filter configured for "every axis" would otherwise
        smooth along it.
        """
        if self._data is not None:
            return self._data

        self._diag.load_all()
        source = np.asarray(self._diag.data, dtype=np.float64)
        self._data = np.stack([self._smooth(source[i]) for i in range(source.shape[0])])
        self._all_loaded = True
        return self._data


class Filtered_Species_Handler:
    """Species-level filtered view. Built by :class:`Filtered_Simulation`."""

    def __init__(self, species_handler: Any, filter: SpatialFilter, periodic_axes: Sequence[int]) -> None:  # noqa: A002
        self._species_handler = species_handler
        self._filter = filter
        self._periodic_axes = tuple(periodic_axes)
        self._filtered: dict[str, Filtered_Diagnostic] = {}
        self._custom: dict[str, Diagnostic] = {}

    def __getitem__(self, key: str) -> Diagnostic:
        if key in self._custom:
            return self._custom[key]
        if key not in self._filtered:
            self._filtered[key] = Filtered_Diagnostic(
                _raw_quantity(self._species_handler, key),
                self._filter,
                self._periodic_axes,
            )
        return self._filtered[key]

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Register a diagnostic on this view, exempt from smoothing.

        Use it for anything built *from* this view (products, derivatives,
        e_vlasov): the smoothing already happened in its leaves.  Stored
        here rather than on the wrapped species handler, so two views with
        different filters cannot pick up each other's composites.
        """
        return _add_custom(self._custom, diagnostic, name)

    def delete_all_diagnostics(self) -> None:
        self._filtered = {}
        self._custom = {}

    @property
    def species(self) -> Any:
        return getattr(self._species_handler, "species", None)

    @property
    def loaded_diagnostics(self) -> dict[str, Diagnostic]:
        return {**self._filtered, **self._custom}


class Filtered_Simulation(PostProcess):
    """Simulation-like view whose raw quantities come out smoothed.

    Parameters
    ----------
    simulation :
        The :class:`~osiris_utils.data.simulation.Simulation` (or another
        Simulation-like wrapper) to view.
    filter :
        The :class:`~osiris_utils.filters.SpatialFilter` to apply, or a
        sequence of them (chained in order).  ``None`` / ``()`` gives
        :class:`~osiris_utils.filters.NoFilter`, i.e. an identity view.
    mft_axis :
        Averaging axis (OSIRIS 1-indexed).  Used only to pick the default
        periodic axis, matching the database convention.
    periodic_axes :
        Explicit list of periodic axes (OSIRIS 1-indexed), overriding
        *mft_axis*.

    Examples
    --------
    >>> smooth = Filtered_Simulation(sim, GaussianFilter(sigma=2.0))
    >>> smooth["b3"][10]                 # smoothed field frame
    >>> smooth["electrons"]["vfl1"][10]  # smoothed moment frame
    """

    def __init__(
        self,
        simulation: Any,
        filter: SpatialFilter | Sequence[SpatialFilter] | None = None,  # noqa: A002
        mft_axis: int = 2,
        periodic_axes: Sequence[int] | None = None,
    ) -> None:
        filt = as_filter(filter)
        super().__init__(f"Filtered({filt!r})", simulation)

        self._filter = filt
        self._periodic_axes = (int(mft_axis),) if periodic_axes is None else tuple(int(a) for a in periodic_axes)
        if isinstance(filt, NoFilter):
            logger.debug("Filtered_Simulation built with NoFilter: frames pass through unchanged.")

        self._filtered: dict[str, Filtered_Diagnostic] = {}
        self._custom: dict[str, Diagnostic] = {}
        self._species_handler: dict[str, Filtered_Species_Handler] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._species:
            if key not in self._species_handler:
                self._species_handler[key] = Filtered_Species_Handler(
                    self._simulation[key],
                    self._filter,
                    self._periodic_axes,
                )
            return self._species_handler[key]

        if key in self._custom:
            return self._custom[key]
        if key not in self._filtered:
            self._filtered[key] = Filtered_Diagnostic(
                _raw_quantity(self._simulation, key),
                self._filter,
                self._periodic_axes,
            )
        return self._filtered[key]

    def add_diagnostic(self, diagnostic: Diagnostic, name: str | None = None) -> str:
        """Register a diagnostic on this view, exempt from smoothing.

        Stored on the view, **not** delegated to the wrapped simulation:
        the value of a derived diagnostic depends on the filter it was
        built with, so it belongs to this view alone.  Register it on the
        underlying simulation as well if other code should see it.
        """
        return _add_custom(self._custom, diagnostic, name)

    def delete_all_diagnostics(self) -> None:
        self._filtered = {}
        self._custom = {}
        self._species_handler = {}

    def delete_diagnostic(self, key: str) -> None:
        for store in (self._filtered, self._custom, self._species_handler):
            if key in store:
                del store[key]
                return
        logger.warning("Diagnostic '%s' not found in %s.", key, self._name)

    @property
    def filter(self) -> SpatialFilter:
        return self._filter

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        return self._periodic_axes

    @property
    def loaded_diagnostics(self) -> dict[str, Diagnostic]:
        return {**self._filtered, **self._custom}


def _add_custom(store: dict[str, Diagnostic], diagnostic: Diagnostic, name: str | None) -> str:
    if not isinstance(diagnostic, Diagnostic):
        raise ValueError("Only Diagnostic objects are supported")
    if name is None:
        i = 1
        while f"custom_diag_{i}" in store:
            i += 1
        name = f"custom_diag_{i}"
    store[name] = diagnostic
    return name


def _raw_quantity(container: Any, key: str) -> Diagnostic:
    """Fetch *key* from *container*, refusing anything but a raw quantity.

    A file-backed diagnostic records the quantity it was opened for in
    ``_quantity``; anything derived carries whichever operand's metadata it
    was cloned from.  Smoothing a derived diagnostic would apply the kernel
    on top of the one its leaves already carry, so it is refused instead.
    """
    diag = container[key]
    if getattr(diag, "_quantity", None) != key:
        raise KeyError(
            f"'{key}' is a derived diagnostic on the wrapped simulation, not a raw OSIRIS quantity. "
            f"A filtered view only smooths raw quantities — a derived one is already built from "
            f"smoothed leaves. Register it on the view with add_diagnostic() instead."
        )
    return diag
